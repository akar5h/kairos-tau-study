"""LLM provider abstraction for the tau-bench host runtime.

This module is the single seam where every outgoing chat-completions call in
the host picks its credentials, endpoint, and client class. It exists so that
the agent loop (`openai_agent.py`), the user simulator (`openai_user.py`), and
the cascade reranker (`cascade_retriever.py`) all funnel through one
``build_client(provider)`` call rather than each reaching for env vars on
their own — making provider swaps a one-string change at the CLI rather than
a code change in three files.

The "provider" here is just a tag string — ``"openai"``, ``"openrouter"``,
or ``"azure"`` — that selects which env vars are read for credentials and
base URL. Azure is handled as a plain OpenAI-compatible endpoint via AI
Foundry's ``/openai/v1`` data plane (not the legacy ``AzureOpenAI`` SDK
class), so a single ``openai.OpenAI`` construction works across all three
providers. Each branch is a few lines; a richer Provider-class hierarchy
would only earn its keep when we add a non-OpenAI-compatible backend
(Bedrock's Converse API would be the trigger).

Inputs: environment variables — ``OPENAI_API_KEY`` / ``OPENROUTER_API_KEY``
/ ``AZURE_OPENAI_API_KEY``, plus per-provider base-URL counterparts. The
Azure key is sent as a bearer token, which the v1 endpoint accepts as an
OpenAI-compatible auth header. Provider tag arrives from CLI args
(``--provider``) plumbed through ``RunConfig.model_provider`` and
``RunConfig.user_model_provider``.

Outputs: a fully-configured ``openai.OpenAI`` instance with retries,
headers, and timeouts wired. Callers use ``.chat.completions.create(model=...)``
identically across providers — Azure interprets ``model`` as a deployment
name (the alias you created in AI Foundry); OpenAI/OpenRouter treat it as a
model identifier. We let the caller pass the right string.

Feature flags consulted here: none. Provider selection is config, not a
gated subsystem — but adding a new provider is the kind of host-wiring
change that belongs to `tau_harness/feature_flags.py` if we ever need
to default it on/off per environment.

How it plugs in: ``run.py::configure_provider_env`` validates required env
vars up front and aborts cleanly on misconfiguration; ``build_client`` is
called downstream by the three consumer modules. Adding a fourth provider
means extending three small functions (``provider_api_key``,
``provider_base_url``, ``build_client``) and updating the CLI choices list.
"""

import json
import os
import random
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

T = TypeVar("T")


SUPPORTED_PROVIDERS = ("openai", "openrouter", "azure")


def provider_api_key(provider: str) -> str:
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY is required for provider=openai.")
        return api_key
    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY is required for provider=openrouter.")
        return api_key
    if provider == "azure":
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("AZURE_OPENAI_API_KEY is required for provider=azure.")
        return api_key
    raise SystemExit(f"Unsupported provider: {provider}")


def provider_base_url(provider: str) -> str | None:
    if provider == "openai":
        return os.getenv("OPENAI_API_BASE")
    if provider == "openrouter":
        return os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    if provider == "azure":
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise SystemExit("AZURE_OPENAI_ENDPOINT is required for provider=azure.")
        # The AI Foundry v1 endpoint is OpenAI-compatible only when the path
        # includes /openai/v1. Auto-append if the user gave us a bare resource
        # URL so we don't 404 on the first request.
        if "/openai/v1" not in endpoint:
            endpoint = endpoint.rstrip("/") + "/openai/v1"
        return endpoint
    raise SystemExit(f"Unsupported provider: {provider}")


def is_nvidia_openai_provider(provider: str) -> bool:
    if provider != "openai":
        return False
    base_url = provider_base_url(provider) or ""
    return "integrate.api.nvidia.com" in base_url


def provider_headers(provider: str) -> dict[str, str] | None:
    if provider != "openrouter":
        return None
    headers: dict[str, str] = {}
    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    title = os.getenv("OPENROUTER_APP_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers or None


def build_client(provider: str) -> OpenAI:
    # Azure on AI Foundry's /openai/v1 endpoint is fully OpenAI-compatible —
    # plain OpenAI client + Azure API key as bearer works, no AzureOpenAI
    # class needed. The legacy {resource}.openai.azure.com data plane is the
    # only thing that would require AzureOpenAI, and we deliberately don't
    # support it (the v1 endpoint is the new default).
    return OpenAI(
        api_key=provider_api_key(provider),
        base_url=provider_base_url(provider),
        default_headers=provider_headers(provider),
        max_retries=sdk_retry_count(),
    )


def env_float(name: str) -> float | None:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else None


def env_int(name: str) -> int | None:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else None


def env_bool(name: str) -> bool | None:
    value = os.getenv(name)
    if value in (None, ""):
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_str(name: str) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else None


def sdk_retry_count() -> int:
    value = env_int("TAU_BENCH_SDK_MAX_RETRIES")
    return 0 if value is None else value


def openrouter_fallback_enabled(provider: str) -> bool:
    enabled = env_bool("TAU_BENCH_OPENROUTER_FALLBACK")
    if enabled is not None:
        return enabled
    return is_nvidia_openai_provider(provider) and bool(os.getenv("OPENROUTER_API_KEY"))


def openrouter_fallback_model(model: str) -> str:
    override = env_str("TAU_BENCH_OPENROUTER_FALLBACK_MODEL")
    if override:
        return override
    aliases = {
        "moonshotai/kimi-k2-instruct": "moonshotai/kimi-k2",
    }
    return aliases.get(model, model)


def env_name(prefix: str, field: str) -> str:
    return f"{prefix}{field}"


def chat_kwargs(
    prefix: str = "TAU_BENCH_",
    temperature: float | None = None,
    *,
    provider: str | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    top_p = env_float(env_name(prefix, "TOP_P")) or env_float("TAU_BENCH_TOP_P")
    if top_p is not None:
        kwargs["top_p"] = top_p
    max_tokens = env_int(env_name(prefix, "MAX_TOKENS")) or env_int("TAU_BENCH_MAX_TOKENS")
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    timeout = env_float(env_name(prefix, "TIMEOUT")) or env_float("TAU_BENCH_TIMEOUT")
    if timeout is not None:
        kwargs["timeout"] = timeout
    extra_body = build_extra_body(prefix, provider=provider)
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


def provider_supports_chat_template_kwargs(provider: str | None) -> bool:
    # ``chat_template_kwargs`` is a vLLM/SGLang convention exposed by OpenRouter
    # and NVIDIA NIM to control reasoning chains on models like kimi-k2. Stock
    # OpenAI servers (including Azure's v1 endpoint) reject unknown extra_body
    # keys with HTTP 400, so we strip it for those providers.
    if provider is None:
        return True  # legacy callers: keep prior behavior
    if provider == "openrouter":
        return True
    if provider == "openai":
        return is_nvidia_openai_provider(provider)
    return False


def build_extra_body(
    prefix: str = "TAU_BENCH_",
    *,
    provider: str | None = None,
) -> dict[str, Any] | None:
    if not provider_supports_chat_template_kwargs(provider):
        return None
    chat_template_kwargs: dict[str, Any] = {}
    thinking = env_bool(env_name(prefix, "THINKING"))
    if thinking is None:
        thinking = env_bool("TAU_BENCH_THINKING")
    if thinking is not None:
        chat_template_kwargs["thinking"] = thinking
    reasoning_effort = os.getenv(env_name(prefix, "REASONING_EFFORT")) or os.getenv("TAU_BENCH_REASONING_EFFORT")
    # NVIDIA's OpenAI-compatible endpoint accepts reasoning controls, but sending
    # them while "thinking" is disabled can still route requests through a slower
    # reasoning path. Keep the user simulator on the simple path by omitting
    # reasoning settings unless thinking is explicitly enabled.
    if reasoning_effort and thinking:
        chat_template_kwargs["reasoning_effort"] = reasoning_effort
    if chat_template_kwargs:
        return {"chat_template_kwargs": chat_template_kwargs}
    return None


def request_settings(prefix: str = "TAU_BENCH_", temperature: float | None = None) -> dict[str, Any]:
    thinking = env_bool(env_name(prefix, "THINKING"))
    if thinking is None:
        thinking = env_bool("TAU_BENCH_THINKING")
    reasoning_effort = env_str(env_name(prefix, "REASONING_EFFORT")) or env_str("TAU_BENCH_REASONING_EFFORT")
    return {
        "temperature": temperature,
        "top_p": env_float(env_name(prefix, "TOP_P")) or env_float("TAU_BENCH_TOP_P"),
        "max_tokens": env_int(env_name(prefix, "MAX_TOKENS")) or env_int("TAU_BENCH_MAX_TOKENS"),
        "timeout": env_float(env_name(prefix, "TIMEOUT")) or env_float("TAU_BENCH_TIMEOUT"),
        "thinking": thinking,
        "reasoning_effort": reasoning_effort if thinking else None,
        "retries": env_int(env_name(prefix, "RETRIES")) or env_int("TAU_BENCH_RETRIES") or 0,
    }


def maybe_json(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except TypeError:
        return str(value)


class _RequestLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self._interval_s = 60.0 / requests_per_minute
        self._next_allowed_at = 0.0
        self._lock = threading.Lock()

    def wait_for_turn(self, context: str) -> None:
        with self._lock:
            now = time.monotonic()
            reserved_at = max(now, self._next_allowed_at)
            self._next_allowed_at = reserved_at + self._interval_s
        delay_s = reserved_at - now
        if delay_s > 0:
            print(f"Rate limiter sleeping {delay_s:.1f}s before {context} request")
            time.sleep(delay_s)


_REQUEST_LIMITER: _RequestLimiter | None = None


def requests_per_minute() -> int:
    value = env_int("TAU_BENCH_REQUESTS_PER_MINUTE")
    return 12 if value in (None, 0) else value


def wait_for_rate_limit(context: str) -> None:
    global _REQUEST_LIMITER
    rpm = requests_per_minute()
    if rpm < 1:
        return
    if _REQUEST_LIMITER is None:
        _REQUEST_LIMITER = _RequestLimiter(requests_per_minute=rpm)
    _REQUEST_LIMITER.wait_for_turn(context)


def should_log_api_headers() -> bool:
    enabled = env_bool("TAU_BENCH_LOG_API_HEADERS")
    return True if enabled is None else enabled


def error_header_snapshot(exc: Exception) -> dict[str, Any]:
    response = getattr(exc, "response", None)
    if response is None or getattr(response, "headers", None) is None:
        return {}
    headers = response.headers
    interesting = [
        "retry-after",
        "x-request-id",
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-tokens",
        "server",
        "date",
    ]
    snapshot = {key: headers.get(key) for key in interesting if headers.get(key) is not None}
    request_id = getattr(exc, "request_id", None)
    if request_id and "x-request-id" not in snapshot:
        snapshot["x-request-id"] = request_id
    status_code = getattr(response, "status_code", None)
    if status_code is not None:
        snapshot["status_code"] = status_code
    return snapshot


def log_api_error(context: str, exc: Exception) -> None:
    if not should_log_api_headers():
        return
    snapshot = error_header_snapshot(exc)
    if snapshot:
        print(f"{context} API error headers: {snapshot}")
    else:
        print(f"{context} API error headers: <none available>")


def rate_limit_retry_count() -> int:
    value = env_int("TAU_BENCH_RATE_LIMIT_RETRIES")
    return 6 if value is None else value


def rate_limit_retry_count_for_nvidia_fallback() -> int:
    value = env_int("TAU_BENCH_NVIDIA_RATE_LIMIT_RETRIES_BEFORE_FALLBACK")
    return 3 if value is None else value


def rate_limit_backoff_base_s() -> float:
    value = env_float("TAU_BENCH_RATE_LIMIT_BACKOFF_BASE")
    return 5.0 if value is None else value


def rate_limit_backoff_max_s() -> float:
    value = env_float("TAU_BENCH_RATE_LIMIT_BACKOFF_MAX")
    return 60.0 if value is None else value


def rate_limit_backoff_jitter_s() -> float:
    value = env_float("TAU_BENCH_RATE_LIMIT_BACKOFF_JITTER")
    return 1.0 if value is None else value


def transient_retry_count() -> int:
    value = env_int("TAU_BENCH_TRANSIENT_RETRIES")
    return 4 if value is None else value


def transient_backoff_base_s() -> float:
    value = env_float("TAU_BENCH_TRANSIENT_BACKOFF_BASE")
    return 1.0 if value is None else value


def transient_backoff_max_s() -> float:
    value = env_float("TAU_BENCH_TRANSIENT_BACKOFF_MAX")
    return 20.0 if value is None else value


def transient_backoff_jitter_s() -> float:
    value = env_float("TAU_BENCH_TRANSIENT_BACKOFF_JITTER")
    return 0.5 if value is None else value


def retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    retry_after = headers.get("retry-after")
    if not retry_after:
        return None
    try:
        return max(0.0, float(retry_after))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(retry_after)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            return None


def _backoff_delay(
    *,
    attempt: int,
    base_s: float,
    max_s: float,
    jitter_s: float,
    exc: Exception | None = None,
) -> float:
    delay_s = min(max_s, base_s * (2**attempt))
    if exc is not None:
        header_delay_s = retry_after_seconds(exc)
        if header_delay_s is not None:
            delay_s = max(delay_s, header_delay_s)
    if jitter_s > 0:
        delay_s += random.uniform(0.0, jitter_s)
    return delay_s


def call_with_retry(
    context: str,
    fn: Callable[[], T],
    *,
    rate_limit_retries: int | None = None,
    transient_retries: int | None = None,
    fallback_fn: Callable[[], T] | None = None,
    fallback_label: str = "OpenRouter fallback",
) -> T:
    max_rate_limit_retries = rate_limit_retry_count() if rate_limit_retries is None else rate_limit_retries
    max_transient_retries = transient_retry_count() if transient_retries is None else transient_retries
    rate_limit_attempt = 0
    transient_attempt = 0
    while True:
        try:
            return fn()
        except RateLimitError as exc:
            log_api_error(context, exc)
            if rate_limit_attempt >= max_rate_limit_retries:
                if fallback_fn is not None:
                    print(f"{context} exhausted primary rate-limit retries; trying {fallback_label}")
                    return fallback_fn()
                raise
            delay_s = _backoff_delay(
                attempt=rate_limit_attempt,
                base_s=rate_limit_backoff_base_s(),
                max_s=rate_limit_backoff_max_s(),
                jitter_s=rate_limit_backoff_jitter_s(),
                exc=exc,
            )
            print(
                f"{context} hit rate limit on attempt {rate_limit_attempt + 1}/{max_rate_limit_retries + 1}; "
                f"retrying in {delay_s:.1f}s"
            )
            time.sleep(delay_s)
            rate_limit_attempt += 1
        except (APITimeoutError, APIConnectionError, InternalServerError) as exc:
            log_api_error(context, exc)
            if transient_attempt >= max_transient_retries:
                raise
            delay_s = _backoff_delay(
                attempt=transient_attempt,
                base_s=transient_backoff_base_s(),
                max_s=transient_backoff_max_s(),
                jitter_s=transient_backoff_jitter_s(),
            )
            print(
                f"{context} transient API failure on attempt {transient_attempt + 1}/{max_transient_retries + 1}: "
                f"{exc}. Retrying in {delay_s:.1f}s"
            )
            time.sleep(delay_s)
            transient_attempt += 1
