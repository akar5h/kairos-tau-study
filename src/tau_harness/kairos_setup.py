"""OTel + Phoenix wiring for tau-agent runs."""

import os
from urllib.error import URLError
from urllib.request import urlopen

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.otel import register

PHOENIX_OTLP_ENDPOINT = os.getenv(
    "PHOENIX_OTLP_ENDPOINT",
    "http://localhost:6006/v1/traces",
)

_PROVIDER: TracerProvider | None = None
_OPENAI_INSTRUMENTED = False


def _warn_if_phoenix_unreachable() -> None:
    base = PHOENIX_OTLP_ENDPOINT.removesuffix("/v1/traces")
    try:
        with urlopen(base, timeout=2) as response:
            status = getattr(response, "status", "ok")
            print(f"Phoenix collector reachable at {base} (status={status})")
    except URLError:
        print(
            f"Warning: Phoenix collector is not reachable at {base}. "
            "Start Phoenix first or traces will not appear in the UI."
        )


def install_kairos() -> None:
    global _PROVIDER, _OPENAI_INSTRUMENTED
    _warn_if_phoenix_unreachable()
    tracer_provider = register(
        project_name="tau_agent",
        endpoint=PHOENIX_OTLP_ENDPOINT,
        auto_instrument=False,
    )
    tracer_provider = tracer_provider or trace.get_tracer_provider()
    if not _OPENAI_INSTRUMENTED:
        instrumentor = OpenAIInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)
        print(
            "OpenAI OpenInference instrumented:",
            instrumentor.is_instrumented_by_opentelemetry,
        )
        _OPENAI_INSTRUMENTED = True
    _PROVIDER = tracer_provider


def shutdown_kairos() -> None:
    if _PROVIDER is not None:
        _PROVIDER.force_flush()
        _PROVIDER.shutdown()
