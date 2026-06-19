"""OpenAI-compatible tau-bench agent loop with optional kairos hook points.

This module is the *inner loop* of a tau-bench evaluation: it implements the
two agent strategies tau-bench supports against an OpenRouter-compatible
chat-completions API — :class:`OpenAIToolCallingAgent` (the tool-calling
strategy used in this experiment) and :class:`OpenAIChatReActAgent` (the
legacy chat-only ReAct strategy, kept for parity but not exercised by the
memory-only ablation).

Why this module is the integration point for kairos: the SDK's session
lifecycle maps onto the agent loop's natural boundaries — one
:class:`KairosSession` per task, ``before_tool_call`` per proposed tool
call, ``after_tool_result`` per environment observation. Wiring kairos at
``benchmark.py`` level would require intercepting the agent's internal
state; wiring it here keeps the SDK touch to five call sites and leaves
the kairos-disabled control path completely unchanged.

Inputs: a tau-bench :class:`Env`, an optional :class:`KairosHost` handle
constructed by ``benchmark.py``, and a task index. Per-turn inputs are the
LLM chat completions response (tool-calls or assistant content) and the
tau-bench environment's step observations.

Outputs: a :class:`SolveResult` carrying the final reward, info dict, and
the full message list (used downstream for checkpoint persistence and the
post-hoc trace debugger).

Feature flags consulted here:
  - ``tau_harness.feature_flags.plan_injection_enabled`` — whether to
    splice the kairos session plan into the system message before the
    first agent turn. Default on; the only memory-related host toggle.

How this plugs in: instantiated by ``benchmark.py::run_benchmark``;
consumes the host handle constructed in ``_build_kairos_host``. When
``host`` is ``None`` the agent runs ``_solve_plain``, a literal copy of
the loop without any SDK calls (used for ``baseline_no_kairos`` runs).
The semantic snapshot read at session start carries the agent plan
rendered by ``kairos.semantic_recovery.plan.build_agent_facing_plan`` —
that string is the entire payload of the memory-only experiment.
"""

import json
from pathlib import Path
from typing import Any

from kairos.host import KairosHost
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import Action, RESPOND_ACTION_FIELD_NAME, RESPOND_ACTION_NAME, SolveResult

from tau_harness import feature_flags as flags
from tau_harness.openai_compat import (
    build_client,
    call_with_retry,
    chat_kwargs,
    is_nvidia_openai_provider,
    log_api_error,
    openrouter_fallback_enabled,
    openrouter_fallback_model,
    rate_limit_retry_count_for_nvidia_fallback,
    wait_for_rate_limit,
)


RETRYABLE_ERRORS = (APITimeoutError, APIConnectionError, InternalServerError, RateLimitError)


def kairos_plan_injection_enabled() -> bool:
    """Backwards-compatible alias for :func:`tau_harness.feature_flags.plan_injection_enabled`."""
    return flags.plan_injection_enabled()


class ToolArgumentsJSONError(ValueError):
    def __init__(self, tool_name: str, raw_arguments: Any, original_error: Exception) -> None:
        super().__init__(f"Invalid JSON arguments for tool {tool_name}: {original_error}")
        self.tool_name = tool_name
        self.raw_arguments = raw_arguments
        self.original_error = original_error


def message_to_action(message: dict[str, Any]) -> Action:
    tool_calls = message.get("tool_calls") or []
    if tool_calls and tool_calls[0].get("function") is not None:
        tool_call = tool_calls[0]
        tool_name = tool_call["function"]["name"]
        raw_arguments = tool_call["function"].get("arguments") or "{}"
        try:
            kwargs = json.loads(raw_arguments)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ToolArgumentsJSONError(tool_name, raw_arguments, exc) from exc
        # Unwrap double-encoded JSON: some models (notably kimi-k2 via
        # OpenRouter) intermittently emit tool arguments as a JSON STRING
        # whose contents are a JSON OBJECT — e.g.
        #     '"{\\"reservation_id\\": \\"DF89BM\\"}"'
        # parses to the string '{"reservation_id": "DF89BM"}' which then
        # parses to the intended dict. Without this unwrap the agent gets
        # stuck in identical-retry loops on the same malformed payload
        # (observed eating ~30 of 42 turns on tasks 1, 26, 33 in the
        # Phase 2 ablation — see docs/phase2.5-parallel-investigation-
        # findings.md §3).
        if isinstance(kwargs, str):
            try:
                unwrapped = json.loads(kwargs)
            except (json.JSONDecodeError, TypeError):
                unwrapped = None
            if isinstance(unwrapped, dict):
                kwargs = unwrapped
        if not isinstance(kwargs, dict):
            raise ToolArgumentsJSONError(
                tool_name,
                raw_arguments,
                TypeError(f"expected JSON object, got {type(kwargs).__name__}"),
            )
        return Action(
            name=tool_name,
            kwargs=kwargs,
        )
    return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message.get("content") or ""})


def append_invalid_tool_arguments_error(
    messages: list[dict[str, Any]],
    next_message: dict[str, Any],
    exc: ToolArgumentsJSONError,
) -> None:
    """Ask the model to repair malformed tool-call arguments instead of crashing."""
    next_message["tool_calls"] = (next_message.get("tool_calls") or [])[:1]
    if not next_message["tool_calls"]:
        raise exc
    messages.extend(
        [
            next_message,
            {
                "role": "tool",
                "tool_call_id": next_message["tool_calls"][0]["id"],
                "name": exc.tool_name,
                "content": (
                    "Error: invalid tool arguments JSON. Retry the same tool call with a valid JSON object "
                    "matching the tool schema. Do not wrap JSON in markdown. Parser error: "
                    f"{exc.original_error}"
                ),
            },
        ]
    )


def task_instruction_from_reset_info(info: dict[str, Any], fallback: str) -> str:
    task = info.get("task")
    if isinstance(task, dict) and isinstance(task.get("instruction"), str):
        return str(task["instruction"])
    return fallback


def write_semantic_session_artifact(
    *,
    run_dir: Path | None,
    session_id: str,
    task_index: int | None,
    task_instruction: str,
    semantic_snapshot: Any,
) -> None:
    if run_dir is None or semantic_snapshot is None:
        return
    output_dir = run_dir / "semantic_sessions"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_session_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in session_id)
    payload = {
        "session_id": session_id,
        "task_index": task_index,
        "task_instruction": task_instruction,
        "semantic_snapshot": semantic_snapshot.model_dump(),
    }
    (output_dir / f"{safe_session_id}.json").write_text(json.dumps(payload, indent=2, default=str))


class OpenAIToolCallingAgent(Agent):
    def __init__(
        self,
        tools_info: list[dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        host: KairosHost | None = None,
    ) -> None:
        self.client = build_client(provider)
        self.fallback_client = build_client("openrouter") if openrouter_fallback_enabled(provider) else None
        self.fallback_model = openrouter_fallback_model(model) if self.fallback_client is not None else None
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.temperature = temperature
        self.host = host
        self.provider = provider

    def _llm_call(self, messages: list[dict[str, Any]]) -> Any:
        return call_with_retry(
            "Agent",
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools_info,
                **chat_kwargs("TAU_BENCH_", self.temperature, provider=self.provider),
            ),
            rate_limit_retries=(
                rate_limit_retry_count_for_nvidia_fallback()
                if self.fallback_client is not None and is_nvidia_openai_provider(self.provider)
                else None
            ),
            fallback_fn=(
                None
                if self.fallback_client is None or self.fallback_model is None
                else lambda: self.fallback_client.chat.completions.create(
                    model=self.fallback_model,
                    messages=messages,
                    tools=self.tools_info,
                    **chat_kwargs("TAU_BENCH_", self.temperature, provider=self.provider),
                )
            ),
            fallback_label=(
                f"OpenRouter fallback ({self.fallback_model})"
                if self.fallback_model is not None
                else "OpenRouter fallback"
            ),
        )

    def solve(
        self,
        env: Env,
        task_index: int | None = None,
        max_num_steps: int = 30,
        *,
        session_id: str | None = None,
    ) -> SolveResult:
        if session_id is None:
            session_id = f"task-{task_index}"
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        task_instruction = task_instruction_from_reset_info(info, obs)
        reward = 0.0
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]

        # No kairos? Plain loop, no SDK calls.
        if self.host is None:
            return self._solve_plain(env, messages, max_num_steps)

        # Kairos session: 5 SDK call sites in the entire loop.
        with self.host.session(
            session_id,
            user_instruction=task_instruction,
            tool_schemas=self.tools_info,
        ) as session:
            # Inject the per-session agent plan into the system prompt if the
            # semantic-recovery runtime produced one.
            snapshot = session.semantic_snapshot
            if snapshot is not None:
                write_semantic_session_artifact(
                    run_dir=self.host.run_path,
                    session_id=session_id,
                    task_index=task_index,
                    task_instruction=task_instruction,
                    semantic_snapshot=snapshot,
                )
                if flags.plan_injection_enabled() and getattr(snapshot, "agent_plan", None) is not None:
                    messages[0]["content"] = (
                        f"{messages[0]['content']}\n\n"
                        "# Kairos Session Plan\n"
                        f"{snapshot.agent_plan.artifact}"
                    )

            try:
                for _ in range(max_num_steps):
                    try:
                        wait_for_rate_limit("agent")
                        res = self._llm_call(messages)
                    except RETRYABLE_ERRORS as exc:
                        if not isinstance(exc, RateLimitError):
                            log_api_error("Agent", exc)
                        raise

                    next_message = res.choices[0].message.model_dump(exclude_none=True)
                    # T-05 (2026-05-20): feed assistant content to kairos so
                    # AP-02 (QuoteWarningAsPolicyBreaker) can substring-match
                    # the agent's words against the injected plan's
                    # absolute-claim lines. Called on every assistant message
                    # — respond turns AND tool-call turns — because the
                    # smoking-gun quote often lands in a respond and the
                    # harmful tool call comes later with empty content.
                    if next_message.get("content"):
                        session.record_assistant_message(next_message["content"])
                    try:
                        action = message_to_action(next_message)
                    except ToolArgumentsJSONError as exc:
                        print(f"tool_call_parse: {exc}")
                        append_invalid_tool_arguments_error(messages, next_message, exc)
                        continue

                    if action.name != RESPOND_ACTION_NAME:
                        # T-05 (2026-05-20): pass assistant content so AP-02
                        # (QuoteWarningAsPolicyBreaker) can substring-match it
                        # against the rendered plan's absolute-claim lines.
                        assistant_content = next_message.get("content") or ""
                        decision = session.before_tool_call(
                            action.name, action.kwargs, assistant_content=assistant_content
                        )
                        if decision.action == "inject_correction":
                            next_message["tool_calls"] = next_message["tool_calls"][:1]
                            messages.extend(
                                [
                                    next_message,
                                    {
                                        "role": "tool",
                                        "tool_call_id": next_message["tool_calls"][0]["id"],
                                        "name": next_message["tool_calls"][0]["function"]["name"],
                                        "content": decision.correction_artifact or "",
                                    },
                                ]
                            )
                            continue
                        # execute or execute_fail_open: fall through to env.step.

                    env_response = env.step(action)
                    reward = env_response.reward
                    info = {**info, **env_response.info.model_dump()}

                    if action.name != RESPOND_ACTION_NAME:
                        next_message["tool_calls"] = next_message["tool_calls"][:1]
                        session.after_tool_result(
                            action.name, action.kwargs, env_response.observation
                        )
                        messages.extend(
                            [
                                next_message,
                                {
                                    "role": "tool",
                                    "tool_call_id": next_message["tool_calls"][0]["id"],
                                    "name": next_message["tool_calls"][0]["function"]["name"],
                                    "content": env_response.observation,
                                },
                            ]
                        )
                    else:
                        session.record_user_turn(env_response.observation)
                        messages.extend(
                            [next_message, {"role": "user", "content": env_response.observation}]
                        )

                    if env_response.done:
                        break
            finally:
                session.end(reward=reward, info=info)

        return SolveResult(reward=reward, info=info, messages=messages, total_cost=None)

    def _solve_plain(
        self,
        env: Env,
        messages: list[dict[str, Any]],
        max_num_steps: int,
    ) -> SolveResult:
        """Kairos-disabled loop. Identical control flow, no SDK calls."""
        reward = 0.0
        info: dict[str, Any] = {}
        for _ in range(max_num_steps):
            try:
                wait_for_rate_limit("agent")
                res = self._llm_call(messages)
            except RETRYABLE_ERRORS as exc:
                if not isinstance(exc, RateLimitError):
                    log_api_error("Agent", exc)
                raise
            next_message = res.choices[0].message.model_dump(exclude_none=True)
            try:
                action = message_to_action(next_message)
            except ToolArgumentsJSONError as exc:
                print(f"tool_call_parse: {exc}")
                append_invalid_tool_arguments_error(messages, next_message, exc)
                continue
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                messages.extend(
                    [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": env_response.observation,
                        },
                    ]
                )
            else:
                messages.extend(
                    [next_message, {"role": "user", "content": env_response.observation}]
                )
            if env_response.done:
                break
        return SolveResult(reward=reward, info=info, messages=messages, total_cost=None)


REACT_INSTRUCTION = """
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision making. Do not include extra lines.>
Action:
{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}
"""


ACT_INSTRUCTION = """
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:
Action:
{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}
"""


class OpenAIChatReActAgent(Agent):
    def __init__(
        self,
        tools_info: list[dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        use_reasoning: bool = True,
        temperature: float = 0.0,
    ) -> None:
        instruction = REACT_INSTRUCTION if use_reasoning else ACT_INSTRUCTION
        self.prompt = wiki + "\n#Available tools\n" + json.dumps(tools_info) + instruction
        self.client = build_client(provider)
        self.fallback_client = build_client("openrouter") if openrouter_fallback_enabled(provider) else None
        self.fallback_model = openrouter_fallback_model(model) if self.fallback_client is not None else None
        self.model = model
        self.temperature = temperature
        self.provider = provider

    def generate_next_step(self, messages: list[dict[str, Any]]) -> tuple[dict[str, Any], Action]:
        try:
            wait_for_rate_limit("agent")
            res = call_with_retry(
                "Agent",
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **chat_kwargs("TAU_BENCH_", self.temperature, provider=self.provider),
                ),
                rate_limit_retries=(
                    rate_limit_retry_count_for_nvidia_fallback()
                    if self.fallback_client is not None and is_nvidia_openai_provider(self.provider)
                    else None
                ),
                fallback_fn=(
                    None
                    if self.fallback_client is None or self.fallback_model is None
                    else lambda: self.fallback_client.chat.completions.create(
                        model=self.fallback_model,
                        messages=messages,
                        **chat_kwargs("TAU_BENCH_", self.temperature, provider=self.provider),
                    )
                ),
                fallback_label=(
                    f"OpenRouter fallback ({self.fallback_model})"
                    if self.fallback_model is not None
                    else "OpenRouter fallback"
                ),
            )
        except RETRYABLE_ERRORS as exc:
            if not isinstance(exc, RateLimitError):
                log_api_error("Agent", exc)
            raise
        message = res.choices[0].message
        content = message.content or ""
        action_str = content.split("Action:")[-1].strip()
        try:
            parsed = json.loads(action_str)
        except json.JSONDecodeError:
            parsed = {"name": RESPOND_ACTION_NAME, "arguments": {RESPOND_ACTION_FIELD_NAME: action_str}}
        return message.model_dump(exclude_none=True), Action(name=parsed["name"], kwargs=parsed["arguments"])

    def solve(self, env: Env, task_index: int | None = None, max_num_steps: int = 30) -> SolveResult:
        response = env.reset(task_index=task_index)
        reward = 0.0
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": response.observation},
        ]
        info: dict[str, Any] = {}
        for _ in range(max_num_steps):
            message, action = self.generate_next_step(messages)
            response = env.step(action)
            obs = response.observation
            reward = response.reward
            info = {**info, **response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                obs = "API output: " + obs
            messages.extend([message, {"role": "user", "content": obs}])
            if response.done:
                break
        return SolveResult(messages=messages, reward=reward, info=info)
