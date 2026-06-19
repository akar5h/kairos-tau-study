from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Protocol

from pydantic import SecretStr, ValidationError

from kairos.config import settings
from kairos.models.semantic_recovery import (
    IntentTemplate,
    MemoryRetrievalResult,
    PolicyPack,
    SessionExpectation,
)
from kairos.semantic_recovery.memory import _intent_action_class, extract_constraints

DEFAULT_SEMANTIC_MODEL = "openai/gpt-4o-mini"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ExpectationLLMClient(Protocol):
    """Minimal strict-JSON client used by the expectation builder."""

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return a JSON object as text."""


class SessionExpectationBuilder:
    """Build one session-start expectation from policy, user intent, and memory.

    Three-layer robustness against weak/empty LLM responses:

      1. **Strict-JSON one-shot**: the LLM is asked once with the structured
         system prompt.
      2. **Re-prompt on empty critical fields**: if the LLM returned
         ``expected_terminal_actions=[]`` while the user instruction has a
         recognizable intent class, re-prompt once with a sharpened
         instruction and the candidate tools listed. Bounded by a single
         retry so we never burn unbounded LLM budget.
      3. **Deterministic intent-template fallback**: if the LLM still
         returns empty critical fields, fill them from the host-supplied
         ``intent_template_map``. The template only fills empty fields; it
         never overrides values the LLM produced.
    """

    def __init__(
        self,
        *,
        client: ExpectationLLMClient | None = None,
        intent_template_map: dict[str, IntentTemplate] | None = None,
    ) -> None:
        self.client = client
        self.intent_template_map = intent_template_map

    def build(
        self,
        *,
        policy_pack: PolicyPack,
        user_instruction: str,
        tool_descriptions: list[dict[str, Any]],
        memory_hits: list[MemoryRetrievalResult],
        deterministic_gate_ids: list[str],
    ) -> SessionExpectation | None:
        """Return a parsed expectation, or fail open when the SLM output is invalid."""
        if self.client is None:
            return None
        try:
            primary = self._llm_call(
                policy_pack=policy_pack,
                user_instruction=user_instruction,
                tool_descriptions=tool_descriptions,
                memory_hits=memory_hits,
                deterministic_gate_ids=deterministic_gate_ids,
                sharpen=False,
            )
            if primary is None:
                return None

            intent_class = _intent_action_class(user_instruction, [])

            # Layer 2: re-prompt once if the critical field is missing AND
            # the user has a recognizable intent class. Skips when the intent
            # is unknown (no signal for the judge to ground on) or when the
            # LLM already returned a non-empty value.
            if not primary.expected_terminal_actions and intent_class != "unknown":
                retry = self._llm_call(
                    policy_pack=policy_pack,
                    user_instruction=user_instruction,
                    tool_descriptions=tool_descriptions,
                    memory_hits=memory_hits,
                    deterministic_gate_ids=deterministic_gate_ids,
                    sharpen=True,
                    intent_class=intent_class,
                )
                if retry is not None:
                    primary = retry

            static_expectation = build_static_session_expectation(
                policy_pack=policy_pack,
                user_instruction=user_instruction,
                memory_hits=memory_hits,
                intent_template_map=self.intent_template_map,
            )
            return _merge_expectations(primary, static_expectation)
        except (json.JSONDecodeError, OSError, RuntimeError, TypeError, ValidationError, ValueError):
            return None

    def _llm_call(
        self,
        *,
        policy_pack: PolicyPack,
        user_instruction: str,
        tool_descriptions: list[dict[str, Any]],
        memory_hits: list[MemoryRetrievalResult],
        deterministic_gate_ids: list[str],
        sharpen: bool,
        intent_class: str | None = None,
    ) -> SessionExpectation | None:
        assert self.client is not None
        system_prompt = _sharpened_system_prompt(intent_class) if sharpen else _system_prompt()
        user_prompt = _user_prompt(
            policy_pack=policy_pack,
            user_instruction=user_instruction,
            tool_descriptions=tool_descriptions,
            memory_hits=memory_hits,
            deterministic_gate_ids=deterministic_gate_ids,
            sharpen=sharpen,
        )
        raw = self.client.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return None
        return SessionExpectation.model_validate(parsed)


class OpenRouterExpectationClient:
    """Tiny OpenRouter JSON client used by the session-start expectation
    builder and the runtime drift detector.

    Default model is mid-tier (``openai/gpt-4o-mini``) — strong enough to fill
    the structured expectation schema reliably and to reason about drift, weak
    enough to stay cheap (~$0.001 per call). Override per-deployment with
    ``KAIROS_SEMANTIC_MODEL``.
    """

    def __init__(
        self,
        *,
        api_key: SecretStr,
        model: str = DEFAULT_SEMANTIC_MODEL,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        timeout_s: float = 60.0,
        temperature: float = 0.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.temperature = temperature

    @classmethod
    def from_env(cls) -> OpenRouterExpectationClient | None:
        """Create a client from Kairos/OpenRouter env, or None when disabled."""
        if not _env_enabled("KAIROS_SEMANTIC_EXPECTATION_ENABLED"):
            return None
        provider = os.getenv("KAIROS_SEMANTIC_PROVIDER", settings.semantic_provider)
        if provider.lower() != "openrouter":
            return None
        api_key = _secret_value(settings.semantic_openrouter_api_key) or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return None
        return cls(
            api_key=SecretStr(api_key),
            model=os.getenv("KAIROS_SEMANTIC_MODEL", settings.semantic_model),
            base_url=os.getenv("KAIROS_SEMANTIC_OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL),
            timeout_s=float(os.getenv("KAIROS_SEMANTIC_TIMEOUT_S", str(settings.semantic_timeout_s))),
            temperature=float(os.getenv("KAIROS_SEMANTIC_TEMPERATURE", str(settings.semantic_temperature))),
        )

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> str:
        """Call OpenRouter and return the assistant JSON text."""
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        request = urllib.request.Request(  # noqa: S310
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenRouter expectation request failed: {exc}") from exc
        parsed = json.loads(body)
        try:
            content = parsed["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("OpenRouter expectation response missing message content") from exc
        if not isinstance(content, str):
            raise TypeError("OpenRouter expectation response content was not a string")
        return content


def _system_prompt() -> str:
    return (
        "You build Kairos SessionExpectation JSON for a tool-using agent. "
        "Return exactly one JSON object with keys: user_constraints, likely_workflow, "
        "must_read_tools, allowed_write_tools, forbidden_shortcuts, optimization_target, "
        "expected_terminal_actions, success_lock_conditions, danger_points. Use only provided evidence. "
        "Memory is advisory and lower priority than the current user instruction. "
        "If the user instruction implies a concrete goal (book, update, cancel, etc.), "
        "expected_terminal_actions MUST list at least one write tool that completes that goal."
    )


def _sharpened_system_prompt(intent_class: str | None) -> str:
    """Re-prompt variant that explicitly names the intent class observed in
    the user instruction and demands a non-empty terminal actions list.
    """
    intent_hint = f"observed intent class: {intent_class!r}" if intent_class else "intent class: ambiguous"
    return (
        "You build Kairos SessionExpectation JSON. Your previous response left "
        "expected_terminal_actions empty, but the user instruction has a clear "
        f"action goal ({intent_hint}). Return a fresh JSON object with the same keys "
        "as before. expected_terminal_actions MUST contain at least one write tool "
        "name selected from the provided tool_descriptions that would complete the "
        "user's stated goal. If no write tool matches, set the field to "
        "['transfer_to_human_agents']. Do not leave it empty."
    )


def build_static_session_expectation(
    *,
    policy_pack: PolicyPack,
    user_instruction: str,
    memory_hits: list[MemoryRetrievalResult],
    intent_template_map: dict[str, IntentTemplate] | None = None,
) -> SessionExpectation:
    """Build a deterministic contract when the SLM is disabled or fails open.

    When ``intent_template_map`` is supplied, an intent-class-derived template
    fills critical fields (``expected_terminal_actions``, ``likely_workflow``,
    ``success_lock_conditions``) that the memory-derived path leaves empty.
    The template only fills empties — it never overrides values already
    derived from the memory hits.
    """
    user_constraints = extract_constraints(user_instruction)
    intent_class = _intent_action_class(user_instruction, [])
    template = (intent_template_map or {}).get(intent_class)

    likely_workflow = _likely_workflow(memory_hits)
    if not likely_workflow and template is not None:
        likely_workflow = list(template.likely_workflow)

    terminal_actions = _terminal_actions(likely_workflow)
    if not terminal_actions and template is not None and template.expected_terminal_actions:
        terminal_actions = list(template.expected_terminal_actions)

    success_locks = _success_lock_conditions(terminal_actions)
    if not success_locks and template is not None and template.success_lock_conditions:
        success_locks = list(template.success_lock_conditions)

    must_read_tools = _must_read_tools(policy_pack, likely_workflow)
    return SessionExpectation(
        user_constraints=user_constraints,
        likely_workflow=likely_workflow,
        must_read_tools=must_read_tools,
        allowed_write_tools=_allowed_write_tools(policy_pack),
        forbidden_shortcuts=_forbidden_shortcuts(policy_pack, memory_hits),
        optimization_target=_optional_str(user_constraints.get("optimization")),
        expected_terminal_actions=terminal_actions,
        success_lock_conditions=success_locks,
        danger_points=_danger_points(policy_pack, memory_hits),
    )


def _merge_expectations(
    primary: SessionExpectation,
    fallback: SessionExpectation,
) -> SessionExpectation:
    return SessionExpectation(
        user_constraints={**fallback.user_constraints, **primary.user_constraints},
        likely_workflow=primary.likely_workflow or fallback.likely_workflow,
        must_read_tools=primary.must_read_tools or fallback.must_read_tools,
        allowed_write_tools=primary.allowed_write_tools or fallback.allowed_write_tools,
        forbidden_shortcuts=_merge_lists(fallback.forbidden_shortcuts, primary.forbidden_shortcuts),
        optimization_target=primary.optimization_target or fallback.optimization_target,
        expected_terminal_actions=primary.expected_terminal_actions or fallback.expected_terminal_actions,
        success_lock_conditions=primary.success_lock_conditions or fallback.success_lock_conditions,
        danger_points=_merge_lists(fallback.danger_points, primary.danger_points),
    )


def _merge_lists(first: list[str], second: list[str]) -> list[str]:
    merged: list[str] = []
    for item in [*first, *second]:
        if item not in merged:
            merged.append(item)
    return merged


def _user_prompt(
    *,
    policy_pack: PolicyPack,
    user_instruction: str,
    tool_descriptions: list[dict[str, Any]],
    memory_hits: list[MemoryRetrievalResult],
    deterministic_gate_ids: list[str],
    sharpen: bool = False,
) -> str:
    base_instructions = (
        "Infer the expected workflow and constraints for this one session. Do not invent IDs, dates, prices, or tools. "
    )
    if sharpen:
        instructions = (
            base_instructions + "Your previous response had empty expected_terminal_actions. "
            "Pick at least one terminal write tool from tool_descriptions that "
            "matches the user's stated goal. Do not leave the field empty."
        )
    else:
        instructions = (
            base_instructions + "If uncertain about a non-critical field, leave it empty. "
            "expected_terminal_actions is critical: it must list at least one "
            "write tool when the user has a clear action goal."
        )
    return json.dumps(
        {
            "user_instruction": user_instruction,
            "policy_pack": policy_pack.model_dump(),
            "tool_descriptions": tool_descriptions,
            "retrieved_workflow_memories": [hit.model_dump() for hit in memory_hits],
            "deterministic_gate_ids": deterministic_gate_ids,
            "instructions": instructions,
        },
        sort_keys=True,
        default=str,
    )


def _likely_workflow(memory_hits: list[MemoryRetrievalResult]) -> list[str]:
    if not memory_hits:
        return []
    return list(dict.fromkeys(memory_hits[0].memory.expected_tool_sequence))


def _terminal_actions(workflow: list[str]) -> list[str]:
    return [tool for tool in workflow if _is_terminal_tool(tool)]


def _must_read_tools(policy_pack: PolicyPack, workflow: list[str]) -> list[str]:
    reads: list[str] = []
    candidate_tools = workflow or list(policy_pack.required_read_before_write)
    for tool in candidate_tools:
        for read_tool in policy_pack.required_read_before_write.get(tool, []):
            if read_tool not in reads:
                reads.append(read_tool)
    return reads


def _allowed_write_tools(policy_pack: PolicyPack) -> list[str]:
    tools = list(policy_pack.required_read_before_write)
    for tool in policy_pack.write_tool_constraints:
        if tool not in tools:
            tools.append(tool)
    return tools


def _forbidden_shortcuts(policy_pack: PolicyPack, memory_hits: list[MemoryRetrievalResult]) -> list[str]:
    shortcuts: list[str] = []
    for expectation in policy_pack.static_expectations:
        if expectation not in shortcuts:
            shortcuts.append(expectation)
    for hit in memory_hits[:1]:
        for trap in hit.memory.failure_traps:
            if trap not in shortcuts:
                shortcuts.append(trap)
    return shortcuts


def _danger_points(policy_pack: PolicyPack, memory_hits: list[MemoryRetrievalResult]) -> list[str]:
    danger_points: list[str] = []
    for trap in policy_pack.failure_traps:
        if trap not in danger_points:
            danger_points.append(trap)
    for hit in memory_hits[:1]:
        for trap in hit.memory.failure_traps:
            if trap not in danger_points:
                danger_points.append(trap)
    return danger_points


def _success_lock_conditions(terminal_actions: list[str]) -> list[str]:
    return [f"after {tool} succeeds, do not undo it unless the user changes intent" for tool in terminal_actions]


def _is_terminal_tool(tool_name: str) -> bool:
    return tool_name.startswith(("book_", "update_", "cancel_", "send_")) or tool_name == "transfer_to_human_agents"


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _env_enabled(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _secret_value(value: SecretStr | None) -> str | None:
    if value is None:
        return None
    secret = value.get_secret_value()
    return secret or None
