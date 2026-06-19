"""Phase 1 runtime-correction helpers built on top of kairos.intercept.

This package intentionally does not extend the interceptor engine. It gives
hosts a small experiment harness for the Phase 1 thesis probe: block a narrow
write-tool pattern, inject a synthetic correction once, fail open on a bad
retry, and remember successful retry shapes for later review.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from kairos.intercept import (
    Gate,
    GateEvaluation,
    GateStatus,
    InterceptResult,
    KairosInterceptor,
    SessionContext,
)
from kairos.runtime_correction.primitives import (
    InterventionAwarenessTracker,
    RetryBudget,
    RetryBudgetDecision,
    SingleCallCorrection,
    build_single_call_correction_artifact,
    matches_single_call_suggestion,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

PHASE1_PATTERN_ID = "ungrounded_id_before_write.v0"

AIRLINE_MUTATION_TOOLS = frozenset(
    {
        "book_reservation",
        "cancel_reservation",
        "send_certificate",
        "update_reservation_baggages",
        "update_reservation_flights",
        "update_reservation_passengers",
    }
)

_DEFAULT_GATE_ERROR_STRING = (
    "Error (kairos correction): tool call references an identifier not grounded in trusted context."
)
_GROUNDED_LIST_LIMIT = 10
_KWARGS_ECHO_LIMIT = 240

_RETRY_BUDGETS_KEY = "phase1_retry_budgets"
_PENDING_INTERVENTIONS_KEY = "phase1_pending_interventions"
_FAIL_OPEN_MISSES_KEY = "phase1_fail_open_misses"
_WINNING_PATH_MEMORY_KEY = "phase1_winning_path_memory"


class CorrectionAction(enum.StrEnum):
    """Host action to take after a Phase 1 pre-tool-call check."""

    EXECUTE = "execute"
    INJECT_CORRECTION = "inject_correction"
    EXECUTE_FAIL_OPEN = "execute_fail_open"


@dataclass(frozen=True)
class CorrectionDecision:
    """Decision returned to the host before it executes a tool."""

    action: CorrectionAction
    intercept_result: InterceptResult
    correction_artifact: str | None = None
    retry_allowed: bool = False
    retry_key: str | None = None
    fired_gate: GateEvaluation | None = None
    pattern_id: str | None = None


def find_ungrounded_id_values(kwargs: dict[str, Any], ctx: SessionContext) -> tuple[str, ...]:
    """Return unique ID-like kwarg values not grounded in trusted context."""
    known_ids = ctx.read_cache | ctx.user_supplied_ids
    missing: list[str] = []
    seen: set[str] = set()
    for value in iter_id_values(kwargs):
        if value in known_ids or value in seen:
            continue
        missing.append(value)
        seen.add(value)
    return tuple(missing)


def ungrounded_id_before_write(kwargs: dict[str, Any], ctx: SessionContext) -> bool:
    """Deterministic Phase 1 detector for write-tool ID provenance."""
    return bool(find_ungrounded_id_values(kwargs, ctx))


def build_ungrounded_id_before_write_gates(
    *,
    target_tools: Iterable[str],
    status: GateStatus = GateStatus.SHADOW,
    error_string: str = _DEFAULT_GATE_ERROR_STRING,
    timeout_ms: int = 500,
) -> list[Gate]:
    """Build one targeted gate per write tool.

    Phase 1 should not activate broad wildcard gates. Hosts pass explicit
    mutation tools so read tools remain available for verification.
    """
    tools = tuple(dict.fromkeys(target_tools))
    if not tools:
        raise ValueError("target_tools must contain at least one write tool")
    if "*" in tools:
        raise ValueError("Phase 1 ungrounded-ID correction must target explicit write tools, not '*'")

    return [
        Gate(
            gate_id=PHASE1_PATTERN_ID,
            gate_type="deterministic",
            target_tool=tool,
            evaluation_logic=ungrounded_id_before_write,
            error_string=error_string,
            status=status,
            timeout_ms=timeout_ms,
        )
        for tool in tools
    ]


def build_ungrounded_id_correction_artifact(
    *,
    kwargs: dict[str, Any],
    ungrounded_values: tuple[str, ...],
    ctx: SessionContext,
    winning_path_hint: str | None = None,
) -> str:
    """Build the synthetic tool-error artifact injected by a host.

    Mirrors the shape of phase1_recovery exact-family artifacts: name the
    failure family, echo the actual ungrounded values, list grounded IDs
    available from prior reads, summarize the call, and prescribe a recipe.
    """
    flagged = ", ".join(ungrounded_values) if ungrounded_values else "none"
    grounded_sorted = sorted(ctx.read_cache | ctx.user_supplied_ids)
    if len(grounded_sorted) > _GROUNDED_LIST_LIMIT:
        truncated = ", ".join(grounded_sorted[:_GROUNDED_LIST_LIMIT])
        overflow = len(grounded_sorted) - _GROUNDED_LIST_LIMIT
        grounded_display = f"{truncated}, ... (+{overflow} more)"
    else:
        grounded_display = ", ".join(grounded_sorted) if grounded_sorted else "none"
    kwargs_echo = _format_kwargs_echo(kwargs)

    artifact = (
        "Error (kairos correction): Do not execute this tool call as written.\n\n"
        "What went wrong:\n"
        "The proposed call references identifier(s) not grounded in trusted context.\n\n"
        "Trusted evidence:\n"
        f"- Ungrounded ID(s) in this call: {flagged}\n"
        f"- Grounded IDs available from prior reads: {grounded_display}\n"
        f"- Tool kwargs: {kwargs_echo}\n\n"
        "What to do next:\n"
        "1. For each ungrounded ID listed above, replace it with a grounded ID "
        "from the available list that matches the user's stated intent.\n"
        "2. If no listed ID matches, call the right read tool FIRST "
        "(get_user_details for user/payment IDs, get_reservation_details for "
        "reservation IDs, list_* for searches) and use an ID it returns.\n"
        "3. Do NOT retry with the same ungrounded value — the result will be "
        "identical to this error."
    )
    if winning_path_hint:
        artifact = f"{artifact}\n\nPrior successful recovery: {winning_path_hint}"
    return artifact


def _format_kwargs_echo(kwargs: dict[str, Any]) -> str:
    try:
        rendered = json.dumps(kwargs, default=str, separators=(", ", ": "), ensure_ascii=False)
    except (TypeError, ValueError):
        rendered = repr(kwargs)
    if len(rendered) > _KWARGS_ECHO_LIMIT:
        return rendered[: _KWARGS_ECHO_LIMIT - 1] + "…"
    return rendered


class Phase1CorrectionController:
    """Host-side Phase 1 controller over an existing KairosInterceptor."""

    def __init__(self, interceptor: KairosInterceptor, *, max_retries_per_intervention: int = 1) -> None:
        if max_retries_per_intervention < 0:
            raise ValueError("max_retries_per_intervention must be non-negative")
        self._interceptor = interceptor
        self._max_retries_per_intervention = max_retries_per_intervention

    def before_tool_call(
        self,
        session_id: str,
        tool_name: str,
        kwargs: dict[str, Any],
    ) -> CorrectionDecision:
        """Evaluate a proposed tool call and return the host's next action."""
        result = self._interceptor.evaluate(session_id, tool_name, kwargs)
        ctx = self._interceptor.get_session_context(session_id)

        blocking_gate = _first_blocking_gate(result)
        if blocking_gate is None:
            self._record_winning_path_if_pending(ctx, tool_name, kwargs)
            return CorrectionDecision(action=CorrectionAction.EXECUTE, intercept_result=result)

        retry_key = _retry_key(ctx, blocking_gate.gate_id, tool_name)
        budgets = _retry_budgets(ctx)
        retries_used = budgets.get(retry_key, 0)
        if retries_used >= self._max_retries_per_intervention:
            _append_fail_open_miss(ctx, retry_key, blocking_gate, tool_name, kwargs)
            _pending_interventions(ctx).pop(retry_key, None)
            return CorrectionDecision(
                action=CorrectionAction.EXECUTE_FAIL_OPEN,
                intercept_result=result,
                retry_allowed=False,
                retry_key=retry_key,
                fired_gate=blocking_gate,
            )

        budgets[retry_key] = retries_used + 1
        _pending_interventions(ctx)[retry_key] = {
            "pattern_id": blocking_gate.gate_id,
            "tool_name": tool_name,
            "turn_idx": ctx.turn_idx,
            "original_kwargs": dict(kwargs),
        }
        ungrounded_values = find_ungrounded_id_values(kwargs, ctx)
        artifact = build_ungrounded_id_correction_artifact(
            kwargs=kwargs,
            ungrounded_values=ungrounded_values,
            ctx=ctx,
            winning_path_hint=_winning_path_hint(ctx, tool_name),
        )
        return CorrectionDecision(
            action=CorrectionAction.INJECT_CORRECTION,
            intercept_result=result,
            correction_artifact=artifact,
            retry_allowed=True,
            retry_key=retry_key,
            fired_gate=blocking_gate,
        )

    def _record_winning_path_if_pending(self, ctx: SessionContext, tool_name: str, kwargs: dict[str, Any]) -> None:
        pending = _pending_interventions(ctx)
        for retry_key, intervention in tuple(pending.items()):
            if intervention.get("tool_name") != tool_name:
                continue
            grounded_ids = _grounded_id_values(kwargs, ctx)
            memory = _winning_path_memory(ctx)
            memory.append(
                {
                    "pattern_id": intervention["pattern_id"],
                    "tool_name": tool_name,
                    "turn_idx": intervention["turn_idx"],
                    "original_kwargs": intervention["original_kwargs"],
                    "corrected_kwargs": dict(kwargs),
                    "grounded_ids_used": grounded_ids,
                }
            )
            pending.pop(retry_key, None)
            return


def iter_id_values(node: Any) -> Iterable[str]:
    """Yield every ID-shaped string value reachable from ``node``.

    Walks dicts and lists/tuples recursively; yields values whose key ends in
    ``_id``. This is the single source of truth for what counts as an
    extractable ID — both the gate evaluator and the host-side context
    extractor must use this to stay symmetric.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if key.endswith("_id") and isinstance(value, str):
                yield value
            yield from iter_id_values(value)
    elif isinstance(node, list | tuple):
        for item in node:
            yield from iter_id_values(item)


def _grounded_id_values(kwargs: dict[str, Any], ctx: SessionContext) -> list[str]:
    known_ids = ctx.read_cache | ctx.user_supplied_ids
    values = [value for value in iter_id_values(kwargs) if value in known_ids]
    return sorted(dict.fromkeys(values))


def _first_blocking_gate(result: InterceptResult) -> GateEvaluation | None:
    return next((ev for ev in result.fired_gates if ev.blocked), None)


def _retry_key(ctx: SessionContext, pattern_id: str, tool_name: str) -> str:
    return f"retry_budget:{ctx.session_id}:{pattern_id}:{ctx.turn_idx}:{tool_name}"


def _retry_budgets(ctx: SessionContext) -> dict[str, int]:
    raw = ctx.extras.setdefault(_RETRY_BUDGETS_KEY, {})
    if not isinstance(raw, dict):
        raise TypeError(f"{_RETRY_BUDGETS_KEY} must be a dict")
    return cast("dict[str, int]", raw)


def _pending_interventions(ctx: SessionContext) -> dict[str, dict[str, Any]]:
    raw = ctx.extras.setdefault(_PENDING_INTERVENTIONS_KEY, {})
    if not isinstance(raw, dict):
        raise TypeError(f"{_PENDING_INTERVENTIONS_KEY} must be a dict")
    return cast("dict[str, dict[str, Any]]", raw)


def _winning_path_memory(ctx: SessionContext) -> list[dict[str, Any]]:
    raw = ctx.extras.setdefault(_WINNING_PATH_MEMORY_KEY, [])
    if not isinstance(raw, list):
        raise TypeError(f"{_WINNING_PATH_MEMORY_KEY} must be a list")
    return cast("list[dict[str, Any]]", raw)


def _append_fail_open_miss(
    ctx: SessionContext,
    retry_key: str,
    blocking_gate: GateEvaluation,
    tool_name: str,
    kwargs: dict[str, Any],
) -> None:
    raw = ctx.extras.setdefault(_FAIL_OPEN_MISSES_KEY, [])
    if not isinstance(raw, list):
        raise TypeError(f"{_FAIL_OPEN_MISSES_KEY} must be a list")
    misses = cast("list[dict[str, Any]]", raw)
    misses.append(
        {
            "retry_key": retry_key,
            "pattern_id": blocking_gate.gate_id,
            "tool_name": tool_name,
            "turn_idx": ctx.turn_idx,
            "kwargs": dict(kwargs),
        }
    )


def _winning_path_hint(ctx: SessionContext, tool_name: str) -> str | None:
    for memory in reversed(_winning_path_memory(ctx)):
        if memory.get("tool_name") == tool_name:
            return "a prior retry recovered after reading trusted records and reusing only identifiers from context."
    return None


__all__ = [
    "AIRLINE_MUTATION_TOOLS",
    "CorrectionAction",
    "CorrectionDecision",
    "InterventionAwarenessTracker",
    "PHASE1_PATTERN_ID",
    "Phase1CorrectionController",
    "RetryBudget",
    "RetryBudgetDecision",
    "SingleCallCorrection",
    "build_single_call_correction_artifact",
    "build_ungrounded_id_before_write_gates",
    "build_ungrounded_id_correction_artifact",
    "find_ungrounded_id_values",
    "iter_id_values",
    "matches_single_call_suggestion",
    "ungrounded_id_before_write",
]
