"""Generic runtime-correction primitives shared by host integrations.

These helpers intentionally know nothing about a host domain such as airline or
retail. Hosts decide which pattern fired and what the suggested next call is;
Kairos owns the safe single-call artifact shape, retry budgeting, and
post-intervention awareness telemetry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from kairos.intercept import GateEvaluation, GateStatus, KairosInterceptor, SessionContext

CorrectionConfidence = Literal["low", "medium", "high"]

_RETRY_BUDGETS_KEY = "kairos_runtime_retry_budgets"
_PENDING_INTERVENTIONS_KEY = "kairos_runtime_pending_interventions"
_COMPLETED_INTERVENTIONS_KEY = "kairos_runtime_completed_interventions"
_DEFAULT_IGNORED_FOLLOWUP_TOOLS = frozenset({"think"})


@dataclass(frozen=True)
class SingleCallCorrection:
    """One safe runtime correction suggestion.

    Runtime injection is only safe when ``next_kwargs`` is exactly one JSON
    object and ``planner_required`` is false. Multi-step plans may be represented
    elsewhere, but the injected recovery channel must always point to one next
    tool call.
    """

    pattern_id: str
    blocked_summary: str
    next_tool: str
    next_kwargs: dict[str, Any] | None
    why: str
    evidence_refs: dict[str, Any] = field(default_factory=dict)
    confidence: CorrectionConfidence = "medium"
    planner_required: bool = False
    after_success: str | None = None

    @property
    def is_injectable(self) -> bool:
        """Whether this correction is safe to inject as a tool error."""
        return self.next_kwargs is not None and not self.planner_required


@dataclass(frozen=True)
class RetryBudgetDecision:
    """Result of checking or consuming a retry budget."""

    retry_key: str
    retry_allowed: bool
    retries_used: int
    max_retries: int


class RetryBudget:
    """Per-session retry budget backed by ``SessionContext.extras``."""

    def __init__(self, *, max_retries: int = 1, extras_key: str = _RETRY_BUDGETS_KEY) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.max_retries = max_retries
        self.extras_key = extras_key

    def retry_key(self, ctx: SessionContext, *, pattern_id: str, tool_name: str) -> str:
        """Build the stable retry key for a turn-scoped intervention."""
        return f"runtime_retry:{ctx.session_id}:{pattern_id}:{ctx.turn_idx}:{tool_name}"

    def check(self, ctx: SessionContext, *, pattern_id: str, tool_name: str) -> RetryBudgetDecision:
        """Return budget state without mutating it."""
        retry_key = self.retry_key(ctx, pattern_id=pattern_id, tool_name=tool_name)
        retries_used = int(_dict_extra(ctx.extras, self.extras_key).get(retry_key, 0))
        return RetryBudgetDecision(
            retry_key=retry_key,
            retry_allowed=retries_used < self.max_retries,
            retries_used=retries_used,
            max_retries=self.max_retries,
        )

    def consume(self, ctx: SessionContext, *, pattern_id: str, tool_name: str) -> RetryBudgetDecision:
        """Consume one retry if available and return the pre-consumption state."""
        decision = self.check(ctx, pattern_id=pattern_id, tool_name=tool_name)
        if decision.retry_allowed:
            _dict_extra(ctx.extras, self.extras_key)[decision.retry_key] = decision.retries_used + 1
        return decision


class InterventionAwarenessTracker:
    """Track what happens after a Kairos nudge and emit telemetry records."""

    def __init__(
        self,
        interceptor: KairosInterceptor,
        *,
        pending_key: str = _PENDING_INTERVENTIONS_KEY,
        completed_key: str = _COMPLETED_INTERVENTIONS_KEY,
        ignored_followup_tools: frozenset[str] = _DEFAULT_IGNORED_FOLLOWUP_TOOLS,
    ) -> None:
        self._interceptor = interceptor
        self._pending_key = pending_key
        self._completed_key = completed_key
        self._ignored_followup_tools = ignored_followup_tools

    def record_pending(
        self,
        *,
        session_id: str,
        retry_key: str,
        pattern_id: str,
        blocked_tool_name: str,
        blocked_kwargs: dict[str, Any],
        suggested_tool_name: str | None,
        suggested_kwargs: Any,
        confidence: CorrectionConfidence,
        planner_required: bool,
    ) -> dict[str, Any]:
        """Store a pending intervention until the agent takes a real follow-up action."""
        ctx = self._interceptor.get_session_context(session_id)
        pending = {
            "retry_key": retry_key,
            "pattern_id": pattern_id,
            "blocked_tool_name": blocked_tool_name,
            "blocked_kwargs": dict(blocked_kwargs),
            "suggested_tool_name": suggested_tool_name,
            "suggested_kwargs": suggested_kwargs,
            "confidence": confidence,
            "planner_required": planner_required,
            "agent_intermediate_tools": [],
            "agent_followed_continuation": None,
            "agent_next_tool_name": None,
            "agent_next_kwargs": None,
            "same_failure_recurred": False,
            "task_passed_post_nudge": None,
        }
        _list_extra(ctx.extras, self._pending_key).append(pending)
        return pending

    def record_follow_up(self, session_id: str, tool_name: str, kwargs: dict[str, Any]) -> dict[str, Any] | None:
        """Record the first meaningful tool call after a pending intervention."""
        ctx = self._interceptor.get_session_context(session_id)
        for pending in _list_extra(ctx.extras, self._pending_key):
            if pending.get("agent_next_tool_name") is not None:
                continue
            if tool_name in self._ignored_followup_tools:
                _list_extra(pending, "agent_intermediate_tools").append(
                    {"tool_name": tool_name, "kwargs": dict(kwargs)}
                )
                return None
            pending["agent_next_tool_name"] = tool_name
            pending["agent_next_kwargs"] = dict(kwargs)
            pending["agent_followed_continuation"] = matches_single_call_suggestion(
                suggested_tool_name=pending.get("suggested_tool_name"),
                suggested_kwargs=pending.get("suggested_kwargs"),
                actual_tool_name=tool_name,
                actual_kwargs=kwargs,
            )
            self._emit_awareness_event(session_id=session_id, event_name="followup", payload=dict(pending))
            return dict(pending)
        return None

    def mark_same_failure_recurred(self, session_id: str, pattern_id: str) -> dict[str, Any] | None:
        """Mark that a pattern fired again after the agent already responded to a nudge."""
        ctx = self._interceptor.get_session_context(session_id)
        for pending in _list_extra(ctx.extras, self._pending_key):
            if pending.get("agent_next_tool_name") is None:
                continue
            if pending.get("pattern_id") != pattern_id:
                continue
            if pending.get("same_failure_recurred"):
                continue
            pending["same_failure_recurred"] = True
            self._emit_awareness_event(session_id=session_id, event_name="recurrence", payload=dict(pending))
            return dict(pending)
        return None

    def record_task_outcome(self, session_id: str, *, reward: float, info: dict[str, Any]) -> list[dict[str, Any]]:
        """Attach terminal task outcome to every still-unresolved intervention."""
        ctx = self._interceptor.get_session_context(session_id)
        updated: list[dict[str, Any]] = []
        for pending in _list_extra(ctx.extras, self._pending_key):
            if pending.get("task_passed_post_nudge") is not None:
                continue
            pending["task_passed_post_nudge"] = bool(reward >= 1.0)
            pending["task_reward"] = reward
            pending["task_info"] = info
            snapshot = dict(pending)
            _list_extra(ctx.extras, self._completed_key).append(snapshot)
            self._emit_awareness_event(session_id=session_id, event_name="outcome", payload=snapshot)
            updated.append(snapshot)
        return updated

    def _emit_awareness_event(self, *, session_id: str, event_name: str, payload: dict[str, Any]) -> None:
        ctx = self._interceptor.get_session_context(session_id)
        pattern_id = str(payload.get("pattern_id") or "unknown_pattern")
        self._interceptor.emit_evaluation(
            GateEvaluation(
                session_id=session_id,
                turn_idx=ctx.turn_idx,
                gate_id=f"{pattern_id}.{event_name}",
                status=GateStatus.SHADOW,
                fired=False,
                blocked=False,
                kwargs_snapshot=payload,
                latency_ms=0.0,
                tool_name=str(payload.get("agent_next_tool_name") or payload.get("blocked_tool_name") or ""),
                error=json.dumps(payload, sort_keys=True, default=str),
            )
        )


def build_single_call_correction_artifact(correction: SingleCallCorrection) -> str:
    """Render a correction artifact that is safe for tool-calling models."""
    structured = {
        "pattern_id": correction.pattern_id,
        "continuation_source": "A" if correction.next_kwargs is not None else "planner_required",
        "continuation_confidence": correction.confidence,
        "planner_required": correction.planner_required,
        "tool_name": correction.next_tool,
        "suggested_kwargs": correction.next_kwargs,
        "evidence_refs": correction.evidence_refs,
    }
    if correction.after_success is not None:
        structured["after_success"] = correction.after_success

    next_arguments = _json_inline(correction.next_kwargs) if correction.next_kwargs is not None else "null"
    lines = [
        "KAIROS RUNTIME CORRECTION: proposed tool call is blocked.",
        "Do not retry the identical call.",
        f"BLOCKED: {correction.blocked_summary}",
        f"NEXT_TOOL: {correction.next_tool}",
        f"NEXT_ARGUMENTS_JSON: {next_arguments}",
    ]
    if correction.next_kwargs is not None:
        lines.append("ACTION: Emit exactly one tool call next, using NEXT_ARGUMENTS_JSON as the argument object.")
    else:
        lines.append(
            "ACTION: No safe exact JSON is available. Pause for the winning-path planner; do not invent arguments."
        )
    lines.extend(
        [
            f"WHY: {correction.why}",
            f"EVIDENCE_JSON: {_json_inline(correction.evidence_refs)}",
        ]
    )
    if correction.after_success is not None:
        lines.append(f"AFTER_SUCCESS: {correction.after_success}")
    lines.append(f"KAIROS_CONTINUATION_JSON: {_json_inline(structured)}")
    return "\n".join(lines)


def matches_single_call_suggestion(
    *,
    suggested_tool_name: Any,
    suggested_kwargs: Any,
    actual_tool_name: str,
    actual_kwargs: dict[str, Any],
) -> bool | None:
    """Return whether a follow-up call matched a single-call suggestion."""
    if suggested_tool_name in (None, "respond"):
        return None
    if suggested_tool_name != actual_tool_name:
        return False
    if isinstance(suggested_kwargs, dict):
        return suggested_kwargs == actual_kwargs
    return None


def _json_inline(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _dict_extra(extras: dict[str, Any], key: str) -> dict[str, Any]:
    value = extras.setdefault(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a dict")
    return cast("dict[str, Any]", value)


def _list_extra(extras: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = extras.setdefault(key, [])
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list")
    return cast("list[dict[str, Any]]", value)
