"""Internal orchestrator: retry budget + intervention awareness + block telemetry.

Wraps :class:`~kairos.runtime_correction.primitives.RetryBudget` and
:class:`~kairos.runtime_correction.primitives.InterventionAwarenessTracker`
for the host SDK. Hosts never construct or call this directly — they use
:class:`~kairos.host.KairosSession`, which routes block decisions through here.

What this orchestrator owns:
  * Per-pattern, per-tool retry budget (consumed when a domain extension or
    the semantic verifier fires). If the budget is exhausted on a recurring
    block, the host gets ``execute_fail_open`` instead of a fresh inject.
  * In-flight intervention awareness — the tracker auto-emits the
    ``.followup``, ``.recurrence`` and ``.outcome`` meta-evaluations via
    ``interceptor.emit_evaluation``. The orchestrator just calls the right
    methods at the right times (every tool attempt, every recurrence, every
    task end).
  * Block-time GateEvaluation emission. The tracker only emits meta-events;
    the orchestrator emits the primary block eval (and the fail-open eval
    when budget is exhausted) so every host gate firing has a primary record
    in the sink.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from kairos.intercept import (
    GateEvaluation,
    GateStatus,
    KairosInterceptor,
    SessionContext,
)
from kairos.runtime_correction import (
    InterventionAwarenessTracker,
    RetryBudget,
)

if TYPE_CHECKING:
    from kairos.runtime_correction.primitives import CorrectionConfidence

BlockOutcome = Literal["inject_correction", "execute_fail_open"]


class ExtensionBlockOrchestrator:
    """Retry budget + intervention awareness for host-side block decisions."""

    def __init__(
        self,
        interceptor: KairosInterceptor,
        *,
        max_retries_per_intervention: int = 1,
    ) -> None:
        self._interceptor = interceptor
        self._retry_budget = RetryBudget(max_retries=max_retries_per_intervention)
        self._tracker = InterventionAwarenessTracker(interceptor)

    # ---- per-call hooks --------------------------------------------------- #

    def record_follow_up(
        self,
        session_id: str,
        tool_name: str,
        kwargs: dict[str, Any],
    ) -> None:
        """Tell the tracker the agent attempted a tool.

        Triggers ``.followup`` meta-eval emission if the call matches (or
        diverges from) a pending intervention's suggestion. Called on every
        ``KairosSession.before_tool_call`` regardless of whether anything
        eventually blocks.
        """
        self._tracker.record_follow_up(session_id, tool_name, kwargs)

    def handle_block(
        self,
        session_id: str,
        tool_name: str,
        kwargs: dict[str, Any],
        *,
        pattern_id: str,
        confidence: str | None,
        suggested_tool_name: str | None = None,
        suggested_kwargs: Any = None,
    ) -> tuple[BlockOutcome, GateEvaluation]:
        """Run the block bookkeeping for one host-side decision.

        Order of operations:
          1. ``mark_same_failure_recurred`` — emits ``.recurrence`` if this
             pattern already had an in-flight intervention.
          2. ``RetryBudget.consume`` — returns whether the host should
             ``inject_correction`` or ``execute_fail_open``.
          3. If retrying is allowed: ``record_pending`` so the next tool call
             is tracked; emit a primary block ``GateEvaluation``.
          4. If budget exhausted: emit a fail-open eval tagged
             ``error="fail_open_after_retry_budget"``.
        """
        ctx = self._interceptor.get_session_context(session_id)

        # Recurrence detection runs first so it fires even when budget is empty.
        self._tracker.mark_same_failure_recurred(session_id, pattern_id)

        budget = self._retry_budget.consume(ctx, pattern_id=pattern_id, tool_name=tool_name)
        if not budget.retry_allowed:
            ev = self._emit_block_eval(
                session_id=session_id,
                ctx=ctx,
                pattern_id=pattern_id,
                tool_name=tool_name,
                kwargs=kwargs,
                blocked=False,
                error="fail_open_after_retry_budget",
            )
            return "execute_fail_open", ev

        self._tracker.record_pending(
            session_id=session_id,
            retry_key=budget.retry_key,
            pattern_id=pattern_id,
            blocked_tool_name=tool_name,
            blocked_kwargs=kwargs,
            suggested_tool_name=suggested_tool_name or tool_name,
            suggested_kwargs=suggested_kwargs,
            confidence=cast("CorrectionConfidence", confidence or "medium"),
            planner_required=False,
        )

        ev = self._emit_block_eval(
            session_id=session_id,
            ctx=ctx,
            pattern_id=pattern_id,
            tool_name=tool_name,
            kwargs=kwargs,
            blocked=True,
            error=None,
        )
        return "inject_correction", ev

    def record_task_outcome(
        self,
        session_id: str,
        *,
        reward: float,
        info: dict[str, Any] | None,
    ) -> None:
        """Emit ``.outcome`` meta-evals for every still-pending intervention."""
        self._tracker.record_task_outcome(session_id, reward=reward, info=info or {})

    # ---- helpers --------------------------------------------------------- #

    def _emit_block_eval(
        self,
        *,
        session_id: str,
        ctx: SessionContext,
        pattern_id: str,
        tool_name: str,
        kwargs: dict[str, Any],
        blocked: bool,
        error: str | None,
    ) -> GateEvaluation:
        evaluation = GateEvaluation(
            session_id=session_id,
            turn_idx=ctx.turn_idx,
            gate_id=pattern_id,
            status=GateStatus.ACTIVE if blocked else GateStatus.SHADOW,
            fired=blocked,
            blocked=blocked,
            kwargs_snapshot=dict(kwargs),
            latency_ms=0.0,
            tool_name=tool_name,
            error=error,
        )
        self._interceptor.emit_evaluation(evaluation)
        return evaluation
