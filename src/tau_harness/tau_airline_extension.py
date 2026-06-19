"""DomainExtension that wraps TauPhase1RecoveryController for the kairos SDK.

This V1 adapter delegates to the existing tau-internal recovery controller
and semantic-recovery runtime. Returns ``ToolDecision`` with
``bypass_orchestrator=True`` so kairos.host's orchestrator doesn't
double-emit retry/follow-up/recurrence/outcome telemetry — the legacy
controller already drives that pipeline directly through the interceptor.

Future cleanup: pull the airline-specific detectors and suggestion engines
out of phase1_recovery.py and into this module, and switch
``bypass_orchestrator`` to ``False`` so kairos owns retry/awareness for
free. For now we just consolidate the call sites at the host SDK boundary.
"""

from __future__ import annotations

from typing import Any

from kairos.host import DomainExtension, ToolDecision
from kairos.intercept import Gate, KairosInterceptor, SessionContext
from kairos.runtime_correction import CorrectionAction
from kairos.semantic_recovery import (
    SEMANTIC_PREWRITE_PATTERN_ID,
    SemanticRecoveryRuntime,
    build_semantic_decision_artifact,
)

from tau_harness.phase1_recovery import TauPhase1RecoveryController


class TauAirlineExtension(DomainExtension):
    """Bridge between the legacy TauPhase1 stack and kairos.host."""

    def __init__(
        self,
        *,
        interceptor: KairosInterceptor,
        semantic_runtime: SemanticRecoveryRuntime | None,
    ) -> None:
        self._interceptor = interceptor
        self._controller = TauPhase1RecoveryController(interceptor)
        self._semantic_runtime = semantic_runtime

    # ---- DomainExtension protocol ---------------------------------------- #

    def gates(self) -> list[Gate]:
        # Static phase1 ungrounded-id gates are loaded by tau today via the
        # kairos_gates.json + runtime build path inside the host bootstrap.
        # The extension contributes none of its own.
        return []

    def before_tool_call(
        self,
        ctx: SessionContext,
        tool_name: str,
        kwargs: dict[str, Any],
    ) -> ToolDecision | None:
        session_id = ctx.session_id

        # Semantic prewrite verification (Phase 3) — runs first today.
        if self._semantic_runtime is not None:
            evidence = self._controller.semantic_prewrite_evidence(
                session_id, tool_name, kwargs
            )
            try:
                semantic_decision = self._semantic_runtime.verify_tool_call(
                    ctx,
                    tool_name=tool_name,
                    kwargs=kwargs,
                    evidence=evidence,
                )
            except Exception:  # noqa: BLE001 - never crash the host on a judge fault.
                semantic_decision = None
            if semantic_decision is not None and semantic_decision.is_injectable:
                artifact = build_semantic_decision_artifact(
                    decision=semantic_decision,
                    blocked_tool_name=tool_name,
                    blocked_kwargs=kwargs,
                )
                corr = self._controller.record_semantic_prewrite_intervention(
                    session_id=session_id,
                    tool_name=tool_name,
                    kwargs=kwargs,
                    artifact=artifact,
                    suggested_tool_name=semantic_decision.next_tool or tool_name,
                    suggested_kwargs=semantic_decision.next_kwargs or {},
                    confidence=semantic_decision.confidence,
                )
                if corr.action == CorrectionAction.INJECT_CORRECTION:
                    print(
                        "kairos_recovery: injecting semantic Phase 3 correction "
                        f"{SEMANTIC_PREWRITE_PATTERN_ID} for {tool_name}"
                    )
                    return ToolDecision(
                        action="inject_correction",
                        correction_artifact=corr.correction_artifact or "",
                        pattern_id=SEMANTIC_PREWRITE_PATTERN_ID,
                        confidence=semantic_decision.confidence,
                        bypass_orchestrator=True,
                    )
                if corr.action == CorrectionAction.EXECUTE_FAIL_OPEN:
                    print(
                        "kairos_intercept: fail-open after Phase 3 semantic retry budget "
                        f"for {tool_name} ({corr.retry_key})"
                    )
                    return ToolDecision(
                        action="execute_fail_open",
                        pattern_id=SEMANTIC_PREWRITE_PATTERN_ID,
                        bypass_orchestrator=True,
                    )

        # Phase 1 exact-family detection. ``record_attempt=True`` so the
        # legacy controller's awareness tracker records the follow-up against
        # its own ``tau_phase1_pending_interventions`` extras key. (The
        # kairos.host orchestrator uses different keys and so doesn't see the
        # pendings recorded by this extension — without this, the tracker
        # never observes the agent's subsequent tool call as a follow-up
        # and ``.followup`` evaluations never emit.)
        corr = self._controller.before_tool_call(
            session_id, tool_name, kwargs, record_attempt=True
        )
        if corr.action == CorrectionAction.INJECT_CORRECTION:
            pattern_id = getattr(corr, "pattern_id", None)
            if pattern_id:
                print(
                    "kairos_recovery: injecting exact Phase 1 correction "
                    f"{pattern_id} for {tool_name}"
                )
            return ToolDecision(
                action="inject_correction",
                correction_artifact=corr.correction_artifact or "",
                pattern_id=pattern_id,
                bypass_orchestrator=True,
            )
        if corr.action == CorrectionAction.EXECUTE_FAIL_OPEN:
            print(
                "kairos_intercept: fail-open after Phase 1 retry budget "
                f"for {tool_name} ({corr.retry_key})"
            )
            return ToolDecision(
                action="execute_fail_open",
                pattern_id=getattr(corr, "pattern_id", None),
                bypass_orchestrator=True,
            )

        # Nothing matched — let kairos.host's pipeline continue (static gates, etc.).
        return None

    def semantic_evidence(
        self,
        ctx: SessionContext,
        tool_name: str,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        # We run the semantic verifier ourselves above; kairos.host's own
        # semantic stage uses this only if it owned the runtime. With this
        # adapter the runtime is owned by the extension, so there is nothing
        # to contribute via the host's verifier path.
        return {}

    def after_tool_result(
        self,
        ctx: SessionContext,
        tool_name: str,
        kwargs: dict[str, Any],
        observation: Any,
    ) -> None:
        self._controller.after_tool_result(
            ctx.session_id, tool_name, observation, kwargs
        )

    # ---- convenience ----------------------------------------------------- #

    def after_task(
        self,
        session_id: str,
        *,
        reward: float,
        info: dict[str, Any],
    ) -> None:
        """Forward task outcome to the legacy controller's awareness tracker.

        kairos.host's ``KairosSession.end`` already calls its own tracker's
        ``record_task_outcome``; the legacy tracker uses the same extras keys
        so a second call here is idempotent (the outcome flag is already set
        on each pending entry).
        """
        self._controller.after_task(session_id, reward=reward, info=info)
