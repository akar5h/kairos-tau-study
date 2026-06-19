from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kairos.runtime_correction import SingleCallCorrection, build_single_call_correction_artifact

if TYPE_CHECKING:
    from kairos.models.semantic_recovery import SemanticDecision

SEMANTIC_PREWRITE_PATTERN_ID = "semantic_prewrite_planner.v0"


def build_semantic_decision_artifact(
    *,
    decision: SemanticDecision,
    blocked_tool_name: str,
    blocked_kwargs: dict[str, Any],
) -> str:
    """Render an injectable semantic decision as a Kairos correction artifact."""
    if not decision.is_injectable:
        raise ValueError("semantic decision is not injectable")
    assert decision.next_tool is not None
    assert decision.next_kwargs is not None
    return build_single_call_correction_artifact(
        SingleCallCorrection(
            pattern_id=SEMANTIC_PREWRITE_PATTERN_ID,
            blocked_summary=(
                f"{blocked_tool_name} kwargs do not satisfy the session expectation: "
                f"{_compact_diff(decision.contract_diffs)}"
            ),
            next_tool=decision.next_tool,
            next_kwargs=decision.next_kwargs,
            confidence=decision.confidence,
            planner_required=False,
            why=decision.reason,
            evidence_refs={
                "failure_class": decision.failure_class,
                "blocked_tool_name": blocked_tool_name,
                "blocked_kwargs": blocked_kwargs,
                "contract_diffs": decision.contract_diffs,
                "decision_evidence": decision.evidence_refs,
            },
        )
    )


def _compact_diff(diffs: list[str]) -> str:
    if not diffs:
        return "semantic continuation differs from proposed call"
    return "; ".join(diffs[:4])
