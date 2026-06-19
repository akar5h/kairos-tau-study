from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from kairos.config import settings
from kairos.models.semantic_recovery import (
    IntentTemplate,
    PolicyPack,
    SemanticDecision,
    SemanticSessionSnapshot,
    SessionExpectation,
    ToolPolicyPack,
)
from kairos.semantic_recovery.expectation import build_static_session_expectation
from kairos.semantic_recovery.memory import WorkflowMemoryStore
from kairos.semantic_recovery.plan import build_agent_facing_plan
from kairos.semantic_recovery.tool_policy import ToolPolicyAuditor, build_tool_policy_pack
from kairos.semantic_recovery.verifier import RuntimeVerifier

if TYPE_CHECKING:
    from kairos.intercept import SessionContext
    from kairos.semantic_recovery.expectation import SessionExpectationBuilder

POLICY_PACK_EXTRA_KEY = "semantic_recovery_policy_pack"
MEMORY_HITS_EXTRA_KEY = "semantic_recovery_memory_hits"
SESSION_EXPECTATION_EXTRA_KEY = "semantic_recovery_session_expectation"
AGENT_PLAN_EXTRA_KEY = "semantic_recovery_agent_plan"
TOOL_POLICY_PACK_EXTRA_KEY = "semantic_recovery_tool_policy_pack"
RETRIEVAL_TELEMETRY_EXTRA_KEY = "semantic_recovery_retrieval_telemetry"

_logger = logging.getLogger("kairos.semantic_recovery.retrieval")


class SemanticRecoveryRuntime:
    """Session-start semantic recovery state attachment."""

    def __init__(
        self,
        *,
        policy_pack: PolicyPack,
        memory_store: WorkflowMemoryStore | None = None,
        expectation_builder: SessionExpectationBuilder | None = None,
        tool_descriptions: list[dict[str, Any]] | None = None,
        deterministic_gate_ids: list[str] | None = None,
        memory_min_semantic_score: float = 0.6,
        runtime_verifier: RuntimeVerifier | None = None,
        tool_policy_pack: ToolPolicyPack | None = None,
        tool_policy_auditor: ToolPolicyAuditor | None = None,
        intent_template_map: dict[str, IntentTemplate] | None = None,
    ) -> None:
        self.policy_pack = policy_pack
        self.memory_store = memory_store or WorkflowMemoryStore()
        self.expectation_builder = expectation_builder
        self.tool_descriptions = list(tool_descriptions or [])
        self.deterministic_gate_ids = list(deterministic_gate_ids or [])
        self.memory_min_semantic_score = memory_min_semantic_score
        self.runtime_verifier = runtime_verifier or RuntimeVerifier()
        self.intent_template_map = intent_template_map
        if (
            self.expectation_builder is not None
            and intent_template_map is not None
            and getattr(self.expectation_builder, "intent_template_map", None) is None
        ):
            # If the caller built the SessionExpectationBuilder before having
            # the template map, retroactively attach it. The builder reads
            # the map on every build() call so the assignment takes effect.
            self.expectation_builder.intent_template_map = intent_template_map
        self.tool_policy_pack = tool_policy_pack or build_tool_policy_pack(
            policy_pack=policy_pack,
            tool_descriptions=self.tool_descriptions,
        )
        auditor_client = self.expectation_builder.client if self.expectation_builder is not None else None
        self.tool_policy_auditor = tool_policy_auditor or ToolPolicyAuditor(
            client=auditor_client if settings.semantic_tool_policy_auditor_enabled else None,
            blocking=settings.semantic_tool_policy_auditor_blocking,
        )

    def start_session(
        self,
        ctx: SessionContext,
        *,
        user_instruction: str,
        top_k: int = 1,
    ) -> SemanticSessionSnapshot:
        """Attach policy and retrieved workflow memories to the session context."""
        memory_hits = self.memory_store.retrieve(
            user_instruction,
            top_k=1 if top_k > 1 else top_k,
            toolset_hash=_optional_str(self.policy_pack.provenance.get("toolset_hash")),
            prompt_hash=_optional_str(self.policy_pack.provenance.get("prompt_hash")),
            min_semantic_score=self.memory_min_semantic_score,
        )
        session_expectation = None
        if self.expectation_builder is not None:
            session_expectation = self.expectation_builder.build(
                policy_pack=self.policy_pack,
                user_instruction=user_instruction,
                tool_descriptions=self.tool_descriptions,
                memory_hits=memory_hits,
                deterministic_gate_ids=self.deterministic_gate_ids,
            )
        if session_expectation is None:
            session_expectation = build_static_session_expectation(
                policy_pack=self.policy_pack,
                user_instruction=user_instruction,
                memory_hits=memory_hits,
                intent_template_map=self.intent_template_map,
            )
        agent_plan = build_agent_facing_plan(
            policy_pack=self.policy_pack,
            memory_hits=memory_hits,
            session_expectation=session_expectation,
        )
        snapshot = SemanticSessionSnapshot(
            policy_pack_id=self.policy_pack.pack_id,
            memory_hit_count=len(memory_hits),
            memory_hits=memory_hits,
            session_expectation=session_expectation,
            agent_plan=agent_plan,
        )
        ctx.extras[POLICY_PACK_EXTRA_KEY] = self.policy_pack.model_dump()
        ctx.extras[TOOL_POLICY_PACK_EXTRA_KEY] = self.tool_policy_pack.model_dump()
        ctx.extras[MEMORY_HITS_EXTRA_KEY] = [hit.model_dump() for hit in memory_hits]
        if session_expectation is not None:
            ctx.extras[SESSION_EXPECTATION_EXTRA_KEY] = session_expectation.model_dump()
        if agent_plan is not None:
            ctx.extras[AGENT_PLAN_EXTRA_KEY] = agent_plan.model_dump()

        # Retrieval telemetry: one structured event per session-start. Captures
        # hits count, top score, top hit's action_class, and whether the plan
        # was rendered. Stashed on ctx.extras so it lands in the host's
        # semantic_session JSON artifact alongside the snapshot, AND emitted
        # via stdlib logging for live observability. Lets us answer "how often
        # do queries fall through with zero hits?" without re-parsing
        # trajectories — the bug behind tau-airline task 21 was invisible
        # before this line existed.
        top_hit = memory_hits[0] if memory_hits else None
        telemetry = {
            "session_id": getattr(ctx, "session_id", None),
            "user_instruction_head": user_instruction[:120],
            "hits": len(memory_hits),
            "top_semantic_score": top_hit.semantic_score if top_hit is not None else None,
            "top_utility_score": top_hit.utility_score if top_hit is not None else None,
            "top_action_class": (
                top_hit.memory.intent_action_class if top_hit is not None else None
            ),
            "top_memory_id": top_hit.memory.memory_id if top_hit is not None else None,
            "agent_plan_injected": agent_plan is not None,
            "min_semantic_score_threshold": self.memory_min_semantic_score,
        }
        ctx.extras[RETRIEVAL_TELEMETRY_EXTRA_KEY] = telemetry
        _logger.info("kairos.semantic_recovery.retrieval", extra={"telemetry": telemetry})
        return snapshot

    def verify_tool_call(
        self,
        ctx: SessionContext,
        *,
        tool_name: str,
        kwargs: dict[str, Any],
        evidence: dict[str, Any] | None = None,
    ) -> SemanticDecision:
        """Verify a proposed runtime write against session expectation and evidence."""
        expectation = _session_expectation_from_extras(ctx.extras.get(SESSION_EXPECTATION_EXTRA_KEY))
        decision = self.runtime_verifier.verify(
            tool_name=tool_name,
            kwargs=kwargs,
            session_expectation=expectation,
            evidence=evidence,
        )
        if decision.verdict == "block_with_next_call":
            return decision
        policy_decision = self.tool_policy_auditor.audit(
            tool_name=tool_name,
            kwargs=kwargs,
            tool_policy_pack=self.tool_policy_pack,
            user_transcript=ctx.full_transcript,
            evidence=evidence,
        )
        if policy_decision is not None:
            return policy_decision
        return decision


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _session_expectation_from_extras(value: Any) -> SessionExpectation | None:
    if isinstance(value, SessionExpectation):
        return value
    if isinstance(value, dict):
        return SessionExpectation.model_validate(value)
    return None
