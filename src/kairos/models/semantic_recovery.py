"""Pydantic models for Kairos semantic recovery."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PolicyPack(BaseModel):
    """Static host policy extracted from prompts, tools, and docs."""

    pack_id: str
    source_name: str
    source_hash: str
    tool_names: list[str] = Field(default_factory=list)
    static_expectations: list[str] = Field(default_factory=list)
    required_read_before_write: dict[str, list[str]] = Field(default_factory=dict)
    write_tool_constraints: dict[str, list[str]] = Field(default_factory=dict)
    failure_traps: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


class ArgPolicy(BaseModel):
    """Grounding contract for one tool argument."""

    field: str
    allowed_sources: list[str] = Field(default_factory=list)
    rule: str = ""
    required_evidence: bool = True
    can_be_inferred: bool = False
    conflict_rule: str = "current user instruction > latest trusted tool result > memory"


class ToolPolicy(BaseModel):
    """Per-tool policy contract compiled from prompt, schema, and docs."""

    tool_name: str
    purpose: str = ""
    side_effect_type: Literal["read", "write", "handoff", "final_response"] = "read"
    required_preconditions: list[str] = Field(default_factory=list)
    arg_policies: dict[str, ArgPolicy] = Field(default_factory=dict)
    continuity_rules: list[str] = Field(default_factory=list)
    forbidden_shortcuts: list[str] = Field(default_factory=list)
    output_facts_produced: list[str] = Field(default_factory=list)
    common_failure_modes: list[str] = Field(default_factory=list)
    input_schema: dict[str, Any] = Field(default_factory=dict)


class ToolPolicyPack(BaseModel):
    """Detailed per-tool policy memory compiled from host context."""

    pack_id: str
    source_policy_pack_id: str
    source_hash: str
    prompt_hash: str | None = None
    toolset_hash: str | None = None
    tools: dict[str, ToolPolicy] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


class ToolPolicyViolation(BaseModel):
    """One field-level policy violation returned by the L2 auditor."""

    field: str
    reason: str
    suggested_value: Any = None


class ToolPolicyVerdict(BaseModel):
    """Strict JSON result from the L2 tool policy auditor."""

    compliant: bool
    confidence: Literal["low", "medium", "high"]
    violations: list[ToolPolicyViolation] = Field(default_factory=list)


class WorkflowMemory(BaseModel):
    """Reusable workflow mined from a labeled trajectory."""

    memory_id: str
    title: str | None = None
    description: str | None = None
    content: str | None = None
    # Full curator-authored instruction text (advisory voice in trace-grounded
    # entries; canonical playbook prose in older hand-curated entries). Kept
    # separate from `description` (which is a 180-char summary) so the
    # plan-builder can render the rich prose verbatim instead of relying on
    # the flag-soup of expected_constraints + failure_traps. Optional;
    # entries ingested before this field was added will be None and the
    # plan-builder falls back to `description`.
    instruction_text: str | None = None
    category: Literal["strategy", "recovery", "optimization"] = "strategy"
    intent_action_class: Literal["book", "update", "cancel", "send_certificate", "handoff", "read_only", "unknown"] = (
        "unknown"
    )
    intent_signature: str
    expected_tool_sequence: list[str] = Field(default_factory=list)
    expected_constraints: dict[str, Any] = Field(default_factory=dict)
    canonical_kwargs_hints: dict[str, Any] = Field(default_factory=dict)
    failure_traps: list[str] = Field(default_factory=list)
    negative_example: str | None = None
    source_trajectory: str
    source_trajectory_ids: list[str] = Field(default_factory=list)
    prompt_hash: str | None = None
    toolset_hash: str | None = None
    utility_score: float = 1.0
    provenance: dict[str, Any] = Field(default_factory=dict)


class MemoryRetrievalResult(BaseModel):
    """Ranked retrieval result for a workflow memory."""

    memory: WorkflowMemory
    semantic_score: float
    utility_score: float
    matched_constraints: list[str] = Field(default_factory=list)
    hash_compatible: bool = True
    source_trajectory: str


class IntentTemplate(BaseModel):
    """Per-intent-class deterministic template for the expectation fallback.

    Hosts supply a map ``intent_class -> IntentTemplate`` to fill the
    critical fields when the LLM expectation builder returns empty for those
    fields. The fallback runs after the LLM call, so a well-behaved LLM is
    not overridden — the template only fills gaps.

    Example::

        {
            "book":   IntentTemplate(expected_terminal_actions=["book_reservation"]),
            "cancel": IntentTemplate(expected_terminal_actions=["cancel_reservation"]),
            ...
        }
    """

    expected_terminal_actions: list[str] = Field(default_factory=list)
    likely_workflow: list[str] = Field(default_factory=list)
    success_lock_conditions: list[str] = Field(default_factory=list)


class SessionExpectation(BaseModel):
    """Compact session-start policy expectation for runtime recovery."""

    user_constraints: dict[str, Any] = Field(default_factory=dict)
    likely_workflow: list[str] = Field(default_factory=list)
    must_read_tools: list[str] = Field(default_factory=list)
    allowed_write_tools: list[str] = Field(default_factory=list)
    forbidden_shortcuts: list[str] = Field(default_factory=list)
    optimization_target: str | None = None
    expected_terminal_actions: list[str] = Field(default_factory=list)
    success_lock_conditions: list[str] = Field(default_factory=list)
    danger_points: list[str] = Field(default_factory=list)


class SemanticDecision(BaseModel):
    """Runtime semantic prewrite decision.

    This is the generic Kairos contract. Host adapters may provide
    domain-specific evidence, but Kairos owns the decision shape and the safety
    rule that active recovery emits at most one next tool call.
    """

    verdict: Literal["allow", "block_with_next_call", "observe_only"]
    failure_class: str | None = None
    reason: str = ""
    next_tool: str | None = None
    next_kwargs: dict[str, Any] | None = None
    confidence: Literal["low", "medium", "high"] = "medium"
    evidence_refs: dict[str, Any] = Field(default_factory=dict)
    contract_diffs: list[str] = Field(default_factory=list)

    @property
    def is_injectable(self) -> bool:
        """Whether this decision can safely become a tool-error nudge."""
        return (
            self.verdict == "block_with_next_call"
            and self.confidence == "high"
            and isinstance(self.next_tool, str)
            and isinstance(self.next_kwargs, dict)
        )


class AgentFacingPlan(BaseModel):
    """Compact plan artifact injected into the host agent prompt."""

    artifact: str
    expected_tool_sequence: list[str] = Field(default_factory=list)
    expected_terminal_actions: list[str] = Field(default_factory=list)
    source_trajectories: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


class SemanticSessionSnapshot(BaseModel):
    """Observe-only semantic recovery state attached at session start."""

    policy_pack_id: str
    memory_hit_count: int = 0
    memory_hits: list[MemoryRetrievalResult] = Field(default_factory=list)
    session_expectation: SessionExpectation | None = None
    agent_plan: AgentFacingPlan | None = None


class DriftObservation(BaseModel):
    """One drift-detection observation written per evaluated tool call.

    Pure observation: this record is sunk to ``drift_observations.jsonl``
    and to the per-task entry of ``summary.json``. It never feeds the
    intervention path. The judge picks ``drift_label`` freely; we do not
    enumerate categories upfront so the taxonomy can emerge from data.
    """

    session_id: str
    turn_idx: int
    tool_name: str
    kwargs_snapshot: dict[str, Any] = Field(default_factory=dict)
    verdict_status: Literal["clean", "judge_error", "invalid_verdict_json"] = "clean"
    consistent: bool | None
    drift_label: str | None = None
    matched_pattern_ids: list[str] = Field(default_factory=list)
    severity: Literal["low", "medium", "high"] = "low"
    would_break_task: bool = False
    recoverable: bool = True
    reason: str = ""
    confidence: Literal["low", "medium", "high"] = "low"
    evidence_pointers: list[str] = Field(default_factory=list)
    judge_model: str | None = None
    judge_latency_ms: float = 0.0
    error: str | None = None
