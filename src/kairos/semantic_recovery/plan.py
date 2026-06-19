from __future__ import annotations

import json
from typing import Any

from kairos.models.semantic_recovery import AgentFacingPlan, MemoryRetrievalResult, PolicyPack, SessionExpectation

MAX_MEMORY_HITS_IN_PLAN = 1
MAX_FAILURE_TRAPS_IN_PLAN = 6


def build_agent_facing_plan(
    *,
    policy_pack: PolicyPack,
    memory_hits: list[MemoryRetrievalResult],
    session_expectation: SessionExpectation | None,
) -> AgentFacingPlan | None:
    """Render retrieved winning paths and expectations as an agent-facing plan.

    Returns ``None`` when ``memory_hits`` is empty. The static-fallback plan
    (policy_pack + static session expectation, no retrieved trajectory) was
    found to bias the agent toward mutation on tasks whose correct answer is
    refusal — e.g. tau-airline task 21 where the agent fabricated a parallel
    booking instead of declining a basic-economy modification. Without a
    confirmed prior trajectory we have no positive evidence that mutation is
    the right move, so the host should fall back to running the agent
    against the policy text alone.
    """
    if not memory_hits:
        return None

    # ADVISORY RENDERING (Phase 3 Fix A, 2026-05-19).
    # Lead with the memory's rich instruction prose ("when this pattern applies
    # + the right move") so scenario-specific reasoning sits at the head of
    # the system prompt where attention is highest. Surface the negative_example
    # as a labeled "common failure" section. Demote generic flag bullets (the
    # old "Constraints to preserve / Do not" lists) to a brief tail.
    # See docs/phase2.5-parallel-investigation-findings.md §6 layer 3 for
    # the motivation: Phase 2 cascade picks were correct on covered tasks
    # but the agent's behavior didn't shift because the plan never showed
    # the memory's actual prose — only its derived flag-list.
    primary = memory_hits[0].memory
    expected_sequence = _expected_sequence(memory_hits, session_expectation)
    expected_terminal_actions = _expected_terminal_actions(memory_hits, session_expectation)
    source_trajectories = _source_trajectories(memory_hits)

    # Prefer the rich `instruction_text` field (added 2026-05-19). Fall back
    # to `description` (180-char truncation of the same prose) for older
    # entries that don't have instruction_text populated.
    headline_prose = (primary.instruction_text or primary.description or "").strip()
    negative_prose = (primary.negative_example or "").strip()

    parts: list[str] = []
    parts.append(
        "# Kairos session memory\n"
        "(advisory only — current user message takes priority over anything below)"
    )
    parts.append("")

    if headline_prose:
        parts.append("## When this pattern applies + the right move")
        parts.append(headline_prose)
        parts.append("")

    if negative_prose:
        parts.append("## Common failure mode to avoid")
        parts.append(negative_prose)
        parts.append("")

    # Fix A v3 — surgical relabel (Phase 7, 2026-05-20).
    # Phase 4-6 forensics identified two specific renderer-level bugs that
    # caused the v4 cascade regressions; neither was content-related, both
    # are LABEL bugs:
    #
    #   (1) ``preserve from current evidence: X=Y`` falsely framed
    #       memory.expected_constraints (which are SOURCE-TRAJECTORY
    #       constraints) as if they were CURRENT-SESSION evidence. Task 13
    #       in Phase 5 saw ``basic_economy=true`` on a non-basic-economy
    #       reservation and acted on it. Fixed by relabeling so the agent
    #       knows the provenance.
    #
    #   (2) ``watch for: X`` turned cautionary memory observations into
    #       authoritative-sounding policy claims. Task 4 in Phase 4 saw
    #       ``watch for: basic economy cannot be modified`` and quoted it
    #       back as policy fact (tau-airline actually PERMITS passenger
    #       updates on basic economy). Fixed by relabeling the line as a
    #       prior-trace caution requiring verification, not a fact.
    #
    #   (3) ``## Typical tool sequence (reference shape, expect variation)``
    #       was read as a procedure to execute, not a reference. Task 2 in
    #       Phase 4 stripped its own search step to match the listed
    #       sequence verbatim. Relabeled to make the "this is one example,
    #       agent decides" framing impossible to miss.
    #
    # The actual CONTENT of all three sections is unchanged — same
    # constraints, same failure_traps, same expected_sequence. Only the
    # framing labels move. Phase 6 showed full strip is net-negative
    # (kills helpful elements with the harmful ones); this surgical
    # relabel keeps the helpful information while disarming the
    # misleading-provenance bug.
    pre_write_checks = _pre_write_checks(memory_hits, session_expectation)
    constraints = _constraint_lines(memory_hits, session_expectation)
    failure_traps = _failure_traps(policy_pack, memory_hits, session_expectation)
    if pre_write_checks or constraints or failure_traps:
        parts.append("## Verifications before any write")
        for check in pre_write_checks:
            parts.append(f"- {check}")
        for cstr in constraints[:6]:
            parts.append(
                f"- memory says this pattern requires (verify in current session): {cstr}"
            )
        for trap in failure_traps[:4]:
            parts.append(
                f"- prior trace had this caution (verify against current task before acting): {trap}"
            )
        parts.append("")

    if expected_sequence:
        parts.append(
            "## Prior successful trace used this sequence (one example shape; agent decides actual order)"
        )
        parts.append(" -> ".join(expected_sequence))
        parts.append("")

    success_locks = _success_locks(session_expectation)
    if success_locks:
        parts.append("## After completion")
        for lock in success_locks:
            parts.append(f"- {lock}")
        parts.append("")

    # Universal refusal-aware tiebreaker. Kept as a final guardrail —
    # if the rendered plan above doesn't fit the current user's actual
    # situation, the agent has a clear escape hatch.
    parts.append("## Universal fallback")
    parts.append(
        "If the user's request has no policy entitlement, decline politely "
        "or call transfer_to_human_agents. Do not invent workarounds such as "
        "parallel bookings or alternative tools that achieve the forbidden "
        "outcome. Use current tool results as evidence; do not copy IDs, "
        "prices, or dates from memory unless this session confirms them."
    )

    if source_trajectories:
        parts.append("")
        parts.append(f"_source: {', '.join(source_trajectories)}_")

    artifact = "\n".join(parts)

    return AgentFacingPlan(
        artifact=artifact,
        expected_tool_sequence=expected_sequence,
        expected_terminal_actions=expected_terminal_actions,
        source_trajectories=source_trajectories,
        provenance={
            "policy_pack_id": policy_pack.pack_id,
            "memory_hit_count": len(memory_hits),
            "has_session_expectation": session_expectation is not None,
            "rendering_style": "advisory_v3_surgical_relabel",
        },
    )


def _expected_sequence(
    memory_hits: list[MemoryRetrievalResult],
    session_expectation: SessionExpectation | None,
) -> list[str]:
    if session_expectation is not None and session_expectation.likely_workflow:
        return list(dict.fromkeys(session_expectation.likely_workflow))
    if not memory_hits:
        return []
    return list(dict.fromkeys(memory_hits[0].memory.expected_tool_sequence))


def _expected_terminal_actions(
    memory_hits: list[MemoryRetrievalResult],
    session_expectation: SessionExpectation | None,
) -> list[str]:
    if session_expectation is not None and session_expectation.expected_terminal_actions:
        return list(dict.fromkeys(session_expectation.expected_terminal_actions))
    sequence = _expected_sequence(memory_hits, session_expectation)
    return [tool for tool in sequence if _is_write_or_handoff(tool)]


def _source_trajectories(memory_hits: list[MemoryRetrievalResult]) -> list[str]:
    sources: list[str] = []
    for hit in memory_hits[:MAX_MEMORY_HITS_IN_PLAN]:
        if hit.source_trajectory not in sources:
            sources.append(hit.source_trajectory)
    return sources


def _constraint_lines(
    memory_hits: list[MemoryRetrievalResult],
    session_expectation: SessionExpectation | None,
) -> list[str]:
    constraints: list[str] = []
    if session_expectation is not None:
        for key, value in session_expectation.user_constraints.items():
            constraints.append(f"{key}={_json_inline(value)}")
        if session_expectation.optimization_target:
            constraints.append(f"optimization_target={session_expectation.optimization_target}")
        # forbidden_shortcuts is misnamed but populated from
        # policy_pack.static_expectations — positive policy hints, not
        # negations. Keep them under "Constraints to preserve" because
        # rendering "verify eligibility before cancel" under "Do not:"
        # inverts the intent. The primary task-21 mitigation lives in the
        # zero-hit guard at the top of build_agent_facing_plan.
        constraints.extend(session_expectation.forbidden_shortcuts)
    for hit in memory_hits[:MAX_MEMORY_HITS_IN_PLAN]:
        for key, value in hit.memory.expected_constraints.items():
            item = f"{key}={_json_inline(value)}"
            if item not in constraints:
                constraints.append(item)
    return constraints


def _failure_traps(
    policy_pack: PolicyPack,
    memory_hits: list[MemoryRetrievalResult],
    session_expectation: SessionExpectation | None,
) -> list[str]:
    traps: list[str] = []
    traps.extend(policy_pack.failure_traps)
    if session_expectation is not None:
        traps.extend(session_expectation.danger_points)
    for hit in memory_hits[:MAX_MEMORY_HITS_IN_PLAN]:
        traps.extend(hit.memory.failure_traps)

    unique: list[str] = []
    for trap in traps:
        if trap and trap not in unique:
            unique.append(trap)
    return unique[:MAX_FAILURE_TRAPS_IN_PLAN]


def _pre_write_checks(
    memory_hits: list[MemoryRetrievalResult],
    session_expectation: SessionExpectation | None,
) -> list[str]:
    checks: list[str] = [
        "ground IDs and amounts in current session evidence",
        "preserve passenger/payment/baggage/insurance fields from current evidence unless user changes them",
    ]
    if session_expectation is not None:
        for tool in session_expectation.must_read_tools:
            item = f"call or use fresh {tool} evidence before mutation"
            if item not in checks:
                checks.append(item)
    for hit in memory_hits[:MAX_MEMORY_HITS_IN_PLAN]:
        if hit.memory.category == "optimization" and "compare all candidate options" not in checks:
            checks.append("compare all candidate options against the optimization target")
    return checks


def _success_locks(session_expectation: SessionExpectation | None) -> list[str]:
    if session_expectation is None:
        return []
    return list(dict.fromkeys(session_expectation.success_lock_conditions))


def _is_write_or_handoff(tool_name: str) -> bool:
    return tool_name.startswith(("book_", "update_", "cancel_", "send_")) or tool_name == "transfer_to_human_agents"


def _json_inline(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
