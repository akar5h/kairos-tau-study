"""Layer 1 drift detector — generic, judge-driven, observation-only.

Runs once per evaluated tool call. Asks a small LLM judge (the existing
:class:`~kairos.semantic_recovery.expectation.ExpectationLLMClient`) whether
the agent's proposed call is consistent with the session contract and the
evidence gathered so far. The judge picks ``drift_label`` freely from
free-form kebab-case strings; we deliberately do not enumerate categories
so the taxonomy can emerge from the data.

Detection-only by design. Output is written to
``data/runs/<id>/drift_observations.jsonl`` and into the per-task
``summary.json`` entry. It is never consulted by the intervention path,
which keeps grading honest — the detector cannot accidentally close a
control loop on itself.

The detector observes *every* proposed mutating/terminal action, regardless
of whether another layer later blocks it. This makes the observation an
unbiased record of agent intent rather than a filtered record of execution.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable  # noqa: TC003
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import TYPE_CHECKING, Any, Literal

from kairos.models.semantic_recovery import (
    DriftObservation,
    SessionExpectation,
)

if TYPE_CHECKING:
    from kairos.intercept import SessionContext
    from kairos.semantic_recovery.expectation import ExpectationLLMClient

__all__ = ["DriftDetector", "default_monitor_predicate"]


# Tier-1 default: monitor mutating writes + the terminal handoff action.
# Reads are NOT monitored by default — the failure cost of a bad read is
# recoverable (the agent can search again). The cost of a bad write is not.
# Hosts that want to monitor entity-scoping reads (get_user_details,
# get_reservation_details) supply their own predicate via the host kwarg.
_TIER1_PREFIXES: tuple[str, ...] = ("book_", "update_", "cancel_", "send_")
_TIER1_NAMES = frozenset({"transfer_to_human_agents"})


def default_monitor_predicate(tool_name: str) -> bool:
    """Tier-1 default monitored-tool predicate.

    Returns True for any tool that mutates state or terminates the session
    (handoff). Returns False for reads, ``think``, and any tool whose name
    doesn't match the tier-1 shape. Hosts can pass a custom predicate to
    :class:`DriftDetector` or :func:`kairos.host.host` to widen the set.
    """
    if tool_name.startswith(_TIER1_PREFIXES):
        return True
    return tool_name in _TIER1_NAMES


_DEFAULT_USER_MESSAGE_WINDOW = 5
_DEFAULT_TOOL_RESULT_WINDOW = 5


class DriftDetector:
    """Judge-driven runtime drift detector. One LLM call per monitored tool."""

    def __init__(
        self,
        *,
        client: ExpectationLLMClient | None = None,
        judge_model: str | None = None,
        user_message_window: int = _DEFAULT_USER_MESSAGE_WINDOW,
        tool_result_window: int = _DEFAULT_TOOL_RESULT_WINDOW,
        monitor_predicate: Callable[[str], bool] | None = None,
        diagnostic_patterns: list[dict[str, Any]] | None = None,
    ) -> None:
        self.client = client
        self.judge_model = judge_model
        self._user_message_window = user_message_window
        self._tool_result_window = tool_result_window
        self._monitor_predicate = monitor_predicate or default_monitor_predicate
        self._diagnostic_patterns = _compact_diagnostic_patterns(diagnostic_patterns or [])
        self._diagnostic_pattern_index = {str(pattern["pattern_id"]): pattern for pattern in self._diagnostic_patterns}

    def is_enabled(self) -> bool:
        return self.client is not None

    def is_monitored(self, tool_name: str) -> bool:
        return self._monitor_predicate(tool_name)

    def observe(
        self,
        *,
        session_id: str,
        ctx: SessionContext,
        tool_name: str,
        kwargs: dict[str, Any],
        session_expectation: SessionExpectation | None = None,
        memory_plan_artifact: str | None = None,
    ) -> DriftObservation | None:
        """Run one drift check. Returns ``None`` when the detector is disabled
        or when the tool is not in the monitored set.

        Never raises on judge failure — emits a record with
        ``verdict_status != "clean"``, ``consistent=None``, and an ``error``
        field so the failure is visible in telemetry without polluting the
        "drift detected" signal.
        """
        if self.client is None:
            return None
        if not self._monitor_predicate(tool_name):
            return None

        recent_user_messages = list(ctx.user_messages)[-self._user_message_window :]
        # Structured per-tool log maintained by ``KairosInterceptor.update_context``.
        # No transcript parsing — the engine writes the records we read.
        recent_tool_results = list(ctx.recent_tool_results)[-self._tool_result_window :]

        contract_mismatches = _contract_constraint_evidence(
            tool_name=tool_name,
            kwargs=kwargs,
            session_expectation=session_expectation,
        )
        prompt_payload = _build_judge_payload(
            tool_name=tool_name,
            kwargs=kwargs,
            session_expectation=session_expectation,
            recent_user_messages=recent_user_messages,
            recent_tool_results=recent_tool_results,
            memory_plan_artifact=memory_plan_artifact,
            diagnostic_patterns=self._diagnostic_patterns,
            deterministic_contract_mismatches=contract_mismatches,
        )

        started = time.perf_counter()
        error: str | None = None
        raw: str | None = None
        try:
            raw = _complete_json_with_deadline(
                self.client,
                system_prompt=_judge_system_prompt(),
                user_prompt=prompt_payload,
            )
        except Exception as exc:  # noqa: BLE001 - detection must never crash the agent loop.
            error = f"judge_error: {type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - started) * 1000.0

        if error is not None:
            return _error_observation(
                session_id=session_id,
                ctx=ctx,
                tool_name=tool_name,
                kwargs=kwargs,
                error=error,
                judge_model=self.judge_model,
                latency_ms=latency_ms,
            )

        verdict = _parse_verdict(raw)
        if verdict is None:
            return _error_observation(
                session_id=session_id,
                ctx=ctx,
                tool_name=tool_name,
                kwargs=kwargs,
                error="invalid_verdict_json",
                judge_model=self.judge_model,
                latency_ms=latency_ms,
            )

        consistent = bool(verdict.get("consistent", True))
        matched_pattern_ids = _normalize_matched_pattern_ids(
            verdict.get("matched_pattern_ids"),
            drift_label=verdict.get("drift_label"),
            pattern_index=self._diagnostic_pattern_index,
            tool_name=tool_name,
        )
        severity = _normalize_confidence(verdict.get("severity"))
        would_break_task = bool(verdict.get("would_break_task", False))
        recoverable = bool(verdict.get("recoverable", True))
        hard_constraint_evidence = _hard_constraint_evidence(
            tool_name=tool_name,
            kwargs=kwargs,
            verdict=verdict,
            session_expectation=session_expectation,
            recent_user_messages=recent_user_messages,
            recent_tool_results=recent_tool_results,
        )
        hard_constraint_evidence.extend(contract_mismatches)
        if self._diagnostic_pattern_index and not consistent and not matched_pattern_ids:
            # Keep raw drift visible, but do not let an uncalibrated/free-form
            # judge label count as predicted task-breaking drift.
            severity = "medium" if severity == "high" else severity
            would_break_task = False
            recoverable = True
        elif (
            self._diagnostic_pattern_index
            and not consistent
            and "constraint-violation" in matched_pattern_ids
            and not hard_constraint_evidence
        ):
            # Precision-first calibration: an LLM saying "value-not-grounded"
            # or "constraint-violation" is not enough. Treat representation
            # ambiguity (for example multi-leg tool payloads for a "nonstop"
            # user goal) as suspicious but not outcome-breaking unless it is
            # anchored to a typed hard-constraint mismatch.
            matched_pattern_ids = [
                pattern_id for pattern_id in matched_pattern_ids if pattern_id != "constraint-violation"
            ]
            severity = "medium" if severity == "high" else severity
            would_break_task = False
            recoverable = True

        return DriftObservation(
            session_id=session_id,
            turn_idx=ctx.turn_idx,
            tool_name=tool_name,
            kwargs_snapshot=dict(kwargs),
            verdict_status="clean",
            consistent=consistent,
            drift_label=_optional_str(verdict.get("drift_label")),
            matched_pattern_ids=matched_pattern_ids,
            severity=severity,
            would_break_task=would_break_task,
            recoverable=recoverable,
            reason=str(verdict.get("reason") or ""),
            confidence=_normalize_confidence(verdict.get("confidence")),
            evidence_pointers=_merge_unique(
                _normalize_pointers(verdict.get("evidence_pointers")),
                contract_mismatches,
            ),
            judge_model=self.judge_model,
            judge_latency_ms=latency_ms,
            error=None,
        )

    def observe_missing_actions(
        self,
        *,
        session_id: str,
        ctx: SessionContext,
        session_expectation: SessionExpectation,
    ) -> DriftObservation | None:
        """Emit one end-of-session missing-action observation.

        This catches a different failure mode from pre-tool bad-action drift:
        the agent terminates or hands off without ever attempting concrete
        expected terminal writes. It is intentionally conservative and skips
        sessions where the user supplied a conditional no-op instruction such
        as "if over budget, don't make any changes".
        """
        expected = _expected_write_actions(session_expectation)
        if not expected:
            return None
        ledger = _update_obligation_ledger(ctx, expected)
        missing = [tool for tool in expected if ledger[tool]["state"] == "open"]
        if not missing:
            return None
        if _allows_conditional_noop(ctx.full_transcript):
            return None
        attempted = set(ctx.attempted_tools)
        terminal_attempted = "transfer_to_human_agents" in attempted or "respond" in attempted

        reason = f"Session ended without attempting expected terminal action(s): {', '.join(missing)}."
        if terminal_attempted:
            reason += " A terminal handoff/final action occurred first."
        return DriftObservation(
            session_id=session_id,
            turn_idx=ctx.turn_idx,
            tool_name="__session_end__",
            kwargs_snapshot={
                "expected_terminal_actions": expected,
                "attempted_tools": list(ctx.attempted_tools),
                "executed_tools": list(ctx.executed_tools),
                "missing_terminal_actions": missing,
                "obligation_ledger": ledger,
            },
            verdict_status="clean",
            consistent=False,
            drift_label="missing-action-drift",
            matched_pattern_ids=["missing-action-drift"],
            severity="high",
            would_break_task=True,
            recoverable=False,
            reason=reason,
            confidence="high",
            evidence_pointers=[
                f"expected_terminal_actions={expected}",
                f"attempted_tools={list(ctx.attempted_tools)}",
            ],
            judge_model=self.judge_model,
            judge_latency_ms=0.0,
            error=None,
        )


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #


def _judge_system_prompt() -> str:
    return (
        "You are a runtime monitor for an autonomous agent. Decide whether "
        "the agent's proposed next tool call is consistent with the recent "
        "user messages and trusted tool evidence gathered so far. You are "
        "observation-only — never produce a corrected call. Treat any "
        "advisory session expectation as a weak hint only: it can be wrong, "
        "and it must never override recent user messages or tool evidence.\n\n"
        "Return STRICT JSON only, with these fields:\n"
        "  consistent: bool\n"
        "  drift_label: short kebab-case string of your own choosing "
        "(e.g. 'wrong-terminal-action', 'unjustified-write', "
        "'entity-mismatch', 'value-not-grounded'). Null if consistent=true.\n"
        "  matched_pattern_ids: list of exact pattern_id strings from "
        "diagnostic_patterns that match this call; [] if none. Only match "
        "patterns whose pattern_role is 'failure' and whose tool_scopes include "
        "the proposed tool when tool_scopes is non-empty. "
        "Never invent pattern IDs. Never return a 'negative_calibration' pattern_id.\n"
        "  severity: 'low' | 'medium' | 'high'. Use diagnostic pattern defaults when a pattern matches.\n"
        "  would_break_task: bool. True only when this drift is likely to make the current task fail if uncorrected.\n"
        "  recoverable: bool. True when the agent can likely recover without intervention.\n"
        "  reason: one sentence explaining the verdict.\n"
        "  confidence: 'low' | 'medium' | 'high'.\n"
        "  evidence_pointers: list of short strings citing user messages or "
        "tool results you grounded on (e.g. 'user_msg_2 said \"compensation\"').\n\n"
        "A call is CONSISTENT only when every kwarg value is justified by "
        "an explicit user message or a prior trusted tool result, AND the "
        "tool's action fits the inferred current session intent. For terminal "
        "handoff tools, do an extra check: the call is consistent only if the "
        "recent evidence shows the whole remaining user intent cannot be "
        "handled by available tools. A failed or impossible sub-request is not "
        "enough if the user provided a fallback path or there is still "
        "actionable remaining work. When an advisory expectation lists concrete "
        "non-handoff terminal actions, treat a handoff as drift unless recent "
        "user/tool evidence proves those actions are impossible. When unsure, "
        "set consistent=false with confidence='low' rather than approving. "
        "Calibration rule: diagnostic_patterns with pattern_role='negative_calibration' "
        "are examples of suspicious-looking calls that should usually NOT be "
        "treated as outcome-breaking drift. Use them to reduce false positives, "
        "not as matchable drift IDs. Do not set would_break_task=true for every "
        "suspicious action. Use it only for outcome-breaking patterns, destructive "
        "writes, unrecoverable terminal actions, violated hard constraints, or "
        "skipped required prior obligations. If the call is suspicious but "
        "historically recoverable or low-risk, set consistent=false, "
        "severity='low' or 'medium', would_break_task=false, recoverable=true. "
        "Representation ambiguity is not task-breaking by itself: for example, "
        "a tool payload containing multiple flight entries does not prove a "
        "nonstop-flight constraint was violated unless it contradicts a concrete "
        "date, route, selected option, passenger, price, cabin, payment, baggage, "
        "insurance, cancellation, refund, or certificate constraint."
    )


def _build_judge_payload(
    *,
    tool_name: str,
    kwargs: dict[str, Any],
    session_expectation: SessionExpectation | None,
    recent_user_messages: list[str],
    recent_tool_results: list[dict[str, Any]],
    memory_plan_artifact: str | None,
    diagnostic_patterns: list[dict[str, Any]],
    deterministic_contract_mismatches: list[str],
) -> str:
    payload: dict[str, Any] = {
        "proposed_call": {"tool": tool_name, "kwargs": kwargs},
        "advisory_session_expectation": (session_expectation.model_dump() if session_expectation is not None else None),
        "deterministic_contract_mismatches": deterministic_contract_mismatches,
        "recent_user_messages": recent_user_messages,
        "recent_tool_results": recent_tool_results,
        "diagnostic_patterns": diagnostic_patterns,
    }
    return json.dumps(payload, sort_keys=True, default=str)


# --------------------------------------------------------------------------- #
# Verdict parsing
# --------------------------------------------------------------------------- #


def _parse_verdict(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _normalize_confidence(value: Any) -> Literal["low", "medium", "high"]:
    text = str(value or "").strip().lower()
    if text in {"low", "medium", "high"}:
        return text  # type: ignore[return-value]
    return "low"


def _normalize_pointers(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, (str, int, float))]


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _compact_diagnostic_patterns(patterns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only prompt-useful fields from the offline diagnostic catalog."""
    compact: list[dict[str, Any]] = []
    for entry in patterns:
        pattern_id = entry.get("pattern_id")
        if not isinstance(pattern_id, str) or not pattern_id:
            continue
        compact.append(
            {
                "pattern_id": pattern_id,
                "pattern_role": _normalize_pattern_role(entry),
                "description": entry.get("description") or "",
                "runtime_features": entry.get("runtime_features") or [],
                "tool_scopes": entry.get("tool_scopes") or [],
                "aliases": entry.get("aliases") or [],
                "severity_default": entry.get("severity_default") or "low",
                "recoverability_default": bool(entry.get("recoverability_default", True)),
                "positive_examples": entry.get("positive_examples") or [],
                "negative_examples": entry.get("negative_examples") or [],
            }
        )
    return compact


def _normalize_pattern_role(entry: dict[str, Any]) -> Literal["failure", "negative_calibration"]:
    role = str(entry.get("pattern_role") or "").strip()
    if role in {"failure", "negative_calibration"}:
        return role  # type: ignore[return-value]
    positive_count = entry.get("positive_count")
    if isinstance(positive_count, int) and positive_count <= 0:
        return "negative_calibration"
    return "failure"


def _normalize_matched_pattern_ids(
    value: Any,
    *,
    drift_label: Any,
    pattern_index: dict[str, dict[str, Any]],
    tool_name: str,
) -> list[str]:
    candidates: list[str] = []
    if not isinstance(value, list):
        value = []
    for item in value:
        candidates.append(str(item))
    label = _optional_str(drift_label)
    if label is not None:
        candidates.append(label)

    matched: list[str] = []
    for candidate in candidates:
        pattern_id = _resolve_pattern_id(candidate, pattern_index)
        if pattern_id is None:
            continue
        pattern = pattern_index.get(pattern_id)
        if pattern is None:
            continue
        if pattern.get("pattern_role") != "failure":
            continue
        if not _pattern_allows_tool(pattern, tool_name):
            continue
        if pattern_id not in matched:
            matched.append(pattern_id)
    return matched


def _resolve_pattern_id(candidate: str, pattern_index: dict[str, dict[str, Any]]) -> str | None:
    if candidate in pattern_index:
        return candidate
    for pattern_id, pattern in pattern_index.items():
        aliases = pattern.get("aliases")
        if isinstance(aliases, list) and candidate in aliases:
            return pattern_id
    return None


def _pattern_allows_tool(pattern: dict[str, Any], tool_name: str) -> bool:
    scopes = pattern.get("tool_scopes")
    if not isinstance(scopes, list) or not scopes:
        return True
    for scope in scopes:
        if not isinstance(scope, str) or not scope:
            continue
        if scope.endswith("*") and tool_name.startswith(scope[:-1]):
            return True
        if scope == tool_name:
            return True
    return False


def _expected_write_actions(session_expectation: SessionExpectation) -> list[str]:
    expected: list[str] = []
    for tool_name in session_expectation.expected_terminal_actions:
        if not isinstance(tool_name, str) or not tool_name:
            continue
        if tool_name in {"transfer_to_human_agents", "respond"}:
            continue
        if default_monitor_predicate(tool_name):
            expected.append(tool_name)
    return expected


def _update_obligation_ledger(ctx: SessionContext, expected: list[str]) -> dict[str, dict[str, str]]:
    raw = ctx.extras.setdefault("obligation_ledger", {})
    if not isinstance(raw, dict):
        raw = {}
        ctx.extras["obligation_ledger"] = raw

    ledger: dict[str, dict[str, str]] = {}
    for tool_name in expected:
        entry = raw.get(tool_name)
        if not isinstance(entry, dict):
            entry = {"state": "open", "reason": "expected_terminal_action"}
        ledger[tool_name] = {
            "state": str(entry.get("state") or "open"),
            "reason": str(entry.get("reason") or "expected_terminal_action"),
        }

    attempted = set(ctx.attempted_tools)
    executed = set(ctx.executed_tools)
    user_accepted_handoff = _user_accepted_handoff(ctx.full_transcript) and (
        "transfer_to_human_agents" in attempted or "transfer_to_human_agents" in executed
    )
    direct_action_impossible = _direct_action_impossible(ctx.full_transcript)
    for tool_name, entry in ledger.items():
        if tool_name in attempted or tool_name in executed:
            entry["state"] = "satisfied"
            entry["reason"] = "terminal_action_attempted"
        elif _tool_can_be_closed_by_impossibility(tool_name) and user_accepted_handoff and direct_action_impossible:
            entry["state"] = "accepted_handoff"
            entry["reason"] = "user accepted handoff after impossible direct action"
        elif _tool_can_be_closed_by_impossibility(tool_name) and direct_action_impossible:
            entry["state"] = "impossible"
            entry["reason"] = "transcript says direct action is impossible"
        raw[tool_name] = dict(entry)
    return ledger


def _tool_can_be_closed_by_impossibility(tool_name: str) -> bool:
    return tool_name.startswith(("update_", "cancel_"))


def _contract_constraint_evidence(
    *,
    tool_name: str,
    kwargs: dict[str, Any],
    session_expectation: SessionExpectation | None,
) -> list[str]:
    if session_expectation is None:
        return []
    constraints = session_expectation.user_constraints
    if not constraints:
        return []

    mismatches: list[str] = []
    comparable_fields = (
        "reservation_id",
        "user_id",
        "origin",
        "destination",
        "cabin",
        "insurance",
        "total_baggages",
        "nonfree_baggages",
    )
    for field in comparable_fields:
        if field not in constraints or field not in kwargs:
            continue
        expected = _normalize_contract_value(constraints[field])
        proposed = _normalize_contract_value(kwargs[field])
        if expected is not None and proposed is not None and expected != proposed:
            mismatches.append(f"contract_mismatch.{field}: expected {constraints[field]!r}, proposed {kwargs[field]!r}")

    budget = _coerce_number(constraints.get("budget") or constraints.get("willingness_to_pay"))
    if budget is not None:
        proposed_amount = _max_proposed_amount(kwargs)
        if proposed_amount is not None and proposed_amount > budget:
            mismatches.append(f"contract_mismatch.budget: expected <= {budget:g}, proposed {proposed_amount:g}")

    _ = tool_name
    return mismatches


def _allows_conditional_noop(transcript: str) -> bool:
    text = transcript.lower()
    no_change = (
        "don't make any changes",
        "do not make any changes",
        "dont make any changes",
        "don't make changes",
        "do not make changes",
        "no changes",
    )
    conditional = (
        "if",
        "above",
        "over",
        "exceed",
        "budget",
        "fee",
        "cost",
        "charge",
    )
    return any(phrase in text for phrase in no_change) and any(word in text for word in conditional)


def _user_accepted_handoff(transcript: str) -> bool:
    text = transcript.lower()
    handoff_terms = (
        "human agent",
        "connect me",
        "transfer me",
        "talk to a human",
        "speak to a human",
        "representative",
    )
    acceptance_terms = (
        "yes",
        "please",
        "go ahead",
        "connect",
        "transfer",
    )
    return any(term in text for term in handoff_terms) and any(term in text for term in acceptance_terms)


def _direct_action_impossible(transcript: str) -> bool:
    text = transcript.lower()
    impossibility_terms = (
        "cannot be changed",
        "can't be changed",
        "cannot be modified",
        "can't be modified",
        "cannot modify",
        "can't modify",
        "cannot cancel",
        "can't cancel",
        "not allowed",
        "not eligible",
        "not possible",
    )
    policy_terms = (
        "basic economy",
        "policy",
        "travel insurance",
        "without travel insurance",
        "more than 24 hours",
    )
    return any(term in text for term in impossibility_terms) and any(term in text for term in policy_terms)


_HARD_CONSTRAINT_TERMS: dict[str, tuple[str, ...]] = {
    "budget": (
        "budget",
        "cost",
        "price",
        "amount",
        "fee",
        "charge",
        "total",
        "over",
        "exceed",
        "waste",
    ),
    "passenger_entity": (
        "passenger",
        "traveler",
        "companion",
        "entity",
        "name",
        "dob",
        "remove",
        "wrong person",
        "user_id",
        "reservation_id",
    ),
    "date": (
        "date",
        "departure date",
        "return date",
        "same day",
        "earlier",
        "later",
    ),
    "route": (
        "origin",
        "destination",
        "route",
        "wrong city",
        "from ",
        " to ",
    ),
    "payment": (
        "payment",
        "credit card",
        "gift card",
        "certificate",
        "visa",
        "mastercard",
    ),
    "cabin": (
        "cabin",
        "basic economy",
        "economy",
        "business",
        "upgrade",
    ),
    "baggage": (
        "baggage",
        "bag",
        "checked",
    ),
    "insurance": ("insurance",),
    "cancellation_refund": (
        "cancel",
        "cancellation",
        "refund",
    ),
    "selected_option": (
        "selected",
        "flight number",
        "flight_number",
    ),
}

_REPRESENTATION_AMBIGUITY_TERMS = (
    "multiple flights",
    "multiple flight",
    "multiple segments",
    "multiple legs",
    "one-stop",
    "one stop",
    "nonstop",
    "not confirmed",
    "has not confirmed",
    "not explicitly confirmed",
)


def _hard_constraint_evidence(
    *,
    tool_name: str,
    kwargs: dict[str, Any],
    verdict: dict[str, Any],
    session_expectation: SessionExpectation | None,
    recent_user_messages: list[str],
    recent_tool_results: list[dict[str, Any]],
) -> list[str]:
    """Return typed evidence buckets that can justify task-breaking drift.

    This is intentionally conservative. The judge may be right that a call is
    suspicious, but task-breaking status needs an anchor in a hard contract
    field. Generic "not grounded", "not confirmed", or representation-shape
    complaints do not qualify on their own.
    """
    reason = str(verdict.get("reason") or "")
    evidence = " ".join(str(item) for item in _normalize_pointers(verdict.get("evidence_pointers")))
    text = f"{reason} {evidence}".lower()
    if not text.strip():
        return []

    matches: list[str] = []
    for bucket, terms in _HARD_CONSTRAINT_TERMS.items():
        if any(term in text for term in terms):
            matches.append(bucket)

    if set(matches) <= {"route", "date", "cabin", "selected_option"} and _is_only_representation_ambiguity(text):
        return []

    kwargs_text = json.dumps(kwargs, sort_keys=True, default=str).lower()
    expectation_constraints = session_expectation.user_constraints if session_expectation is not None else {}
    if "payment" in matches and not _contains_any(kwargs_text, ("payment", "card", "gift", "certificate")):
        matches.remove("payment")
    if "budget" in matches and not _contains_any(kwargs_text, ("amount", "payment", "total", "price")):
        matches.remove("budget")
    if "passenger_entity" in matches and not _contains_any(kwargs_text, ("passenger", "user_id", "reservation_id")):
        matches.remove("passenger_entity")

    # When the expectation explicitly captured the field, trust that as an
    # additional anchor. This keeps the validator generic while avoiding pure
    # keyword matching against judge prose.
    constraint_keys = {str(key).lower() for key in expectation_constraints}
    if "route" in matches and not (
        {"origin", "destination", "route"} & constraint_keys or _contains_any(kwargs_text, ("origin", "destination"))
    ):
        matches.remove("route")
    if "date" in matches and not (
        any("date" in key for key in constraint_keys) or _contains_any(kwargs_text, ("date",))
    ):
        matches.remove("date")
    if "cabin" in matches and "cabin" not in kwargs_text:
        matches.remove("cabin")
    if "baggage" in matches and "baggage" not in kwargs_text:
        matches.remove("baggage")
    if "insurance" in matches and "insurance" not in kwargs_text:
        matches.remove("insurance")

    # Tool result/user-message parameters are intentionally accepted only via
    # the judge's cited text above. They are included in the signature so this
    # validator can later grow structured evidence checks without changing the
    # call site.
    _ = tool_name, recent_user_messages, recent_tool_results
    return matches


def _is_only_representation_ambiguity(text: str) -> bool:
    return any(term in text for term in _REPRESENTATION_AMBIGUITY_TERMS)


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _merge_unique(first: list[str], second: list[str]) -> list[str]:
    merged: list[str] = []
    for item in [*first, *second]:
        if item not in merged:
            merged.append(item)
    return merged


def _normalize_contract_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value:g}"
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"yes", "y"}:
        return "true"
    if text in {"no", "n"}:
        return "false"
    return text


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace("$", "").replace(",", "").strip())
        except ValueError:
            return None
    return None


def _max_proposed_amount(value: Any) -> float | None:
    amounts: list[float] = []

    def visit(item: Any, key: str | None = None) -> None:
        if isinstance(item, dict):
            for child_key, child in item.items():
                visit(child, str(child_key).lower())
            return
        if isinstance(item, list):
            for child in item:
                visit(child, key)
            return
        if key in {"amount", "total", "total_cost", "price", "fee", "cost"}:
            amount = _coerce_number(item)
            if amount is not None:
                amounts.append(amount)

    visit(value)
    return max(amounts) if amounts else None


def _complete_json_with_deadline(
    client: ExpectationLLMClient,
    *,
    system_prompt: str,
    user_prompt: str,
) -> str:
    timeout_s = float(getattr(client, "timeout_s", 60.0))
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        client.complete_json,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    try:
        return future.result(timeout=timeout_s)
    except FuturesTimeoutError as exc:
        future.cancel()
        raise TimeoutError(f"semantic judge exceeded {timeout_s:.2f}s deadline") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _error_observation(
    *,
    session_id: str,
    ctx: SessionContext,
    tool_name: str,
    kwargs: dict[str, Any],
    error: str,
    judge_model: str | None,
    latency_ms: float,
) -> DriftObservation:
    verdict_status: Literal["judge_error", "invalid_verdict_json"] = (
        "invalid_verdict_json" if error == "invalid_verdict_json" else "judge_error"
    )
    return DriftObservation(
        session_id=session_id,
        turn_idx=ctx.turn_idx,
        tool_name=tool_name,
        kwargs_snapshot=dict(kwargs),
        verdict_status=verdict_status,
        consistent=None,  # don't claim drift or consistency on detector failure
        drift_label=None,
        matched_pattern_ids=[],
        severity="low",
        would_break_task=False,
        recoverable=True,
        reason="",
        confidence="low",
        evidence_pointers=[],
        judge_model=judge_model,
        judge_latency_ms=latency_ms,
        error=error,
    )
