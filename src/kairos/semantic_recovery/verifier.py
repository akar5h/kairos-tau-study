from __future__ import annotations

from typing import Any, Literal, cast

from kairos.models.semantic_recovery import SemanticDecision, SessionExpectation

_WRITE_PREFIXES = ("update_reservation_",)
_WRITE_TOOLS = {
    "book_reservation",
    "cancel_reservation",
    "send_certificate",
    "transfer_to_human_agents",
}


class RuntimeVerifier:
    """Generic semantic prewrite verifier.

    Hosts provide evidence such as ``expected_next_calls`` after translating
    live tool/search state into one or more safe continuations. Kairos compares
    the proposed write against that evidence and returns a strict one-call
    decision.
    """

    def verify(
        self,
        *,
        tool_name: str,
        kwargs: dict[str, Any],
        session_expectation: SessionExpectation | None,
        evidence: dict[str, Any] | None = None,
    ) -> SemanticDecision:
        """Return a runtime semantic decision for a proposed tool call."""
        if not _is_runtime_target(tool_name):
            return SemanticDecision(verdict="allow", reason="Tool is not a semantic prewrite target.")
        evidence = dict(evidence or {})
        terminal_decision = _terminal_lock_decision(tool_name=tool_name, kwargs=kwargs, evidence=evidence)
        if terminal_decision is not None:
            return terminal_decision

        expected_calls = _expected_next_calls(evidence)
        if not expected_calls:
            return SemanticDecision(
                verdict="allow",
                reason="No trusted expected-next-call evidence is available.",
                evidence_refs=_public_evidence_refs(evidence),
            )

        for expected_call in expected_calls:
            expected_tool = expected_call.get("tool_name")
            expected_kwargs = expected_call.get("kwargs")
            if expected_tool != tool_name:
                continue
            if not isinstance(expected_kwargs, dict):
                continue
            if expected_kwargs == kwargs:
                return SemanticDecision(
                    verdict="allow",
                    reason="Proposed tool call matches trusted expected-next-call evidence.",
                    confidence="high",
                    evidence_refs=_expected_evidence_refs(expected_call, evidence),
                )
            return _block_for_expected_call(
                expected_call=expected_call,
                failure_class="expected_kwargs_mismatch",
                reason="Proposed write differs from the trusted next-call continuation.",
                diffs=_kwargs_diffs(expected_kwargs, kwargs),
                evidence=evidence,
            )

        first_call = expected_calls[0]
        expected_tool = first_call.get("tool_name")
        expected_kwargs = first_call.get("kwargs")
        if isinstance(expected_tool, str) and isinstance(expected_kwargs, dict):
            return _block_for_expected_call(
                expected_call=first_call,
                failure_class="expected_tool_mismatch",
                reason=(
                    f"Expected next tool is {expected_tool}, but the agent proposed {tool_name}."
                    if session_expectation is None
                    else f"Session expectation and live evidence require {expected_tool} before {tool_name}."
                ),
                diffs=[f"tool_name: expected {expected_tool}, got {tool_name}"],
                evidence=evidence,
            )

        return SemanticDecision(
            verdict="observe_only",
            failure_class="incomplete_expected_call",
            reason="Expected-call evidence exists, but it is not a single safe JSON tool call.",
            confidence="low",
            evidence_refs=_public_evidence_refs(evidence),
        )


def _terminal_lock_decision(
    *,
    tool_name: str,
    kwargs: dict[str, Any],
    evidence: dict[str, Any],
) -> SemanticDecision | None:
    lock = evidence.get("terminal_lock")
    if not isinstance(lock, dict) or not lock.get("active"):
        return None
    allowed_tool = lock.get("allowed_tool")
    allowed_kwargs = lock.get("allowed_kwargs")
    if allowed_tool == tool_name and (not isinstance(allowed_kwargs, dict) or allowed_kwargs == kwargs):
        return SemanticDecision(
            verdict="allow",
            reason="Proposed call respects terminal lock.",
            confidence="high",
            evidence_refs={"terminal_lock": lock},
        )
    if isinstance(allowed_tool, str) and isinstance(allowed_kwargs, dict):
        return SemanticDecision(
            verdict="block_with_next_call",
            failure_class="terminal_drift",
            reason="A prior Kairos correction locked the workflow to a specific next call.",
            next_tool=allowed_tool,
            next_kwargs=allowed_kwargs,
            confidence="high",
            evidence_refs={"terminal_lock": lock},
            contract_diffs=[f"tool_name: expected {allowed_tool}, got {tool_name}"],
        )
    return SemanticDecision(
        verdict="observe_only",
        failure_class="terminal_drift",
        reason="A terminal lock is active, but it does not contain one safe next call.",
        confidence="low",
        evidence_refs={"terminal_lock": lock},
    )


def _block_for_expected_call(
    *,
    expected_call: dict[str, Any],
    failure_class: str,
    reason: str,
    diffs: list[str],
    evidence: dict[str, Any],
) -> SemanticDecision:
    return SemanticDecision(
        verdict="block_with_next_call",
        failure_class=failure_class,
        reason=str(expected_call.get("reason") or reason),
        next_tool=str(expected_call["tool_name"]),
        next_kwargs=dict(expected_call["kwargs"]),
        confidence=_confidence(expected_call.get("confidence")),
        evidence_refs=_expected_evidence_refs(expected_call, evidence),
        contract_diffs=diffs,
    )


def _expected_next_calls(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    raw = evidence.get("expected_next_calls")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _expected_evidence_refs(expected_call: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    refs = _public_evidence_refs(evidence)
    call_refs = expected_call.get("evidence_refs")
    if isinstance(call_refs, dict):
        refs["expected_call"] = call_refs
    return refs


def _public_evidence_refs(evidence: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in evidence.items() if key not in {"expected_next_calls"}}


def _kwargs_diffs(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    diffs: list[str] = []
    for key in sorted(set(expected) | set(actual)):
        if expected.get(key) != actual.get(key):
            diffs.append(f"{key}: expected {expected.get(key)!r}, got {actual.get(key)!r}")
    return diffs


def _is_runtime_target(tool_name: str) -> bool:
    return tool_name in _WRITE_TOOLS or tool_name.startswith(_WRITE_PREFIXES)


def _confidence(value: Any) -> Literal["low", "medium", "high"]:
    if value in {"low", "medium", "high"}:
        return cast("Literal['low', 'medium', 'high']", value)
    return "high"
