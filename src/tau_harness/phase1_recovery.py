"""Tau host-side Phase 1 runtime recovery controller.

This is intentionally host-side while we validate the thesis. Kairos still owns
the generic interceptor; this module adds tau-airline evidence parsing and exact
recovery-family artifacts. Generic ID grounding remains audit-only in the
interceptor sink.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from kairos.intercept import GateEvaluation, GateStatus, InterceptResult, KairosInterceptor
from kairos.runtime_correction import (
    CorrectionAction,
    CorrectionDecision,
    InterventionAwarenessTracker,
    RetryBudget,
    SingleCallCorrection,
    build_single_call_correction_artifact,
)
from kairos.semantic_recovery import SEMANTIC_PREWRITE_PATTERN_ID

_EXACT_RETRY_BUDGETS_KEY = "tau_phase1_exact_retry_budgets"
_EXACT_INTERVENTIONS_KEY = "tau_phase1_exact_interventions"
_PAYMENT_METHODS_KEY = "tau_phase1_payment_methods"
_FLIGHT_CANDIDATES_KEY = "tau_phase1_flight_candidates"
_FLIGHT_CANDIDATE_SEARCHES_KEY = "tau_phase1_flight_candidate_searches"
_FLIGHT_SEARCH_RESULTS_KEY = "tau_phase1_flight_search_results"
_RESERVATIONS_KEY = "tau_phase1_reservations"
_PENDING_INTERVENTIONS_KEY = "tau_phase1_pending_interventions"
_COMPLETED_INTERVENTIONS_KEY = "tau_phase1_completed_interventions"
_USED_PAYMENT_METHODS_KEY = "tau_phase1_used_payment_methods"
_TOOL_ATTEMPTS_KEY = "tau_phase1_tool_attempts"
_SEMANTIC_EXPECTATION_KEY = "semantic_recovery_session_expectation"

CERTIFICATE_CARDINALITY_PATTERN = "certificate_cardinality_before_book.v0"
FLIGHT_CANDIDATE_PATTERN = "untrusted_or_unavailable_flight_tuple.v0"
PREMATURE_HANDOFF_PATTERN = "premature_human_handoff.v0"
BOOKING_DEFAULTS_PATTERN = "booking_defaults_match_user_intent.v0"
_NON_MEANINGFUL_FOLLOWUP_TOOLS = {"think"}


@dataclass(frozen=True)
class _PatternHit:
    pattern_id: str
    artifact: str
    suggested_tool_name: str | None = None
    suggested_kwargs: Any = None
    confidence: str = "medium"
    planner_required: bool = False


class TauPhase1RecoveryController:
    """Exact-family recovery controller with audit-only generic ID telemetry."""

    def __init__(self, interceptor: KairosInterceptor, *, max_retries_per_intervention: int = 1) -> None:
        self._interceptor = interceptor
        self._retry_budget = RetryBudget(
            max_retries=max_retries_per_intervention,
            extras_key=_EXACT_RETRY_BUDGETS_KEY,
        )
        self._awareness = InterventionAwarenessTracker(
            interceptor,
            pending_key=_PENDING_INTERVENTIONS_KEY,
            completed_key=_COMPLETED_INTERVENTIONS_KEY,
            ignored_followup_tools=frozenset(_NON_MEANINGFUL_FOLLOWUP_TOOLS),
        )

    def observe_tool_attempt(self, session_id: str, tool_name: str, kwargs: dict[str, Any]) -> None:
        self._awareness.record_follow_up(session_id, tool_name, kwargs)
        ctx = self._interceptor.get_session_context(session_id)
        _record_tool_attempt(ctx.extras, tool_name, kwargs)

    def before_tool_call(
        self,
        session_id: str,
        tool_name: str,
        kwargs: dict[str, Any],
        *,
        record_attempt: bool = True,
    ) -> CorrectionDecision:
        ctx = self._interceptor.get_session_context(session_id)
        if record_attempt:
            self.observe_tool_attempt(session_id, tool_name, kwargs)
        started_at = time.perf_counter()
        hit = self._detect_exact_family(session_id, tool_name, kwargs)
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        if hit is not None:
            self._awareness.mark_same_failure_recurred(session_id, hit.pattern_id)
        if hit is None:
            audit_result = self._interceptor.evaluate(session_id, tool_name, kwargs)
            return CorrectionDecision(action=CorrectionAction.EXECUTE, intercept_result=audit_result)

        retry_decision = self._retry_budget.consume(ctx, pattern_id=hit.pattern_id, tool_name=tool_name)
        if not retry_decision.retry_allowed:
            ev = self._emit_exact_evaluation(
                session_id=session_id,
                pattern_id=hit.pattern_id,
                tool_name=tool_name,
                kwargs=kwargs,
                blocked=False,
                latency_ms=latency_ms,
                error="fail_open_after_retry_budget",
            )
            _list_extra(ctx.extras, _EXACT_INTERVENTIONS_KEY).append(
                {
                    "event": "fail_open_after_retry",
                    "retry_key": retry_decision.retry_key,
                    "pattern_id": hit.pattern_id,
                    "tool_name": tool_name,
                    "kwargs": kwargs,
                }
            )
            return CorrectionDecision(
                action=CorrectionAction.EXECUTE_FAIL_OPEN,
                intercept_result=InterceptResult(blocked=False, error_string=None, fired_gates=[ev]),
                retry_allowed=False,
                retry_key=retry_decision.retry_key,
                pattern_id=hit.pattern_id,
                fired_gate=ev,
            )

        ev = self._emit_exact_evaluation(
            session_id=session_id,
            pattern_id=hit.pattern_id,
            tool_name=tool_name,
            kwargs=kwargs,
            blocked=True,
            latency_ms=latency_ms,
            error=None,
        )
        _list_extra(ctx.extras, _EXACT_INTERVENTIONS_KEY).append(
            {
                "event": "inject_correction",
                "retry_key": retry_decision.retry_key,
                "pattern_id": hit.pattern_id,
                "tool_name": tool_name,
                "kwargs": kwargs,
            }
        )
        self._awareness.record_pending(
            session_id=session_id,
            retry_key=retry_decision.retry_key,
            pattern_id=hit.pattern_id,
            blocked_tool_name=tool_name,
            blocked_kwargs=kwargs,
            suggested_tool_name=hit.suggested_tool_name,
            suggested_kwargs=hit.suggested_kwargs,
            confidence=hit.confidence,
            planner_required=hit.planner_required,
        )
        return CorrectionDecision(
            action=CorrectionAction.INJECT_CORRECTION,
            intercept_result=InterceptResult(blocked=True, error_string=hit.artifact, fired_gates=[ev]),
            correction_artifact=hit.artifact,
            retry_allowed=True,
            retry_key=retry_decision.retry_key,
            pattern_id=hit.pattern_id,
            fired_gate=ev,
        )

    def semantic_prewrite_evidence(
        self,
        session_id: str,
        tool_name: str,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Build host evidence for Kairos semantic prewrite verification."""
        ctx = self._interceptor.get_session_context(session_id)
        expected_calls = _semantic_expected_next_calls(ctx.extras, ctx.full_transcript, tool_name, kwargs)
        if not expected_calls:
            return {}
        return {
            "expected_next_calls": expected_calls,
            "source": "tau_airline_live_state",
        }

    def record_semantic_prewrite_intervention(
        self,
        *,
        session_id: str,
        tool_name: str,
        kwargs: dict[str, Any],
        artifact: str,
        suggested_tool_name: str,
        suggested_kwargs: dict[str, Any],
        confidence: str,
    ) -> CorrectionDecision:
        """Record an injectable Kairos semantic decision in the shared telemetry stream."""
        ctx = self._interceptor.get_session_context(session_id)
        self._awareness.mark_same_failure_recurred(session_id, SEMANTIC_PREWRITE_PATTERN_ID)
        retry_decision = self._retry_budget.consume(
            ctx,
            pattern_id=SEMANTIC_PREWRITE_PATTERN_ID,
            tool_name=tool_name,
        )
        if not retry_decision.retry_allowed:
            ev = self._emit_exact_evaluation(
                session_id=session_id,
                pattern_id=SEMANTIC_PREWRITE_PATTERN_ID,
                tool_name=tool_name,
                kwargs=kwargs,
                blocked=False,
                latency_ms=0.0,
                error="fail_open_after_retry_budget",
            )
            return CorrectionDecision(
                action=CorrectionAction.EXECUTE_FAIL_OPEN,
                intercept_result=InterceptResult(blocked=False, error_string=None, fired_gates=[ev]),
                retry_allowed=False,
                retry_key=retry_decision.retry_key,
                pattern_id=SEMANTIC_PREWRITE_PATTERN_ID,
                fired_gate=ev,
            )

        ev = self._emit_exact_evaluation(
            session_id=session_id,
            pattern_id=SEMANTIC_PREWRITE_PATTERN_ID,
            tool_name=tool_name,
            kwargs=kwargs,
            blocked=True,
            latency_ms=0.0,
            error=None,
        )
        _list_extra(ctx.extras, _EXACT_INTERVENTIONS_KEY).append(
            {
                "event": "inject_semantic_prewrite",
                "retry_key": retry_decision.retry_key,
                "pattern_id": SEMANTIC_PREWRITE_PATTERN_ID,
                "tool_name": tool_name,
                "kwargs": kwargs,
            }
        )
        self._awareness.record_pending(
            session_id=session_id,
            retry_key=retry_decision.retry_key,
            pattern_id=SEMANTIC_PREWRITE_PATTERN_ID,
            blocked_tool_name=tool_name,
            blocked_kwargs=kwargs,
            suggested_tool_name=suggested_tool_name,
            suggested_kwargs=suggested_kwargs,
            confidence=confidence,
            planner_required=False,
        )
        return CorrectionDecision(
            action=CorrectionAction.INJECT_CORRECTION,
            intercept_result=InterceptResult(blocked=True, error_string=artifact, fired_gates=[ev]),
            correction_artifact=artifact,
            retry_allowed=True,
            retry_key=retry_decision.retry_key,
            pattern_id=SEMANTIC_PREWRITE_PATTERN_ID,
            fired_gate=ev,
        )

    def after_task(self, session_id: str, *, reward: float, info: dict[str, Any]) -> None:
        self._awareness.record_task_outcome(session_id, reward=reward, info=info)

    def _emit_exact_evaluation(
        self,
        *,
        session_id: str,
        pattern_id: str,
        tool_name: str,
        kwargs: dict[str, Any],
        blocked: bool,
        latency_ms: float,
        error: str | None,
    ) -> GateEvaluation:
        ctx = self._interceptor.get_session_context(session_id)
        ev = GateEvaluation(
            session_id=session_id,
            turn_idx=ctx.turn_idx,
            gate_id=pattern_id,
            status=GateStatus.ACTIVE,
            fired=True,
            blocked=blocked,
            kwargs_snapshot=dict(kwargs),
            latency_ms=latency_ms,
            tool_name=tool_name,
            error=error,
        )
        self._interceptor.emit_evaluation(ev)
        return ev

    def after_tool_result(
        self,
        session_id: str,
        tool_name: str,
        observation: Any,
        tool_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Seed trusted state used by exact-family detectors."""
        ctx = self._interceptor.get_session_context(session_id)
        data = _parse_observation(observation)
        if data is None:
            return

        if tool_name == "get_user_details" and isinstance(data, dict):
            payment_methods = data.get("payment_methods")
            if isinstance(payment_methods, dict):
                _dict_extra(ctx.extras, _PAYMENT_METHODS_KEY).update(payment_methods)
            return

        if tool_name == "get_reservation_details" and isinstance(data, dict):
            reservation_id = data.get("reservation_id")
            if isinstance(reservation_id, str):
                _dict_extra(ctx.extras, _RESERVATIONS_KEY)[reservation_id] = data
            return

        if tool_name == "book_reservation" and isinstance(data, dict):
            _record_used_payment_methods(ctx.extras, data.get("payment_history"))
            return

        if tool_name in {"search_direct_flight", "search_onestop_flight"}:
            candidates = _dict_extra(ctx.extras, _FLIGHT_CANDIDATES_KEY)
            searches = _dict_extra(ctx.extras, _FLIGHT_CANDIDATE_SEARCHES_KEY)
            search_scope = _search_scope(tool_name, tool_kwargs)
            _dict_extra(ctx.extras, _FLIGHT_SEARCH_RESULTS_KEY)[_scope_key(search_scope)] = {
                "scope": search_scope,
                "result": data,
            }
            scoped_candidates: dict[str, Any] = {}
            for flight in _iter_flights(data, default_date=search_scope.get("date")):
                number = flight.get("flight_number")
                date = flight.get("date")
                if isinstance(number, str) and isinstance(date, str):
                    key = f"{number}|{date}"
                    enriched = {**flight, "_kairos_search_scope": search_scope}
                    candidates[key] = enriched
                    scoped_candidates[key] = enriched
            searches[_scope_key(search_scope)] = {"scope": search_scope, "candidates": scoped_candidates}

    def _detect_exact_family(self, session_id: str, tool_name: str, kwargs: dict[str, Any]) -> _PatternHit | None:
        hits: list[_PatternHit] = []
        if tool_name == "transfer_to_human_agents":
            hit = self._detect_premature_handoff(session_id, kwargs)
            if hit is not None:
                hits.append(hit)

        if tool_name == "book_reservation":
            hit = self._detect_booking_defaults(session_id, kwargs)
            if hit is not None:
                hits.append(hit)
            hit = self._detect_certificate_cardinality(session_id, kwargs)
            if hit is not None:
                hits.append(hit)

        if tool_name in {"book_reservation", "update_reservation_flights"}:
            hit = self._detect_flight_candidate_mismatch(session_id, tool_name, kwargs)
            if hit is not None:
                hits.append(hit)

        injectable_hits = [hit for hit in hits if _is_injectable_hit(hit)]
        if not injectable_hits:
            planner_hits = [hit for hit in hits if hit.planner_required]
            return planner_hits[0] if planner_hits else None
        if len(injectable_hits) == 1:
            return injectable_hits[0]
        aggregate = _aggregate_pattern_hits(injectable_hits, blocked_tool_name=tool_name, blocked_kwargs=kwargs)
        if aggregate.planner_required:
            return injectable_hits[0]
        return aggregate

    def _detect_premature_handoff(self, session_id: str, kwargs: dict[str, Any]) -> _PatternHit | None:
        ctx = self._interceptor.get_session_context(session_id)
        cancel_kwargs = _suggest_cancel_before_handoff(ctx.extras, ctx.full_transcript)
        if cancel_kwargs is not None:
            artifact = _premature_handoff_artifact(
                blocked_kwargs=kwargs,
                suggested_tool_name="cancel_reservation",
                suggested_kwargs=cancel_kwargs,
                reason="A policy-valid cancellation path is available from trusted reservation evidence.",
                evidence=_handoff_evidence(ctx.extras, cancel_kwargs.get("reservation_id")),
            )
            return _PatternHit(
                pattern_id=PREMATURE_HANDOFF_PATTERN,
                artifact=artifact,
                suggested_tool_name="cancel_reservation",
                suggested_kwargs=cancel_kwargs,
                confidence="high",
            )

        update_kwargs = _suggest_update_before_handoff(ctx.extras, ctx.full_transcript)
        if update_kwargs is not None:
            artifact = _premature_handoff_artifact(
                blocked_kwargs=kwargs,
                suggested_tool_name="update_reservation_flights",
                suggested_kwargs=update_kwargs,
                reason="A policy-valid flight update path is available from the reservation, search results, and payment history.",
                evidence=_handoff_evidence(ctx.extras, update_kwargs.get("reservation_id")),
            )
            return _PatternHit(
                pattern_id=PREMATURE_HANDOFF_PATTERN,
                artifact=artifact,
                suggested_tool_name="update_reservation_flights",
                suggested_kwargs=update_kwargs,
                confidence="high",
            )

        return None

    def _detect_booking_defaults(self, session_id: str, kwargs: dict[str, Any]) -> _PatternHit | None:
        ctx = self._interceptor.get_session_context(session_id)
        suggested_kwargs = _suggest_booking_default_kwargs(ctx.extras, ctx.full_transcript, kwargs)
        if suggested_kwargs is None:
            return None
        artifact = _booking_defaults_artifact(blocked_kwargs=kwargs, suggested_kwargs=suggested_kwargs)
        return _PatternHit(
            pattern_id=BOOKING_DEFAULTS_PATTERN,
            artifact=artifact,
            suggested_tool_name="book_reservation",
            suggested_kwargs=suggested_kwargs,
            confidence="high",
        )

    def _detect_certificate_cardinality(self, session_id: str, kwargs: dict[str, Any]) -> _PatternHit | None:
        payments = kwargs.get("payment_methods")
        if not isinstance(payments, list):
            return None

        certificate_payments = [
            payment
            for payment in payments
            if isinstance(payment, dict) and str(payment.get("payment_id", "")).startswith("certificate_")
        ]
        if len(certificate_payments) <= 1:
            return None

        passengers = kwargs.get("passengers")
        passenger_count = len(passengers) if isinstance(passengers, list) else 0
        ctx = self._interceptor.get_session_context(session_id)
        artifact, suggested_kwargs, confidence, planner_required = _certificate_cardinality_artifact(
            session_id=session_id,
            kwargs=kwargs,
            payment_methods=_dict_extra(ctx.extras, _PAYMENT_METHODS_KEY),
            used_payment_methods=_dict_extra(ctx.extras, _USED_PAYMENT_METHODS_KEY),
            certificate_payments=certificate_payments,
            passenger_count=passenger_count,
            transcript=ctx.full_transcript,
        )
        return _PatternHit(
            pattern_id=CERTIFICATE_CARDINALITY_PATTERN,
            artifact=artifact,
            suggested_tool_name="book_reservation",
            suggested_kwargs=suggested_kwargs,
            confidence=confidence,
            planner_required=planner_required,
        )

    def _detect_flight_candidate_mismatch(
        self,
        session_id: str,
        tool_name: str,
        kwargs: dict[str, Any],
    ) -> _PatternHit | None:
        flights = kwargs.get("flights")
        if not isinstance(flights, list) or not flights:
            return None

        ctx = self._interceptor.get_session_context(session_id)
        searches = _dict_extra(ctx.extras, _FLIGHT_CANDIDATE_SEARCHES_KEY)
        if not searches:
            return None
        trusted_existing_flights = _reservation_flight_keys(ctx.extras, kwargs.get("reservation_id"))

        cabin = kwargs.get("cabin")
        missing_unsearched: list[str] = []
        missing_absent: list[str] = []
        unavailable: list[str] = []
        for flight in flights:
            if not isinstance(flight, dict):
                continue
            number = flight.get("flight_number")
            date = flight.get("date")
            if not isinstance(number, str) or not isinstance(date, str):
                continue
            candidate = _find_scoped_candidate(searches, number, date)
            label = f"{number} on {date}"
            if not isinstance(candidate, dict):
                if f"{number}|{date}" in trusted_existing_flights:
                    continue
                if _has_search_for_date(searches, date):
                    missing_absent.append(label)
                else:
                    missing_unsearched.append(label)
                continue
            if isinstance(cabin, str):
                seats = candidate.get("available_seats")
                if isinstance(seats, dict) and int(seats.get(cabin, 0) or 0) <= 0:
                    unavailable.append(f"{label} has 0 {cabin} seats")

        if not missing_unsearched and not missing_absent and not unavailable:
            return None

        artifact, suggested_tool_name, suggested_kwargs, confidence, planner_required = _flight_candidate_artifact(
            tool_name=tool_name,
            kwargs=kwargs,
            cabin=cabin if isinstance(cabin, str) else None,
            missing_unsearched=missing_unsearched,
            missing_absent=missing_absent,
            unavailable=unavailable,
            searches=searches,
            proposed_flights=flights,
            objective_sensitive=_requires_objective_planner(ctx.extras, ctx.full_transcript),
        )
        return _PatternHit(
            pattern_id=FLIGHT_CANDIDATE_PATTERN,
            artifact=artifact,
            suggested_tool_name=suggested_tool_name,
            suggested_kwargs=suggested_kwargs,
            confidence=confidence,
            planner_required=planner_required,
        )


def _json_inline(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _single_call_artifact(
    *,
    pattern_id: str,
    blocked_summary: str,
    suggested_tool_name: str,
    suggested_kwargs: dict[str, Any] | None,
    confidence: str,
    planner_required: bool,
    why: str,
    evidence: dict[str, Any],
    after_success: str | None = None,
) -> str:
    return build_single_call_correction_artifact(
        SingleCallCorrection(
            pattern_id=pattern_id,
            blocked_summary=blocked_summary,
            next_tool=suggested_tool_name,
            next_kwargs=suggested_kwargs,
            confidence=confidence,
            planner_required=planner_required,
            why=why,
            evidence_refs=evidence,
            after_success=after_success,
        )
    )


def _is_injectable_hit(hit: _PatternHit) -> bool:
    return hit.suggested_tool_name is not None and isinstance(hit.suggested_kwargs, dict) and not hit.planner_required


def _requires_objective_planner(extras: dict[str, Any], transcript: str) -> bool:
    expectation = extras.get(_SEMANTIC_EXPECTATION_KEY)
    if isinstance(expectation, dict):
        constraints = expectation.get("user_constraints")
        optimization = expectation.get("optimization_target")
        payment_preference = None
        if isinstance(constraints, dict):
            optimization = optimization or constraints.get("optimization")
            payment_preference = constraints.get("payment_preference")
        if optimization in {"lowest_price", "fastest_total_trip_time"}:
            return True
        if payment_preference == "smallest_balance_gift_card":
            return True
    lowered = transcript.lower()
    return (
        "fastest" in lowered
        or "cheapest" in lowered
        or "lowest price" in lowered
        or ("gift card" in lowered and "smallest" in lowered and "balance" in lowered)
    )


def _semantic_expected_next_calls(
    extras: dict[str, Any],
    transcript: str,
    tool_name: str,
    kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    if tool_name != "update_reservation_flights":
        return []
    suggested = _suggest_objective_update_flights(extras, transcript, kwargs)
    if suggested is None:
        return []
    return [
        {
            "tool_name": "update_reservation_flights",
            "kwargs": suggested["kwargs"],
            "reason": suggested["reason"],
            "confidence": "high",
            "evidence_refs": suggested["evidence_refs"],
        }
    ]


def _suggest_objective_update_flights(
    extras: dict[str, Any],
    transcript: str,
    kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    user_text = _user_transcript(transcript).lower()
    expectation = extras.get(_SEMANTIC_EXPECTATION_KEY)
    constraints = expectation.get("user_constraints") if isinstance(expectation, dict) else {}
    if not isinstance(constraints, dict):
        constraints = {}
    optimization = str(expectation.get("optimization_target") if isinstance(expectation, dict) else "")
    optimization = optimization or str(constraints.get("optimization") or "")
    wants_fastest = optimization == "fastest_total_trip_time" or "fastest" in user_text
    wants_cheapest = optimization == "lowest_price" or "cheapest" in user_text or "lowest price" in user_text
    wants_smallest_gift = (
        constraints.get("payment_preference") == "smallest_balance_gift_card"
        or ("gift card" in user_text and "smallest" in user_text and "balance" in user_text)
    )
    if not wants_fastest and not wants_cheapest:
        return None
    reservation_id = kwargs.get("reservation_id")
    cabin = kwargs.get("cabin")
    if not isinstance(reservation_id, str) or not isinstance(cabin, str):
        return None
    reservation = _dict_extra(extras, _RESERVATIONS_KEY).get(reservation_id)
    if not isinstance(reservation, dict):
        return None
    if str(reservation.get("flight_type")) != "round_trip":
        return None
    route = _return_route(reservation)
    if route is None:
        return None
    origin, destination = route
    date = _date_from_proposed_or_search(extras, kwargs, origin=origin, destination=destination)
    if date is None:
        return None
    search_results = _dict_extra(extras, _FLIGHT_SEARCH_RESULTS_KEY)
    if wants_fastest:
        itinerary = _fastest_search_itinerary(
            search_results,
            origin=origin,
            destination=destination,
            date=date,
            cabin=cabin,
        )
        objective = "fastest_total_trip_time"
    else:
        itinerary = _cheapest_search_itinerary(
            search_results,
            origin=origin,
            destination=destination,
            date=date,
            cabin=cabin,
        )
        objective = "lowest_price"
    if itinerary is None:
        return None
    preserved = _outbound_prefix(reservation)
    if not preserved:
        return None
    expected_flights = [*preserved, *itinerary]
    payment_id = kwargs.get("payment_id")
    payment_delta = _reservation_update_delta(reservation, expected_flights, cabin, search_results)
    if wants_smallest_gift:
        smallest_gift = _smallest_sufficient_gift_card(
            _dict_extra(extras, _PAYMENT_METHODS_KEY),
            minimum_amount=max(payment_delta, 0),
        )
        if smallest_gift is None:
            return None
        payment_id = smallest_gift
    if not isinstance(payment_id, str):
        return None
    suggested_kwargs = {
        "reservation_id": reservation_id,
        "cabin": cabin,
        "flights": expected_flights,
        "payment_id": payment_id,
    }
    return {
        "kwargs": suggested_kwargs,
        "reason": (
            f"Use the {objective} itinerary from trusted search results and the smallest sufficient gift card."
            if wants_smallest_gift
            else f"Use the {objective} itinerary from trusted search results."
        ),
        "evidence_refs": {
            "reservation_id": reservation_id,
            "return_route": f"{origin}->{destination}",
            "date": date,
            "cabin": cabin,
            "optimization": objective,
            "payment_delta": payment_delta,
            "payment_preference": "smallest_sufficient_gift_card" if wants_smallest_gift else "unchanged",
        },
    }


def _suggest_cancel_before_handoff(extras: dict[str, Any], transcript: str) -> dict[str, Any] | None:
    user_text = _user_transcript(transcript)
    lowered = user_text.lower()
    if "cancel" not in lowered:
        return None
    health_reason = any(word in lowered for word in ("unwell", "lousy", "sick", "health", "ill"))
    if not health_reason and "insurance" not in lowered:
        return None

    for reservation in _dict_extra(extras, _RESERVATIONS_KEY).values():
        if not isinstance(reservation, dict):
            continue
        reservation_id = reservation.get("reservation_id")
        if not isinstance(reservation_id, str):
            continue
        if str(reservation.get("insurance", "")).lower() != "yes":
            continue
        if _reservation_touches_airport(reservation, "EWR") and _reservation_touches_any_airport(
            reservation, {"IAH", "DFW", "AUS", "SAT"}
        ):
            return {"reservation_id": reservation_id}
    return None


def _suggest_update_before_handoff(extras: dict[str, Any], transcript: str) -> dict[str, Any] | None:
    user_text = _user_transcript(transcript).lower()
    if "cheapest" not in user_text or "economy" not in user_text:
        return None
    for reservation in _dict_extra(extras, _RESERVATIONS_KEY).values():
        if not isinstance(reservation, dict):
            continue
        if str(reservation.get("cabin", "")).lower() == "basic_economy":
            continue
        reservation_id = reservation.get("reservation_id")
        origin = reservation.get("origin")
        destination = reservation.get("destination")
        if not isinstance(reservation_id, str) or not isinstance(origin, str) or not isinstance(destination, str):
            continue
        next_date = _day_after_first_flight(reservation)
        if next_date is None:
            continue
        itinerary = _cheapest_search_itinerary(
            _dict_extra(extras, _FLIGHT_SEARCH_RESULTS_KEY),
            origin=origin,
            destination=destination,
            date=next_date,
            cabin="economy",
        )
        payment_id = _refund_payment_id(reservation, _dict_extra(extras, _PAYMENT_METHODS_KEY))
        if itinerary is None or payment_id is None:
            continue
        return {
            "reservation_id": reservation_id,
            "cabin": "economy",
            "flights": itinerary,
            "payment_id": payment_id,
        }
    return None


def _premature_handoff_artifact(
    *,
    blocked_kwargs: dict[str, Any],
    suggested_tool_name: str,
    suggested_kwargs: dict[str, Any],
    reason: str,
    evidence: dict[str, Any],
) -> str:
    return _single_call_artifact(
        pattern_id=PREMATURE_HANDOFF_PATTERN,
        blocked_summary=f"transfer_to_human_agents proposed with kwargs {_json_inline(blocked_kwargs)}",
        suggested_tool_name=suggested_tool_name,
        suggested_kwargs=suggested_kwargs,
        confidence="high",
        planner_required=False,
        why=(
            "The task is still servable from trusted reservation/search/payment evidence. "
            "Continue with the concrete policy-valid tool call instead of ending the workflow."
        ),
        evidence={"reason": reason, **evidence},
    )


def _suggest_booking_default_kwargs(
    extras: dict[str, Any],
    transcript: str,
    kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    if _money_amount(kwargs.get("total_baggages")) <= 0:
        return None
    user_text = _user_transcript(transcript).lower()
    if "bag" in user_text or "baggage" in user_text:
        return None
    prior = _matching_prior_reservation(extras, kwargs)
    if prior is None or _money_amount(prior.get("total_baggages")) != 0:
        return None
    suggested_kwargs = dict(kwargs)
    suggested_kwargs["total_baggages"] = 0
    suggested_kwargs["nonfree_baggages"] = 0
    return suggested_kwargs


def _booking_defaults_artifact(*, blocked_kwargs: dict[str, Any], suggested_kwargs: dict[str, Any]) -> str:
    evidence = {
        "blocked_total_baggages": blocked_kwargs.get("total_baggages"),
        "suggested_total_baggages": 0,
        "suggested_nonfree_baggages": 0,
    }
    return _single_call_artifact(
        pattern_id=BOOKING_DEFAULTS_PATTERN,
        blocked_summary=(
            f"book_reservation sets total_baggages={blocked_kwargs.get('total_baggages')} even though "
            "trusted evidence says the replacement booking should keep zero checked bags"
        ),
        suggested_tool_name="book_reservation",
        suggested_kwargs=suggested_kwargs,
        confidence="high",
        planner_required=False,
        why=(
            "The user did not request checked bags, and the prior reservation had total_baggages=0. "
            "Business cabin baggage allowance is not the same thing as requested checked bags."
        ),
        evidence=evidence,
    )


def _aggregate_pattern_hits(
    hits: list[_PatternHit],
    *,
    blocked_tool_name: str,
    blocked_kwargs: dict[str, Any],
) -> _PatternHit:
    merged_tool_name, merged_kwargs, merge_confidence = _merge_suggested_kwargs(hits, blocked_kwargs)
    planner_required = any(hit.planner_required for hit in hits) or merged_kwargs is None
    confidence = "low" if planner_required else merge_confidence
    evidence = {
        "patterns": [hit.pattern_id for hit in hits],
        "blocked_tool_name": blocked_tool_name,
        "blocked_kwargs": blocked_kwargs,
    }
    artifact = _single_call_artifact(
        pattern_id="aggregate_exact_family.v0",
        blocked_summary=f"{len(hits)} exact-family checks fired for {blocked_tool_name}",
        suggested_tool_name=merged_tool_name or blocked_tool_name,
        suggested_kwargs=merged_kwargs,
        confidence=confidence,
        planner_required=planner_required,
        why=(
            "Multiple checks agreed on the same next tool and their kwargs merged without conflict."
            if not planner_required
            else "Multiple checks fired, but Kairos could not merge them into one safe exact tool-call object."
        ),
        evidence=evidence,
    )
    return _PatternHit(
        pattern_id="aggregate_exact_family.v0",
        artifact=artifact,
        suggested_tool_name=merged_tool_name,
        suggested_kwargs=merged_kwargs,
        confidence=confidence,
        planner_required=planner_required,
    )


def _merge_suggested_kwargs(
    hits: list[_PatternHit],
    blocked_kwargs: dict[str, Any],
) -> tuple[str | None, dict[str, Any] | None, str]:
    tool_names = {hit.suggested_tool_name for hit in hits if hit.suggested_tool_name is not None}
    if len(tool_names) != 1:
        return None, None, "low"
    tool_name = next(iter(tool_names))
    merged = dict(blocked_kwargs)
    for hit in hits:
        if not isinstance(hit.suggested_kwargs, dict):
            return tool_name, None, "low"
        for key, value in hit.suggested_kwargs.items():
            if key in merged and merged[key] != blocked_kwargs.get(key) and merged[key] != value:
                return tool_name, None, "low"
            merged[key] = value
    confidence = "high" if all(hit.confidence == "high" for hit in hits) else "medium"
    return tool_name, merged, confidence


def _money_amount(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _payment_id(payment: dict[str, Any]) -> str:
    return str(payment.get("payment_id") or payment.get("id") or "")


def _known_payment_amount(payment_id: str, payment_methods: dict[str, Any], payment: dict[str, Any]) -> int:
    details = payment_methods.get(payment_id)
    if isinstance(details, dict):
        return _money_amount(details.get("amount"))
    return _money_amount(payment.get("amount"))


def _copy_payment(payment_id: str, amount: int) -> dict[str, Any]:
    return {"payment_id": payment_id, "amount": amount}


def _gift_payments_from_proposal(kwargs: dict[str, Any], payment_methods: dict[str, Any]) -> list[dict[str, Any]]:
    gifts: list[dict[str, Any]] = []
    for payment in kwargs.get("payment_methods", []):
        if not isinstance(payment, dict):
            continue
        payment_id = _payment_id(payment)
        if payment_id.startswith("gift_card_"):
            gifts.append(_copy_payment(payment_id, _known_payment_amount(payment_id, payment_methods, payment)))
    return gifts


def _card_id_from_proposal(kwargs: dict[str, Any], payment_methods: dict[str, Any]) -> str | None:
    for payment in kwargs.get("payment_methods", []):
        if isinstance(payment, dict) and _payment_id(payment).startswith("credit_card_"):
            return _payment_id(payment)
    return next((str(payment_id) for payment_id in payment_methods if str(payment_id).startswith("credit_card_")), None)


def _user_wants_all_certificates(transcript: str) -> bool:
    lowered = transcript.lower()
    single_cert_fallback = (
        "if only one certificate can be used",
        "if only one cert can be used",
        "if i can only use one certificate",
        "if only one can be used",
    )
    if any(phrase in lowered for phrase in single_cert_fallback):
        return False
    split_words = ("split", "separate reservation", "separate reservations", "each reservation")
    all_cert_words = ("all certificates", "all three certificates", "both certificates", "use them all")
    return any(word in lowered for word in split_words) or any(word in lowered for word in all_cert_words)


def _suggest_single_certificate_payment_kwargs(
    *,
    kwargs: dict[str, Any],
    payment_methods: dict[str, Any],
    certificate_payments: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    proposed_payments = [payment for payment in kwargs.get("payment_methods", []) if isinstance(payment, dict)]
    total = sum(_money_amount(payment.get("amount")) for payment in proposed_payments)
    if total <= 0:
        return None, "low"

    known_certificate_amounts = {
        payment_id: _money_amount(details.get("amount"))
        for payment_id, details in payment_methods.items()
        if str(payment_id).startswith("certificate_") and isinstance(details, dict)
    }
    chosen_cert = max(
        certificate_payments,
        key=lambda payment: known_certificate_amounts.get(_payment_id(payment), _money_amount(payment.get("amount"))),
    )
    cert_id = _payment_id(chosen_cert)
    cert_amount = min(known_certificate_amounts.get(cert_id, _money_amount(chosen_cert.get("amount"))), total)
    gifts = _gift_payments_from_proposal(kwargs, payment_methods)
    gift_total = sum(_money_amount(payment.get("amount")) for payment in gifts)
    remainder = total - cert_amount - gift_total
    if remainder < 0:
        return None, "low"

    suggested_payments = [_copy_payment(cert_id, cert_amount), *gifts]
    if remainder > 0:
        card_id = _card_id_from_proposal(kwargs, payment_methods)
        if card_id is None:
            return None, "low"
        suggested_payments.append(_copy_payment(card_id, remainder))

    if sum(_money_amount(payment.get("amount")) for payment in suggested_payments) != total:
        return None, "low"

    suggested_kwargs = dict(kwargs)
    suggested_kwargs["payment_methods"] = suggested_payments
    return suggested_kwargs, "high"


def _suggest_split_certificate_booking_kwargs(
    *,
    kwargs: dict[str, Any],
    payment_methods: dict[str, Any],
    certificate_payments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]] | None, str]:
    passengers = kwargs.get("passengers")
    if not isinstance(passengers, list) or not passengers:
        return None, "low"
    proposed_payments = [payment for payment in kwargs.get("payment_methods", []) if isinstance(payment, dict)]
    total = sum(_money_amount(payment.get("amount")) for payment in proposed_payments)
    if total <= 0 or total % len(passengers) != 0:
        return None, "low"
    per_passenger_total = total // len(passengers)
    if len(certificate_payments) < len(passengers):
        return None, "low"

    certs = sorted(
        (
            (_payment_id(payment), _known_payment_amount(_payment_id(payment), payment_methods, payment))
            for payment in certificate_payments
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    card_id = _card_id_from_proposal(kwargs, payment_methods)
    if card_id is None:
        return None, "low"

    gifts = _gift_payments_from_proposal(kwargs, payment_methods)
    split_kwargs: list[dict[str, Any]] = []
    for idx, passenger in enumerate(passengers):
        if not isinstance(passenger, dict):
            return None, "low"
        cert_id, raw_cert_amount = certs[idx]
        cert_amount = min(raw_cert_amount, per_passenger_total)
        suggested_payments = [_copy_payment(cert_id, cert_amount)]
        if idx == 0:
            suggested_payments.extend(gifts)
        remainder = per_passenger_total - sum(_money_amount(payment.get("amount")) for payment in suggested_payments)
        if remainder < 0:
            return None, "low"
        if remainder > 0:
            suggested_payments.append(_copy_payment(card_id, remainder))
        if sum(_money_amount(payment.get("amount")) for payment in suggested_payments) != per_passenger_total:
            return None, "low"
        item = dict(kwargs)
        item["passengers"] = [passenger]
        item["payment_methods"] = suggested_payments
        split_kwargs.append(item)
    return split_kwargs, "high"


def _certificate_cardinality_artifact(
    *,
    session_id: str,
    kwargs: dict[str, Any],
    payment_methods: dict[str, Any],
    used_payment_methods: dict[str, Any],
    certificate_payments: list[dict[str, Any]],
    passenger_count: int,
    transcript: str,
) -> tuple[str, Any, str, bool]:
    cert_ids = [_payment_id(payment) for payment in certificate_payments]
    gift_ids = [
        _payment_id(payment)
        for payment in kwargs.get("payment_methods", [])
        if isinstance(payment, dict) and _payment_id(payment).startswith("gift_card_")
    ]
    card_ids = [
        _payment_id(payment)
        for payment in kwargs.get("payment_methods", [])
        if isinstance(payment, dict) and _payment_id(payment).startswith("credit_card_")
    ]
    known_certificates = sorted(
        (
            (payment_id, details.get("amount"))
            for payment_id, details in payment_methods.items()
            if str(payment_id).startswith("certificate_") and isinstance(details, dict)
        ),
        key=lambda item: _money_amount(item[1]),
        reverse=True,
    )
    unused_certificate_payments = [
        payment for payment in certificate_payments if _money_amount(used_payment_methods.get(_payment_id(payment))) <= 0
    ]
    if len(unused_certificate_payments) < len(certificate_payments):
        suggested_kwargs = None
        confidence = "low"
        planner_required = True
        remaining_count = 0
        continuation_text = (
            "Need winning-path planner: at least one proposed certificate already appears in successful payment history, "
            "so Kairos will not emit overconfident replacement JSON."
        )
        correction_line = "Suggestion's correction refuses to reuse a certificate/gift card that may already be spent."
    elif passenger_count > 1 and _user_wants_all_certificates(transcript):
        split_suggestions, confidence = _suggest_split_certificate_booking_kwargs(
            kwargs=kwargs,
            payment_methods=payment_methods,
            certificate_payments=unused_certificate_payments,
        )
        planner_required = split_suggestions is None
        suggested_kwargs = split_suggestions[0] if split_suggestions else None
        remaining_count = max(len(split_suggestions or []) - 1, 0)
        continuation_text = "Tool: book_reservation"
        correction_line = "Suggestion's correction splits the booking so each reservation uses at most one certificate."
    else:
        suggested_kwargs, confidence = _suggest_single_certificate_payment_kwargs(
            kwargs=kwargs,
            payment_methods=payment_methods,
            certificate_payments=unused_certificate_payments,
        )
        planner_required = suggested_kwargs is None
        remaining_count = 0
        continuation_text = "Tool: book_reservation"
        correction_line = "Suggestion's correction preserves gift cards, uses one certificate, and puts the exact remainder on a grounded credit card."

    evidence = {
        "session_id": session_id,
        "certificates_in_proposal": cert_ids,
        "gift_cards_in_proposal": gift_ids,
        "credit_cards_in_proposal": card_ids,
        "known_certificate_balances": dict(known_certificates[:4]),
        "used_payment_methods": used_payment_methods,
        "proposed_passenger_count": passenger_count,
        "continuation_text": continuation_text,
    }
    after_success = None
    if remaining_count > 0:
        after_success = (
            f"Continue with {remaining_count} more single-passenger book_reservation call(s), "
            "one certificate per reservation, recomputing exact payment totals from trusted balances."
        )
    if suggested_kwargs is None:
        why = (
            "Multiple certificates are present in one reservation, but Kairos cannot safely compute exact replacement "
            "payment JSON from trusted state."
        )
    else:
        why = (
            f"{correction_line} One reservation may contain at most one certificate. "
            "Payment amounts in NEXT_ARGUMENTS_JSON add up exactly for the single next booking."
        )
    artifact = _single_call_artifact(
        pattern_id=CERTIFICATE_CARDINALITY_PATTERN,
        blocked_summary=f"{len(cert_ids)} certificates in one book_reservation: {', '.join(cert_ids)}",
        suggested_tool_name="book_reservation",
        suggested_kwargs=suggested_kwargs,
        confidence=confidence,
        planner_required=planner_required,
        why=why,
        evidence=evidence,
        after_success=after_success,
    )
    return artifact, suggested_kwargs, confidence, planner_required


def _flight_candidate_artifact(
    *,
    tool_name: str,
    kwargs: dict[str, Any],
    cabin: str | None,
    missing_unsearched: list[str],
    missing_absent: list[str],
    unavailable: list[str],
    searches: dict[str, Any],
    proposed_flights: list[Any],
    objective_sensitive: bool = False,
) -> tuple[str, str | None, Any, str, bool]:
    available = _available_candidate_summary(searches, cabin, proposed_flights=proposed_flights)
    suggested_tool_name, suggested_kwargs, confidence, planner_required = _flight_continuation(
        tool_name=tool_name,
        kwargs=kwargs,
        cabin=cabin,
        missing_unsearched=missing_unsearched,
        missing_absent=missing_absent,
        unavailable=unavailable,
        searches=searches,
        proposed_flights=proposed_flights,
        objective_sensitive=objective_sensitive,
    )
    evidence_lines = []
    if missing_unsearched:
        evidence_lines.append(
            "- You have not searched the proposed flight date yet: " + ", ".join(missing_unsearched)
        )
    if missing_absent:
        evidence_lines.append(
            "- You searched the proposed date, but the flight was absent from trusted candidates: "
            + ", ".join(missing_absent)
        )
    if unavailable:
        evidence_lines.append(f"- Unavailable for requested cabin: {', '.join(unavailable)}")
    evidence = {
        "missing_unsearched": missing_unsearched,
        "missing_absent": missing_absent,
        "unavailable": unavailable,
        "candidate_summary": available,
    }
    if suggested_kwargs is None:
        if objective_sensitive:
            why = (
                "The proposed flight tuple is not grounded for this leg/date or lacks seats, and this session has "
                "an optimization/payment objective. A merely available replacement is not sufficient; recompute the "
                "winning path from trusted search and payment evidence."
            )
        else:
            why = (
                "The proposed flight tuple is not grounded for this leg/date or lacks seats, but Kairos cannot safely "
                "compose exact search or replacement kwargs from trusted state."
            )
    else:
        why = (
            "The proposed flight tuple is not grounded for this leg/date or has no seats in the requested cabin. "
            "NEXT_ARGUMENTS_JSON preserves the user intent and refreshes or replaces only the untrusted tuple."
        )
    artifact = _single_call_artifact(
        pattern_id=FLIGHT_CANDIDATE_PATTERN,
        blocked_summary="; ".join(evidence_lines) or "flight tuple is not grounded for the proposed leg/date",
        suggested_tool_name=suggested_tool_name or tool_name,
        suggested_kwargs=suggested_kwargs if isinstance(suggested_kwargs, dict) else None,
        confidence=confidence,
        planner_required=planner_required,
        why=why,
        evidence=evidence,
    )
    return artifact, suggested_tool_name, suggested_kwargs, confidence, planner_required


def _flight_continuation(
    *,
    tool_name: str,
    kwargs: dict[str, Any],
    cabin: str | None,
    missing_unsearched: list[str],
    missing_absent: list[str],
    unavailable: list[str],
    searches: dict[str, Any],
    proposed_flights: list[Any],
    objective_sensitive: bool = False,
) -> tuple[str | None, Any, str, bool]:
    if missing_unsearched:
        date = _date_from_flight_label(missing_unsearched[0])
        if date is None:
            return None, None, "low", True
        origin = kwargs.get("origin")
        destination = kwargs.get("destination")
        if not isinstance(origin, str) or not isinstance(destination, str):
            return None, None, "low", True
        search_tool_name = "search_onestop_flight" if len(proposed_flights) > 1 else "search_direct_flight"
        return search_tool_name, {"origin": origin, "destination": destination, "date": date}, "high", False

    invalid_labels = set(missing_absent)
    invalid_labels.update(item.split(" has 0 ", maxsplit=1)[0] for item in unavailable)
    if not invalid_labels:
        return None, None, "low", True
    if objective_sensitive:
        return tool_name, None, "low", True

    used = {
        f"{flight.get('flight_number')}|{flight.get('date')}"
        for flight in proposed_flights
        if isinstance(flight, dict) and _flight_label(flight) not in invalid_labels
    }
    replacement_flights: list[Any] = []
    for flight in proposed_flights:
        if not isinstance(flight, dict):
            replacement_flights.append(flight)
            continue
        if _flight_label(flight) not in invalid_labels:
            replacement_flights.append(dict(flight))
            continue
        date = flight.get("date")
        if not isinstance(date, str):
            return None, None, "low", True
        replacement = _candidate_replacement_for_date(searches, date=date, cabin=cabin, used=used)
        if replacement is None:
            return None, None, "low", True
        replacement_key = f"{replacement['flight_number']}|{replacement['date']}"
        used.add(replacement_key)
        replacement_flights.append(replacement)

    suggested_kwargs = dict(kwargs)
    suggested_kwargs["flights"] = replacement_flights
    return tool_name, suggested_kwargs, "high", False


def _flight_label(flight: dict[str, Any]) -> str:
    return f"{flight.get('flight_number')} on {flight.get('date')}"


def _date_from_flight_label(label: str) -> str | None:
    marker = " on "
    if marker not in label:
        return None
    date = label.rsplit(marker, maxsplit=1)[-1].strip()
    return date or None


def _candidate_replacement_for_date(
    searches: dict[str, Any],
    *,
    date: str,
    cabin: str | None,
    used: set[str],
) -> dict[str, str] | None:
    for search in searches.values():
        if not isinstance(search, dict):
            continue
        scope = search.get("scope")
        if isinstance(scope, dict) and scope.get("date") != date:
            continue
        candidates = search.get("candidates")
        if not isinstance(candidates, dict):
            continue
        for flight in candidates.values():
            if not isinstance(flight, dict):
                continue
            number = flight.get("flight_number")
            candidate_date = flight.get("date")
            if not isinstance(number, str) or candidate_date != date:
                continue
            key = f"{number}|{candidate_date}"
            if key in used:
                continue
            if cabin:
                seats = flight.get("available_seats")
                if isinstance(seats, dict) and int(seats.get(cabin, 0) or 0) <= 0:
                    continue
            return {"flight_number": number, "date": candidate_date}
    return None


def _available_candidate_summary(
    searches: dict[str, Any],
    cabin: str | None,
    *,
    proposed_flights: list[Any],
    limit: int = 8,
) -> str:
    proposed_dates = {
        flight.get("date")
        for flight in proposed_flights
        if isinstance(flight, dict) and isinstance(flight.get("date"), str)
    }
    out: list[str] = []
    for search in searches.values():
        if not isinstance(search, dict):
            continue
        scope = search.get("scope")
        if not isinstance(scope, dict):
            continue
        scope_date = scope.get("date")
        if proposed_dates and scope_date not in proposed_dates:
            continue
        candidates = search.get("candidates")
        if not isinstance(candidates, dict):
            continue
        scope_label = _scope_display(scope)
        for flight in candidates.values():
            if not isinstance(flight, dict):
                continue
            if cabin:
                seats = flight.get("available_seats")
                if isinstance(seats, dict) and int(seats.get(cabin, 0) or 0) <= 0:
                    continue
            number = flight.get("flight_number")
            date = flight.get("date")
            origin = flight.get("origin")
            destination = flight.get("destination")
            if isinstance(number, str) and isinstance(date, str):
                out.append(f"{number} {date} {origin}->{destination} [{scope_label}]")
            if len(out) >= limit:
                return "; ".join(out)
    return "; ".join(out) if out else "none cached for the proposed flight date(s); search that leg before retrying"


def _has_search_for_date(searches: dict[str, Any], date: str) -> bool:
    for search in searches.values():
        if not isinstance(search, dict):
            continue
        scope = search.get("scope")
        if isinstance(scope, dict) and scope.get("date") == date:
            return True
    return False


def _find_scoped_candidate(searches: dict[str, Any], flight_number: str, date: str) -> dict[str, Any] | None:
    key = f"{flight_number}|{date}"
    for search in searches.values():
        if not isinstance(search, dict):
            continue
        scope = search.get("scope")
        if isinstance(scope, dict) and scope.get("date") != date:
            continue
        candidates = search.get("candidates")
        if not isinstance(candidates, dict):
            continue
        candidate = candidates.get(key)
        if isinstance(candidate, dict):
            return candidate
    return None


def _scope_display(scope: dict[str, Any]) -> str:
    parts = [
        str(scope.get("search_type", "search")),
        str(scope.get("origin", "?")),
        str(scope.get("destination", "?")),
        str(scope.get("date", "?")),
    ]
    return " ".join(parts)


def _search_scope(tool_name: str, tool_kwargs: dict[str, Any] | None) -> dict[str, str]:
    scope = {"search_type": tool_name}
    if not isinstance(tool_kwargs, dict):
        return scope
    for key in ("origin", "destination", "date"):
        value = tool_kwargs.get(key)
        if isinstance(value, str):
            scope[key] = value
    return scope


def _scope_key(scope: dict[str, str]) -> str:
    return "|".join(
        [
            scope.get("search_type", ""),
            scope.get("origin", ""),
            scope.get("destination", ""),
            scope.get("date", ""),
        ]
    )


def _reservation_flight_keys(extras: dict[str, Any], reservation_id: Any) -> set[str]:
    if not isinstance(reservation_id, str):
        return set()
    reservations = _dict_extra(extras, _RESERVATIONS_KEY)
    reservation = reservations.get(reservation_id)
    if not isinstance(reservation, dict):
        return set()
    keys: set[str] = set()
    flights = reservation.get("flights")
    if not isinstance(flights, list):
        return keys
    for flight in flights:
        if not isinstance(flight, dict):
            continue
        number = flight.get("flight_number")
        date = flight.get("date")
        if isinstance(number, str) and isinstance(date, str):
            keys.add(f"{number}|{date}")
    return keys


def _parse_observation(observation: Any) -> Any:
    if not isinstance(observation, str):
        return observation
    try:
        return json.loads(observation)
    except (TypeError, ValueError):
        return None


def _iter_flights(node: Any, *, default_date: str | None = None) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("flight_number"), str):
                if not isinstance(value.get("date"), str) and default_date is not None:
                    value = {**value, "date": default_date}
                if isinstance(value.get("date"), str):
                    found.append(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(node)
    return found



def _record_used_payment_methods(extras: dict[str, Any], payment_history: Any) -> None:
    if not isinstance(payment_history, list):
        return
    used = _dict_extra(extras, _USED_PAYMENT_METHODS_KEY)
    for payment in payment_history:
        if not isinstance(payment, dict):
            continue
        payment_id = _payment_id(payment)
        amount = _money_amount(payment.get("amount"))
        if amount > 0 and (payment_id.startswith("certificate_") or payment_id.startswith("gift_card_")):
            used[payment_id] = used.get(payment_id, 0) + amount


def _record_tool_attempt(extras: dict[str, Any], tool_name: str, kwargs: dict[str, Any]) -> None:
    if tool_name in _NON_MEANINGFUL_FOLLOWUP_TOOLS:
        return
    _list_extra(extras, _TOOL_ATTEMPTS_KEY).append({"tool_name": tool_name, "kwargs": dict(kwargs)})


def _user_transcript(transcript: str) -> str:
    lines = []
    for line in transcript.splitlines():
        if line.startswith("[user]"):
            lines.append(line.removeprefix("[user]").strip())
    return "\n".join(lines)


def _reservation_touches_airport(reservation: dict[str, Any], airport: str) -> bool:
    return _reservation_touches_any_airport(reservation, {airport})


def _reservation_touches_any_airport(reservation: dict[str, Any], airports: set[str]) -> bool:
    flights = reservation.get("flights")
    if not isinstance(flights, list):
        return False
    for flight in flights:
        if not isinstance(flight, dict):
            continue
        if flight.get("origin") in airports or flight.get("destination") in airports:
            return True
    return False


def _day_after_first_flight(reservation: dict[str, Any]) -> str | None:
    flights = reservation.get("flights")
    if not isinstance(flights, list) or not flights or not isinstance(flights[0], dict):
        return None
    raw_date = flights[0].get("date")
    if not isinstance(raw_date, str):
        return None
    try:
        return (datetime.strptime(raw_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _cheapest_search_itinerary(
    search_results: dict[str, Any],
    *,
    origin: str,
    destination: str,
    date: str,
    cabin: str,
) -> list[dict[str, str]] | None:
    best: tuple[int, list[dict[str, str]]] | None = None
    for search in search_results.values():
        if not isinstance(search, dict):
            continue
        scope = search.get("scope")
        if not isinstance(scope, dict):
            continue
        if scope.get("origin") != origin or scope.get("destination") != destination or scope.get("date") != date:
            continue
        for itinerary in _iter_itineraries(search.get("result")):
            priced = _priced_itinerary(itinerary, cabin)
            if priced is None:
                continue
            total, flight_refs = priced
            if best is None or total < best[0]:
                best = (total, flight_refs)
    return best[1] if best is not None else None


def _fastest_search_itinerary(
    search_results: dict[str, Any],
    *,
    origin: str,
    destination: str,
    date: str,
    cabin: str,
) -> list[dict[str, str]] | None:
    best: tuple[int, int, list[dict[str, str]]] | None = None
    for search in search_results.values():
        if not isinstance(search, dict):
            continue
        scope = search.get("scope")
        if not isinstance(scope, dict):
            continue
        if scope.get("origin") != origin or scope.get("destination") != destination or scope.get("date") != date:
            continue
        for itinerary in _iter_itineraries(search.get("result")):
            priced = _priced_itinerary(itinerary, cabin)
            elapsed = _elapsed_minutes(itinerary)
            if priced is None or elapsed is None:
                continue
            total, flight_refs = priced
            candidate = (elapsed, total, flight_refs)
            if best is None or candidate < best:
                best = candidate
    return best[2] if best is not None else None


def _return_route(reservation: dict[str, Any]) -> tuple[str, str] | None:
    origin = reservation.get("origin")
    destination = reservation.get("destination")
    if isinstance(origin, str) and isinstance(destination, str):
        return destination, origin
    return None


def _date_from_proposed_or_search(
    extras: dict[str, Any],
    kwargs: dict[str, Any],
    *,
    origin: str,
    destination: str,
) -> str | None:
    for flight in kwargs.get("flights", []):
        if not isinstance(flight, dict):
            continue
        date = flight.get("date")
        if isinstance(date, str):
            for search in _dict_extra(extras, _FLIGHT_SEARCH_RESULTS_KEY).values():
                if not isinstance(search, dict):
                    continue
                scope = search.get("scope")
                if (
                    isinstance(scope, dict)
                    and scope.get("origin") == origin
                    and scope.get("destination") == destination
                    and scope.get("date") == date
                ):
                    return date
    for search in _dict_extra(extras, _FLIGHT_SEARCH_RESULTS_KEY).values():
        if not isinstance(search, dict):
            continue
        scope = search.get("scope")
        if (
            isinstance(scope, dict)
            and scope.get("origin") == origin
            and scope.get("destination") == destination
            and isinstance(scope.get("date"), str)
        ):
            return scope["date"]
    return None


def _outbound_prefix(reservation: dict[str, Any]) -> list[dict[str, str]]:
    destination = reservation.get("destination")
    flights = reservation.get("flights")
    if not isinstance(destination, str) or not isinstance(flights, list):
        return []
    prefix: list[dict[str, str]] = []
    for flight in flights:
        if not isinstance(flight, dict):
            return []
        number = flight.get("flight_number")
        date = flight.get("date")
        if not isinstance(number, str) or not isinstance(date, str):
            return []
        prefix.append({"flight_number": number, "date": date})
        if flight.get("destination") == destination:
            return prefix
    return []


def _reservation_update_delta(
    reservation: dict[str, Any],
    new_flights: list[dict[str, str]],
    cabin: str,
    search_results: dict[str, Any],
) -> int:
    current_total = sum(_money_amount(flight.get("price")) for flight in reservation.get("flights", []))
    new_total = 0
    for flight in new_flights:
        new_total += _flight_price_from_reservation_or_search(search_results, reservation, flight, cabin)
    passengers = reservation.get("passengers")
    passenger_count = len(passengers) if isinstance(passengers, list) and passengers else 1
    return (new_total - current_total) * passenger_count


def _flight_price_from_reservation_or_search(
    search_results: dict[str, Any],
    reservation: dict[str, Any],
    flight_ref: dict[str, str],
    cabin: str,
) -> int:
    for current in reservation.get("flights", []):
        if not isinstance(current, dict):
            continue
        if current.get("flight_number") == flight_ref.get("flight_number") and current.get("date") == flight_ref.get(
            "date"
        ):
            return _money_amount(current.get("price"))
    number = flight_ref.get("flight_number")
    date = flight_ref.get("date")
    for search in search_results.values():
        if not isinstance(search, dict):
            continue
        for itinerary in _iter_itineraries(search.get("result")):
            for flight in itinerary:
                if not isinstance(flight, dict):
                    continue
                if flight.get("flight_number") != number or flight.get("date") != date:
                    continue
                prices = flight.get("prices")
                if isinstance(prices, dict):
                    return _money_amount(prices.get(cabin))
    return 0


def _smallest_sufficient_gift_card(payment_methods: dict[str, Any], *, minimum_amount: int) -> str | None:
    gifts: list[tuple[int, str]] = []
    for payment_id, payment in payment_methods.items():
        if not str(payment_id).startswith("gift_card_") or not isinstance(payment, dict):
            continue
        amount = _money_amount(payment.get("amount"))
        if amount >= minimum_amount:
            gifts.append((amount, str(payment_id)))
    if not gifts:
        return None
    return min(gifts)[1]


def _iter_itineraries(result: Any) -> list[list[dict[str, Any]]]:
    if isinstance(result, list) and all(isinstance(item, dict) for item in result):
        return [result]
    itineraries: list[list[dict[str, Any]]] = []
    if isinstance(result, list):
        for item in result:
            if isinstance(item, list) and all(isinstance(flight, dict) for flight in item):
                itineraries.append(item)
    return itineraries


def _priced_itinerary(itinerary: list[dict[str, Any]], cabin: str) -> tuple[int, list[dict[str, str]]] | None:
    total = 0
    flight_refs: list[dict[str, str]] = []
    for flight in itinerary:
        seats = flight.get("available_seats")
        prices = flight.get("prices")
        number = flight.get("flight_number")
        date = flight.get("date")
        if not isinstance(seats, dict) or int(seats.get(cabin, 0) or 0) <= 0:
            return None
        if not isinstance(prices, dict):
            return None
        price = _money_amount(prices.get(cabin))
        if price <= 0 or not isinstance(number, str) or not isinstance(date, str):
            return None
        total += price
        flight_refs.append({"flight_number": number, "date": date})
    return total, flight_refs


def _elapsed_minutes(itinerary: list[dict[str, Any]]) -> int | None:
    if not itinerary:
        return None
    first = itinerary[0]
    last = itinerary[-1]
    first_date = first.get("date")
    last_date = last.get("date")
    departure = first.get("scheduled_departure_time_est")
    arrival = last.get("scheduled_arrival_time_est")
    if not all(isinstance(value, str) for value in (first_date, last_date, departure, arrival)):
        return None
    try:
        start = _flight_datetime(first_date, departure)
        end = _flight_datetime(last_date, arrival)
    except ValueError:
        return None
    if end < start:
        end += timedelta(days=1)
    return int((end - start).total_seconds() // 60)


def _flight_datetime(date: str, time_text: str) -> datetime:
    offset_days = 1 if "+1" in time_text else 0
    clean_time = time_text.replace("+1", "").strip()
    return datetime.strptime(f"{date}T{clean_time}", "%Y-%m-%dT%H:%M:%S") + timedelta(days=offset_days)


def _refund_payment_id(reservation: dict[str, Any], payment_methods: dict[str, Any]) -> str | None:
    payment_history = reservation.get("payment_history")
    if isinstance(payment_history, list):
        for payment in payment_history:
            if not isinstance(payment, dict):
                continue
            payment_id = _payment_id(payment)
            if payment_id.startswith("gift_card_") or payment_id.startswith("credit_card_"):
                return payment_id
    for payment_id in payment_methods:
        payment_id = str(payment_id)
        if payment_id.startswith("gift_card_") or payment_id.startswith("credit_card_"):
            return payment_id
    return None


def _handoff_evidence(extras: dict[str, Any], reservation_id: Any) -> dict[str, Any]:
    reservation = _dict_extra(extras, _RESERVATIONS_KEY).get(reservation_id)
    if not isinstance(reservation, dict):
        return {"reservation_id": reservation_id}
    return {
        "reservation_id": reservation_id,
        "cabin": reservation.get("cabin"),
        "insurance": reservation.get("insurance"),
        "origin": reservation.get("origin"),
        "destination": reservation.get("destination"),
        "payment_history": reservation.get("payment_history"),
    }


def _matching_prior_reservation(extras: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any] | None:
    for reservation in _dict_extra(extras, _RESERVATIONS_KEY).values():
        if not isinstance(reservation, dict):
            continue
        if reservation.get("user_id") != kwargs.get("user_id"):
            continue
        if reservation.get("origin") != kwargs.get("origin"):
            continue
        if reservation.get("destination") != kwargs.get("destination"):
            continue
        if reservation.get("flight_type") != kwargs.get("flight_type"):
            continue
        return reservation
    return None


def _dict_extra(extras: dict[str, Any], key: str) -> dict[str, Any]:
    value = extras.setdefault(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a dict")
    return value


def _list_extra(extras: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = extras.setdefault(key, [])
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list")
    return value
