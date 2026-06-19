"""Thin host-facing SDK for kairos integration.

Public surface for harness authors: one :class:`KairosHost` per process,
one :class:`KairosSession` per task, one :class:`ToolDecision` per intercept,
a :class:`DomainExtension` protocol for host-specific detectors, and an
:class:`IdExtractor` protocol for pluggable entity extraction.

See ``docs/host-sdk-design.md`` for the integration target and rationale.

Implementation status:
  - lifecycle, manifest, summary, sink, IdExtractor wiring: implemented.
  - decision routing through domain extensions + interceptor gates: implemented.
  - semantic-recovery verifier integration: implemented when a judge is given.
  - retry-budget + intervention-awareness telemetry around domain extension
    blocks: TODO — port from ``runtime_correction.primitives`` in a follow-up
    commit on this branch.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import time
from collections.abc import Callable, Iterator  # noqa: TC003
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from kairos.config import settings
from kairos.diagnostic.catalog import load_catalog
from kairos.host._orchestrator import ExtensionBlockOrchestrator
from kairos.host.id_extract import DefaultIdExtractor
from kairos.host.run_dir import make_run_id, write_manifest, write_summary
from kairos.host.sinks import DriftObservationSink, StreamingJSONLSink
from kairos.intercept import (
    Gate,
    GateEvaluation,
    KairosInterceptor,
    SessionContext,
    gate_from_config,
)

try:  # semantic recovery is optional — only required if a judge is configured.
    from kairos.semantic_recovery import (
        SEMANTIC_PREWRITE_PATTERN_ID,
        DriftDetector,
        SemanticRecoveryRuntime,
        SessionExpectationBuilder,
        WorkflowMemoryStore,
        build_policy_pack,
        build_semantic_decision_artifact,
    )

    _SEMANTIC_AVAILABLE = True
except ImportError:  # pragma: no cover - defensive
    _SEMANTIC_AVAILABLE = False

# Phase 7 / T-03 (2026-05-20): deterministic breakers for the active harness.
# Always-on import (no heavy deps — just stdlib + the AP DB JSON).
from kairos.semantic_recovery.breakers import (
    BreakerState,
    DeterministicBreakers,
    Trip,
    observation_had_error,
)
# Phase 7 / T-04 (2026-05-20): hypothesis-driven progress monitor.
from kairos.semantic_recovery.progress_monitor import ProgressMonitor

__all__ = [
    "DomainExtension",
    "IdExtractor",
    "KairosHost",
    "KairosSession",
    "ToolDecision",
    "host",
]


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ToolDecision:
    """Result of :meth:`KairosSession.before_tool_call`.

    ``bypass_orchestrator`` is for extensions that already drive their own
    retry budget + intervention-awareness telemetry (e.g. ports of legacy
    host controllers that pre-date this SDK). When True, the session returns
    the decision verbatim and skips the kairos.host orchestrator's wrapper.
    Most extensions should leave it ``False`` and let kairos handle retry +
    follow-up + recurrence + outcome telemetry for them.
    """

    action: Literal["execute", "inject_correction", "execute_fail_open"]
    correction_artifact: str | None = None
    fired_gates: list[GateEvaluation] = field(default_factory=list)
    pattern_id: str | None = None
    confidence: Literal["low", "medium", "high"] | None = None
    bypass_orchestrator: bool = False


# --------------------------------------------------------------------------- #
# Protocols
# --------------------------------------------------------------------------- #


class IdExtractor(Protocol):
    """Pluggable entity-ID extractor."""

    def from_user_text(self, text: str) -> list[str]:  # pragma: no cover - protocol
        ...

    def from_tool_result(self, observation: Any) -> list[str]:  # pragma: no cover - protocol
        ...


class DomainExtension(Protocol):
    """Host plug-in for application-specific gates and detectors."""

    def gates(self) -> list[Gate]:  # pragma: no cover - protocol
        ...

    def before_tool_call(
        self,
        ctx: SessionContext,
        tool_name: str,
        kwargs: dict[str, Any],
    ) -> ToolDecision | None:  # pragma: no cover - protocol
        ...

    def semantic_evidence(
        self,
        ctx: SessionContext,
        tool_name: str,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:  # pragma: no cover - protocol
        ...

    def after_tool_result(
        self,
        ctx: SessionContext,
        tool_name: str,
        kwargs: dict[str, Any],
        observation: Any,
    ) -> None:  # pragma: no cover - protocol
        ...


# --------------------------------------------------------------------------- #
# KairosSession
# --------------------------------------------------------------------------- #


class KairosSession:
    """One task / session against a :class:`KairosHost`.

    Construct via :meth:`KairosHost.session` (preferred — context-managed).
    Owns implicit ``update_context`` calls on every tool result / user turn,
    and runs the registered decision pipeline on each :meth:`before_tool_call`.
    """

    def __init__(
        self,
        host: KairosHost,
        session_id: str,
        *,
        user_instruction: str,
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> None:
        self._host = host
        self._session_id = session_id
        self._user_instruction = user_instruction
        self._tool_schemas = tool_schemas or []
        self._ended = False
        self._semantic_snapshot: Any | None = None

        # Seed the initial user turn.
        ids = host._id_extractor.from_user_text(user_instruction)
        host._interceptor.update_context(
            session_id,
            {"role": "user", "content": user_instruction, "ids": ids},
        )

        # Phase 7 / T-03 + T-04: per-session breaker scratchpad. Fresh state
        # per session — never reuse a BreakerState across tasks. Created
        # when EITHER subsystem is enabled (both subsystems share the
        # state struct but are independently flag-gated).
        self._breaker_state: BreakerState | None = None
        if host._deterministic_breakers is not None or host._progress_monitor is not None:
            self._breaker_state = BreakerState(max_steps=20)
            self._breaker_state.parse_instruction(user_instruction)
            # T-05 (2026-05-20): snapshot the rendered plan's absolute-claim
            # lines so AP-02 (QuoteWarningAsPolicyBreaker) can substring-match
            # assistant content against them. Done AFTER the semantic-
            # snapshot starts (see further below); deferred to a setter
            # called once the snapshot is built.

        # Semantic-recovery session-start if enabled.
        if host._semantic_runtime is not None:
            ctx = host._interceptor.get_session_context(session_id)
            try:
                self._semantic_snapshot = host._semantic_runtime.start_session(ctx, user_instruction=user_instruction)
            except Exception:  # noqa: BLE001 - never crash session bring-up on a judge fault.
                self._semantic_snapshot = None

        # T-05 (2026-05-20): now that the snapshot is built, snapshot its
        # plan artifact's absolute-claim lines onto BreakerState so AP-02
        # has its substring corpus ready before the agent starts emitting
        # tool calls.
        if (
            self._breaker_state is not None
            and self._semantic_snapshot is not None
            and getattr(self._semantic_snapshot, "agent_plan", None) is not None
        ):
            artifact = getattr(self._semantic_snapshot.agent_plan, "artifact", "") or ""
            self._breaker_state.snapshot_plan_absolute_claims(artifact)

    # ---- handles --------------------------------------------------------- #

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def context(self) -> SessionContext:
        return self._host._interceptor.get_session_context(self._session_id)

    @property
    def semantic_snapshot(self) -> Any | None:
        """The :class:`SemanticSessionSnapshot` returned by ``start_session``.

        ``None`` if no judge is configured or ``start_session`` failed. Hosts
        that want to persist the snapshot or inject the agent plan into the
        system prompt read it from here.
        """
        return self._semantic_snapshot

    # ---- per-turn API ---------------------------------------------------- #

    def record_user_turn(self, content: str) -> None:
        ids = self._host._id_extractor.from_user_text(content)
        self._host._interceptor.update_context(
            self._session_id,
            {"role": "user", "content": content, "ids": ids},
        )

    def record_assistant_message(self, content: str) -> None:
        """T-05 (2026-05-20): host calls this on every assistant message
        (respond turns AND tool-call turns) so AP-02's substring corpus
        is complete even on traces where the smoking-gun quote lands in
        a respond message turns before any tool call.

        Safe to call with empty content (no-op).
        """
        if self._breaker_state is not None and (content or "").strip():
            self._breaker_state.record_assistant_message(content)

    def before_tool_call(
        self,
        tool_name: str,
        kwargs: dict[str, Any],
        *,
        assistant_content: str = "",
    ) -> ToolDecision:
        """Run every pre-tool check and return what the host should do.

        ``assistant_content`` (T-05, 2026-05-20): the content of the assistant
        message that emitted this tool call. AP-02 reads it to detect when
        the agent quoted an injected ``watch for:`` warning as its own
        policy claim. Hosts that don't supply it lose AP-02 coverage but
        everything else still works.
        """
        host = self._host
        ctx = self.context
        ctx.attempted_tools.append(tool_name)
        # T-05: stash the assistant content on BreakerState so AP-02 can
        # read it without changing the Breaker protocol signature.
        if self._breaker_state is not None and assistant_content:
            self._breaker_state.record_assistant_message(assistant_content)

        # Drift observation runs FIRST so every proposed monitored action is
        # observed — even if a later layer blocks it. Detection is decoupled
        # from intervention by design; the observation never affects the
        # ToolDecision we eventually return.
        self._observe_drift(ctx, tool_name, kwargs)

        # Phase 7 / T-04 refactor (2026-05-20): record per-tool-call state
        # ONCE (turn counter + canonical hash) so BOTH the deterministic
        # breakers and the progress monitor see fresh state regardless of
        # which subsystem is enabled.
        if self._breaker_state is not None:
            self._breaker_state.record_before_tool(tool_name, kwargs or {})

        # Phase 7 / T-04: progress monitor may have flipped a stall flag
        # in the previous after_tool_result. Consume it BEFORE deterministic
        # breakers (the monitor's verdict is independent of breaker shapes;
        # if it fired, that takes precedence). One-shot: reset the flag so
        # the agent gets one chance to course-correct after the injection.
        if (
            self._breaker_state is not None
            and self._breaker_state.progress_monitor_stalled
        ):
            correction = self._breaker_state.progress_monitor_correction_text
            pattern_id = self._breaker_state.progress_monitor_pattern_id or "AP-PROGRESS"
            self._breaker_state.progress_monitor_stalled = False
            if settings.progress_monitor_verbose:
                print(
                    f"[progress_monitor] inject correction (pattern={pattern_id})",
                    flush=True,
                )
            return self._wrap_host_block(
                tool_name=tool_name,
                kwargs=kwargs,
                pattern_id=pattern_id,
                correction_artifact=correction,
                confidence="medium",  # LLM judgement, not deterministic
            )

        # Phase 7 / T-03: deterministic breakers run BEFORE the rest of the
        # decision pipeline so their hash-window state always updates (the
        # aggregator records the canonical hash unconditionally, even on
        # turns that would later be blocked by a domain extension or
        # semantic gate). If a breaker trips here, we short-circuit and
        # route through ``_wrap_host_block`` for telemetry parity with
        # other intervention sources.
        breaker_trip = self._run_breakers_before(tool_name, kwargs)
        if breaker_trip is not None:
            return self._wrap_host_block(
                tool_name=tool_name,
                kwargs=kwargs,
                pattern_id=breaker_trip.ap_id,
                correction_artifact=breaker_trip.jit_correction_text,
                confidence="high",  # deterministic match → high confidence
            )

        # Tracker follow-up: every tool attempt is a potential follow-up to a
        # previously blocked call. This auto-emits ``.followup`` evals.
        host._orchestrator.record_follow_up(self._session_id, tool_name, kwargs)

        # 1. Domain extensions — first non-None decision wins.
        for ext in host._domain_extensions:
            decision = ext.before_tool_call(ctx, tool_name, kwargs)
            if decision is None:
                continue
            if decision.bypass_orchestrator:
                # Extension already drove its own retry/telemetry; pass through.
                return decision
            if decision.action == "inject_correction":
                return self._wrap_host_block(
                    tool_name=tool_name,
                    kwargs=kwargs,
                    pattern_id=decision.pattern_id or "extension",
                    correction_artifact=decision.correction_artifact,
                    confidence=decision.confidence,
                )
            # Extension explicitly said execute or execute_fail_open — pass through.
            return decision

        # 2. Static gates via the interceptor.
        result = host._interceptor.evaluate(self._session_id, tool_name, kwargs)
        if result.blocked:
            return ToolDecision(
                action="inject_correction",
                correction_artifact=result.error_string,
                fired_gates=list(result.fired_gates),
            )

        # 3. Semantic verifier — gated by settings.semantic_recovery_enabled
        # (default off). The runtime itself may exist for memory retrieval
        # only; verify_tool_call only runs when the flag is on AND a judge is
        # wired (the verifier needs an LLM client).
        if (
            host._semantic_runtime is not None
            and _SEMANTIC_AVAILABLE
            and settings.semantic_recovery_enabled
            and host._judge is not None
        ):
            evidence: dict[str, Any] = {}
            for ext in host._domain_extensions:
                try:
                    extra = ext.semantic_evidence(ctx, tool_name, kwargs)
                except Exception:  # noqa: BLE001 - fall through on evidence faults.
                    extra = {}
                if extra:
                    evidence.update(extra)
            try:
                semantic_decision = host._semantic_runtime.verify_tool_call(
                    ctx, tool_name=tool_name, kwargs=kwargs, evidence=evidence
                )
            except Exception:  # noqa: BLE001 - never crash a host on a judge fault.
                semantic_decision = None
            if semantic_decision is not None and semantic_decision.is_injectable:
                artifact = build_semantic_decision_artifact(
                    decision=semantic_decision,
                    blocked_tool_name=tool_name,
                    blocked_kwargs=kwargs,
                )
                return self._wrap_host_block(
                    tool_name=tool_name,
                    kwargs=kwargs,
                    pattern_id=SEMANTIC_PREWRITE_PATTERN_ID,
                    correction_artifact=artifact,
                    confidence=semantic_decision.confidence,
                    suggested_tool_name=semantic_decision.next_tool,
                    suggested_kwargs=semantic_decision.next_kwargs,
                )

        # 4. Nothing blocked.
        return ToolDecision(
            action="execute",
            fired_gates=list(result.fired_gates) if result else [],
        )

    def _observe_drift(
        self,
        ctx: SessionContext,
        tool_name: str,
        kwargs: dict[str, Any],
    ) -> None:
        host = self._host
        detector = host._drift_detector
        if detector is None or not detector.is_enabled():
            return
        try:
            observation = detector.observe(
                session_id=self._session_id,
                ctx=ctx,
                tool_name=tool_name,
                kwargs=kwargs,
                session_expectation=(
                    self._semantic_snapshot.session_expectation if self._semantic_snapshot is not None else None
                ),
                memory_plan_artifact=(
                    self._semantic_snapshot.agent_plan.artifact
                    if self._semantic_snapshot is not None and self._semantic_snapshot.agent_plan is not None
                    else None
                ),
            )
        except Exception:  # noqa: BLE001 - drift detection must never crash the agent loop.
            return
        if observation is None:
            return
        if host._drift_sink is not None:
            with contextlib.suppress(Exception):
                host._drift_sink.emit(observation)

    def _wrap_host_block(
        self,
        *,
        tool_name: str,
        kwargs: dict[str, Any],
        pattern_id: str,
        correction_artifact: str | None,
        confidence: str | None,
        suggested_tool_name: str | None = None,
        suggested_kwargs: Any = None,
    ) -> ToolDecision:
        """Run retry-budget + intervention awareness telemetry around a block.

        Used for domain-extension blocks and semantic-verifier blocks (the
        static-gate path emits its own telemetry through the interceptor).
        Returns ``inject_correction`` if budget allows or ``execute_fail_open``
        on a recurring block whose retries are exhausted.
        """
        action, evaluation = self._host._orchestrator.handle_block(
            self._session_id,
            tool_name,
            kwargs,
            pattern_id=pattern_id,
            confidence=confidence,
            suggested_tool_name=suggested_tool_name,
            suggested_kwargs=suggested_kwargs,
        )
        return ToolDecision(
            action=action,
            correction_artifact=correction_artifact if action == "inject_correction" else None,
            fired_gates=[evaluation],
            pattern_id=pattern_id,
            confidence=confidence,  # type: ignore[arg-type]
        )

    def after_tool_result(
        self,
        tool_name: str,
        kwargs: dict[str, Any],
        observation: Any,
    ) -> None:
        host = self._host
        ids = host._id_extractor.from_tool_result(observation)
        host._interceptor.update_context(
            self._session_id,
            {
                "role": "tool",
                "tool_name": tool_name,
                "result": observation,
                "ids": ids,
            },
        )
        ctx = self.context
        for ext in host._domain_extensions:
            with contextlib.suppress(Exception):
                ext.after_tool_result(ctx, tool_name, kwargs, observation)

        # Phase 7 / T-04 refactor (2026-05-20): record the post-tool state
        # ONCE — both deterministic breakers and the progress monitor read
        # this. Done HERE (not inside either subsystem's check_after) so
        # toggling one subsystem off without the other doesn't strand the
        # active one with stale tool_history / recent_observations.
        had_error = observation_had_error(observation)
        if self._breaker_state is not None:
            self._breaker_state.record_after_tool(tool_name, observation, had_error)

        # Phase 7 / T-03: deterministic breakers post-result hook. Detection
        # only — state mutation already done above. The trip return value is
        # informational here — corrections only ever fire from
        # ``before_tool_call``. Errors here MUST NOT crash the agent loop.
        if (
            host._deterministic_breakers is not None
            and self._breaker_state is not None
        ):
            try:
                trip = host._deterministic_breakers.check_after(
                    self._breaker_state, tool_name, kwargs, observation, had_error
                )
                if trip is not None and settings.deterministic_breakers_verbose:
                    print(
                        f"[breakers] after-trip {trip.ap_id} on {tool_name}: {trip.reason}",
                        flush=True,
                    )
            except Exception:  # noqa: BLE001 - never crash the loop on a breaker fault
                pass

        # Phase 7 / T-04: progress monitor — every N turns, ask Haiku
        # "is the agent moving toward expected_terminal_actions". Trip sets
        # state.progress_monitor_stalled = True; next before_tool_call
        # converts to inject_correction. Never crash the agent loop on
        # judge fault.
        if (
            host._progress_monitor is not None
            and host._progress_monitor.is_enabled()
            and self._breaker_state is not None
            and self._semantic_snapshot is not None
        ):
            try:
                expectation = getattr(self._semantic_snapshot, "session_expectation", None)
                terminals = (
                    list(getattr(expectation, "expected_terminal_actions", []))
                    if expectation is not None
                    else []
                )
                host._progress_monitor.check(
                    self._breaker_state,
                    expected_terminal_actions=terminals,
                    last_actions=list(self._breaker_state.tool_history),
                    last_observations=list(self._breaker_state.recent_observations),
                    max_steps=self._breaker_state.max_steps,
                    verbose=settings.progress_monitor_verbose,
                )
            except Exception:  # noqa: BLE001 - never crash the loop on a monitor fault
                pass

    def _run_breakers_before(
        self, tool_name: str, kwargs: dict[str, Any]
    ) -> Trip | None:
        """Return a Trip if any deterministic breaker fires in check_before.

        Always-on state update (hash window + turn counter) happens inside
        the aggregator's :meth:`DeterministicBreakers.check_before`; the
        side effect runs even when no trip fires, which is critical for
        AP-05's adversarial-bypass mitigation (see breakers.py MED-1 fix
        comment).
        """
        host = self._host
        if host._deterministic_breakers is None or self._breaker_state is None:
            return None
        try:
            trip = host._deterministic_breakers.check_before(
                self._breaker_state, tool_name, kwargs
            )
        except Exception:  # noqa: BLE001 - never crash the loop on a breaker fault
            return None
        if trip is not None and settings.deterministic_breakers_verbose:
            print(
                f"[breakers] before-trip {trip.ap_id} on {tool_name}: {trip.reason}",
                flush=True,
            )
        return trip

    def end(
        self,
        *,
        reward: float = 0.0,
        info: dict[str, Any] | None = None,
    ) -> None:
        if self._ended:
            return
        self._ended = True
        self._observe_missing_action()
        # Let extensions emit their own task-outcome telemetry first. This is
        # important for adapters wrapping a legacy controller whose awareness
        # tracker uses different ``ctx.extras`` keys than the kairos.host
        # orchestrator — without this, the legacy ``.outcome`` events never
        # emit.
        for ext in self._host._domain_extensions:
            hook = getattr(ext, "after_task", None)
            if hook is None:
                continue
            with contextlib.suppress(Exception):
                hook(self._session_id, reward=reward, info=info or {})
        # Then emit ``.outcome`` meta-evals from the kairos orchestrator for
        # any still-pending interventions tracked under the kairos keys.
        with contextlib.suppress(Exception):
            self._host._orchestrator.record_task_outcome(self._session_id, reward=reward, info=info)
        self._host._interceptor.reset_session(self._session_id)

    def _observe_missing_action(self) -> None:
        host = self._host
        detector = host._drift_detector
        if detector is None or not detector.is_enabled() or self._semantic_snapshot is None:
            return
        expectation = getattr(self._semantic_snapshot, "session_expectation", None)
        if expectation is None:
            return
        with contextlib.suppress(Exception):
            observation = detector.observe_missing_actions(
                session_id=self._session_id,
                ctx=self.context,
                session_expectation=expectation,
            )
            if observation is not None and host._drift_sink is not None:
                host._drift_sink.emit(observation)

    # ---- context-manager sugar ------------------------------------------ #

    def __enter__(self) -> KairosSession:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if not self._ended:
            self.end()


# --------------------------------------------------------------------------- #
# KairosHost
# --------------------------------------------------------------------------- #


class KairosHost:
    """Process-level kairos integration handle.

    Construct once via :func:`host`. Use as a context manager or call
    :meth:`finalize` explicitly to flush the streaming sink and write
    ``summary.json``. ``atexit`` registers a fallback flush so the run dir
    always contains a summary, even on uncaught exceptions.
    """

    def __init__(
        self,
        *,
        run_dir: Path,
        gates_config: list[dict[str, Any]],
        domain_extensions: list[DomainExtension],
        id_extractor: IdExtractor,
        judge: Any | None = None,
        policy_pack: Any | None = None,
        memory_store: Any | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
        manifest: dict[str, Any] | None = None,
        drift_monitor_predicate: Callable[[str], bool] | None = None,
        intent_template_map: dict[str, Any] | None = None,
        diagnostic_pattern_catalog: list[dict[str, Any]] | None = None,
    ) -> None:
        self._gates_config_raw = list(gates_config)
        self._domain_extensions = list(domain_extensions)
        self._id_extractor = id_extractor
        self._judge = judge
        self._policy_pack = policy_pack
        self._memory_store = memory_store
        self._tool_schemas = list(tool_schemas or [])
        self._manifest = manifest or {}
        self._drift_monitor_predicate = drift_monitor_predicate
        self._intent_template_map = intent_template_map
        self._diagnostic_pattern_catalog = list(diagnostic_pattern_catalog or [])
        if diagnostic_pattern_catalog is None and settings.diagnostic_pattern_catalog:
            self._diagnostic_pattern_catalog = load_catalog(settings.diagnostic_pattern_catalog)
        self._started_at = time.time()
        self._finalized = False

        # Static gates from config + extension-supplied gates.
        gates: list[Gate] = [gate_from_config(cfg) for cfg in self._gates_config_raw]
        for ext in self._domain_extensions:
            with contextlib.suppress(Exception):
                gates.extend(ext.gates())

        # Per-run directory.
        run_id = make_run_id(self._manifest)
        self._run_path = run_dir / run_id
        self._run_path.mkdir(parents=True, exist_ok=True)

        # Streaming JSONL sink for evaluations.
        self._sink = StreamingJSONLSink(self._run_path / "gate_evaluations.jsonl")

        # Interceptor.
        self._interceptor = KairosInterceptor(gates=gates, log_sink=self._sink, judge=judge)

        # Block orchestrator (retry budgets + intervention awareness).
        self._orchestrator = ExtensionBlockOrchestrator(self._interceptor)

        # Optional semantic-recovery runtime. Built when EITHER a judge OR a
        # memory_store is provided so the memory-only experiment (memory loaded,
        # no LLM judge) can still retrieve hits and render the agent plan.
        # When judge is None, expectation_builder is None and start_session
        # falls back to build_static_session_expectation.
        self._semantic_runtime: Any | None = None
        if _SEMANTIC_AVAILABLE and (judge is not None or memory_store is not None):
            try:
                pack = policy_pack or build_policy_pack(
                    source_name="host",
                    system_prompt="",
                    tool_descriptions=self._tool_schemas,
                )
                store = memory_store if memory_store is not None else WorkflowMemoryStore()
                expectation_builder = (
                    SessionExpectationBuilder(
                        client=judge,
                        intent_template_map=self._intent_template_map,
                    )
                    if judge is not None
                    else None
                )
                self._semantic_runtime = SemanticRecoveryRuntime(
                    policy_pack=pack,
                    memory_store=store,
                    expectation_builder=expectation_builder,
                    tool_descriptions=self._tool_schemas,
                    deterministic_gate_ids=[g.gate_id for g in gates],
                    intent_template_map=self._intent_template_map,
                )
            except Exception:  # noqa: BLE001 - never crash bring-up on judge wiring.
                self._semantic_runtime = None

        # Drift detector — observation-only Layer 1. Gated by
        # ``settings.drift_detection_enabled`` (default off) AND requires a
        # judge client. Writes to a separate JSONL artifact so the observation
        # stream stays grep-able and never feeds intervention.
        self._drift_detector: Any | None = None
        self._drift_sink: DriftObservationSink | None = None
        if judge is not None and _SEMANTIC_AVAILABLE and settings.drift_detection_enabled:
            try:
                judge_model = getattr(judge, "model", None)
                self._drift_detector = DriftDetector(
                    client=judge,
                    judge_model=judge_model,
                    monitor_predicate=self._drift_monitor_predicate,
                    diagnostic_patterns=self._diagnostic_pattern_catalog,
                )
                self._drift_sink = DriftObservationSink(self._run_path / "drift_observations.jsonl")
            except Exception:  # noqa: BLE001 - drift detection is optional; never crash bring-up.
                self._drift_detector = None
                self._drift_sink = None

        # Phase 7 / T-03 (2026-05-20): deterministic breakers — runtime trap
        # layer for the active harness. Loads detector specs from the
        # host-owned anti_patterns.json (path supplied via
        # ``settings.anti_patterns_path``). Construction is gated on
        # ``settings.deterministic_breakers_enabled``; if missing or
        # malformed the aggregator degrades to zero breakers (graceful) so
        # the agent loop never crashes on a bad DB. The aggregator is
        # stateless across sessions — per-session :class:`BreakerState`
        # lives on :class:`KairosSession`.
        self._deterministic_breakers: DeterministicBreakers | None = None
        if settings.deterministic_breakers_enabled and settings.anti_patterns_path:
            try:
                self._deterministic_breakers = DeterministicBreakers(settings.anti_patterns_path)
            except Exception:  # noqa: BLE001 - never crash bring-up on AP-DB faults
                self._deterministic_breakers = None

        # Phase 7 / T-04 (2026-05-20): progress monitor — LLM-based stall
        # detection that complements the deterministic breakers. Requires a
        # judge client (passed via ``judge=`` to KairosHost). When enabled
        # without a judge, construction succeeds but ``is_enabled()`` is
        # False and the monitor never runs.
        self._progress_monitor: ProgressMonitor | None = None
        if settings.progress_monitor_enabled and judge is not None:
            try:
                self._progress_monitor = ProgressMonitor(
                    client=judge,
                    model=settings.progress_monitor_model,
                    min_turns_between_checks=settings.progress_monitor_min_turns_between_checks,
                )
            except Exception:  # noqa: BLE001 - never crash bring-up
                self._progress_monitor = None

        # Write manifest.
        gate_summary = [(g.gate_id, g.status.value, g.target_tool) for g in gates]
        write_manifest(
            self._run_path,
            run_metadata=self._manifest,
            gates_config=self._gates_config_raw,
            gate_summary=gate_summary,
        )

        atexit.register(self._atexit_finalize)

    # ---- public API ------------------------------------------------------ #

    @property
    def run_path(self) -> Path:
        return self._run_path

    @property
    def interceptor(self) -> KairosInterceptor:
        """Read-only handle on the underlying :class:`KairosInterceptor`.

        Exposed for adapters that wrap a pre-SDK host controller (e.g.
        tau-agent's TauPhase1RecoveryController) which takes an interceptor
        in its constructor. New extensions should rely on ``ctx`` passed into
        :meth:`DomainExtension.before_tool_call` and avoid touching the
        interceptor directly.
        """
        return self._interceptor

    @property
    def semantic_runtime(self) -> Any | None:
        """Read-only handle on the :class:`SemanticRecoveryRuntime` if enabled.

        Returns ``None`` if no judge was configured at construction. Hosts
        that need the per-session snapshot from ``start_session`` (e.g. to
        persist artifacts or inject an agent plan into the system prompt)
        can call it themselves; otherwise :class:`KairosSession` handles
        ``start_session`` automatically on session open.
        """
        return self._semantic_runtime

    @contextlib.contextmanager
    def session(
        self,
        session_id: str,
        *,
        user_instruction: str,
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> Iterator[KairosSession]:
        sess = KairosSession(
            self,
            session_id,
            user_instruction=user_instruction,
            tool_schemas=tool_schemas,
        )
        try:
            yield sess
        finally:
            if not sess._ended:
                sess.end()

    def finalize(
        self,
        *,
        task_results: list[dict[str, Any]] | None = None,
    ) -> Path | None:
        if self._finalized:
            return self._run_path / "summary.json"
        self._finalized = True
        records = list(self._sink.records)
        drift_observations = list(self._drift_sink.records) if self._drift_sink is not None else None
        path = write_summary(
            self._run_path,
            records=records,
            task_results=task_results,
            started_at_monotonic=self._started_at,
            finalized_via="explicit",
            drift_observations=drift_observations,
        )
        self._sink.close()
        if self._drift_sink is not None:
            self._drift_sink.close()
        return path

    # ---- context-manager sugar ------------------------------------------ #

    def __enter__(self) -> KairosHost:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if not self._finalized:
            self.finalize()

    # ---- atexit fallback ------------------------------------------------- #

    def _atexit_finalize(self) -> None:
        if self._finalized:
            return
        try:
            self.finalize()
        except Exception:  # noqa: BLE001 - atexit must never raise.
            return
        # Tag as having come from atexit so callers can tell.
        summary_path = self._run_path / "summary.json"
        try:
            data = json.loads(summary_path.read_text())
            data["finalized_via"] = "atexit"
            summary_path.write_text(json.dumps(data, indent=2, default=str))
        except (OSError, ValueError):
            pass


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def host(
    *,
    run_dir: str | Path = "data/runs",
    gates_config: str | Path | list[dict[str, Any]] | None = None,
    domain_extensions: list[DomainExtension] | None = None,
    id_extractor: IdExtractor | None = None,
    judge: Any | None = None,
    policy_pack: Any | None = None,
    memory_store: Any | None = None,
    tool_schemas: list[dict[str, Any]] | None = None,
    manifest: dict[str, Any] | None = None,
    drift_monitor_predicate: Callable[[str], bool] | None = None,
    intent_template_map: dict[str, Any] | None = None,
    diagnostic_pattern_catalog: list[dict[str, Any]] | None = None,
) -> KairosHost:
    """Build a :class:`KairosHost`.

    ``gates_config`` can be a path to a JSON file, an already-parsed list of
    gate config dicts, or ``None``. Missing files are treated as an empty list
    so callers can run gate-less SHADOW configurations.

    When ``judge`` is provided the semantic-recovery runtime is enabled. Pass
    ``policy_pack`` (a :class:`~kairos.semantic_recovery.PolicyPack`) and
    ``memory_store`` (a :class:`~kairos.semantic_recovery.WorkflowMemoryStore`)
    to give the judge real policy text and prior-trajectory memories instead
    of the empty defaults. ``tool_schemas`` is passed through to the runtime
    so it can reference tool shapes when reasoning.
    """
    if isinstance(gates_config, (str, Path)):
        path = Path(gates_config)
        parsed: list[dict[str, Any]] = json.loads(path.read_text()) if path.exists() else []
    elif isinstance(gates_config, list):
        parsed = gates_config
    else:
        parsed = []

    return KairosHost(
        run_dir=Path(run_dir),
        gates_config=parsed,
        domain_extensions=domain_extensions or [],
        id_extractor=id_extractor or DefaultIdExtractor(),
        judge=judge,
        policy_pack=policy_pack,
        memory_store=memory_store,
        tool_schemas=tool_schemas,
        manifest=manifest,
        drift_monitor_predicate=drift_monitor_predicate,
        intent_template_map=intent_template_map,
        diagnostic_pattern_catalog=diagnostic_pattern_catalog,
    )
