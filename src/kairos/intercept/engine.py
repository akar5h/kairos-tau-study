"""KairosInterceptor — the runtime gate-evaluation engine.

Contract with the agent harness:
  - The harness calls update_context() on every user turn and tool result.
  - Before invoking a tool, the harness calls evaluate() (or aevaluate()).
  - If the returned InterceptResult.blocked is True, the harness injects
    InterceptResult.error_string back into the agent context instead of
    calling the tool. Otherwise the tool runs normally.

Invariants this engine enforces, regardless of what gates do:
  - DISABLED gates never run.
  - SHADOW gates evaluate and log but never block.
  - Gate exceptions never crash the engine — they fail-open and are logged.
  - semantic_async gates honour their timeout and failure-mode policy:
      timeout in SHADOW → no block (data only)
      timeout in ACTIVE → blocked iff active_failure_mode == "fail_closed"
  - sync evaluate() never runs semantic_async gates (they require an event
    loop); use aevaluate() for those.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from kairos.log import get_logger

from .gates import (
    Gate,
    GateEvaluation,
    GateStatus,
    SessionContext,
)
from .sinks import InMemoryLogSink, LogSink

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


def _combined_error_string(error_strings: list[str]) -> str:
    return "\n\n".join(error_strings)


@dataclass
class InterceptResult:
    """Returned by evaluate()/aevaluate() — drives the harness loop."""

    blocked: bool
    error_string: str | None
    fired_gates: list[GateEvaluation] = field(default_factory=list)


class KairosInterceptor:
    def __init__(
        self,
        gates: list[Gate],
        judge: Callable[..., Any] | None = None,
        log_sink: LogSink | None = None,
    ) -> None:
        self._gates: list[Gate] = list(gates)
        self._judge = judge  # injected dependency for semantic_async gates that need it
        self._sink: LogSink = log_sink if log_sink is not None else InMemoryLogSink()
        self._sessions: dict[str, SessionContext] = {}

    # ------------------------------------------------------------------
    # Session context management
    # ------------------------------------------------------------------

    def get_session_context(self, session_id: str) -> SessionContext:
        ctx = self._sessions.get(session_id)
        if ctx is None:
            ctx = SessionContext(session_id=session_id)
            self._sessions[session_id] = ctx
        return ctx

    def reset_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def emit_evaluation(self, evaluation: GateEvaluation) -> None:
        """Emit a host-side evaluation into the interceptor's telemetry sink.

        Hosts may run narrow experiment detectors outside the generic engine
        while keeping the same GateEvaluation stream for summaries and review.
        """
        self._sink.emit(evaluation)

    def update_context(self, session_id: str, message: dict[str, Any]) -> None:
        """Update session state from a user turn or tool result.

        Expected shapes:
          {"role": "user",      "content": str, "ids": [str, ...]?}
          {"role": "assistant", "content": str}
          {"role": "tool",      "tool_name": str, "result": Any, "ids": [str, ...]?}

        IDs are passed in explicitly rather than parsed here — extraction
        (regex, NER, whatever) is the harness's job, not the engine's.
        """
        ctx = self.get_session_context(session_id)
        role = message.get("role")
        if role == "user":
            ctx.turn_idx += 1
            content = str(message.get("content", ""))
            ctx.full_transcript += f"\n[user] {content}"
            if content:
                ctx.user_messages.append(content)
            mentioned = set(message.get("ids", []) or [])
            # An id mentioned by the user but not yet in read_cache is "user-supplied"
            # — useful for gates that block actions on ids the agent hallucinated.
            ctx.user_supplied_ids |= mentioned - ctx.read_cache
        elif role == "tool":
            tool_name = str(message.get("tool_name", ""))
            if tool_name:
                ctx.executed_tools.append(tool_name)
            ids = set(message.get("ids", []) or [])
            ctx.read_cache |= ids
            # Once an id has been read it's no longer merely user-supplied.
            ctx.user_supplied_ids -= ids
            result = message.get("result")
            if tool_name == "get_user_details" and isinstance(result, dict):
                ctx.user_profile.update(result)
            ctx.full_transcript += f"\n[tool:{tool_name}] {result!r}"
            # Structured tool-result log for drift detection and other
            # observers that need the recent reads without re-parsing
            # ``full_transcript``. Capped at the last 20 entries so it's
            # bounded across long sessions.
            result_text = repr(result)
            if len(result_text) > 600:
                result_text = result_text[:600] + "...<truncated>"
            ctx.recent_tool_results.append(
                {
                    "tool_name": tool_name,
                    "result_repr": result_text,
                    "ids": sorted(ids),
                    "turn_idx": ctx.turn_idx,
                }
            )
            if len(ctx.recent_tool_results) > 20:
                ctx.recent_tool_results = ctx.recent_tool_results[-20:]
        elif role == "assistant":
            content = str(message.get("content", ""))
            ctx.full_transcript += f"\n[assistant] {content}"
        # Unknown roles are ignored — harness may send other event types we don't track.

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        session_id: str,
        tool_name: str,
        kwargs: dict[str, Any],
    ) -> InterceptResult:
        """Synchronous evaluation. Skips semantic_async gates by design."""
        ctx = self.get_session_context(session_id)
        kwargs_snapshot = dict(kwargs)
        fired_gates: list[GateEvaluation] = []
        blocked_overall = False
        error_strings: list[str] = []

        for gate in self._gates:
            if not self._should_run(gate, tool_name):
                continue
            if gate.gate_type == "semantic_async":
                # Sync path can't drive an event loop; skip to keep the contract simple.
                continue

            ev = self._run_sync_gate(gate, kwargs_snapshot, ctx, tool_name)
            self._sink.emit(ev)
            fired_gates.append(ev)

            if ev.blocked:
                blocked_overall = True
                error_strings.append(gate.error_string)

        return InterceptResult(
            blocked=blocked_overall,
            error_string=_combined_error_string(error_strings) if blocked_overall else None,
            fired_gates=fired_gates,
        )

    async def aevaluate(
        self,
        session_id: str,
        tool_name: str,
        kwargs: dict[str, Any],
    ) -> InterceptResult:
        """Async evaluation. Runs deterministic gates inline and awaits semantic_async."""
        ctx = self.get_session_context(session_id)
        kwargs_snapshot = dict(kwargs)
        fired_gates: list[GateEvaluation] = []
        blocked_overall = False
        error_strings: list[str] = []

        for gate in self._gates:
            if not self._should_run(gate, tool_name):
                continue

            if gate.gate_type == "semantic_async":
                ev = await self._run_async_gate(gate, kwargs_snapshot, ctx, tool_name)
            else:
                ev = self._run_sync_gate(gate, kwargs_snapshot, ctx, tool_name)
            self._sink.emit(ev)
            fired_gates.append(ev)

            if ev.blocked:
                blocked_overall = True
                error_strings.append(gate.error_string)

        return InterceptResult(
            blocked=blocked_overall,
            error_string=_combined_error_string(error_strings) if blocked_overall else None,
            fired_gates=fired_gates,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _should_run(gate: Gate, tool_name: str) -> bool:
        if gate.status == GateStatus.DISABLED:
            return False
        return gate.target_tool == "*" or gate.target_tool == tool_name

    def _run_sync_gate(
        self,
        gate: Gate,
        kwargs_snapshot: dict[str, Any],
        ctx: SessionContext,
        tool_name: str,
    ) -> GateEvaluation:
        t0 = time.perf_counter()
        fired = False
        error: str | None = None
        try:
            raw = gate.evaluation_logic(kwargs_snapshot, ctx)
            if inspect.iscoroutine(raw):
                # A semantic_async evaluator slipped into the deterministic path
                # — refuse to run it here, fail-open, log loudly.
                error = "async_evaluator_in_sync_path"
                raw.close()
            elif inspect.isawaitable(raw):
                error = "async_evaluator_in_sync_path"
            else:
                fired = bool(raw)
        except Exception as exc:  # noqa: BLE001 — engine contract: never crash
            error = f"gate_error: {type(exc).__name__}: {exc}"
            logger.warning(
                "intercept.gate_error",
                gate_id=gate.gate_id,
                tool=tool_name,
                error=error,
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        blocked = self._decide_block(gate, fired, error, timeout=False)
        return GateEvaluation(
            session_id=ctx.session_id,
            turn_idx=ctx.turn_idx,
            gate_id=gate.gate_id,
            status=gate.status,
            fired=fired,
            blocked=blocked,
            kwargs_snapshot=kwargs_snapshot,
            latency_ms=latency_ms,
            tool_name=tool_name,
            error=error,
        )

    async def _run_async_gate(
        self,
        gate: Gate,
        kwargs_snapshot: dict[str, Any],
        ctx: SessionContext,
        tool_name: str,
    ) -> GateEvaluation:
        t0 = time.perf_counter()
        fired = False
        error: str | None = None
        timed_out = False
        try:
            raw = gate.evaluation_logic(kwargs_snapshot, ctx)
            if inspect.isawaitable(raw):
                fired = bool(await asyncio.wait_for(raw, timeout=gate.timeout_ms / 1000.0))
            else:
                # Sync evaluator under semantic_async type — accept it.
                fired = bool(raw)
        except TimeoutError:
            timed_out = True
            error = "timeout"
            logger.info(
                "intercept.gate_timeout",
                gate_id=gate.gate_id,
                tool=tool_name,
                timeout_ms=gate.timeout_ms,
            )
        except Exception as exc:  # noqa: BLE001
            error = f"gate_error: {type(exc).__name__}: {exc}"
            logger.warning(
                "intercept.gate_error",
                gate_id=gate.gate_id,
                tool=tool_name,
                error=error,
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        blocked = self._decide_block(gate, fired, error, timeout=timed_out)
        return GateEvaluation(
            session_id=ctx.session_id,
            turn_idx=ctx.turn_idx,
            gate_id=gate.gate_id,
            status=gate.status,
            fired=fired,
            blocked=blocked,
            kwargs_snapshot=kwargs_snapshot,
            latency_ms=latency_ms,
            tool_name=tool_name,
            error=error,
        )

    @staticmethod
    def _decide_block(
        gate: Gate,
        fired: bool,
        error: str | None,
        *,
        timeout: bool,
    ) -> bool:
        """Apply mode + failure-mode rules to compute the final block decision.

        Rules (precedence top-down):
          - DISABLED: never block (caller already filtered, but defensive).
          - SHADOW: never block, regardless of fire/error/timeout.
          - Non-timeout error (gate raised exception): fail-open, never block.
          - ACTIVE + timeout: per active_failure_mode (default fail_closed).
          - ACTIVE + clean: block iff fired.
        """
        if gate.status == GateStatus.DISABLED:
            return False
        if gate.status == GateStatus.SHADOW:
            return False
        if error is not None and not timeout:
            # Spec: gate exceptions always fail-open.
            return False
        if timeout:
            return gate.active_failure_mode == "fail_closed"
        return fired
