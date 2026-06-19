"""Gate primitives for the Kairos runtime interceptor.

Defines:
  - SessionContext: per-session state read by gates
  - Gate: the unit of policy (id, type, target tool, evaluator callable, mode)
  - GateEvaluation: telemetry record emitted on every gate run
  - register_gate_callable / get_gate_callable / gate_from_config:
      the registry pattern that lets JSON/dict configs reference compiled
      Python evaluators by gate_id without ever calling eval().

The interceptor itself (engine.py) knows nothing about specific policies;
it only knows how to evaluate Gate objects built from this registry.
"""

from __future__ import annotations

import enum
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

GateType = Literal["deterministic", "semantic_async", "corpus"]
FailureMode = Literal["fail_open", "fail_closed"]


class GateStatus(enum.StrEnum):
    """Lifecycle state of a gate.

    SHADOW   — evaluate and log, never block. Default for new gates.
    ACTIVE   — evaluate and block on fire.
    DISABLED — skip entirely, do not log.
    """

    SHADOW = "SHADOW"
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"


@dataclass
class SessionContext:
    """Per-session state visible to gates during evaluation.

    Gates read freely; mutation goes through KairosInterceptor.update_context
    so the engine controls the invariants (e.g. read_cache ⊥ user_supplied_ids).
    The `extras` dict is the controlled extension point for gates that need
    to stash derived state — engine writes, gates read.
    """

    session_id: str
    read_cache: set[str] = field(default_factory=set)
    user_supplied_ids: set[str] = field(default_factory=set)
    full_transcript: str = ""
    user_messages: list[str] = field(default_factory=list)
    recent_tool_results: list[dict[str, Any]] = field(default_factory=list)
    attempted_tools: list[str] = field(default_factory=list)
    executed_tools: list[str] = field(default_factory=list)
    user_profile: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)
    turn_idx: int = 0


# Evaluator callable signatures. Deterministic gates return bool synchronously;
# semantic_async gates return an awaitable that resolves to bool.
DeterministicEvaluator = Callable[[dict[str, Any], SessionContext], bool]
AsyncEvaluator = Callable[[dict[str, Any], SessionContext], Awaitable[bool]]
GateEvaluator = DeterministicEvaluator | AsyncEvaluator


@dataclass
class Gate:
    """A single policy guarding tool execution.

    `evaluation_logic` is a compiled Python callable (never a string to eval).
    Configs reference gates by `gate_id`; the registry resolves the callable
    at instantiation time via `gate_from_config`.
    """

    gate_id: str
    gate_type: GateType
    target_tool: str  # specific tool name, or "*" for all tools
    evaluation_logic: GateEvaluator
    error_string: str
    status: GateStatus = GateStatus.SHADOW
    timeout_ms: int = 500
    shadow_failure_mode: FailureMode = "fail_open"
    active_failure_mode: FailureMode = "fail_closed"


@dataclass
class GateEvaluation:
    """Telemetry record for a single gate evaluation.

    Shape is the contract with the shadow→active promoter: it consumes streams
    of these to compute fire rates, false-positive rates, and latency p99 per
    gate before flipping status from SHADOW to ACTIVE.
    """

    session_id: str
    turn_idx: int
    gate_id: str
    status: GateStatus
    fired: bool
    blocked: bool
    kwargs_snapshot: dict[str, Any]
    latency_ms: float
    tool_name: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# Gate-callable registry
# ---------------------------------------------------------------------------
#
# Gates ship as compiled Python in a separate gate-library module. Each
# library module decorates its evaluators with @register_gate_callable("id"),
# and config files reference them by gate_id. The interceptor never sees
# the registry — it only sees the resolved Gate dataclass.

_REGISTRY: dict[str, GateEvaluator] = {}


def register_gate_callable(gate_id: str) -> Callable[[GateEvaluator], GateEvaluator]:
    """Decorator: register a gate evaluator under a stable gate_id."""

    def _wrap(fn: GateEvaluator) -> GateEvaluator:
        _REGISTRY[gate_id] = fn
        return fn

    return _wrap


def get_gate_callable(gate_id: str) -> GateEvaluator:
    """Look up a registered evaluator by gate_id. Raises KeyError if missing."""
    if gate_id not in _REGISTRY:
        raise KeyError(f"No gate callable registered for gate_id={gate_id!r}")
    return _REGISTRY[gate_id]


def gate_from_config(config: dict[str, Any]) -> Gate:
    """Build a Gate from a config dict (e.g. parsed from JSON).

    Required keys: gate_id, gate_type, target_tool, error_string.
    Optional keys: status (defaults to SHADOW), timeout_ms,
                   shadow_failure_mode, active_failure_mode.

    The evaluator callable is resolved from the registry by gate_id —
    config never contains executable code.
    """
    status_value = config.get("status", "SHADOW")
    status = status_value if isinstance(status_value, GateStatus) else GateStatus(status_value)

    return Gate(
        gate_id=config["gate_id"],
        gate_type=config["gate_type"],
        target_tool=config["target_tool"],
        evaluation_logic=get_gate_callable(config["gate_id"]),
        error_string=config["error_string"],
        status=status,
        timeout_ms=int(config.get("timeout_ms", 500)),
        shadow_failure_mode=config.get("shadow_failure_mode", "fail_open"),
        active_failure_mode=config.get("active_failure_mode", "fail_closed"),
    )
