"""Kairos runtime interceptor — deterministic control plane between an LLM
agent and its tool execution layer.

Public surface:

    from kairos.intercept import (
        KairosInterceptor,
        InterceptResult,
        Gate,
        GateStatus,
        GateEvaluation,
        SessionContext,
        InMemoryLogSink,
        LogSink,
        register_gate_callable,
        get_gate_callable,
        gate_from_config,
    )
"""

from __future__ import annotations

from .engine import InterceptResult, KairosInterceptor
from .gates import (
    Gate,
    GateEvaluation,
    GateStatus,
    GateType,
    SessionContext,
    gate_from_config,
    get_gate_callable,
    register_gate_callable,
)
from .sinks import InMemoryLogSink, LogSink

__all__ = [
    "Gate",
    "GateEvaluation",
    "GateStatus",
    "GateType",
    "InMemoryLogSink",
    "InterceptResult",
    "KairosInterceptor",
    "LogSink",
    "SessionContext",
    "gate_from_config",
    "get_gate_callable",
    "register_gate_callable",
]
