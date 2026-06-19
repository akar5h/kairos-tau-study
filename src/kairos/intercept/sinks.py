"""Telemetry sinks for the Kairos interceptor.

A sink consumes GateEvaluation records produced on every gate run. The
default in-memory sink is fine for tests and dev; production deployments
swap in a persistent sink (e.g. JSONL on disk, Phoenix, ClickHouse) by
passing it to KairosInterceptor(log_sink=...).
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gates import GateEvaluation


class LogSink(abc.ABC):
    """Pluggable destination for GateEvaluation records.

    Implementations must be cheap and non-throwing — the interceptor calls
    `emit` on the hot path of every tool call.
    """

    @abc.abstractmethod
    def emit(self, evaluation: GateEvaluation) -> None: ...


class InMemoryLogSink(LogSink):
    """Stores evaluations in a list. Default sink for tests and local dev."""

    def __init__(self) -> None:
        self.records: list[GateEvaluation] = []

    def emit(self, evaluation: GateEvaluation) -> None:
        self.records.append(evaluation)

    def clear(self) -> None:
        self.records.clear()
