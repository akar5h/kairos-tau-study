"""Streaming sinks for kairos host integration.

``StreamingJSONLSink`` writes every :class:`~kairos.intercept.GateEvaluation`
to a JSONL file as it arrives. ``DriftObservationSink`` does the same for
drift observations. Both are line-buffered + flushed per emit so a SIGKILL
only loses the last partial line. Both register ``atexit`` to close cleanly
on shutdown.

This module was extracted from tau-agent's ``kairos_intercept.py`` because
the behaviour (run-scoped streaming JSONL) is generic to any host
integration, not application-specific.
"""

from __future__ import annotations

import atexit
import json
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

from kairos.intercept import GateEvaluation, LogSink

if TYPE_CHECKING:
    from kairos.models.semantic_recovery import DriftObservation

__all__ = [
    "DriftObservationSink",
    "StreamingJSONLSink",
    "evaluation_to_dict",
]


def evaluation_to_dict(evaluation: GateEvaluation) -> dict[str, object]:
    """Serialize a :class:`GateEvaluation` into a JSON-compatible dict."""
    status = evaluation.status.value if hasattr(evaluation.status, "value") else str(evaluation.status)
    return {
        "session_id": evaluation.session_id,
        "turn_idx": evaluation.turn_idx,
        "gate_id": evaluation.gate_id,
        "tool_name": evaluation.tool_name,
        "status": status,
        "fired": evaluation.fired,
        "blocked": evaluation.blocked,
        "kwargs_snapshot": evaluation.kwargs_snapshot,
        "latency_ms": evaluation.latency_ms,
        "error": evaluation.error,
    }


class StreamingJSONLSink(LogSink):
    """Append each evaluation to ``path`` as a JSONL line, flushed per emit.

    Keeps an in-memory ``records`` list for callers that want to scan after the
    run completes (the JSONL on disk is the authoritative artifact). Closes the
    file handle in ``atexit`` if the host never gets to call ``close()``.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: list[GateEvaluation] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", buffering=1, encoding="utf-8")
        atexit.register(self.close)

    def emit(self, evaluation: GateEvaluation) -> None:
        self.records.append(evaluation)
        try:
            self._fh.write(json.dumps(evaluation_to_dict(evaluation)) + "\n")
            self._fh.flush()
        except (ValueError, OSError):
            # File closed mid-shutdown — keep the in-memory copy and move on.
            pass

    def close(self) -> None:
        try:
            if self._fh and not self._fh.closed:
                self._fh.flush()
                self._fh.close()
        except (ValueError, OSError):
            pass


class DriftObservationSink:
    """Append-per-emit JSONL sink for :class:`DriftObservation` records.

    Parallel artifact to :class:`StreamingJSONLSink` — the gate sink writes
    intervention telemetry, this sink writes observation-only drift records.
    Kept separate so the two streams stay grep-able and consumers can't
    accidentally mix block decisions with observations.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: list[DriftObservation] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", buffering=1, encoding="utf-8")
        atexit.register(self.close)

    def emit(self, observation: DriftObservation) -> None:
        self.records.append(observation)
        try:
            self._fh.write(observation.model_dump_json() + "\n")
            self._fh.flush()
        except (ValueError, OSError):
            pass

    def close(self) -> None:
        try:
            if self._fh and not self._fh.closed:
                self._fh.flush()
                self._fh.close()
        except (ValueError, OSError):
            pass
