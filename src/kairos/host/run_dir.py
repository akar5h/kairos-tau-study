"""Run directory + manifest + summary helpers for kairos host integration.

Functions here own the on-disk shape of a run: directory naming, manifest
serialization, summary aggregation, and a tiny env-var snapshot helper. All
extracted from tau-agent's ``kairos_intercept.py`` because the behaviour
(timestamped run dir + manifest + summary aggregate) is generic to any host.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable, Sequence  # noqa: TC003
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003
from typing import Any

from kairos.intercept import GateEvaluation  # noqa: TC001

__all__ = [
    "capture_env_subset",
    "make_run_id",
    "write_manifest",
    "write_summary",
]


def make_run_id(manifest: dict[str, Any] | None) -> str:
    """Return ``<UTC timestamp>[_<env>][_<model basename>]``."""
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parts = [ts]
    if manifest:
        env_name = manifest.get("env")
        if env_name:
            parts.append(str(env_name))
        model = manifest.get("model")
        if model:
            parts.append(str(model).split("/")[-1])
    return "_".join(parts)


def capture_env_subset(
    *,
    prefixes: Sequence[str] = ("TAU_BENCH_", "KAIROS_", "OPENAI_API_BASE", "PHOENIX_"),
    redact: Iterable[str] = (
        "KAIROS_SEMANTIC_OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
    ),
) -> dict[str, str]:
    """Snapshot the env vars matching any of ``prefixes``, redacting secrets."""
    redact_set = set(redact)
    out: dict[str, str] = {}
    for key, value in os.environ.items():
        if not any(key.startswith(prefix) or key == prefix for prefix in prefixes):
            continue
        out[key] = "<redacted>" if key in redact_set else value
    return out


def write_manifest(
    run_dir: Path,
    *,
    run_metadata: dict[str, Any] | None,
    gates_config: list[dict[str, Any]],
    gate_summary: list[tuple[str, str, str]],
) -> Path:
    """Write ``manifest.json`` into ``run_dir``."""
    full: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "run_dir": str(run_dir),
        "gates_config": gates_config,
        "gate_summary": gate_summary,
        "env": capture_env_subset(),
    }
    if run_metadata:
        full["run"] = run_metadata
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(full, indent=2, default=str))
    return path


def write_summary(
    run_dir: Path,
    *,
    records: list[GateEvaluation],
    task_results: list[dict[str, Any]] | None,
    started_at_monotonic: float,
    finalized_via: str = "explicit",
    drift_observations: list[Any] | None = None,
) -> Path:
    """Write ``summary.json`` aggregating gate evaluations, drift
    observations, and task results.

    ``drift_observations`` is a list of :class:`DriftObservation` Pydantic
    models when the drift detector ran during the run; pass ``None`` when
    drift detection was disabled.
    """
    fired = sum(1 for ev in records if ev.fired)
    blocked = sum(1 for ev in records if ev.blocked)

    by_gate: dict[str, dict[str, int]] = {}
    for ev in records:
        bucket = by_gate.setdefault(ev.gate_id, {"evaluations": 0, "fired": 0, "blocked": 0})
        bucket["evaluations"] += 1
        if ev.fired:
            bucket["fired"] += 1
        if ev.blocked:
            bucket["blocked"] += 1

    summary: dict[str, Any] = {
        "finished_at": datetime.now(UTC).isoformat(),
        "finalized_via": finalized_via,
        "duration_s": time.time() - started_at_monotonic,
        "gate_evaluations_total": len(records),
        "gate_evaluations_fired": fired,
        "gate_evaluations_blocked": blocked,
        "gate_breakdown": by_gate,
    }

    # Drift aggregate — run-wide breakdown + per-session bucket below.
    drift_per_session: dict[str, list[dict[str, Any]]] = {}
    if drift_observations:
        by_label: dict[str, int] = {}
        by_confidence: dict[str, int] = {"low": 0, "medium": 0, "high": 0}
        total_observations = 0
        total_drifts = 0
        total_predicted_failures = 0
        total_recoverable = 0
        total_errors = 0
        by_severity: dict[str, int] = {"low": 0, "medium": 0, "high": 0}
        by_verdict_status: dict[str, int] = {
            "clean": 0,
            "judge_error": 0,
            "invalid_verdict_json": 0,
        }
        for obs in drift_observations:
            obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)
            total_observations += 1
            verdict_status = str(obs_dict.get("verdict_status") or "clean")
            by_verdict_status[verdict_status] = by_verdict_status.get(verdict_status, 0) + 1
            session_id = obs_dict.get("session_id", "")
            drift_per_session.setdefault(session_id, []).append(obs_dict)
            if obs_dict.get("error") or verdict_status != "clean":
                total_errors += 1
                continue
            if obs_dict.get("consistent") is False:
                total_drifts += 1
                if obs_dict.get("would_break_task"):
                    total_predicted_failures += 1
                if obs_dict.get("recoverable"):
                    total_recoverable += 1
                label = obs_dict.get("drift_label") or "unlabeled"
                by_label[label] = by_label.get(label, 0) + 1
                severity = obs_dict.get("severity", "low")
                if severity in by_severity:
                    by_severity[severity] += 1
                confidence = obs_dict.get("confidence", "low")
                if confidence in by_confidence:
                    by_confidence[confidence] += 1
        summary["drift_summary"] = {
            "total_observations": total_observations,
            "drift_detected": total_drifts,
            "predicted_task_breaking_drift": total_predicted_failures,
            "recoverable_drift": total_recoverable,
            "judge_errors": total_errors,
            "classification_failed": total_errors,
            "by_verdict_status": by_verdict_status,
            "by_label": dict(sorted(by_label.items(), key=lambda kv: -kv[1])),
            "by_severity": by_severity,
            "by_confidence": by_confidence,
        }

    if task_results is not None:
        rewards = [float(r.get("reward", 0.0)) for r in task_results]
        summary["tasks_total"] = len(rewards)
        summary["tasks_passed"] = int(sum(1 for r in rewards if r == 1.0))
        summary["avg_reward"] = (sum(rewards) / len(rewards)) if rewards else 0.0
        per_task_entries: list[dict[str, Any]] = []
        for r in task_results:
            entry: dict[str, Any] = {
                "task_id": r.get("task_id"),
                "reward": r.get("reward"),
                "trial": r.get("trial"),
            }
            # Match drift observations to this task by session_id convention
            # `task-<id>-trial-<n>`. If the host uses a different scheme the
            # observations are still in the per-session bucket below.
            task_session_id = (
                f"task-{entry['task_id']}-trial-{entry['trial']}"
                if entry.get("task_id") is not None and entry.get("trial") is not None
                else None
            )
            task_obs = drift_per_session.get(task_session_id, []) if task_session_id else []
            entry["drift_observation_count"] = len(task_obs)
            entry["drift_detected_count"] = sum(
                1
                for o in task_obs
                if o.get("consistent") is False and not o.get("error") and o.get("verdict_status", "clean") == "clean"
            )
            entry["drift_predicted_task_breaking_count"] = sum(
                1
                for o in task_obs
                if o.get("consistent") is False
                and o.get("would_break_task")
                and not o.get("error")
                and o.get("verdict_status", "clean") == "clean"
            )
            entry["drift_classification_failed_count"] = sum(
                1 for o in task_obs if o.get("error") or o.get("verdict_status", "clean") != "clean"
            )
            entry["drift_observations"] = task_obs
            per_task_entries.append(entry)
        summary["per_task"] = per_task_entries

    path = run_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str))
    return path
