#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A/B measurement for the tau-airline runtime-intervention experiment.

MEASUREMENT / REPORTING ONLY — does not touch controller, detector, or
intervention logic. Reads two tau-bench result sets (intervention OFF vs ON,
same 50 airline test tasks, identical config except KAIROS_TAU_INTERVENTION_ENABLED)
and prints the honest headline numbers:

  - overall pass rate A vs B + delta
  - per-task win / loss / no-change table
  - count of tasks intervention made WORSE (A passed, B failed) — the key signal
  - detector firing stats from the kairos run dirs (data/runs/*) for run B

Usage:
    python scripts/ab_measure.py \
        --a results/ab_runA_full.json \
        --b results/ab_runB/<file>.json \
        [--b-run-dirs data/runs/<B run dirs...>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict[int, float]:
    data = json.loads(Path(path).read_text())
    rows = data if isinstance(data, list) else data.get("results", data)
    return {r["task_id"]: float(r["reward"]) for r in rows}


def _passed(reward: float) -> bool:
    return reward >= 1.0


def _drift_stats(run_dirs: list[str]) -> dict[str, Any]:
    """Aggregate detector firing across the given kairos run dirs."""
    total_obs = drift_detected = task_breaking = injections = judge_errors = 0
    gates_fired_total = 0
    fired_tasks: set[str] = set()
    for d in run_dirs:
        summ = Path(d) / "summary.json"
        if summ.exists():
            s = json.loads(summ.read_text()).get("drift_summary", {})
            total_obs += s.get("total_observations", 0)
            drift_detected += s.get("drift_detected", 0)
            task_breaking += s.get("predicted_task_breaking_drift", 0)
            judge_errors += s.get("judge_errors", 0)
        gates = Path(d) / "gate_evaluations.jsonl"
        gates_fired = gates_blocked = 0
        if gates.exists():
            for line in gates.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    g = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # A gate that fired+blocked == a correction injected into the loop.
                if g.get("fired"):
                    gates_fired += 1
                if g.get("blocked"):
                    gates_blocked += 1
        injections += gates_blocked
        gates_fired_total += gates_fired
        drift = Path(d) / "drift_observations.jsonl"
        if drift.exists():
            for line in drift.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if o.get("session_id") and o.get("would_break_task") is True:
                    fired_tasks.add(o["session_id"])
    return {
        "total_observations": total_obs,
        "drift_detected": drift_detected,
        "predicted_task_breaking": task_breaking,
        "gates_fired": gates_fired_total,
        "injections_blocked": injections,
        "judge_errors": judge_errors,
        "tasks_with_task_breaking_drift": len(fired_tasks),
    }


def _detection_matrix(run_dirs: list[str], rewards: dict[int, float]) -> dict[str, Any]:
    """Detection-as-failure-predictor: flag = >=1 task-breaking drift on a task;
    ground truth = task failed (reward < 1). Task-level confusion matrix.
    """
    flagged: set[int] = set()
    for d in run_dirs:
        drift = Path(d) / "drift_observations.jsonl"
        if not drift.exists():
            continue
        for line in drift.read_text().splitlines():
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("would_break_task") is True and o.get("session_id"):
                tid = int(o["session_id"].replace("task-", "").split("-trial")[0])
                flagged.add(tid)
    tp = fp = tn = fn = 0
    for t, r in rewards.items():
        failed, flag = r < 1.0, t in flagged
        if flag and failed:
            tp += 1
        elif flag and not failed:
            fp += 1
        elif not flag and not failed:
            tn += 1
        else:
            fn += 1
    p = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "flagged": len(flagged),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(p, 3), "recall": round(rec, 3),
        "f1": round(2 * p * rec / (p + rec), 3) if (p + rec) else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="baseline (intervention OFF) result JSON")
    ap.add_argument("--b", required=True, help="intervention (ON) result JSON")
    ap.add_argument("--a-run-dirs", nargs="*", default=[], help="kairos run dirs for run A")
    ap.add_argument("--b-run-dirs", nargs="*", default=[], help="kairos run dirs for run B")
    ap.add_argument("--out", default="RESULTS.md")
    args = ap.parse_args()

    a, b = _load(args.a), _load(args.b)
    common = sorted(set(a) & set(b))
    n = len(common)
    if set(a) != set(b):
        print(f"WARNING: A has {len(a)} tasks, B has {len(b)}; comparing {n} common task_ids only.")

    a_pass = sum(_passed(a[i]) for i in common)
    b_pass = sum(_passed(b[i]) for i in common)
    a_rate, b_rate = a_pass / n, b_pass / n

    helped = [i for i in common if not _passed(a[i]) and _passed(b[i])]  # A fail -> B pass
    hurt = [i for i in common if _passed(a[i]) and not _passed(b[i])]  # A pass -> B fail
    same = [i for i in common if _passed(a[i]) == _passed(b[i])]

    drift = _drift_stats(args.b_run_dirs) if args.b_run_dirs else None
    det_a = _detection_matrix(args.a_run_dirs, a) if args.a_run_dirs else None
    det_b = _detection_matrix(args.b_run_dirs, b) if args.b_run_dirs else None

    # ---- console summary ----
    print(f"\n# A/B Measurement — N={n}\n")
    print(f"Baseline (OFF)      : {a_pass}/{n} = {a_rate:.3f}")
    print(f"Intervention (ON)   : {b_pass}/{n} = {b_rate:.3f}")
    print(f"Delta               : {(b_rate - a_rate) * 100:+.1f} pp\n")
    print(f"Intervention HELPED : {len(helped)} tasks  {helped}")
    print(f"Intervention HURT   : {len(hurt)} tasks  {hurt}   <-- key signal")
    print(f"No change           : {len(same)} tasks")
    if drift:
        print(
            f"\nDetector (run B): obs={drift['total_observations']} "
            f"detected={drift['drift_detected']} task-breaking={drift['predicted_task_breaking']} "
            f"gates_fired={drift['gates_fired']} injections_blocked={drift['injections_blocked']} "
            f"judge_errors={drift['judge_errors']} "
            f"tasks_with_task_breaking_drift={drift['tasks_with_task_breaking_drift']}"
        )
    for label, det in (("A (OFF)", det_a), ("B (ON)", det_b)):
        if det:
            print(
                f"Detection {label}: flagged={det['flagged']} "
                f"TP={det['tp']} FP={det['fp']} TN={det['tn']} FN={det['fn']} "
                f"precision={det['precision']} recall={det['recall']} f1={det['f1']}"
            )

    # ---- machine-readable ----
    summary = {
        "n": n,
        "baseline_pass": a_pass,
        "baseline_rate": round(a_rate, 4),
        "intervention_pass": b_pass,
        "intervention_rate": round(b_rate, 4),
        "delta_pp": round((b_rate - a_rate) * 100, 2),
        "helped": helped,
        "hurt": hurt,
        "no_change_count": len(same),
        "per_task": {i: {"a": a[i], "b": b[i]} for i in common},
        "detector": drift,
        "detection_a": det_a,
        "detection_b": det_b,
    }
    out_json = REPO_ROOT / "results" / "ab_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_json.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
