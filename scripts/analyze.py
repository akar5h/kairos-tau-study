# SPDX-License-Identifier: Apache-2.0
"""Analyze an ablation bundle or summary JSON into a reward ladder and drift
confusion matrices.

Usage:
    python scripts/analyze.py [bundle_or_summary.json]

Defaults to results/ablation_summary.json when no argument is given.
Output: Markdown report to stdout + results/analysis_output.json.
Pure stdlib — no pandas, no kairos import.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "results" / "ablation_summary.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "analysis_output.json"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_bundle(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_modes(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    return bundle.get("modes", [])


def get_per_task(mode: dict[str, Any]) -> list[dict[str, Any]]:
    return (mode.get("kairos_summary") or {}).get("per_task") or []


# ---------------------------------------------------------------------------
# Core analytics
# ---------------------------------------------------------------------------


def build_reward_ladder(modes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows sorted descending by average_reward."""
    rows = []
    for m in modes:
        ks = m.get("kairos_summary") or {}
        rows.append(
            {
                "mode": m["mode"],
                "avg_reward": m.get("average_reward"),
                "tasks_passed": ks.get("tasks_passed"),
                "tasks_total": ks.get("tasks_total"),
            }
        )
    rows.sort(key=lambda r: (r["avg_reward"] is None, -(r["avg_reward"] or 0)))
    return rows


def compute_confusion(
    per_task: list[dict[str, Any]],
    detector_field: str,
) -> dict[str, Any]:
    """Compute TP/FP/TN/FN for one detector view.

    Positive class = task FAILURE (reward < 1.0).
    Detector fires when the given field > 0.
    """
    tp = fp = tn = fn = 0
    for row in per_task:
        reward = float(row.get("reward", 0.0))
        fired = int(row.get(detector_field) or 0) > 0
        failed = reward < 1.0
        if failed and fired:
            tp += 1
        elif not failed and fired:
            fp += 1
        elif not failed and not fired:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "precision": precision, "recall": recall}


def build_detection_results(modes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each mode with per_task drift data, compute both confusion views."""
    results = []
    for m in modes:
        per_task = get_per_task(m)
        if not per_task:
            continue
        raw = compute_confusion(per_task, "drift_detected_count")
        filtered = compute_confusion(per_task, "drift_predicted_task_breaking_count")
        results.append(
            {
                "mode": m["mode"],
                "raw_drift": raw,
                "filtered_task_breaking": filtered,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def fmt_float(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.4f}"


def render_reward_ladder(rows: list[dict[str, Any]]) -> str:
    lines = [
        "## Reward Ladder",
        "",
        "| Mode | Avg Reward | Tasks Passed / Total |",
        "| ---- | ---------- | -------------------- |",
    ]
    for r in rows:
        passed = r["tasks_passed"]
        total = r["tasks_total"]
        fraction = f"{passed}/{total}" if passed is not None and total is not None else "—"
        lines.append(f"| {r['mode']} | {fmt_float(r['avg_reward'])} | {fraction} |")
    lines.append("")
    return "\n".join(lines)


def render_detection_tables(detection: list[dict[str, Any]]) -> str:
    if not detection:
        return ""
    lines = ["## Drift Detection Confusion Matrices", ""]
    for d in detection:
        lines.append(f"### {d['mode']}")
        lines.append("")
        lines.append("| View | TP | FP | TN | FN | Precision | Recall |")
        lines.append("| ---- | -- | -- | -- | -- | --------- | ------ |")
        for view_key, label in [
            ("raw_drift", "Raw drift"),
            ("filtered_task_breaking", "Filtered task-breaking"),
        ]:
            c = d[view_key]
            lines.append(
                f"| {label} | {c['TP']} | {c['FP']} | {c['TN']} | {c['FN']}"
                f" | {c['precision']:.2f} | {c['recall']:.2f} |"
            )
        lines.append("")
    return "\n".join(lines)


def build_report(
    reward_ladder: list[dict[str, Any]],
    detection: list[dict[str, Any]],
) -> str:
    sections = [
        "# Ablation Analysis Report",
        "",
        render_reward_ladder(reward_ladder),
        render_detection_tables(detection),
    ]
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "input",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help="Path to ablation bundle or summary JSON (default: results/ablation_summary.json)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to write analysis JSON output (default: results/analysis_output.json)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path

    bundle = load_bundle(input_path)
    modes = get_modes(bundle)

    reward_ladder = build_reward_ladder(modes)
    detection = build_detection_results(modes)

    report = build_report(reward_ladder, detection)
    print(report)

    output: dict[str, Any] = {
        "source": Path(input_path).name,
        "reward_ladder": reward_ladder,
        "detection": detection,
    }
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(main())
