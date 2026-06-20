#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""First-divergence scorer for tau-bench airline trajectories.

Deterministic, no LLM, no new runs. Reads a tau-bench checkpoint result file
(written by ``tau_harness.run``) and, for each task, finds the FIRST step where
the agent's write action diverges from tau's gold action set.

Why writes only: tau-bench scores a task on its *side-effect* actions
(``book_reservation``, ``update_*``, ``cancel_*``, ``send_*``,
``transfer_*``) — read-only tool calls (``get_user_details``, ``search_*``) do
not affect reward. So correctness = the agent's ordered write actions matching
the gold write actions by name AND kwargs. The first agent write that mismatches
(wrong kwargs, wrong tool, or an extra write) is the first-error step.

This gives a baseline First-Error signal (see docs/live-pipeline/DIRECTION.md)
with zero new API spend: does first-write-divergence predict tau failure?

Usage:
    python scripts/first_divergence.py [checkpoint.json]
    # defaults to the newest results/tool-calling-*range_0-20*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Side-effect (write) tools — the only ones tau scores. Everything else is a read.
WRITE_PREFIXES = ("book_", "update_", "cancel_", "send_", "transfer_")


def _is_write(tool_name: str) -> bool:
    return tool_name.startswith(WRITE_PREFIXES)


def _gold_writes(task_info: dict[str, Any]) -> list[dict[str, Any]]:
    """Gold side-effect (write) actions: name + kwargs, in order.

    tau gold action lists occasionally include reads (get_*, search_*,
    calculate); those don't affect reward, so we drop them to compare
    write-against-write.
    """
    actions = (task_info.get("task") or {}).get("actions") or []
    return [
        {"name": a["name"], "kwargs": a.get("kwargs", {})}
        for a in actions
        if _is_write(a["name"])
    ]


def _agent_writes(traj: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Agent's actual write tool calls, in order, tagged with traj message index."""
    out: list[dict[str, Any]] = []
    for msg_idx, msg in enumerate(traj):
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        for tc in msg["tool_calls"]:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            if not _is_write(name):
                continue
            raw = fn.get("arguments", "{}")
            try:
                kwargs = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except json.JSONDecodeError:
                kwargs = {"__unparseable__": raw}
            out.append({"name": name, "kwargs": kwargs, "msg_idx": msg_idx})
    return out


def _gold_covered(gold: list[dict[str, Any]], agent: list[dict[str, Any]]) -> bool:
    """Error-propagation: is every gold write satisfied by SOME agent write?

    Match-anywhere (not positional) multiset coverage. If true, any earlier
    wrong write was later corrected — the divergence is a neutral self-correction,
    not a real error.
    """
    pool = [(a["name"], json.dumps(a["kwargs"], sort_keys=True)) for a in agent]
    for g in gold:
        key = (g["name"], json.dumps(g["kwargs"], sort_keys=True))
        if key in pool:
            pool.remove(key)
        else:
            return False
    return True


def first_divergence(task: dict[str, Any]) -> dict[str, Any]:
    """Find the first agent write that diverges from gold. Returns a per-task record."""
    gold = _gold_writes(task.get("info", {}))
    agent = _agent_writes(task.get("traj", []))

    div_step: int | None = None
    reason = "match"
    for i in range(max(len(gold), len(agent))):
        g = gold[i] if i < len(gold) else None
        a = agent[i] if i < len(agent) else None
        if g is None:  # agent did an extra write gold never asked for
            div_step, reason = i, f"extra write: {a['name']}"
            break
        if a is None:  # agent never made a required gold write
            div_step, reason = i, f"missing write: {g['name']}"
            break
        if a["name"] != g["name"]:
            div_step, reason = i, f"wrong tool: {a['name']} != {g['name']}"
            break
        if a["kwargs"] != g["kwargs"]:
            div_step, reason = i, f"wrong args for {a['name']}"
            break

    return {
        "task_id": task.get("task_id"),
        "reward": task.get("reward"),
        "n_gold_writes": len(gold),
        "n_agent_writes": len(agent),
        "first_div_write_idx": div_step,
        "first_div_msg_idx": agent[div_step]["msg_idx"] if div_step is not None and div_step < len(agent) else None,
        "corrected": _gold_covered(gold, agent),
        "propagated_div_idx": None if _gold_covered(gold, agent) else div_step,
        "reason": reason,
    }


def _load_checkpoint(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    rows = data if isinstance(data, list) else data.get("results", data)
    if not isinstance(rows, list):
        raise ValueError(f"unexpected checkpoint shape in {path}")
    return rows


def analyze(path: Path) -> dict[str, Any]:
    rows = _load_checkpoint(path)
    records = [first_divergence(r) for r in rows]

    failed = [r for r in records if (r["reward"] or 0) < 1.0]
    passed = [r for r in records if (r["reward"] or 0) >= 1.0]
    diverged = [r for r in records if r["first_div_write_idx"] is not None]

    # Raw signal: among FAILED tasks, did we localize a divergence?
    failed_with_div = [r for r in failed if r["first_div_write_idx"] is not None]
    # Raw false alarms: PASSED tasks we flagged as diverged.
    passed_with_div = [r for r in passed if r["first_div_write_idx"] is not None]

    # Propagated signal: drop divergences that were later corrected (gold covered).
    failed_with_prop = [r for r in failed if r["propagated_div_idx"] is not None]
    passed_with_prop = [r for r in passed if r["propagated_div_idx"] is not None]

    def _pr(tp: int, fp: int, fn: int) -> dict[str, float | None]:
        return {
            "precision": round(tp / (tp + fp), 3) if (tp + fp) else None,
            "recall": round(tp / (tp + fn), 3) if (tp + fn) else None,
        }

    return {
        "checkpoint": path.name,
        "tasks": len(records),
        "passed": len(passed),
        "failed": len(failed),
        "diverged": len(diverged),
        "failed_localized": len(failed_with_div),
        "passed_false_alarm": len(passed_with_div),
        "raw": {
            "failed_localized": len(failed_with_div),
            "passed_false_alarm": len(passed_with_div),
            **_pr(len(failed_with_div), len(passed_with_div), len(failed) - len(failed_with_div)),
        },
        "propagated": {
            "failed_localized": len(failed_with_prop),
            "passed_false_alarm": len(passed_with_prop),
            "corrected_neutralized": len([r for r in records if r["corrected"]]),
            **_pr(len(failed_with_prop), len(passed_with_prop), len(failed) - len(failed_with_prop)),
        },
        "records": records,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", nargs="?", help="tau-bench result JSON")
    args = ap.parse_args()

    if args.checkpoint:
        path = Path(args.checkpoint)
    else:
        hits = sorted(
            glob.glob(str(REPO_ROOT / "results" / "tool-calling-*range_0-20*.json")),
            key=os.path.getmtime,
            reverse=True,
        )
        if not hits:
            print("no checkpoint found; pass one explicitly", file=sys.stderr)
            return 1
        path = Path(hits[0])

    result = analyze(path)

    raw, prop = result["raw"], result["propagated"]
    print(f"# First-Divergence Report — {result['checkpoint']}\n")
    print(f"tasks={result['tasks']}  passed={result['passed']}  failed={result['failed']}\n")
    print("signal      | localized | false-alarm | precision | recall")
    print("------------|-----------|-------------|-----------|-------")
    print(
        f"raw         | {raw['failed_localized']:>9} | {raw['passed_false_alarm']:>11} "
        f"| {raw['precision']!s:>9} | {raw['recall']}"
    )
    print(
        f"propagated  | {prop['failed_localized']:>9} | {prop['passed_false_alarm']:>11} "
        f"| {prop['precision']!s:>9} | {prop['recall']}"
    )
    print(
        f"\n# RAW = positional first-write divergence (catches everything, "
        f"over-fires on self-correction).\n"
        f"# PROPAGATED = drop divergences later corrected (gold fully covered); "
        f"{prop['corrected_neutralized']} tasks neutralized.\n"
        f"# Ceiling ~0.6: tau scores final DB STATE, not action order — exact-kwargs\n"
        f"# diff can't see state-equivalence. Closing it needs DB replay or a judge.\n"
        f"# See docs/live-pipeline/DIRECTION.md.\n"
    )
    print(f"{'task':>4} {'rew':>4} {'gold':>4} {'agent':>5} {'div@':>5} {'corr':>4}  reason")
    for r in result["records"]:
        div = r["first_div_write_idx"]
        print(
            f"{r['task_id']:>4} {r['reward']!s:>4} {r['n_gold_writes']:>4} "
            f"{r['n_agent_writes']:>5} {('-' if div is None else div)!s:>5} "
            f"{('Y' if r['corrected'] else '-'):>4}  {r['reason']}"
        )

    out = REPO_ROOT / "results" / "first_divergence.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
