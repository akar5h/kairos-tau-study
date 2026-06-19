# ruff: noqa: T201
"""Interactive hand-labeler for drift_observations.jsonl.

Walks each ``consistent=false`` drift row, shows the user the proposed call,
recent ledger evidence, the judge's reason + label, and the task outcome from
the tau-bench results JSON. Prompts for four labels:

  * ``true_failure_signal`` — did the drift actually point at something bad?
  * ``would_break_task``    — uncorrected, would this drift have failed the task?
  * ``recoverable``         — could the agent fix this on its own / did it?
  * ``proposed_pattern_id`` — kebab-case label for the failure shape

Appends each labeled row to the output JSONL as it's entered, so a Ctrl-C
mid-session loses nothing. Skips rows already labeled in the output file
(resumable).

Usage:

    python -m kairos.diagnostic.labeler \\
        --drift  data/runs/<id>/drift_observations.jsonl \\
        --results results/<...>.json \\
        --out    data/diagnostic/labeled_alerts.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def _load_drift(path: Path, only_flagged: bool) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if only_flagged:
        rows = [r for r in rows if r.get("consistent") is False]
    return rows


def _load_results_index(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    """Index tau-bench result entries by ``(task_id, trial)``."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        return {}
    index: dict[tuple[int, int], dict[str, Any]] = {}
    for entry in payload:
        try:
            key = (int(entry["task_id"]), int(entry.get("trial", 0)))
            index[key] = entry
        except (KeyError, TypeError, ValueError):
            continue
    return index


def _kwargs_hash(kwargs: Any) -> str:
    """Stable short identity for the proposed call arguments.

    A single turn can propose the same tool more than once with different
    kwargs, so ``(session_id, turn_idx, tool_name)`` is not unique enough for
    resumable labeling.
    """
    encoded = json.dumps(kwargs or {}, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _label_key(row: dict[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(row.get("session_id", "")),
        int(row.get("turn_idx", 0)),
        str(row.get("tool_name", "")),
        str(row.get("kwargs_hash") or _kwargs_hash(row.get("kwargs_snapshot"))),
    )


def _existing_keys(path: Path) -> tuple[set[tuple[str, int, str, str]], list[str]]:
    """Return previously-labeled keys + the running list of known pattern_ids."""
    if not path.exists():
        return set(), []
    keys: set[tuple[str, int, str, str]] = set()
    known: list[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        keys.add(_label_key(row))
        pid = row.get("proposed_pattern_id")
        if isinstance(pid, str) and pid and pid not in known:
            known.append(pid)
    return keys, known


def _parse_session_id(session_id: str) -> tuple[int, int] | None:
    """``task-N-trial-M`` → ``(N, M)``. Returns ``None`` on shape mismatch."""
    parts = session_id.split("-")
    if len(parts) != 4 or parts[0] != "task" or parts[2] != "trial":
        return None
    try:
        return int(parts[1]), int(parts[3])
    except ValueError:
        return None


def _format_row(drift: dict[str, Any], task_result: dict[str, Any] | None) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"session={drift.get('session_id')}  turn={drift.get('turn_idx')}")
    lines.append(f"tool={drift.get('tool_name')}")
    kwargs_text = json.dumps(drift.get("kwargs_snapshot") or {}, default=str)
    if len(kwargs_text) > 600:
        kwargs_text = kwargs_text[:600] + "..."
    lines.append(f"kwargs={kwargs_text}")
    lines.append("")
    lines.append(f"judge_drift_label={drift.get('drift_label')!r}")
    lines.append(f"judge_confidence={drift.get('confidence')}")
    reason = (drift.get("reason") or "")[:400]
    lines.append(f"judge_reason={reason}")
    evidence = drift.get("evidence_pointers") or []
    lines.append(f"judge_evidence={evidence}")
    lines.append("")
    if task_result is not None:
        reward = task_result.get("reward")
        outcome = "PASSED" if isinstance(reward, (int, float)) and reward >= 1.0 else "FAILED"
        lines.append(f"TASK reward={reward}  ({outcome})")
        instruction = task_result.get("info", {}).get("task", {}).get("instruction", "")
        if instruction:
            short = instruction.strip().replace("\n", " ")[:400]
            lines.append(f"INSTRUCTION: {short}")
    else:
        lines.append("TASK outcome: <unmatched>")
    return "\n".join(lines)


def _prompt_yn(label: str) -> bool:
    while True:
        answer = input(f"  {label} [y/n] ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("    enter y or n")


def _prompt_pattern_id(known: list[str]) -> str:
    if known:
        listing = "  ".join(f"[{i}] {p}" for i, p in enumerate(known))
        print(f"  known patterns this session: {listing}")
        print("    enter an index to reuse, or a new kebab-case label")
    while True:
        raw = input("  proposed_pattern_id: ").strip()
        if not raw:
            print("    cannot be empty")
            continue
        if raw.isdigit() and known and 0 <= int(raw) < len(known):
            return known[int(raw)]
        return raw


def _label_row(
    drift: dict[str, Any],
    task_result: dict[str, Any] | None,
    known_patterns: list[str],
) -> dict[str, Any]:
    tfs = _prompt_yn("true_failure_signal?")
    wbt = _prompt_yn("would_break_task?")
    rec = _prompt_yn("recoverable?")
    pid = _prompt_pattern_id(known_patterns)
    reward = task_result.get("reward") if task_result else None
    return {
        "session_id": drift.get("session_id"),
        "turn_idx": drift.get("turn_idx"),
        "tool_name": drift.get("tool_name"),
        "kwargs_snapshot": drift.get("kwargs_snapshot"),
        "kwargs_hash": _kwargs_hash(drift.get("kwargs_snapshot")),
        "judge_drift_label": drift.get("drift_label"),
        "judge_confidence": drift.get("confidence"),
        "judge_reason": drift.get("reason"),
        "judge_evidence_pointers": drift.get("evidence_pointers"),
        "true_failure_signal": tfs,
        "would_break_task": wbt,
        "recoverable": rec,
        "proposed_pattern_id": pid,
        "task_reward": reward,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drift", type=Path, required=True, help="Path to drift_observations.jsonl")
    parser.add_argument("--results", type=Path, required=True, help="Path to tau-bench results JSON")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/diagnostic/labeled_alerts.jsonl"),
        help="Where to append labeled alerts (resumable).",
    )
    parser.add_argument(
        "--include-consistent",
        action="store_true",
        help="Also label rows where consistent=true. Default: only consistent=false.",
    )
    args = parser.parse_args(argv)

    if not args.drift.exists():
        print(f"drift file not found: {args.drift}", file=sys.stderr)
        return 1
    if not args.results.exists():
        print(f"results file not found: {args.results}", file=sys.stderr)
        return 1

    drift_rows = _load_drift(args.drift, only_flagged=not args.include_consistent)
    results_index = _load_results_index(args.results)
    existing_keys, known_patterns = _existing_keys(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    todo = [row for row in drift_rows if _label_key(row) not in existing_keys]
    print(f"loaded {len(drift_rows)} drift rows  ({len(existing_keys)} already labeled, {len(todo)} to go)")
    if not todo:
        print("nothing to do.")
        return 0

    with args.out.open("a", encoding="utf-8") as out_fh:
        for index, drift in enumerate(todo, start=1):
            sess_key = _parse_session_id(str(drift.get("session_id", "")))
            task_result = results_index.get(sess_key) if sess_key else None
            print()
            print(f"[{index}/{len(todo)}]")
            print(_format_row(drift, task_result))
            print()
            try:
                label = _label_row(drift, task_result, known_patterns)
            except (EOFError, KeyboardInterrupt):
                print("\nstopping; everything labeled so far is already saved.")
                return 0
            out_fh.write(json.dumps(label) + "\n")
            out_fh.flush()
            if label["proposed_pattern_id"] not in known_patterns:
                known_patterns.append(label["proposed_pattern_id"])
            print("  saved.")

    print(f"\ndone. labeled {len(todo)} rows → {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
