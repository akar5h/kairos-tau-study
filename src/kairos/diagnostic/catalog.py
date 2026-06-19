# ruff: noqa: T201
"""Build a pattern catalog from hand-labeled drift alerts.

Reads a ``labeled_alerts.jsonl`` produced by ``kairos.diagnostic.labeler`` and
groups rows by ``proposed_pattern_id``. For each pattern, emits an entry
with:

  * ``description`` — empty by default (user can edit the JSON afterward, or a
    later LLM pass can populate this from examples).
  * ``positive_examples`` — up to N rows where ``true_failure_signal`` is True.
  * ``negative_examples`` — up to N rows where ``true_failure_signal`` is False.
  * ``severity_default`` — derived from the would_break_task rate
    (>=0.66 high, >=0.33 medium, else low).
  * ``recoverability_default`` — derived from the recoverable rate
    (>=0.5 → True else False).
  * ``runtime_features`` — empty by default; filled later by hand or by a
    feature-mining pass.
  * ``pattern_role`` — ``failure`` when at least one positive example exists,
    otherwise ``negative_calibration``. Negative-calibration buckets teach the
    runtime judge what *not* to escalate; they are not matchable failure IDs.
  * ``tool_scopes`` — empty by default; optional hand-edited exact tool names
    or ``prefix*`` patterns that constrain where the pattern can match.
  * ``aliases`` — empty by default; optional hand-edited drift labels that
    should normalize to this pattern.

This file is the seed corpus the runtime drift detector will read at session
start to ground its judge prompt. Re-runnable: existing catalog is overwritten
on each invocation.

Usage:

    python -m kairos.diagnostic.catalog \\
        --labeled data/diagnostic/labeled_alerts.jsonl \\
        --out     data/diagnostic/pattern_catalog_v0.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_DEFAULT_EXAMPLES_PER_BUCKET = 5


def _example_summary(row: dict[str, Any]) -> dict[str, Any]:
    """Tight summary for a labeled alert in the catalog payload."""
    return {
        "session_id": row.get("session_id"),
        "turn_idx": row.get("turn_idx"),
        "tool_name": row.get("tool_name"),
        "kwargs_snapshot": row.get("kwargs_snapshot"),
        "judge_drift_label": row.get("judge_drift_label"),
        "judge_confidence": row.get("judge_confidence"),
        "judge_reason": row.get("judge_reason"),
        "task_reward": row.get("task_reward"),
        "would_break_task": row.get("would_break_task"),
        "recoverable": row.get("recoverable"),
    }


def _severity_default(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "low"
    yes = sum(1 for r in rows if r.get("would_break_task"))
    rate = yes / len(rows)
    if rate >= 0.66:
        return "high"
    if rate >= 0.33:
        return "medium"
    return "low"


def _recoverability_default(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return True
    yes = sum(1 for r in rows if r.get("recoverable"))
    return yes >= (len(rows) / 2)


def _pattern_role(positive_count: int) -> str:
    return "failure" if positive_count > 0 else "negative_calibration"


def build_catalog(
    labeled_rows: list[dict[str, Any]],
    *,
    examples_per_bucket: int = _DEFAULT_EXAMPLES_PER_BUCKET,
) -> list[dict[str, Any]]:
    """Pure function: ``labeled_rows`` → list of catalog entries."""
    by_pattern: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labeled_rows:
        pid = row.get("proposed_pattern_id")
        if not isinstance(pid, str) or not pid:
            continue
        by_pattern[pid].append(row)

    catalog: list[dict[str, Any]] = []
    for pattern_id in sorted(by_pattern):
        examples = by_pattern[pattern_id]
        positives = [e for e in examples if e.get("true_failure_signal")]
        negatives = [e for e in examples if not e.get("true_failure_signal")]
        catalog.append(
            {
                "pattern_id": pattern_id,
                "pattern_role": _pattern_role(len(positives)),
                "description": "",
                "positive_examples": [_example_summary(e) for e in positives[:examples_per_bucket]],
                "negative_examples": [_example_summary(e) for e in negatives[:examples_per_bucket]],
                "runtime_features": [],
                "tool_scopes": [],
                "aliases": [],
                "severity_default": _severity_default(examples),
                "recoverability_default": _recoverability_default(examples),
                "labeled_count": len(examples),
                "positive_count": len(positives),
                "negative_count": len(negatives),
            }
        )
    return catalog


def load_catalog(path: Path | str | None) -> list[dict[str, Any]]:
    """Load a diagnostic pattern catalog JSON file.

    The runtime drift detector accepts the returned list as advisory
    calibration evidence. A missing/invalid explicitly configured path should
    fail loudly at host bring-up; callers that want no catalog should pass
    ``None``.
    """
    if path is None:
        return []
    catalog_path = Path(path)
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("diagnostic pattern catalog must be a JSON list")
    return [entry for entry in payload if isinstance(entry, dict)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labeled", type=Path, required=True, help="Path to labeled_alerts.jsonl")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/diagnostic/pattern_catalog_v0.json"),
    )
    parser.add_argument(
        "--examples-per-bucket",
        type=int,
        default=_DEFAULT_EXAMPLES_PER_BUCKET,
        help="Max positive + max negative examples retained per pattern.",
    )
    args = parser.parse_args(argv)

    if not args.labeled.exists():
        print(f"labeled file not found: {args.labeled}", file=sys.stderr)
        return 1

    rows = [json.loads(line) for line in args.labeled.read_text().splitlines() if line.strip()]
    catalog = build_catalog(rows, examples_per_bucket=args.examples_per_bucket)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(catalog, indent=2, default=str))
    print(f"wrote {len(catalog)} patterns to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
