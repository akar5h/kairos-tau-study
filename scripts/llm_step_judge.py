#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""LLM step-judge for tau-bench trajectories — roadmap move A.

A minimal, drift-detector-style LLM step verifier. For each agent WRITE action
in a trajectory it asks an LLM: *given the user's request and the conversation
so far, does executing this action deviate from correctly completing the task?*
(``would_break_task``). The first ``would_break_task=True`` write is the
predicted first-error step.

This is NOT a faithful replay of kairos's full drift detector — that needs a
live ``SessionContext`` + ``PolicyPack`` + ``SessionExpectation`` rebuilt
step-by-step from the interceptor. It is the same *idea* (per-step LLM
"would this break the task?" verdict), reusing kairos's real OpenRouter client
(``OpenRouterExpectationClient``), so we can measure the LLM-judge signal vs the
deterministic ``first_divergence.py`` baseline (raw 0.55 / propagated 0.60)
cheaply and today. See docs/live-pipeline/DIRECTION.md §5b move A.

Usage:
    KAIROS_SEMANTIC_MODEL=openai/gpt-4o-mini \
      python scripts/llm_step_judge.py [checkpoint.json] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from kairos.semantic_recovery.expectation import OpenRouterExpectationClient

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import first_divergence as fd  # noqa: E402  (sibling script, not a package)

JUDGE_SYSTEM = (
    "You are a step verifier for a customer-service agent. You see the user's "
    "request, the conversation so far, and ONE tool action the agent is about "
    "to take. Decide whether this action deviates from correctly completing the "
    "user's task. Be precise: legitimate information-gathering or a reasonable "
    "step toward the goal is NOT a deviation. Only flag actions that are wrong, "
    "premature, contradict the user's stated constraints, or commit an incorrect "
    "side-effect. Reply ONLY as JSON: "
    '{"verdict": "advance|neutral|wrong", "would_break_task": true|false, '
    '"reason": "<one sentence>"}'
)


def _conversation_before(traj: list[dict[str, Any]], stop_msg_idx: int) -> str:
    """Compact text of the conversation up to (not including) stop_msg_idx."""
    lines: list[str] = []
    for msg in traj[:stop_msg_idx]:
        role = msg.get("role")
        if role == "system":
            continue
        if role == "user":
            lines.append(f"USER: {msg.get('content', '')}")
        elif role == "assistant":
            if msg.get("content"):
                lines.append(f"AGENT: {msg['content']}")
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                lines.append(f"AGENT_CALL: {fn.get('name')}({fn.get('arguments', '')})")
        elif role == "tool":
            content = str(msg.get("content", ""))[:400]
            lines.append(f"TOOL_RESULT: {content}")
    text = "\n".join(lines)
    return text[-6000:]  # keep the prompt bounded


def _instruction(task: dict[str, Any]) -> str:
    return (task.get("info", {}).get("task") or {}).get("instruction", "")


def judge_task(task: dict[str, Any], client: OpenRouterExpectationClient) -> dict[str, Any]:
    """Run the LLM judge over each WRITE step; return a per-task record."""
    instruction = _instruction(task)
    traj = task.get("traj", [])
    gold = fd._gold_writes(task.get("info", {}))
    agent = fd._agent_writes(traj)

    first_break: int | None = None
    verdicts: list[dict[str, Any]] = []
    for i, a in enumerate(agent):
        convo = _conversation_before(traj, a["msg_idx"])
        user_prompt = (
            f"USER REQUEST:\n{instruction}\n\n"
            f"CONVERSATION SO FAR:\n{convo}\n\n"
            f"PROPOSED ACTION:\n{a['name']}({json.dumps(a['kwargs'])})"
        )
        try:
            raw = client.complete_json(system_prompt=JUDGE_SYSTEM, user_prompt=user_prompt)
            v = json.loads(raw)
        except (RuntimeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            v = {"verdict": "error", "would_break_task": False, "reason": str(exc)[:120]}
        v["write_idx"] = i
        verdicts.append(v)
        if first_break is None and v.get("would_break_task") is True:
            first_break = i

    corrected = fd._gold_covered(gold, agent)
    return {
        "task_id": task.get("task_id"),
        "reward": task.get("reward"),
        "n_agent_writes": len(agent),
        "first_break_idx": first_break,
        "corrected": corrected,
        "propagated_break_idx": None if corrected else first_break,
        "verdicts": verdicts,
    }


def _pr(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    return {
        "precision": round(tp / (tp + fp), 3) if (tp + fp) else None,
        "recall": round(tp / (tp + fn), 3) if (tp + fn) else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", nargs="?")
    ap.add_argument("--limit", type=int, help="judge only the first N tasks")
    args = ap.parse_args()

    path = Path(args.checkpoint) if args.checkpoint else fd._newest_checkpoint()
    rows = fd._load_checkpoint(path)
    if args.limit:
        rows = rows[: args.limit]

    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        print("set OPENROUTER_API_KEY", file=sys.stderr)
        return 1
    client = OpenRouterExpectationClient(
        api_key=SecretStr(key),
        model=os.getenv("KAIROS_SEMANTIC_MODEL", "openai/gpt-4o-mini"),
    )

    records = []
    for r in rows:
        rec = judge_task(r, client)
        records.append(rec)
        print(
            f"task {rec['task_id']:>2} rew={rec['reward']} "
            f"writes={rec['n_agent_writes']} break@={rec['first_break_idx']} "
            f"corrected={'Y' if rec['corrected'] else '-'}",
            flush=True,
        )

    failed = [r for r in records if (r["reward"] or 0) < 1.0]
    passed = [r for r in records if (r["reward"] or 0) >= 1.0]
    raw_f = [r for r in failed if r["first_break_idx"] is not None]
    raw_p = [r for r in passed if r["first_break_idx"] is not None]
    prop_f = [r for r in failed if r["propagated_break_idx"] is not None]
    prop_p = [r for r in passed if r["propagated_break_idx"] is not None]

    result = {
        "checkpoint": path.name,
        "model": client.model,
        "tasks": len(records),
        "passed": len(passed),
        "failed": len(failed),
        "raw": {
            "localized": len(raw_f),
            "false_alarm": len(raw_p),
            **_pr(len(raw_f), len(raw_p), len(failed) - len(raw_f)),
        },
        "propagated": {
            "localized": len(prop_f),
            "false_alarm": len(prop_p),
            **_pr(len(prop_f), len(prop_p), len(failed) - len(prop_f)),
        },
        "records": records,
    }

    print(f"\n# LLM step-judge ({result['model']}) — {result['checkpoint']}")
    print(f"tasks={result['tasks']} passed={result['passed']} failed={result['failed']}\n")
    print("signal      | localized | false-alarm | precision | recall")
    print("------------|-----------|-------------|-----------|-------")
    for k in ("raw", "propagated"):
        s = result[k]
        print(f"{k:<11} | {s['localized']:>9} | {s['false_alarm']:>11} | {s['precision']!s:>9} | {s['recall']}")
    print("\n# baseline (first_divergence.py): raw 0.55 / propagated 0.60 precision")

    out = REPO_ROOT / "results" / "llm_step_judge.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
