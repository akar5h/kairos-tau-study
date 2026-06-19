# SPDX-License-Identifier: Apache-2.0
"""Run a small tau/Kairos ablation ladder and bundle raw artifacts.

Example:
    python scripts/run_ablation.py --start-index 19 --count 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY_PATH = REPO_ROOT / "data" / "airline_success_workflows.json"
DEFAULT_DIAGNOSTIC_CATALOG = REPO_ROOT / "data" / "pattern_catalog_v0.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "ablation_bundles"

CHECKPOINT_RE = re.compile(r"checkpoint path: (?P<path>[^)\\s]+)")
RESULT_RE = re.compile(r"Results saved to (?P<path>\S+)")

MODES: dict[str, dict[str, Any]] = {
    "baseline_no_kairos": {
        "enable_kairos": False,
        "env": {
            "TAU_BENCH_ENABLE_KAIROS": "0",
            "KAIROS_TAU_INTERVENTION_ENABLED": "0",
            "KAIROS_TAU_INJECT_PLAN": "0",
            "KAIROS_WORKFLOW_MEMORY_PATHS": "",
            "KAIROS_SEMANTIC_RECOVERY_ENABLED": "0",
            "KAIROS_DRIFT_DETECTION_ENABLED": "0",
            "KAIROS_SEMANTIC_EXPECTATION_ENABLED": "false",
        },
    },
    "memory_only": {
        # Clean memory-only run: memory store loaded + agent plan injected,
        # everything else off. The kairos SDK builds SemanticRecoveryRuntime
        # from the memory store alone (judge=None) so this mode makes zero
        # LLM judge calls; plan injection happens once at session start.
        "enable_kairos": True,
        "env": {
            "TAU_BENCH_ENABLE_KAIROS": "1",
            "KAIROS_TAU_INTERVENTION_ENABLED": "0",
            "KAIROS_TAU_INJECT_PLAN": "1",
            "KAIROS_WORKFLOW_MEMORY_PATHS": str(DEFAULT_MEMORY_PATH),
            "KAIROS_SEMANTIC_RECOVERY_ENABLED": "0",
            "KAIROS_DRIFT_DETECTION_ENABLED": "0",
            "KAIROS_SEMANTIC_EXPECTATION_ENABLED": "false",
            "KAIROS_TAU_CASCADE_RETRIEVAL_ENABLED": "0",  # lexical retriever
        },
    },
    "memory_cascade": {
        # Memory-only with the cascade retriever (embeddings + claude-haiku
        # rerank) — the best retriever we measured (covered p@1 = 0.628
        # against 3-judge consensus on the 29-entry pool).
        "enable_kairos": True,
        "env": {
            "TAU_BENCH_ENABLE_KAIROS": "1",
            "KAIROS_TAU_INTERVENTION_ENABLED": "0",
            "KAIROS_TAU_INJECT_PLAN": "1",
            "KAIROS_WORKFLOW_MEMORY_PATHS": str(DEFAULT_MEMORY_PATH),
            "KAIROS_SEMANTIC_RECOVERY_ENABLED": "0",
            "KAIROS_DRIFT_DETECTION_ENABLED": "0",
            "KAIROS_SEMANTIC_EXPECTATION_ENABLED": "false",
            "KAIROS_TAU_CASCADE_RETRIEVAL_ENABLED": "1",
        },
    },
    "kairos_detect_nomem_noplan": {
        "enable_kairos": True,
        "env": {
            "TAU_BENCH_ENABLE_KAIROS": "1",
            "KAIROS_TAU_INTERVENTION_ENABLED": "0",
            "KAIROS_TAU_INJECT_PLAN": "0",
            "KAIROS_WORKFLOW_MEMORY_PATHS": "",
            "KAIROS_SEMANTIC_RECOVERY_ENABLED": "0",
            "KAIROS_DRIFT_DETECTION_ENABLED": "1",
            "KAIROS_SEMANTIC_EXPECTATION_ENABLED": "true",
        },
    },
    "kairos_detect_memory_noplan": {
        "enable_kairos": True,
        "env": {
            "TAU_BENCH_ENABLE_KAIROS": "1",
            "KAIROS_TAU_INTERVENTION_ENABLED": "0",
            "KAIROS_TAU_INJECT_PLAN": "0",
            "KAIROS_WORKFLOW_MEMORY_PATHS": str(DEFAULT_MEMORY_PATH),
            "KAIROS_SEMANTIC_RECOVERY_ENABLED": "0",
            "KAIROS_DRIFT_DETECTION_ENABLED": "1",
            "KAIROS_SEMANTIC_EXPECTATION_ENABLED": "true",
        },
    },
    "kairos_intervention_memory_plan": {
        "enable_kairos": True,
        "env": {
            "TAU_BENCH_ENABLE_KAIROS": "1",
            "KAIROS_TAU_INTERVENTION_ENABLED": "1",
            "KAIROS_TAU_INJECT_PLAN": "1",
            "KAIROS_WORKFLOW_MEMORY_PATHS": str(DEFAULT_MEMORY_PATH),
            "KAIROS_SEMANTIC_RECOVERY_ENABLED": "1",
            "KAIROS_DRIFT_DETECTION_ENABLED": "1",
            "KAIROS_SEMANTIC_EXPECTATION_ENABLED": "true",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tau-bench/Kairos ablations and write one raw-analysis JSON bundle."
    )
    parser.add_argument("--env", choices=["airline", "retail"], default="airline")
    parser.add_argument("--task-split", choices=["train", "test", "dev"], default="test")
    parser.add_argument("--start-index", type=int, default=19)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument(
        "--requests-per-minute",
        type=str,
        default=os.getenv("TAU_BENCH_REQUESTS_PER_MINUTE", "30"),
    )
    parser.add_argument("--agent-timeout", default=os.getenv("TAU_BENCH_TIMEOUT", "120"))
    parser.add_argument("--user-timeout", default=os.getenv("TAU_BENCH_USER_TIMEOUT", "90"))
    parser.add_argument("--user-retries", default=os.getenv("TAU_BENCH_USER_RETRIES", "0"))
    parser.add_argument("--model", default=os.getenv("TAU_BENCH_MODEL"))
    parser.add_argument("--user-model", default=os.getenv("TAU_BENCH_USER_MODEL"))
    parser.add_argument("--provider", default=os.getenv("TAU_BENCH_PROVIDER"))
    parser.add_argument(
        "--semantic-model",
        default=os.getenv("KAIROS_SEMANTIC_MODEL", "openai/gpt-4o-mini"),
    )
    parser.add_argument("--memory-path", default=str(DEFAULT_MEMORY_PATH))
    parser.add_argument("--diagnostic-catalog", default=str(DEFAULT_DIAGNOSTIC_CATALOG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--mode-timeout-s", type=float, default=1800.0)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=sorted(MODES),
        default=[
            "baseline_no_kairos",
            "memory_only",
            "kairos_detect_nomem_noplan",
            "kairos_detect_memory_noplan",
            "kairos_intervention_memory_plan",
        ],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    bundle_path = output_dir / f"kairos_ablation_{args.env}_{args.start_index}_{end_index(args)}_{run_id}.json"
    bundle: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "repo_root": str(REPO_ROOT),
        "args": vars(args),
        "modes": [],
    }
    write_json(bundle_path, bundle)

    for mode in args.modes:
        started = time.time()
        print(f"\n=== running {mode} ===", flush=True)
        result = run_mode(args, mode, started)
        bundle["modes"].append(result)
        write_json(bundle_path, bundle)
        print(
            f"=== finished {mode}: reward={result.get('average_reward')} bundle={bundle_path} ===",
            flush=True,
        )

    bundle["completed_at"] = datetime.now(UTC).isoformat()
    write_json(bundle_path, bundle)
    print(f"\nWrote ablation bundle: {bundle_path}")
    return 0


def end_index(args: argparse.Namespace) -> int:
    if args.end_index is not None:
        return args.end_index
    return args.start_index + args.count


def run_mode(args: argparse.Namespace, mode: str, started: float) -> dict[str, Any]:
    spec = MODES[mode]
    env = os.environ.copy()
    env.update(common_env(args))
    env.update(mode_env(args, spec["env"]))

    cmd = base_command(args)
    if spec["enable_kairos"]:
        cmd.append("--enable-kairos")

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None
    stdout_thread = threading.Thread(
        target=stream_pipe,
        args=(proc.stdout, stdout_chunks, ""),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=stream_pipe,
        args=(proc.stderr, stderr_chunks, "[stderr] "),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    timed_out = False
    try:
        returncode = proc.wait(timeout=args.mode_timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        returncode = proc.wait()
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)

    stdout = "".join(stdout_chunks)
    stderr = "".join(stderr_chunks)
    checkpoint = resolve_checkpoint(stdout)
    checkpoint_rows = read_json(checkpoint) if checkpoint is not None else None
    kairos_run = find_kairos_run(checkpoint, started) if spec["enable_kairos"] else None
    artifacts = collect_kairos_artifacts(kairos_run) if kairos_run is not None else None
    return {
        "mode": mode,
        "command": cmd,
        "returncode": returncode,
        "timed_out": timed_out,
        "env_overrides": {key: env.get(key) for key in sorted(common_env(args) | mode_env(args, spec["env"]))},
        "stdout": stdout,
        "stderr": stderr,
        "checkpoint_path": str(checkpoint) if checkpoint is not None else None,
        "checkpoint_rows": checkpoint_rows,
        "average_reward": average_reward(checkpoint_rows),
        "kairos_run_dir": str(kairos_run) if kairos_run is not None else None,
        "kairos_artifacts": artifacts,
        "live_trace_artifacts": collect_live_trace_artifacts(started),
    }


def stream_pipe(pipe: Any, chunks: list[str], prefix: str) -> None:
    for line in pipe:
        chunks.append(line)
        print(f"{prefix}{line}", end="", flush=True)


def common_env(args: argparse.Namespace) -> dict[str, str]:
    values: dict[str, str] = {
        "TAU_BENCH_REQUESTS_PER_MINUTE": args.requests_per_minute,
        "TAU_BENCH_TIMEOUT": args.agent_timeout,
        "TAU_BENCH_USER_TIMEOUT": args.user_timeout,
        "TAU_BENCH_USER_RETRIES": args.user_retries,
        "KAIROS_SEMANTIC_EXPECTATION_ENABLED": "true",
        "KAIROS_SEMANTIC_PROVIDER": "openrouter",
        "KAIROS_SEMANTIC_MODEL": args.semantic_model,
        "KAIROS_SEMANTIC_TEMPERATURE": "0",
        "KAIROS_SEMANTIC_TIMEOUT_S": "20",
        "KAIROS_DIAGNOSTIC_PATTERN_CATALOG": args.diagnostic_catalog,
        "KAIROS_SEMANTIC_TOOL_POLICY_AUDITOR_ENABLED": "false",
        "KAIROS_SEMANTIC_TOOL_POLICY_AUDITOR_BLOCKING": "false",
    }
    if args.model:
        values["TAU_BENCH_MODEL"] = args.model
    if args.user_model:
        values["TAU_BENCH_USER_MODEL"] = args.user_model
    if args.provider:
        values["TAU_BENCH_PROVIDER"] = args.provider
    return values


def mode_env(args: argparse.Namespace, values: dict[str, str]) -> dict[str, str]:
    resolved = dict(values)
    if resolved.get("KAIROS_WORKFLOW_MEMORY_PATHS") == str(DEFAULT_MEMORY_PATH):
        resolved["KAIROS_WORKFLOW_MEMORY_PATHS"] = args.memory_path
    return resolved


def base_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        "-m",
        "tau_harness.run",
        "--env",
        args.env,
        "--task-split",
        args.task_split,
        "--max-concurrency",
        str(args.max_concurrency),
    ]
    if args.task_ids:
        cmd.extend(["--task-ids", *[str(task_id) for task_id in args.task_ids]])
    else:
        cmd.extend(["--start-index", str(args.start_index), "--end-index", str(end_index(args))])
    return cmd


def resolve_checkpoint(stdout: str) -> Path | None:
    matches = RESULT_RE.findall(stdout) or CHECKPOINT_RE.findall(stdout)
    if not matches:
        return None
    path = Path(matches[-1])
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def find_kairos_run(checkpoint: Path | None, started: float) -> Path | None:
    if checkpoint is None:
        return newest_run_after(started)
    candidates: list[Path] = []
    runs_dir = REPO_ROOT / "data" / "runs"
    for manifest_path in runs_dir.glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        ckpt = str((manifest.get("run") or {}).get("ckpt_path") or "")
        if ckpt and Path(ckpt).name == checkpoint.name:
            candidates.append(manifest_path.parent)
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return newest_run_after(started)


def newest_run_after(started: float) -> Path | None:
    runs_dir = REPO_ROOT / "data" / "runs"
    candidates = [p for p in runs_dir.glob("*") if p.is_dir() and p.stat().st_mtime >= started]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def collect_kairos_artifacts(run_dir: Path) -> dict[str, Any]:
    return {
        "manifest": read_json(run_dir / "manifest.json"),
        "summary": read_json(run_dir / "summary.json"),
        "drift_observations": read_jsonl(run_dir / "drift_observations.jsonl"),
        "gate_evaluations": read_jsonl(run_dir / "gate_evaluations.jsonl"),
        "semantic_sessions": {
            path.name: read_json(path) for path in sorted((run_dir / "semantic_sessions").glob("*.json"))
        },
    }


def collect_live_trace_artifacts(started: float) -> dict[str, Any]:
    return {
        "raw": collect_files_modified_since(REPO_ROOT / "data" / "live" / "raw", started),
        "normalized": collect_files_modified_since(REPO_ROOT / "data" / "live" / "normalized", started),
    }


def collect_files_modified_since(directory: Path, started: float) -> dict[str, Any]:
    if not directory.exists():
        return {}
    artifacts: dict[str, Any] = {}
    for path in sorted(directory.glob("*")):
        if not path.is_file() or path.stat().st_mtime < started:
            continue
        if path.suffix == ".json":
            artifacts[path.name] = read_json(path)
        elif path.suffix == ".jsonl":
            artifacts[path.name] = read_jsonl(path)
        else:
            artifacts[path.name] = path.read_text(encoding="utf-8", errors="replace")
    return artifacts


def average_reward(rows: Any) -> float | None:
    if not isinstance(rows, list) or not rows:
        return None
    rewards = [float(row.get("reward", 0.0)) for row in rows if isinstance(row, dict)]
    return sum(rewards) / len(rewards) if rewards else None


def read_json(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"_read_error": str(exc), "_path": str(path)}


def read_jsonl(path: Path) -> list[Any]:
    if not path.exists():
        return []
    rows: list[Any] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            rows.append({"_raw": line})
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
