"""Run orchestration for tau-bench evaluations with optional kairos wiring.

This module owns the *outer loop* of a tau-bench run: it parses a
``RunConfig``, resolves the environment, instantiates the agent, optionally
builds a :class:`kairos.host.KairosHost`, and dispatches tasks through a
thread pool while streaming results to a checkpoint JSON. It is the only
place in the host where the kairos SDK is constructed; the agent loop in
``openai_agent.py`` only consumes the already-built host handle.

Why this module exists rather than letting ``run.py`` drive things
directly: tau-bench's upstream ``run`` helper hard-codes its provider
selection and doesn't know about kairos; we wrap it with a parallel entry
point that adds (a) the OpenTelemetry span hierarchy used by Phoenix /
ARMS for live trace inspection, (b) per-task checkpoint streaming so a
crashed run doesn't lose all results, and (c) kairos host lifecycle
management (build at start, finalize on exit, write summary.json).

Inputs: a :class:`RunConfig` from ``tau_harness.run``, environment
variables read via ``tau_harness.feature_flags`` (host-side toggles)
and ``kairos.config.settings`` (kairos-side toggles), and the tau-bench
``Env`` returned by ``get_env``.

Outputs: a list of :class:`EnvRunResult` plus a checkpoint JSON written to
``config.log_dir`` and — when kairos is enabled — a per-run directory
under ``data/runs/`` containing ``manifest.json``, ``summary.json``,
``drift_observations.jsonl``, ``gate_evaluations.jsonl``, and
``semantic_sessions/*.json``.

Feature flags consulted here:
  - ``tau_harness.feature_flags.tau_intervention_enabled`` — whether to
    attach :class:`TauAirlineExtension`.
  - ``tau_harness.feature_flags.memory_loading_enabled`` — whether to
    construct a :class:`WorkflowMemoryStore` from the configured paths.
  - ``KAIROS_SEMANTIC_EXPECTATION_ENABLED`` (read inside kairos) — whether
    an OpenRouter judge client is built at all; when off the kairos host
    runs in memory-only mode with no LLM calls.

How this plugs in: invoked by ``tau_harness.run`` (the CLI entry) and
by ``scripts/run_kairos_ablation_bundle.py`` (the ablation driver) which
sets the env-var matrix before each subprocess. Downstream, the host
handle constructed here is passed to ``OpenAIToolCallingAgent`` so the
agent loop can open a :class:`KairosSession` per task.
"""

import contextlib
import json
import multiprocessing
import os
import random
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import kairos.host
from kairos.config import settings as kairos_settings
from kairos.semantic_recovery import (
    OpenRouterExpectationClient,
    build_policy_pack,
    load_success_path_memory_store,
)
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from tau_bench.envs import get_env
from tau_bench.run import display_metrics
from tau_bench.types import EnvRunResult, RESPOND_ACTION_NAME, RunConfig

from tau_harness import feature_flags as flags
from tau_harness.kairos_setup import install_kairos, shutdown_kairos
from tau_harness.openai_agent import OpenAIChatReActAgent, OpenAIToolCallingAgent
from tau_harness.openai_compat import request_settings
from tau_harness.openai_user import install_user_patch
from tau_harness.tau_airline_extension import TauAirlineExtension
from tau_harness.tau_id_extract import TauAirlineIdExtractor


def _load_configured_memory_store(
    *, toolset_hash: str | None = None, prompt_hash: str | None = None
) -> Any:
    if not flags.memory_loading_enabled():
        return None
    paths_text = os.getenv("KAIROS_WORKFLOW_MEMORY_PATHS", "")
    store = load_success_path_memory_store(
        paths_text, toolset_hash=toolset_hash, prompt_hash=prompt_hash
    )
    # Optionally wrap with cascade retriever (embedding cosine + LLM rerank).
    # Rerank provider is set by KAIROS_RERANKER_PROVIDER (default openrouter →
    # claude-haiku; set to "azure" to route through an Azure deployment).
    if store is not None and flags.cascade_retrieval_enabled():
        from tau_harness.cascade_retriever import CascadeMemoryStore
        try:
            store = CascadeMemoryStore(underlying=store)
            print(
                f"[cascade] CascadeMemoryStore active — stage 1 MiniLM + stage 3 rerank "
                f"({store._rerank_model} via {store._rerank_provider})"  # noqa: SLF001
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[cascade] failed to enable cascade retriever: {exc}; falling back to lexical")
    return store


def _build_kairos_host(
    *,
    env: Any,
    config: RunConfig,
    ckpt_path: str,
) -> kairos.host.KairosHost:
    """Build the per-run :class:`~kairos.host.KairosHost` and register the tau extension.

    The judge is only constructed when at least one LLM-using kairos subsystem
    is enabled (semantic recovery verifier, drift detection, or the legacy
    intervention extension). For the memory-only path the judge stays ``None``
    so the kairos host runs entirely without OpenRouter calls — the SDK now
    builds :class:`SemanticRecoveryRuntime` from the memory store alone in
    that case.
    """
    needs_judge = (
        flags.tau_intervention_enabled()
        or kairos_settings.semantic_recovery_enabled
        or kairos_settings.drift_detection_enabled
        or kairos_settings.progress_monitor_enabled  # T-04 (2026-05-20)
    )
    judge = OpenRouterExpectationClient.from_env() if needs_judge else None
    policy_pack = build_policy_pack(
        source_name="tau-airline",
        system_prompt=env.wiki,
        tool_descriptions=env.tools_info,
    )
    memory_store = _load_configured_memory_store(
        toolset_hash=policy_pack.provenance.get("toolset_hash"),
        prompt_hash=policy_pack.provenance.get("prompt_hash"),
    )
    host = kairos.host.host(
        gates_config="kairos_gates.json",
        id_extractor=TauAirlineIdExtractor(),
        judge=judge,
        policy_pack=policy_pack,
        memory_store=memory_store,
        tool_schemas=env.tools_info,
        manifest={
            "model": config.model,
            "user_model": config.user_model,
            "model_provider": config.model_provider,
            "user_model_provider": config.user_model_provider,
            "env": config.env,
            "agent_strategy": config.agent_strategy,
            "user_strategy": config.user_strategy,
            "task_split": config.task_split,
            "start_index": config.start_index,
            "end_index": config.end_index,
            "task_ids": config.task_ids,
            "num_trials": config.num_trials,
            "temperature": config.temperature,
            "ckpt_path": ckpt_path,
        },
    )
    # Wire the tau-airline intervention extension only when explicitly enabled
    # (see tau_harness.feature_flags.tau_intervention_enabled). Defaulted
    # off after the runtime-correction post-mortem; the legacy extension stays
    # available for diagnostic comparison runs.
    if flags.tau_intervention_enabled():
        extension = TauAirlineExtension(
            interceptor=host.interceptor,
            semantic_runtime=host.semantic_runtime,
        )
        host._domain_extensions.append(extension)
    return host


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except TypeError:
        return str(value)


def _last_agent_reply(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message["content"])
    return None


class TracedEnv:
    def __init__(self, env: Any, tracer: Any) -> None:
        self._env = env
        self._tracer = tracer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def step(self, action: Any) -> Any:
        if action.name == RESPOND_ACTION_NAME or action.name not in self._env.tools_map:
            return self._env.step(action)
        with self._tracer.start_as_current_span(
            f"tool.{action.name}",
            attributes={
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": action.name,
                "gen_ai.tool.call.arguments": _json_text(action.kwargs),
            },
        ) as span:
            result = self._env.step(action)
            span.set_attribute("gen_ai.tool.call.result", _json_text(result.observation))
            if isinstance(result.observation, str) and result.observation.startswith("Error:"):
                span.set_status(Status(StatusCode.ERROR, result.observation))
            return result


def run_benchmark(
    config: RunConfig,
    *,
    enable_kairos: bool = False,
    sleep_between_tasks_s: float = 0.0,
) -> list[EnvRunResult]:
    install_user_patch()
    random.seed(config.seed)
    time_str = datetime.now().strftime("%m%d%H%M%S")
    ckpt_path = (
        f"{config.log_dir}/{config.agent_strategy}-{config.model.split('/')[-1]}-"
        f"{config.temperature}_range_{config.start_index}-{config.end_index}_"
        f"user-{config.user_model.split('/')[-1]}-{config.user_strategy}_{time_str}.json"
    )
    os.makedirs(config.log_dir, exist_ok=True)

    print(f"Loading user with strategy: {config.user_strategy}")
    print(f"Agent model/provider: {config.model} via {config.model_provider}")
    print(f"User model/provider: {config.user_model} via {config.user_model_provider}")
    print(f"Agent request settings: {request_settings('TAU_BENCH_', config.temperature)}")
    print(f"User request settings: {request_settings('TAU_BENCH_USER_')}")
    env = get_env(
        config.env,
        user_strategy="human",
        user_model=config.user_model,
        user_provider=config.user_model_provider,
        task_split=config.task_split,
    )

    # Build the kairos host (or skip when disabled). The 3-line SDK touch.
    host = None
    if enable_kairos:
        install_kairos()
        host = _build_kairos_host(env=env, config=config, ckpt_path=ckpt_path)
    tracer = trace.get_tracer(__name__) if enable_kairos else None

    if config.agent_strategy == "tool-calling":
        agent = OpenAIToolCallingAgent(
            tools_info=env.tools_info,
            wiki=env.wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
            host=host,
        )
    elif config.agent_strategy == "act":
        agent = OpenAIChatReActAgent(
            tools_info=env.tools_info,
            wiki=env.wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=False,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "react":
        agent = OpenAIChatReActAgent(
            tools_info=env.tools_info,
            wiki=env.wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=True,
            temperature=config.temperature,
        )
    else:
        raise ValueError("few-shot is not implemented in the direct OpenAI runtime")
    end_index = len(env.tasks) if config.end_index == -1 else min(config.end_index, len(env.tasks))
    results: list[EnvRunResult] = []
    lock = multiprocessing.Lock()

    if config.task_ids:
        print(f"Running tasks {config.task_ids} (checkpoint path: {ckpt_path})")
    else:
        print(f"Running tasks {config.start_index} to {end_index} (checkpoint path: {ckpt_path})")

    try:
        for trial in range(config.num_trials):
            idxs = config.task_ids or list(range(config.start_index, end_index))
            if config.shuffle:
                random.shuffle(idxs)

            def _run(idx: int) -> EnvRunResult:
                isolated_env = get_env(
                    config.env,
                    user_strategy=config.user_strategy,
                    user_model=config.user_model,
                    task_split=config.task_split,
                    user_provider=config.user_model_provider,
                    task_index=idx,
                )
                current_env = TracedEnv(isolated_env, tracer) if tracer is not None else isolated_env
                task = isolated_env.tasks[idx]
                print(f"Running task {idx}")
                try:
                    max_num_steps = int(os.getenv("TAU_BENCH_MAX_STEPS", "30"))
                    if tracer is None:
                        res = agent.solve(env=current_env, task_index=idx, max_num_steps=max_num_steps, session_id=f"task-{idx}-trial-{trial}")
                    else:
                        with tracer.start_as_current_span(
                            "kairos.task",
                            attributes={
                                "kairos.agent.name": "tau_harness",
                                "kairos.business_op": f"tau_{config.env}",
                                "kairos.user_input": task.instruction,
                                "kairos.output_type": "text",
                                "kairos.metadata.task_id": idx,
                                "kairos.metadata.trial": trial,
                                "kairos.metadata.task_split": config.task_split,
                                "kairos.metadata.agent_strategy": config.agent_strategy,
                                "kairos.metadata.model": config.model,
                                "kairos.metadata.user_model": config.user_model,
                                "openinference.span.kind": "AGENT",
                                "input.value": task.instruction,
                                "input.mime_type": "text/plain",
                                "tag.tags": [f"task-{idx}", f"trial-{trial}", config.env],
                            },
                        ) as span:
                            try:
                                res = agent.solve(env=current_env, task_index=idx, max_num_steps=max_num_steps, session_id=f"task-{idx}-trial-{trial}")
                            except Exception as exc:
                                span.record_exception(exc)
                                span.set_status(Status(StatusCode.ERROR, str(exc)))
                                span.set_attribute(
                                    "tag.tags",
                                    [f"task-{idx}", f"trial-{trial}", config.env, "errored"],
                                )
                                raise
                            final_output = _last_agent_reply(res.messages)
                            if final_output is not None:
                                span.set_attribute("kairos.final_output", final_output)
                                span.set_attribute("output.value", final_output)
                                span.set_attribute("output.mime_type", "text/plain")
                            span.set_attribute("kairos.metadata.reward", res.reward)
                            span.set_attribute("kairos.terminal_status", "completed")
                            if res.total_cost is not None:
                                span.set_attribute("kairos.metadata.agent_cost", res.total_cost)
                            user_cost = res.info.get("user_cost")
                            if user_cost is not None:
                                span.set_attribute("kairos.metadata.user_cost", user_cost)
                            outcome = "passed" if res.reward == 1 else "failed"
                            span.set_attribute(
                                "tag.tags",
                                [f"task-{idx}", f"trial-{trial}", config.env, outcome],
                            )
                            span.set_status(Status(StatusCode.OK))
                    result = EnvRunResult(
                        task_id=idx,
                        reward=res.reward,
                        info=res.info,
                        traj=res.messages,
                        trial=trial,
                    )
                except Exception as exc:
                    result = EnvRunResult(
                        task_id=idx,
                        reward=0.0,
                        info={"error": str(exc), "traceback": traceback.format_exc()},
                        traj=[],
                        trial=trial,
                    )
                print("✅" if result.reward == 1 else "❌", f"task_id={idx}", result.info)
                print("-----")
                with lock:
                    data = []
                    if os.path.exists(ckpt_path):
                        with open(ckpt_path, "r", encoding="utf-8") as handle:
                            data = json.load(handle)
                    with open(ckpt_path, "w", encoding="utf-8") as handle:
                        json.dump(data + [result.model_dump()], handle, indent=2)
                if sleep_between_tasks_s > 0:
                    print(f"Sleeping {sleep_between_tasks_s}s before the next task")
                    time.sleep(sleep_between_tasks_s)
                return result

            if config.max_concurrency <= 1:
                for idx in idxs:
                    results.append(_run(idx))
            else:
                with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
                    results.extend(executor.map(_run, idxs))
    finally:
        if host is not None:
            host.finalize(task_results=[r.model_dump() for r in results])
        if enable_kairos:
            shutdown_kairos()

    display_metrics(results)
    with open(ckpt_path, "w", encoding="utf-8") as handle:
        json.dump([result.model_dump() for result in results], handle, indent=2)
        print(f"\n📄 Results saved to {ckpt_path}\n")
    return results
