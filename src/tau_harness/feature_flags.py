"""Host-side feature flags for the tau-agent / kairos integration.

This module is the single source of truth for *tau-agent-specific* gating
decisions — the choices about how this particular harness wires kairos into
its agent loop, distinct from the kairos SDK's own subsystem flags
(``kairos.config.settings``). The split is deliberate: a behaviour that
could ship inside any host's harness gates in ``kairos.config``; a behaviour
that is specifically about how the tau-bench OpenRouter chat loop assembles
messages or attaches domain extensions gates here. The boundary is
documented in CLAUDE.md (Hard rule 1).

Why this module exists at all: prior to this file, the host had a tangle of
scattered ``os.getenv("KAIROS_TAU_INJECT_PLAN")`` and
``os.getenv("KAIROS_TAU_INTERVENTION_ENABLED")`` checks at three different
call sites (``benchmark.py`` for extension attachment, ``openai_agent.py``
for plan injection, and the ablation script for env-var rendering).
Centralising them here means flipping a default lives in one place, and
``CLAUDE.md`` can document the matrix of host-side toggles in one table.

Inputs: environment variables read at process startup. Each accessor is a
fresh ``os.getenv`` so test code and ablation drivers can monkeypatch the
environment between mode runs.

Outputs: three boolean accessors consumed by ``tau_harness.benchmark``
(extension attachment + judge construction) and
``tau_harness.openai_agent`` (system-prompt plan injection).

Flags defined here (defaults match the memory-only project policy):
  - ``memory_loading_enabled()``: on by default; off when
    ``KAIROS_WORKFLOW_MEMORY_PATHS`` is empty/unset/placeholder. Memory
    loading is the one subsystem that defaults *on* in this experiment.
  - ``plan_injection_enabled()``: on by default
    (``KAIROS_TAU_INJECT_PLAN``); splices ``agent_plan.artifact`` into the
    system message at session start.
  - ``tau_intervention_enabled()``: off by default
    (``KAIROS_TAU_INTERVENTION_ENABLED``); attaches the legacy
    ``TauAirlineExtension`` for runtime intervention. Defaulted off after
    the runtime-correction post-mortem.

How this plugs in: ``benchmark.py::_build_kairos_host`` reads
``tau_intervention_enabled`` to decide whether to attach
``TauAirlineExtension`` and reads ``memory_loading_enabled`` (via the
path-resolution helper) to decide whether to construct a workflow memory
store. ``openai_agent.py::OpenAIToolCallingAgent.solve`` reads
``plan_injection_enabled`` to decide whether to prepend the kairos session
plan to the system prompt before the agent loop starts. No other call
site should read these env vars directly — go through the accessors.
"""

from __future__ import annotations

import os

_WORKFLOW_MEMORY_PLACEHOLDERS = frozenset(
    {
        "/absolute/path/to/passed_trajectories.json",
        "absolute/path/to/passed_trajectories.json",
    }
)


def _env_truthy(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def memory_loading_enabled() -> bool:
    """True iff ``KAIROS_WORKFLOW_MEMORY_PATHS`` resolves to a real path list.

    Memory loading is the only host-side subsystem that defaults *on*;
    "default on" here means the loader proceeds whenever a non-empty,
    non-placeholder value is configured. An empty env var disables loading
    silently so baseline runs and CI smoke tests don't need to clear the
    variable explicitly.
    """
    paths_text = os.getenv("KAIROS_WORKFLOW_MEMORY_PATHS")
    if paths_text is None:
        return False
    stripped = paths_text.strip()
    if not stripped or stripped in _WORKFLOW_MEMORY_PLACEHOLDERS:
        return False
    return True


def plan_injection_enabled() -> bool:
    """True iff the kairos session plan should be spliced into the system prompt."""
    return _env_truthy("KAIROS_TAU_INJECT_PLAN", "1")


def cascade_retrieval_enabled() -> bool:
    """True iff the embedding+LLM cascade retriever should wrap the memory store.

    When enabled, ``CascadeMemoryStore`` replaces the lexical scoring in
    ``WorkflowMemoryStore.retrieve`` with stage-1 embedding cosine + stage-3
    LLM rerank. Cost: ~$0.001/session (claude-haiku-4.5 rerank).

    Defaulted off so existing memory_only ablations keep using the lexical
    scorer; flip on for the cascade ablation specifically.
    """
    return _env_truthy("KAIROS_TAU_CASCADE_RETRIEVAL_ENABLED", "0")


def tau_intervention_enabled() -> bool:
    """True iff the legacy ``TauAirlineExtension`` should be attached.

    Defaulted off after the runtime-correction post-mortem (false-positives
    + fail-open behaviour). Kept available for diagnostic comparison runs
    via the ablation bundle's ``kairos_intervention_memory_plan`` mode.
    """
    return _env_truthy("KAIROS_TAU_INTERVENTION_ENABLED", "0")
