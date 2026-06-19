"""Application configuration via environment variables.

Uses pydantic-settings for type-safe config with .env file support.

Usage:
    from kairos.config import settings
    print(settings.log_level)
"""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class KairosSettings(BaseSettings):
    """Kairos SDK configuration. All values read from env vars prefixed KAIROS_."""

    model_config = SettingsConfigDict(
        env_prefix="KAIROS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "console"

    # Semantic recovery
    semantic_provider: str = "openrouter"
    # Default semantic-recovery judge model. Picked for mid-tier capability +
    # JSON-mode reliability + low cost per call. Hosts override with
    # KAIROS_SEMANTIC_MODEL env var. See docs/host-sdk-design.md for the
    # judge-model selection rationale.
    semantic_model: str = "openai/gpt-4o-mini"
    semantic_temperature: float = 0.0
    semantic_timeout_s: float = 60.0
    semantic_openrouter_api_key: SecretStr | None = None
    semantic_tool_policy_auditor_enabled: bool = False
    semantic_tool_policy_auditor_blocking: bool = False

    # Subsystem enable flags (default off, per host-experiment policy: only
    # workflow memory injection is on by default; every other behaviour is
    # opt-in via env var). Wired into KairosHost.__init__ to gate runtime
    # construction and per-tool-call dispatch.
    #
    # - semantic_recovery_enabled: gates the prewrite LLM verifier
    #   (verify_tool_call) inside KairosSession.before_tool_call. Memory
    #   retrieval and agent-plan rendering still run when the runtime is
    #   built; only the LLM-judge verifier is gated.
    # - drift_detection_enabled: gates DriftDetector construction.
    # - runtime_correction_enabled: reserved for legacy runtime_correction
    #   module; not currently wired into kairos.host but kept in the table
    #   per CLAUDE.md hard rule 1.
    semantic_recovery_enabled: bool = False
    drift_detection_enabled: bool = False
    runtime_correction_enabled: bool = False

    # Active-harness deterministic breakers (Phase 7 / T-03, added 2026-05-20).
    # When enabled, KairosSession.before_tool_call / after_tool_result run a
    # set of hash + counter + regex detectors against the per-session
    # BreakerState. Tripping a breaker LOGS a Trip event in T-03; the
    # correction-injection wiring lands in T-05. Detector specs come from
    # the path below (the JSON DB shipped by host).
    deterministic_breakers_enabled: bool = False
    deterministic_breakers_verbose: bool = False
    anti_patterns_path: str | None = None  # host points kairos at the DB file

    # Active-harness progress monitor (Phase 7 / T-04, added 2026-05-20).
    # When enabled, KairosSession.after_tool_result invokes an LLM-based
    # "is the agent making progress" check every N turns. Trip flips a flag
    # on BreakerState; KairosSession.before_tool_call converts the flag to
    # an inject_correction ToolDecision on the next turn. Requires a judge
    # client (passed via KairosHost(judge=...)); degrades silently if none.
    progress_monitor_enabled: bool = False
    progress_monitor_verbose: bool = False
    progress_monitor_model: str = "anthropic/claude-haiku-4.5"
    progress_monitor_min_turns_between_checks: int = 3

    # Offline diagnostic calibration
    diagnostic_pattern_catalog: str | None = None


settings = KairosSettings()
