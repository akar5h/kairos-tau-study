"""Pydantic IR models: Step, TraceEnvelope, NormalizationReport."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from kairos.models.enums import (  # noqa: TCH001
    OutputType,
    StepStatus,
    StepType,
    TerminalStatus,
)


class Step(BaseModel):
    """Single step in an agent execution trace."""

    step_index: int
    step_type: StepType
    agent_name: str | None = None
    node_name: str | None = None

    # Tool call fields
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_args_normalized: dict | None = None
    tool_output: str | None = None

    # LLM fields
    llm_input: str | None = None
    llm_output: str | None = None
    llm_model: str | None = None

    # Retrieval fields
    retrieval_query: str | None = None
    retrieval_chunks: list[str] | None = None

    # Metrics
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: int | None = None

    # Status
    status: StepStatus = StepStatus.OK
    error_message: str | None = None

    # Hierarchy
    parent_step_index: int | None = None

    # Timestamps
    started_at: datetime | None = None
    ended_at: datetime | None = None

    # Provenance
    source_observation_id: str | None = None


class TraceEnvelope(BaseModel):
    """Normalized representation of one agent execution trace."""

    # Identity
    trace_id: str
    source: str = "langfuse"
    source_trace_id: str | None = None

    # Intent
    user_input: str | None = None
    system_prompt: str | None = None
    agent_type: str | None = None

    # Execution
    steps: list[Step] = Field(default_factory=list)

    # Aggregated metrics
    total_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: int = 0
    step_count: int = 0

    # Terminal state
    terminal_status: TerminalStatus = TerminalStatus.UNKNOWN
    output_type: OutputType = OutputType.UNKNOWN

    # Derived fields (computed in model_post_init)
    tool_sequence: list[str] = Field(default_factory=list)
    tool_bigrams: list[tuple[str, str]] = Field(default_factory=list)
    unique_tool_count: int = 0
    error_count: int = 0
    has_retrieval: bool = False
    retrieval_step_count: int = 0

    # Metadata
    session_id: str | None = None
    user_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict | None = None

    # Timestamps
    started_at: datetime | None = None
    ended_at: datetime | None = None
    normalized_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC),
    )

    # Provenance
    source_metadata: dict | None = None

    # Validation
    is_valid: bool = True
    validation_warnings: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        """Compute derived fields from steps."""
        self.step_count = len(self.steps)

        self.tool_sequence = [
            s.tool_name for s in self.steps if s.step_type == StepType.TOOL_CALL and s.tool_name is not None
        ]

        self.tool_bigrams = [
            (self.tool_sequence[i], self.tool_sequence[i + 1]) for i in range(len(self.tool_sequence) - 1)
        ]

        self.unique_tool_count = len(set(self.tool_sequence))

        self.error_count = sum(1 for s in self.steps if s.status == StepStatus.ERROR)

        retrieval_steps = [s for s in self.steps if s.step_type == StepType.RETRIEVAL]
        self.has_retrieval = len(retrieval_steps) > 0
        self.retrieval_step_count = len(retrieval_steps)

        if self.user_input is None:
            self.validation_warnings.append("Missing user_input: trace cannot be clustered by intent")


class NormalizationReport(BaseModel):
    """Summary of a normalization batch run."""

    total_traces_ingested: int = 0
    total_traces_normalized: int = 0
    total_traces_failed: int = 0
    traces_missing_user_input: int = 0
    traces_missing_system_prompt: int = 0
    traces_missing_tool_calls: int = 0
    traces_with_errors: int = 0
    avg_steps_per_trace: float = 0.0
    avg_tokens_per_trace: float = 0.0
    errors: list[dict] = Field(default_factory=list)
