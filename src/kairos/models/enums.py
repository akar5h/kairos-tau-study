"""Enumerations for the Kairos IR models."""

from enum import StrEnum


class TerminalStatus(StrEnum):
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"
    HUMAN_ESCALATION = "human_escalation"
    UNKNOWN = "unknown"


class OutputType(StrEnum):
    TEXT = "text"
    FILE = "file"
    API_CALL = "api_call"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class StepStatus(StrEnum):
    OK = "ok"
    ERROR = "error"


class StepType(StrEnum):
    LLM = "llm"
    TOOL_CALL = "tool_call"
    RETRIEVAL = "retrieval"
    AGENT = "agent"
    OTHER = "other"
