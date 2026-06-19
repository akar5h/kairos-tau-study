"""Kairos — AI Agent Diagnostics SDK."""

from kairos.config import settings
from kairos.log import setup_logging

__version__ = "0.1.0"

# Initialize structured logging on import
setup_logging(
    level=settings.log_level,
    json_output=settings.log_format == "json",
)
