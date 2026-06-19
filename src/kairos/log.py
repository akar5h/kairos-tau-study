"""Structured logging for Kairos SDK.

Uses structlog wrapping stdlib logging so consumers can configure
logging however they want while we get structured JSON output.

Usage:
    from kairos.log import get_logger
    logger = get_logger(__name__)
    logger.info("lead.dispatched", lead_id=lead_id, batch_size=10)
"""

import logging
import sys

import structlog


def setup_logging(*, level: str = "INFO", json_output: bool = True) -> None:
    """Configure structured logging for the entire application.

    Call once at startup. Safe to call multiple times (idempotent).
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger bound to the given module name."""
    return structlog.get_logger(name)
