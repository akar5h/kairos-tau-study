"""Default :class:`IdExtractor` with pluggable regex patterns.

Hosts with domain-specific IDs (e.g. tau-bench's ``#W\\d+``,
``credit_card_\\d+``) override the patterns list, or supply a fully custom
``IdExtractor`` to :func:`kairos.host.host` directly.
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["DefaultIdExtractor"]


_DEFAULT_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Snake-cased identifiers with trailing numeric suffix (user_42, order_abc_99).
    re.compile(r"\b[a-z][a-z0-9_]*_\d+\b"),
    # UUID-shaped.
    re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        re.IGNORECASE,
    ),
    # Plain 8-12 digit numeric IDs.
    re.compile(r"\b\d{8,12}\b"),
)


class DefaultIdExtractor:
    """Generic regex-based ID extractor.

    Pass a custom ``patterns`` tuple to override.
    """

    def __init__(self, *, patterns: tuple[re.Pattern[str], ...] | None = None) -> None:
        self._patterns: tuple[re.Pattern[str], ...] = patterns or _DEFAULT_PATTERNS

    def from_user_text(self, text: str | None) -> list[str]:
        if not text:
            return []
        out: list[str] = []
        for pattern in self._patterns:
            out.extend(pattern.findall(text))
        return list(dict.fromkeys(out))

    def from_tool_result(self, observation: Any) -> list[str]:
        """Walk a tool result for ``*_id`` keys; fall back to regex on strings."""
        if isinstance(observation, str):
            try:
                data: Any = json.loads(observation)
            except (ValueError, TypeError):
                return self.from_user_text(observation)
        else:
            data = observation

        out: list[str] = []

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key.endswith("_id") and isinstance(value, str):
                        out.append(value)
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(data)
        return list(dict.fromkeys(out))
