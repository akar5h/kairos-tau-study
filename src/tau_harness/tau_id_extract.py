"""Tau-bench specific :class:`~kairos.host.IdExtractor` implementation.

The default :class:`~kairos.host.id_extract.DefaultIdExtractor` recognises
generic snake-cased + UUID + long-numeric IDs. Tau-bench retail and airline
tasks have domain-specific ID shapes (#W-prefixed orders, payment-method
strings like ``credit_card_<digits>``, alphanumeric reservation codes) that
the default doesn't pick up. This subclass adds those patterns.
"""

from __future__ import annotations

import json
import re
from typing import Any

from kairos.host.id_extract import DefaultIdExtractor

_TAU_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"#W\d+"),                          # order_id, e.g. #W2378156
    re.compile(r"\b\d{10}\b"),                     # 10-digit product/item id
    re.compile(r"\b\d{5}\b"),                      # 5-digit zip
    re.compile(r"\b[a-z]+_[a-z]+_\d+\b"),          # user_id, e.g. yusuf_rossi_9620
    re.compile(r"\bcredit_card_\d+\b"),
    re.compile(r"\bgift_card_\d+\b"),
    re.compile(r"\bpaypal_\d+\b"),
    re.compile(r"\bcertificate_\d+\b"),
    re.compile(r"\b[A-Z0-9]{5,8}\b"),              # reservation code, e.g. OBUT9V
)


class TauAirlineIdExtractor(DefaultIdExtractor):
    """ID extractor wired with tau-bench airline / retail patterns."""

    def __init__(self) -> None:
        super().__init__(patterns=_TAU_PATTERNS)

    def from_tool_result(self, observation: Any) -> list[str]:
        # Walk for *_id keys (default behaviour) and also harvest values via
        # the tau patterns from any string leaves.
        base = super().from_tool_result(observation)
        if isinstance(observation, str):
            base.extend(self.from_user_text(observation))
            return list(dict.fromkeys(base))
        try:
            base.extend(self.from_user_text(json.dumps(observation, default=str)))
        except (TypeError, ValueError):
            pass
        return list(dict.fromkeys(base))
