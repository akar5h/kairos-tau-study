"""Host-side gate registry, imported for @register_gate_callable side effects.

No host-specific deterministic gates are currently registered. Phase 1
ungrounded-ID detection lives in kairos.runtime_correction and is wired
in code via build_ungrounded_id_before_write_gates(). Future host-specific
patterns belong here:

    from kairos.intercept import register_gate_callable, SessionContext

    @register_gate_callable("my_pattern_id")
    def my_evaluator(kwargs: dict, ctx: SessionContext) -> bool:
        ...
"""

from __future__ import annotations
