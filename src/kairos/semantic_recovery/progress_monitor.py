"""Hypothesis-driven progress monitor for the active harness (T-04).

This module is the *runtime stall detector* of the active-harness architecture.
Unlike :mod:`kairos.semantic_recovery.breakers` — which catches pre-curated
failure shapes (3-identical-args retry, premature transfer, cancel-after-update)
— the progress monitor asks a fast LLM critic ("is the agent making progress
toward the session's expected terminal action?") on a fixed cadence. It catches
the failure category our deterministic breakers miss: **stalls without errors**.

Why this module exists: code-review of T-03's three landed detectors made
plain that they target the kimi-k2-era pathologies (Phase 2/3 traces).
Phase 4's gpt-4.1 + cascade era surfaces a different failure category — the
agent stops moving toward the goal without producing any error the
deterministic layer can hash on. A confidently-wrong action (task 4) and a
quiet read-spiral (task 33) both look identical to AP-05/07/08: silent.
T-04 trades determinism for coverage: one cheap Haiku call every 3 turns
votes "is_drifting" against the session expectation that kairos already
builds at Turn 0.

Inputs
------
- ``ExpectationLLMClient`` (the same Protocol :mod:`expectation` uses) for
  the strict-JSON drift check. Host supplies this via :class:`KairosHost`
  ``judge`` parameter, same pipeline as the existing semantic verifier.
- ``SessionExpectation.expected_terminal_actions`` from the snapshot kairos
  computes at session start. The hypothesis under test is "did the agent
  move toward one of these in the last 3 turns?"
- Per-session :class:`BreakerState` (shared with deterministic breakers —
  one scratchpad per session, both layers read/write the same fields).
- Per-turn ``(tool_name, observation)`` from
  :meth:`KairosSession.after_tool_result`.

Outputs
-------
- A :class:`Trip` (re-used from breakers.py — same shape so the host's
  inject_correction routing doesn't have to discriminate by source).
  Carries the drift verdict's matched_pattern as the AP id (e.g.
  ``"AP-PROGRESS-stall_no_error"``), a one-line reason summarizing the
  Haiku verdict + last actions, and a JIT correction text steering the
  agent toward a write or transfer.
- A side effect on :class:`BreakerState`: ``progress_monitor_stalled``
  flag flipped True. The next :meth:`KairosSession.before_tool_call`
  consumes this flag and emits the :class:`ToolDecision` (the actual
  intervention is in before_tool_call so the correction lands BEFORE
  the next tool call, not after).

Feature flags
-------------
- ``settings.progress_monitor_enabled`` — top-level on/off (default OFF).
- ``settings.progress_monitor_verbose`` — print per-check decisions for
  Stage-B smoke debugging.
- ``settings.progress_monitor_model`` — Haiku is the recommended default
  (~$0.0001 per check); host can override for cheaper/faster models.

Scope of T-04
-------------
- LLM-based stall check every N turns (default 3).
- Once-per-session: monitor fires at most once per task. If the agent
  ignores the correction, that's a separate problem — re-firing every
  3 turns would just slam the agent with the same nudge.
- Graceful degrade: missing client, malformed JSON, timeout → silent
  pass-through. The monitor MUST NOT crash the agent loop.

What's NOT in scope
-------------------
- ProgressMonitor does NOT track session expectations across turns; it
  trusts the snapshot kairos computed at Turn 0. If that snapshot's
  ``expected_terminal_actions`` is empty (which happens when no judge is
  wired), the monitor returns silently.
- ProgressMonitor does NOT escalate to LLM judgement of WHICH terminal
  to push toward — only of whether progress is happening. T-07's async
  critic handles richer "what should the agent do next" inference.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from kairos.semantic_recovery.expectation import ExpectationLLMClient

from kairos.semantic_recovery.breakers import BreakerState, Trip

_logger = logging.getLogger("kairos.semantic_recovery.progress_monitor")

# JSON-output prompt for the Haiku critic. Kept short — Haiku is reliable on
# strict-JSON when the schema is explicit and the question is binary.
_SYSTEM_PROMPT = """You are a runtime progress monitor for a customer-service agent.

You judge whether the agent is making PROGRESS toward one of its expected
terminal actions, given its most recent tool calls. Output strict JSON.

Rules of the judgement:
- "Progress" = the agent has called (or is about to call) a tool that moves the
  task toward completion. Tools like ``cancel_*``, ``update_*``, ``book_*``,
  ``send_*``, and ``transfer_to_human_agents`` are terminal/progress actions.
- Read tools (``get_*``, ``search_*``, ``list_*``, ``calculate``, ``think``)
  are NOT progress on their own, even though they may be necessary preludes.
- If the last 3 actions are ALL reads/searches AND the session is more than
  half through its turn budget, that's a stall (is_drifting=true).
- If ANY of the last 3 actions is a write or transfer, the agent is
  progressing (is_drifting=false), regardless of turn count.
- If the user's task explicitly requires extensive reads (e.g. "audit all
  reservations before deciding"), allow more read budget — set
  matched_pattern="none" and is_drifting=false unless the reads have
  clearly looped past need.

Output schema (no other text):
{
  "is_drifting": bool,
  "matched_pattern": "read_loop" | "search_spiral" | "stall_no_error" | "off_track" | "none",
  "confidence": float (0.0 to 1.0)
}
"""

_DEFAULT_MIN_TURNS_BETWEEN_CHECKS = 3
_DEFAULT_MODEL = "anthropic/claude-haiku-4.5"
_DEFAULT_TIMEOUT_S = 15.0

_CORRECTION_TEMPLATES = {
    "read_loop": (
        "You have called read-only tools repeatedly without committing to a write. "
        "Decide your next action now: call one of {terminal_actions}, or "
        "transfer_to_human_agents if you genuinely cannot proceed."
    ),
    "search_spiral": (
        "You have searched many flights without committing. Pick the best "
        "available option that satisfies the user's stated constraints, OR "
        "tell the user no option fits and offer transfer."
    ),
    "stall_no_error": (
        "The session has progressed several turns without any progress toward "
        "{terminal_actions}. Reconsider whether the path you're on is correct; "
        "commit to a write or transfer."
    ),
    "off_track": (
        "Your recent actions don't appear to move toward the user's stated "
        "goal. Re-read the user's instruction and commit to a write that "
        "matches one of {terminal_actions}."
    ),
}


def _strip_json_fences(text: str) -> str:
    """Drop a ```json fence if present; some Haiku versions still emit them."""
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return match.group(1).strip() if match else text


def _is_write_or_transfer(tool_name: str) -> bool:
    """Action that counts as progress on its own."""
    if tool_name in {"transfer_to_human_agents", "send_email"}:
        return True
    return any(tool_name.startswith(p) for p in ("book_", "update_", "cancel_", "send_"))


class ProgressMonitor:
    """Async-style stall detector: every N turns, asks Haiku 'is agent stalled?'.

    Owned by :class:`KairosHost`; per-session state lives on the shared
    :class:`BreakerState` (no separate scratchpad). The host calls
    :meth:`check` from inside :meth:`KairosSession.after_tool_result`. A
    trip mutates ``state.progress_monitor_stalled = True``; the next
    :meth:`before_tool_call` then routes the correction to the host
    via ``ToolDecision(action="inject_correction", ...)``.
    """

    def __init__(
        self,
        *,
        client: ExpectationLLMClient | None = None,
        model: str = _DEFAULT_MODEL,
        min_turns_between_checks: int = _DEFAULT_MIN_TURNS_BETWEEN_CHECKS,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self.client = client
        self.model = model
        self.min_turns_between_checks = min_turns_between_checks
        self.timeout_s = timeout_s

    def is_enabled(self) -> bool:
        return self.client is not None

    def check(
        self,
        state: BreakerState,
        *,
        expected_terminal_actions: list[str],
        last_actions: list[str],
        last_observations: list[Any],
        max_steps: int = 20,
        verbose: bool = False,
    ) -> Optional[Trip]:
        """Maybe run the LLM critic; return a Trip if drift is detected.

        Returns None when:
          - client is unset (degraded mode);
          - not enough turns have passed since the last check;
          - expected_terminal_actions is empty (no snapshot to check against);
          - the agent already wrote/transferred in the last 3 actions
            (cheap deterministic skip — no LLM call needed);
          - the LLM call fails or returns malformed JSON;
          - is_drifting==false in the LLM verdict;
          - this session has already tripped the monitor (once-per-session).
        """
        if self.client is None:
            return None
        if "PROGRESS_MONITOR" in state.tripped_aps:
            return None
        if not expected_terminal_actions:
            # No snapshot expectation -> nothing to monitor against.
            return None
        if state.turn - state.last_monitor_check_turn < self.min_turns_between_checks:
            return None
        if any(_is_write_or_transfer(a) for a in last_actions[-3:]):
            # Cheap deterministic skip: agent is already progressing.
            state.last_monitor_check_turn = state.turn
            return None

        state.last_monitor_check_turn = state.turn

        # Build the user prompt with the actual context.
        user_prompt = self._build_user_prompt(
            expected_terminal_actions=expected_terminal_actions,
            last_actions=last_actions,
            last_observations=last_observations,
            current_turn=state.turn,
            max_steps=max_steps,
        )

        try:
            raw = self.client.complete_json(
                system_prompt=_SYSTEM_PROMPT, user_prompt=user_prompt
            )
        except Exception as exc:  # noqa: BLE001 - never crash the agent loop
            _logger.warning("progress_monitor: LLM call failed (%s); silent pass-through", exc)
            return None

        verdict = self._parse_verdict(raw)
        if verdict is None:
            return None

        if verbose:
            _logger.info(
                "progress_monitor: verdict %s at turn %d (last actions: %s)",
                verdict,
                state.turn,
                last_actions[-3:],
            )

        if not verdict.get("is_drifting"):
            return None

        confidence = float(verdict.get("confidence", 0.0))
        if confidence < 0.7:
            # Don't intervene on low-confidence drift verdicts. Tightens the
            # false-positive rate without changing the prompt.
            return None

        matched = verdict.get("matched_pattern", "stall_no_error")
        template = _CORRECTION_TEMPLATES.get(matched) or _CORRECTION_TEMPLATES["stall_no_error"]
        correction = template.format(terminal_actions=", ".join(expected_terminal_actions) or "an appropriate write")

        trip = Trip(
            ap_id=f"AP-PROGRESS-{matched}",
            ap_name=f"Progress monitor: {matched}",
            reason=(
                f"Haiku verdict is_drifting=true conf={confidence:.2f} "
                f"matched={matched}; last actions={last_actions[-3:]}"
            ),
            jit_correction_text=correction,
        )

        # Side-effect: flip the per-session stall flag + record correction
        # so KairosSession.before_tool_call picks it up on the next turn.
        state.progress_monitor_stalled = True
        state.progress_monitor_correction_text = trip.jit_correction_text
        state.progress_monitor_pattern_id = trip.ap_id
        state.tripped_aps.add("PROGRESS_MONITOR")  # once-per-session
        return trip

    def _build_user_prompt(
        self,
        *,
        expected_terminal_actions: list[str],
        last_actions: list[str],
        last_observations: list[Any],
        current_turn: int,
        max_steps: int,
    ) -> str:
        recent = list(zip(last_actions[-3:], last_observations[-3:])) if last_observations else []
        recent_lines = "\n".join(
            f"  turn -{2 - i}: {action} -> {self._truncate(obs)}"
            for i, (action, obs) in enumerate(recent)
        ) or "  (no actions yet)"
        terminals = ", ".join(expected_terminal_actions) or "(none specified)"
        return (
            f"Expected terminal action(s) for this task: {terminals}\n\n"
            f"Recent agent actions and results (oldest first):\n{recent_lines}\n\n"
            f"Current turn: {current_turn} of {max_steps} max\n\n"
            "Is the agent making progress toward one of the expected terminal actions? "
            "Output strict JSON per the schema."
        )

    @staticmethod
    def _truncate(obs: Any, limit: int = 120) -> str:
        text = obs if isinstance(obs, str) else json.dumps(obs, default=str)[:limit]
        text = text.replace("\n", " ").strip()
        return text[:limit] + ("..." if len(text) > limit else "")

    @staticmethod
    def _parse_verdict(raw: str) -> dict | None:
        try:
            payload = json.loads(_strip_json_fences(raw))
        except (json.JSONDecodeError, TypeError):
            _logger.warning("progress_monitor: malformed JSON from critic: %r", raw[:120])
            return None
        if not isinstance(payload, dict):
            return None
        if "is_drifting" not in payload:
            return None
        return payload
