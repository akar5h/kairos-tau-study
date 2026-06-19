"""Deterministic breakers for the active harness.

This module is the *runtime trap layer* of the active-harness architecture.
It owns no LLM calls and no embedding lookups; every check is a hash, a
counter, a regex, or an integer threshold against per-session state. The
design goal is **zero false-positive risk on borderline cases** — anything
fuzzy belongs to the async LLM critic in :mod:`kairos.semantic_recovery.critic`
(T-07), not here.

Why this module exists: Phase 1's LLM-judge runtime correction fail-opened on
borderline cases and was disabled (`KAIROS_RUNTIME_CORRECTION_ENABLED=False`).
Deterministic detection has a different failure-mode shape — it can MISS
real bad patterns but it cannot HALLUCINATE bad patterns. That asymmetry is
what makes it safe to ship into the agent loop's hot path.

Inputs
------
- ``data/anti_patterns.json`` (machine-loadable AP database produced by T-01).
  Each entry's ``detection.spec`` field carries the concrete params this
  module consumes: tool-name globs, counter thresholds, regex strings.
- Per-session ``BreakerState`` (per-task scratchpad — tool history, rolling
  hashes, mutation log, parsed user-instruction attributes).
- Per-call ``(tool_name, kwargs, observation)`` from the
  :class:`~kairos.host.KairosSession` hooks.

Outputs
-------
- ``Trip`` events. A trip carries the AP id, a human-readable reason, and the
  ``jit_correction_text`` ready to be passed back to the host as the
  ``ToolDecision.correction_artifact``. T-03 logs trips but does not yet
  surface them as inject_correction actions; T-05 makes that wiring live.

Feature flags
-------------
- ``settings.deterministic_breakers_enabled`` — top-level on/off.
- ``settings.breakers_verbose`` — emit per-call breaker decisions to stdout
  for Stage-B smoke debugging.

Both default OFF per CLAUDE.md hard rule 1.

How it plugs in
---------------
:class:`KairosSession.before_tool_call` and :class:`KairosSession.after_tool_result`
each call into :func:`DeterministicBreakers.check_before` /
:func:`DeterministicBreakers.check_after`. The aggregator iterates per-AP
:class:`Breaker` instances and short-circuits on the first trip — only one
trip per call site, no compound stacking. T-07's async critic operates on a
separate channel and does not compete with this module for the inject slot.

Scope of T-03 — what THIS commit ships (post Stage-A learnings, 2026-05-20)
--------------------------------------------------------------------------
Three tool-call-time deterministic breakers actually instantiated at runtime:
    AP-05 cycle-without-param-change       (before_tool_call hash window)
    AP-07 premature transfer_to_human      (before_tool_call counter + parse)
    AP-08 mutate-then-undo                 (before_tool_call id-keyed log)

Stage-A replay exposed three originally-planned APs as requiring end-of-
session reasoning or semantic comparison; they now ship_in=T-07 in the JSON
DB but their constructor classes (ReadWithoutWriteBreaker,
ToolSequenceVerbatimBreaker, SearchExplodeBreaker) remain in the registry
for when T-07 wires the LLM-critic path:
    AP-01 read-without-write loop          — false-positives mid-prelude
    AP-03 tool-sequence-verbatim           — needs flight-number diff
    AP-10 search-explode-no-decision       — needs end-of-session view

Deferred to T-05 (needs an assistant-content hook that doesn't yet exist):
    AP-02 quote-warning-as-policy-fact     (substring vs snapshotted plan)
    AP-11 arithmetic-drift                 (regex on assistant content)
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

_logger = logging.getLogger("kairos.semantic_recovery.breakers")


@dataclass(frozen=True)
class Trip:
    """A breaker trip event.

    Returned by every :class:`Breaker` ``check_*`` method when the AP shape
    matches. The host turns this into a ``ToolDecision.inject_correction``
    (T-05) or, for now (T-03), records it as telemetry only.
    """

    ap_id: str
    ap_name: str
    reason: str
    jit_correction_text: str


@dataclass
class BreakerState:
    """Per-session breaker state.

    A new instance is created when ``KairosSession`` opens. The session feeds
    every tool call and observation into the state via the breaker check
    methods, and the state mutates accordingly.
    """

    max_steps: int = 20
    user_instruction: str = ""
    allowed_transfer_tries: Optional[int] = None

    turn: int = 0
    reads_count: int = 0
    writes_count: int = 0
    search_count: int = 0
    transfer_count: int = 0

    recent_canonical_hashes: list[str] = field(default_factory=list)
    last_observation_had_error: bool = False
    # Code-review fix MED-2 (2026-05-20): was dict[str, str] (overwrote prior
    # update tool name on the same id). Now records each update event so the
    # mutate-then-undo trip reason can name the FIRST prior update accurately.
    update_log: dict[str, list[tuple[int, str]]] = field(default_factory=dict)
    tool_history: list[str] = field(default_factory=list)
    tripped_aps: set[str] = field(default_factory=set)  # once-per-session APs that fired

    # T-04 additions (2026-05-20): per-session state for the progress monitor.
    # The monitor lives in progress_monitor.py but writes here so before_tool_call
    # can convert a stall verdict into ToolDecision(inject_correction) without
    # going back through the monitor. Decoupling state from logic keeps the
    # before_tool_call hot path free of LLM calls.
    progress_monitor_stalled: bool = False
    progress_monitor_correction_text: str = ""
    progress_monitor_pattern_id: str = ""
    last_monitor_check_turn: int = 0
    # Per-turn observation log for the monitor's "last 3 observations" lookup.
    # Capped via the aggregator to bound memory.
    recent_observations: list[Any] = field(default_factory=list)

    # T-05 additions (2026-05-20): assistant-content channel for AP-02.
    # ``injected_plan_absolute_claims`` is snapshotted ONCE at session init
    # from the rendered plan artifact — the substring AP-02 will search
    # for in each assistant turn. ``recent_assistant_contents`` accumulates
    # the agent's recent assistant messages (both respond and tool_call
    # turns) so AP-02 can catch a smoking-gun quote even if it happened
    # in a respond message N turns before the harmful tool call.
    injected_plan_absolute_claims: list[str] = field(default_factory=list)
    recent_assistant_contents: list[str] = field(default_factory=list)

    def record_assistant_message(self, content: str) -> None:
        """Append non-empty + non-duplicate assistant content to the rolling
        AP-02 corpus, capped at 10 most-recent items.
        """
        content = (content or "").strip()
        if not content:
            return
        if self.recent_assistant_contents and self.recent_assistant_contents[-1] == content:
            return  # exact-dup safeguard (same content seen twice in a row)
        self.recent_assistant_contents.append(content)
        if len(self.recent_assistant_contents) > 10:
            del self.recent_assistant_contents[:-10]

    def snapshot_plan_absolute_claims(self, plan_artifact: str) -> None:
        """Pull caution/constraint lines out of the rendered plan for AP-02.

        Handles both the legacy ``advisory_v2`` labels and the post-Fix-A-v3
        relabel (advisory_v3_surgical_relabel, 2026-05-20). The new labels
        are designed to discourage the agent from quoting them as policy
        in the first place, but if the agent still does, AP-02 catches
        the residual. Belt-and-suspenders.
        """
        if not plan_artifact:
            return
        # Each (prefix, claim-extractor-fn) pair maps a label-shape to the
        # substring to scan against assistant content.
        legacy_prefixes = ("watch for:", "preserve from current evidence:")
        relabel_prefixes = (
            "prior trace had this caution (verify against current task before acting):",
            "memory says this pattern requires (verify in current session):",
        )
        all_prefixes = legacy_prefixes + relabel_prefixes
        for raw in plan_artifact.splitlines():
            line = raw.strip()
            if line.startswith("- "):
                line = line[2:]
            lc = line.lower()
            for prefix in all_prefixes:
                if lc.startswith(prefix):
                    claim = line[len(prefix):].strip()
                    if claim and claim not in self.injected_plan_absolute_claims:
                        self.injected_plan_absolute_claims.append(claim)
                    break

    def record_before_tool(self, tool_name: str, kwargs: dict[str, Any]) -> None:
        """Update per-tool-call state that BOTH subsystems read.

        Called by :meth:`KairosSession.before_tool_call` BEFORE either the
        deterministic-breakers aggregator or the progress monitor dispatches.
        Lifts state mutation out of either subsystem so toggling one off
        doesn't strand the other.
        """
        self.turn += 1
        self.recent_canonical_hashes.append(_canonical_tool_hash(tool_name, kwargs))
        # Keep window bounded — only the last 3 entries are read.
        if len(self.recent_canonical_hashes) > 16:
            del self.recent_canonical_hashes[:-16]

    def record_after_tool(self, tool_name: str, observation: Any, had_error: bool) -> None:
        """Update per-turn state that BOTH subsystems read.

        Called by :meth:`KairosSession.after_tool_result` BEFORE either the
        deterministic-breakers aggregator or the progress monitor dispatches.
        """
        self.tool_history.append(tool_name)
        self.last_observation_had_error = had_error
        self.recent_observations.append(observation)
        if len(self.recent_observations) > 6:
            del self.recent_observations[:-6]

    def parse_instruction(self, instruction: str) -> None:
        """Extract structured signals from the task instruction at session start.

        Currently extracts: the patience-threshold integer for AP-07
        (``"five times"`` / ``"5 tries"`` / etc., digit or word form).
        """
        self.user_instruction = instruction or ""
        text = self.user_instruction.lower()
        # Word-form numbers (one..ten covers every observed case in our corpus).
        word_to_int = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        }
        match = re.search(r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(times|tries|attempts)", text)
        if match:
            raw = match.group(1)
            # Code-review fix CQ-1 (2026-05-20): explicit fallback rather than
            # ``dict.get() or int()``. The ``or`` chain would mis-handle a
            # word_to_int entry that maps to 0 (e.g. if someone adds "zero": 0
            # later) by falling through to ``int("zero")`` and raising.
            if raw in word_to_int:
                self.allowed_transfer_tries = word_to_int[raw]
            else:
                self.allowed_transfer_tries = int(raw)


def _canonical_tool_hash(tool_name: str, kwargs: dict[str, Any]) -> str:
    """Hash a tool call in a way that's stable across whitespace + key order.

    Used by AP-05 to detect "identical args, identical tool, retried".
    """
    payload = json.dumps([tool_name, kwargs], sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _matches_any_glob(name: str, globs: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, g) for g in globs)


class Breaker(Protocol):
    """Interface every breaker implements.

    Implementations carry their own AP id + name + jit_correction_text loaded
    from ``data/anti_patterns.json``. The check methods receive the shared
    per-session state and the current tool call.
    """

    ap_id: str
    ap_name: str
    jit_correction_text: str

    def check_before(
        self, state: BreakerState, tool_name: str, kwargs: dict[str, Any]
    ) -> Optional[Trip]: ...

    def check_after(
        self,
        state: BreakerState,
        tool_name: str,
        kwargs: dict[str, Any],
        observation: Any,
        had_error: bool,
    ) -> Optional[Trip]: ...


# ---------------------------------------------------------------------------
# Concrete breakers — one class per AP. Each takes its detection.spec from
# the JSON DB at construction time, so swapping params is a JSON edit, not a
# code edit.
# ---------------------------------------------------------------------------


class _BreakerBase:
    """Default no-op implementations so subclasses only override what they use."""

    ap_id: str = "AP-XX"
    ap_name: str = ""
    jit_correction_text: str = ""

    def check_before(
        self, state: BreakerState, tool_name: str, kwargs: dict[str, Any]
    ) -> Optional[Trip]:
        return None

    def check_after(
        self,
        state: BreakerState,
        tool_name: str,
        kwargs: dict[str, Any],
        observation: Any,
        had_error: bool,
    ) -> Optional[Trip]:
        return None

    def _trip(self, reason: str) -> Trip:
        return Trip(
            ap_id=self.ap_id,
            ap_name=self.ap_name,
            reason=reason,
            jit_correction_text=self.jit_correction_text,
        )


class ReadWithoutWriteBreaker(_BreakerBase):
    """AP-01 — reads pile up without any write or transfer."""

    ap_id = "AP-01"
    ap_name = "Read-without-write loop"

    def __init__(self, spec: dict[str, Any], jit_text: str) -> None:
        self.read_globs: list[str] = spec["read_tools"]
        self.write_globs: list[str] = spec["write_tools"]
        self.jit_correction_text = jit_text
        # Trip when reads >= 4 AND writes == 0 AND turn >= 0.5 * max_steps.
        # Parsed loosely from the spec's trip_condition string; we don't
        # need a full expression evaluator for this.
        self._reads_threshold = 4
        self._turn_pct_threshold = 0.5

    def check_after(
        self,
        state: BreakerState,
        tool_name: str,
        kwargs: dict[str, Any],
        observation: Any,
        had_error: bool,
    ) -> Optional[Trip]:
        if _matches_any_glob(tool_name, self.read_globs):
            state.reads_count += 1
        if _matches_any_glob(tool_name, self.write_globs):
            state.writes_count += 1
        if (
            state.reads_count >= self._reads_threshold
            and state.writes_count == 0
            and state.turn >= self._turn_pct_threshold * state.max_steps
        ):
            return self._trip(
                f"reads={state.reads_count} writes=0 turn={state.turn}/{state.max_steps}"
            )
        return None


class ToolSequenceVerbatimBreaker(_BreakerBase):
    """AP-03 — write fires without a prerequisite read."""

    ap_id = "AP-03"
    ap_name = "Tool-sequence-verbatim (write without GT-required read)"

    def __init__(self, spec: dict[str, Any], jit_text: str) -> None:
        self.trigger_tool: str = spec["trigger_tool"]
        self.required_predecessors: list[str] = spec["required_predecessor_tools"]
        self.jit_correction_text = jit_text

    def check_before(
        self, state: BreakerState, tool_name: str, kwargs: dict[str, Any]
    ) -> Optional[Trip]:
        if tool_name != self.trigger_tool:
            return None
        seen = any(t in state.tool_history for t in self.required_predecessors)
        if not seen:
            return self._trip(
                f"{tool_name} called without any of {self.required_predecessors} in prior history"
            )
        return None


class QuoteWarningAsPolicyBreaker(_BreakerBase):
    """AP-02 — agent emits a verbatim substring of an injected ``watch for:``
    or ``preserve from current evidence:`` line as its OWN policy claim.

    The smoking-gun pattern from Phase 4 task 4: the agent says "This
    reservation is in basic economy, which cannot be modified according to
    airline policy" — that exact phrase came from the memory's
    ``watch for: basic economy cannot be modified`` line, but it's not
    actually airline policy (passenger updates ARE permitted on basic
    economy). The agent quoted memory text as authoritative policy.

    Detection: snapshot the rendered plan's absolute-claim lines at
    session start (done by KairosSession via
    BreakerState.snapshot_plan_absolute_claims). Then on every assistant
    message, substring-match the content against the snapshot. Match -> trip.

    This is a SUBSTRING check, not LLM. False-positive risk: an injected
    claim that happens to be a legitimate fact. Mitigated by:
      • Only snapshotting from the explicitly-dangerous label prefixes
        (``watch for:`` / ``preserve from current evidence:``); not the
        free-form headline prose.
      • Once-per-session via tripped_aps.
    """

    ap_id = "AP-02"
    ap_name = "Quote-warning-as-policy-fact"

    # Minimum claim length (chars) to be considered; shorter claims are
    # too likely to substring-match by coincidence.
    _MIN_CLAIM_CHARS = 20
    # Minimum tokens in the claim to bother with the in-order match.
    _MIN_CLAIM_TOKENS = 4
    # Maximum filler tokens permitted between consecutive claim tokens in
    # the content. 3 catches "basic economy, which cannot be modified"
    # (filler="which") but rejects unrelated long passages.
    _MAX_FILLER_BETWEEN_TOKENS = 3

    _TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

    def __init__(self, spec: dict[str, Any], jit_text: str) -> None:
        self.jit_correction_text = jit_text

    def _tokenize(self, text: str) -> list[str]:
        return self._TOKEN_RE.findall(text.lower())

    def _in_order_match(self, claim_tokens: list[str], content_tokens: list[str]) -> bool:
        """True if all claim tokens appear in content in order, with no
        more than ``_MAX_FILLER_BETWEEN_TOKENS`` other tokens between any
        two consecutive claim tokens. Avoids ``X .* Y`` false positives by
        bounding the gap; catches reasonable paraphrases like commas /
        conjunctions / "which"-relatives.
        """
        if not claim_tokens:
            return False
        # Walk content; greedily match claim tokens in order.
        last_pos = -1
        for token in claim_tokens:
            # Find next occurrence of token after last_pos, within filler window.
            search_start = last_pos + 1
            search_end = search_start + self._MAX_FILLER_BETWEEN_TOKENS + 1 if last_pos >= 0 else len(content_tokens)
            window = content_tokens[search_start:search_end]
            try:
                offset = window.index(token)
            except ValueError:
                return False
            last_pos = search_start + offset
        return True

    def check_before(
        self, state: BreakerState, tool_name: str, kwargs: dict[str, Any]
    ) -> Optional[Trip]:
        if not state.injected_plan_absolute_claims:
            return None
        if not state.recent_assistant_contents:
            return None
        # AP-02 scans ALL recent assistant content because the smoking-gun
        # quote often arrives on a respond turn while the harmful tool call
        # arrives several turns later with empty content. State accumulates
        # recent contents (capped); AP-02 walks them.
        for content in state.recent_assistant_contents:
            content_tokens = self._tokenize(content)
            for claim in state.injected_plan_absolute_claims:
                if len(claim) < self._MIN_CLAIM_CHARS:
                    continue
                claim_tokens = self._tokenize(claim)
                if len(claim_tokens) < self._MIN_CLAIM_TOKENS:
                    continue
                if self._in_order_match(claim_tokens, content_tokens):
                    return self._trip(
                        f"prior assistant content quoted injected claim: {claim[:80]!r}"
                    )
        return None


class CycleNoParamChangeBreaker(_BreakerBase):
    """AP-05 — identical-args retry after a tool error."""

    ap_id = "AP-05"
    ap_name = "Cycle-without-param-change"

    def __init__(self, spec: dict[str, Any], jit_text: str) -> None:
        self.window_size: int = spec.get("window_size", 3)
        self.jit_correction_text = jit_text

    def check_before(
        self, state: BreakerState, tool_name: str, kwargs: dict[str, Any]
    ) -> Optional[Trip]:
        # Code-review fix MED-1 (2026-05-20): the hash-recording side effect
        # used to live here; if AP-05 tripped and got suppressed, hashes
        # stopped accumulating and a clever adversary could spam identical
        # calls after the first trip. Hash recording moved to
        # DeterministicBreakers.check_before so it ALWAYS runs (state update
        # decoupled from trip emission). This method now only inspects the
        # state and decides whether to trip.
        if not state.recent_canonical_hashes:
            return None
        current = state.recent_canonical_hashes[-1]
        prior = state.recent_canonical_hashes[-3:-1]
        triple_match = len(prior) == 2 and prior[0] == current and prior[1] == current
        if triple_match:
            return self._trip(
                f"3rd identical {tool_name} in a row; hash={current[:8]} after_error={state.last_observation_had_error}"
            )
        return None


class PrematureTransferBreaker(_BreakerBase):
    """AP-07 — transfer fired before user's stated patience threshold."""

    ap_id = "AP-07"
    ap_name = "Premature transfer_to_human_agents"

    TRANSFER_TOOL = "transfer_to_human_agents"

    def __init__(self, spec: dict[str, Any], jit_text: str) -> None:
        self.jit_correction_text = jit_text

    def check_before(
        self, state: BreakerState, tool_name: str, kwargs: dict[str, Any]
    ) -> Optional[Trip]:
        if tool_name != self.TRANSFER_TOOL:
            return None
        allowed = state.allowed_transfer_tries
        if allowed is not None and state.transfer_count < allowed:
            return self._trip(
                f"transfer called after {state.transfer_count} tries; user allows {allowed}"
            )
        state.transfer_count += 1
        return None


class MutateThenUndoBreaker(_BreakerBase):
    """AP-08 — cancel_reservation on an id that was previously updated."""

    ap_id = "AP-08"
    ap_name = "Mutate-then-undo"

    UPDATE_PREFIX = "update_reservation_"
    CANCEL_TOOL = "cancel_reservation"

    def __init__(self, spec: dict[str, Any], jit_text: str) -> None:
        self.jit_correction_text = jit_text

    def check_before(
        self, state: BreakerState, tool_name: str, kwargs: dict[str, Any]
    ) -> Optional[Trip]:
        if tool_name != self.CANCEL_TOOL:
            return None
        rid = kwargs.get("reservation_id")
        history = state.update_log.get(rid) if rid else None
        if history:
            first_turn, first_tool = history[0]
            extra = f" (+ {len(history) - 1} more)" if len(history) > 1 else ""
            return self._trip(
                f"cancel_reservation({rid}) after {first_tool} at turn {first_turn}{extra}"
            )
        return None

    def check_after(
        self,
        state: BreakerState,
        tool_name: str,
        kwargs: dict[str, Any],
        observation: Any,
        had_error: bool,
    ) -> Optional[Trip]:
        if had_error:
            return None
        if tool_name.startswith(self.UPDATE_PREFIX):
            rid = kwargs.get("reservation_id")
            if rid:
                state.update_log.setdefault(rid, []).append((state.turn, tool_name))
        return None


class SearchExplodeBreaker(_BreakerBase):
    """AP-10 — many searches, few writes."""

    ap_id = "AP-10"
    ap_name = "Search-explode-no-decision"

    def __init__(self, spec: dict[str, Any], jit_text: str) -> None:
        self.search_globs: list[str] = spec["search_tools"]
        self.write_globs: list[str] = spec["write_tools"]
        self.jit_correction_text = jit_text
        # Tightened 2026-05-20: was max_writes=1; Stage A found false-positives
        # on legitimate one-write change-flight tasks (e.g. v4 task 13 baseline
        # 6 searches + 1 update -> reward 1.0). max_writes=0 catches the real
        # "search spiral never decides" pattern; one-write tasks are passing
        # behaviour and should not trip.
        self._search_threshold = 6
        self._max_writes = 0

    def check_after(
        self,
        state: BreakerState,
        tool_name: str,
        kwargs: dict[str, Any],
        observation: Any,
        had_error: bool,
    ) -> Optional[Trip]:
        if _matches_any_glob(tool_name, self.search_globs):
            state.search_count += 1
        if (
            state.search_count >= self._search_threshold
            and state.writes_count <= self._max_writes
        ):
            return self._trip(
                f"searches={state.search_count} writes={state.writes_count}"
            )
        return None


# Map AP-ID → constructor. Order matters only for "first trip wins" tie-breaks.
_BREAKER_REGISTRY: dict[str, type[_BreakerBase]] = {
    "AP-01": ReadWithoutWriteBreaker,
    "AP-02": QuoteWarningAsPolicyBreaker,  # T-05 (2026-05-20)
    "AP-03": ToolSequenceVerbatimBreaker,
    "AP-05": CycleNoParamChangeBreaker,
    "AP-07": PrematureTransferBreaker,
    "AP-08": MutateThenUndoBreaker,
    "AP-10": SearchExplodeBreaker,
}

# APs deferred to a later increment. AP-11 (arithmetic drift) still needs a
# regex-based detector against assistant content + tool-result sums. AP-02
# graduated to T-05 when the assistant-content channel landed.
_DEFERRED_APS = {"AP-11"}


class DeterministicBreakers:
    """Aggregator: load anti_patterns.json, instantiate matching breakers, dispatch.

    A single instance is owned by :class:`KairosSession` for its lifetime; the
    aggregator owns no per-session state itself (that lives in
    :class:`BreakerState`). The session passes its state in on every check.
    """

    def __init__(self, anti_patterns_path: str | Path) -> None:
        path = Path(anti_patterns_path)
        self._breakers: list[Breaker] = []
        # Edge case 2/3 (2026-05-20): graceful degrade if the path is missing
        # or malformed. A host that hasn't wired the AP DB yet shouldn't crash
        # the agent loop — it just gets zero breakers and proceeds as if the
        # deterministic-breakers flag were off.
        if not path.exists():
            _logger.warning("breakers: anti_patterns.json not found at %s; loading 0 breakers", path)
            return
        try:
            db = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _logger.warning("breakers: failed to load %s (%s); loading 0 breakers", path, exc)
            return
        if not isinstance(db, dict) or "anti_patterns" not in db:
            _logger.warning("breakers: %s has wrong shape (missing 'anti_patterns'); loading 0 breakers", path)
            return

        # Only instantiate APs marked ships_in in the currently-shipped set
        # AND not in the deferred subset. Multiple ticket numbers can be
        # "shipped" simultaneously (T-03 deterministic breakers + T-05's
        # AP-02 graduating from deferred); the filter accepts any of them.
        shipped_tickets = {"T-03", "T-05"}
        for entry in db["anti_patterns"]:
            ap_id = entry.get("id", "?")
            if entry.get("ships_in") not in shipped_tickets or ap_id in _DEFERRED_APS:
                continue
            ctor = _BREAKER_REGISTRY.get(ap_id)
            if ctor is None:
                _logger.warning("breakers: no registry entry for %s; skipping", ap_id)
                continue
            try:
                spec = entry["detection"]["spec"]
                jit_text = entry["jit_correction_text"]
                instance = ctor(spec=spec, jit_text=jit_text)
                instance.ap_name = entry["name"]
            except (KeyError, TypeError, ValueError) as exc:
                _logger.warning("breakers: %s instantiation failed (%s); skipping", ap_id, exc)
                continue
            self._breakers.append(instance)
        _logger.info(
            "breakers: loaded %d deterministic breakers from %s",
            len(self._breakers),
            path,
        )

    def check_before(
        self, state: BreakerState, tool_name: str, kwargs: dict[str, Any] | None
    ) -> Optional[Trip]:
        # Edge case (review 2026-05-20): contract says kwargs is dict, but a
        # broken host could pass None. Coerce defensively so AP-08's
        # ``kwargs.get(...)`` doesn't AttributeError.
        kwargs = kwargs or {}
        # T-04 refactor (2026-05-20): state.turn / recent_canonical_hashes
        # are mutated by ``BreakerState.record_before_tool`` in the session
        # BEFORE this method runs. Both subsystems read the already-updated
        # state. MED-1's unconditional-hash-recording invariant is preserved
        # by the session calling record_before_tool BEFORE any subsystem
        # dispatches.
        for breaker in self._breakers:
            if breaker.ap_id in state.tripped_aps:
                # Once-per-session: an AP that already fired in this session
                # does not re-fire. Prevents the agent from being slammed with
                # the same correction every turn after a single trip.
                continue
            # Edge case 1 (2026-05-20): a faulty breaker must not crash the
            # whole aggregator — log and skip to the next.
            try:
                trip = breaker.check_before(state, tool_name, kwargs)
            except Exception as exc:  # noqa: BLE001 - defensive
                _logger.warning("breakers: %s.check_before raised %s; skipping", breaker.ap_id, exc)
                continue
            if trip is not None:
                state.tripped_aps.add(breaker.ap_id)
                return trip
        return None

    def check_after(
        self,
        state: BreakerState,
        tool_name: str,
        kwargs: dict[str, Any] | None,
        observation: Any,
        had_error: bool,
    ) -> Optional[Trip]:
        kwargs = kwargs or {}
        # T-04 refactor (2026-05-20): tool_history / recent_observations /
        # last_observation_had_error are mutated by ``BreakerState.record_after_tool``
        # in the session BEFORE this method runs. The aggregator only
        # dispatches detection logic on the already-updated state.
        for breaker in self._breakers:
            if breaker.ap_id in state.tripped_aps:
                continue
            try:
                trip = breaker.check_after(state, tool_name, kwargs, observation, had_error)
            except Exception as exc:  # noqa: BLE001 - defensive
                _logger.warning("breakers: %s.check_after raised %s; skipping", breaker.ap_id, exc)
                continue
            if trip is not None:
                state.tripped_aps.add(breaker.ap_id)
                return trip
        return None

    @property
    def ap_ids(self) -> list[str]:
        return [b.ap_id for b in self._breakers]


_ERROR_PATTERN = re.compile(
    r"(?:^|\W)"
    r"(?:"
    r"error:"                      # Error: prefix (tau-bench convention)
    r"|invalid\s+(?:tool|json|argument|schema)"  # word "invalid" only when paired
    r"|schemavalidationerror"      # OpenAI tool-schema rejection
    r"|http\s*[45]\d\d"            # HTTP 4xx/5xx with explicit prefix
    r")",
    re.IGNORECASE,
)


def observation_had_error(observation: Any) -> bool:
    """Heuristic: did the env tool return an error?

    Tau-bench formats errors as strings starting with "Error:". OpenAI tool
    schema validation failures come back as the same shape. Code-review
    finding (2026-05-20): the original substring approach matched ``"invalid"``
    inside legitimate observations (e.g. flight number ``HAT404`` previously
    matched ``"404"``; product descriptions containing the word ``"invalid"``
    falsely tripped). Tightened to word-boundary patterns that only fire on
    actual error markers.
    """
    if observation is None:
        return False
    text = observation if isinstance(observation, str) else str(observation)
    return bool(_ERROR_PATTERN.search(text))
