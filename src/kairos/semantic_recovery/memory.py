"""Workflow-memory ingestion and retrieval for semantic recovery."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from kairos.models.semantic_recovery import MemoryRetrievalResult, WorkflowMemory

_WORD_RE = re.compile(r"[a-z0-9_]+")

# Word-boundary match for the action verbs we classify on. Substring matching
# misclassifies "cancel my booking" / "upgrade my booking" as book intent
# because `"book" in "booking"` is True. The classifier uses these patterns
# instead of `in lowered` for the book verbs.
_BOOK_VERB_RE = re.compile(r"\b(book|rebook|re-book)\b")


class WorkflowMemoryStore:
    """Small in-memory workflow memory index.

    The store intentionally uses transparent lexical matching for Phase 1. This
    keeps the layer deterministic while still proving the shape needed by later
    semantic and learned-utility retrieval.
    """

    def __init__(self, memories: list[WorkflowMemory] | None = None) -> None:
        self._memories = list(memories or [])

    def add_many(self, memories: list[WorkflowMemory]) -> None:
        """Append workflow memories to the index."""
        self._memories.extend(memories)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
        toolset_hash: str | None = None,
        prompt_hash: str | None = None,
        min_semantic_score: float = -1.0,
    ) -> list[MemoryRetrievalResult]:
        """Retrieve memories by intent-token overlap plus utility score."""
        query_tokens = _token_set(query)
        query_constraints = _extract_constraints(query)
        query_action_class = _intent_action_class(query, [])
        results: list[MemoryRetrievalResult] = []
        for memory in self._memories:
            if not _hash_compatible(memory, toolset_hash=toolset_hash, prompt_hash=prompt_hash):
                continue
            if not _action_class_compatible(query_action_class, memory.intent_action_class):
                continue
            memory_tokens = _token_set(memory.intent_signature)
            overlap = len(query_tokens & memory_tokens)
            denominator = max(len(query_tokens | memory_tokens), 1)
            constraint_matches = _matched_constraints(query_constraints, memory.expected_constraints)
            matched_keys = _matched_constraint_keys(query_constraints, memory.expected_constraints)
            semantic_score = (overlap / denominator) + _constraint_score(query_constraints, matched_keys)
            if semantic_score < min_semantic_score:
                continue
            results.append(
                MemoryRetrievalResult(
                    memory=memory,
                    semantic_score=semantic_score,
                    utility_score=memory.utility_score,
                    matched_constraints=constraint_matches,
                    source_trajectory=memory.source_trajectory,
                )
            )
        results.sort(key=lambda item: (item.semantic_score, item.utility_score), reverse=True)
        return results[:top_k]


def compute_prompt_hash(system_prompt: str) -> str:
    """Return a stable hash for a host prompt surface."""
    return _stable_hash({"system_prompt": system_prompt})


def compute_toolset_hash(tool_descriptions: list[dict[str, Any]]) -> str:
    """Return a stable hash for host tool names/descriptions/schema."""
    return _stable_hash({"tool_descriptions": tool_descriptions})


def ingest_success_path_memories(
    trajectories: list[dict[str, Any]],
    *,
    toolset_hash: str | None = None,
    prompt_hash: str | None = None,
) -> list[WorkflowMemory]:
    """Convert passed/labeled-success trajectories into workflow memories."""
    memories: list[WorkflowMemory] = []
    for trajectory in trajectories:
        if not _is_success(trajectory):
            continue
        instruction = _instruction(trajectory)
        sequence = _tool_sequence(trajectory)
        if not instruction or not sequence:
            continue
        constraints = _extract_constraints(instruction)
        action_class = _intent_action_class(instruction, sequence)
        source_id = _source_trajectory(trajectory)
        failure_traps = _failure_traps_for_constraints(constraints)
        memory_payload = {
            "source_trajectory": source_id,
            "instruction": instruction,
            "tool_sequence": sequence,
            "constraints": constraints,
        }
        memories.append(
            WorkflowMemory(
                memory_id=f"workflow:{_stable_hash(memory_payload)[:12]}",
                title=_memory_title(trajectory, source_id),
                description=_memory_description(instruction),
                content=_memory_content(sequence, constraints, failure_traps),
                instruction_text=instruction,
                category=_memory_category(trajectory, constraints),
                intent_action_class=action_class,
                intent_signature=_intent_signature(instruction),
                expected_tool_sequence=sequence,
                expected_constraints=constraints,
                canonical_kwargs_hints={},
                failure_traps=failure_traps,
                negative_example=_negative_example(trajectory),
                source_trajectory=source_id,
                source_trajectory_ids=[source_id],
                prompt_hash=prompt_hash,
                toolset_hash=toolset_hash,
                utility_score=float(trajectory.get("utility_score", 1.0)),
                provenance={
                    "reward": trajectory.get("reward"),
                    "passed": trajectory.get("passed"),
                    "tool_count": len(sequence),
                },
            )
        )
    return memories


def load_success_path_memory_store(
    paths_text: str | None,
    *,
    toolset_hash: str | None = None,
    prompt_hash: str | None = None,
) -> WorkflowMemoryStore:
    """Load passed trajectory memories from JSON/JSONL files.

    `paths_text` may be a comma-separated list of files. Each file can contain a
    JSON list, one JSON object, or newline-delimited JSON objects.
    """
    memories: list[WorkflowMemory] = []
    for path in _split_paths(paths_text):
        trajectories = _load_trajectory_file(path)
        memories.extend(ingest_success_path_memories(trajectories, toolset_hash=toolset_hash, prompt_hash=prompt_hash))
    return WorkflowMemoryStore(memories)


def extract_constraints(text: str) -> dict[str, Any]:
    """Extract coarse user constraints for session contracts and memory ranking."""
    return _extract_constraints(text)


def _hash_compatible(memory: WorkflowMemory, *, toolset_hash: str | None, prompt_hash: str | None) -> bool:
    if memory.toolset_hash is not None and toolset_hash is not None and memory.toolset_hash != toolset_hash:
        return False
    return not (memory.prompt_hash is not None and prompt_hash is not None and memory.prompt_hash != prompt_hash)


def _is_success(trajectory: dict[str, Any]) -> bool:
    if trajectory.get("passed") is True:
        return True
    reward = trajectory.get("reward")
    return isinstance(reward, int | float) and float(reward) >= 1.0


def _split_paths(paths_text: str | None) -> list[Path]:
    if not paths_text:
        return []
    paths: list[Path] = []
    for raw_path in paths_text.split(","):
        stripped = raw_path.strip()
        if stripped:
            paths.append(Path(stripped).expanduser())
    return paths


def _load_trajectory_file(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                rows.append(parsed)
        return rows
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        return [parsed]
    return []


def _instruction(trajectory: dict[str, Any]) -> str:
    user_instruction = trajectory.get("user_instruction")
    if isinstance(user_instruction, str):
        return user_instruction
    task = trajectory.get("task")
    if isinstance(task, dict) and isinstance(task.get("instruction"), str):
        return str(task["instruction"])
    info = trajectory.get("info")
    if isinstance(info, dict):
        info_task = info.get("task")
        if isinstance(info_task, dict) and isinstance(info_task.get("instruction"), str):
            return str(info_task["instruction"])
    return ""


def _tool_sequence(trajectory: dict[str, Any]) -> list[str]:
    explicit = trajectory.get("tool_sequence")
    if isinstance(explicit, list) and all(isinstance(item, str) for item in explicit):
        return list(dict.fromkeys(explicit))

    sequence: list[str] = []
    traj = trajectory.get("traj")
    if isinstance(traj, list):
        for message in traj:
            if not isinstance(message, dict):
                continue
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if isinstance(function, dict) and isinstance(function.get("name"), str):
                    name = str(function["name"])
                    if name != "think":
                        sequence.append(name)
    if sequence:
        return sequence

    info = trajectory.get("info")
    task = info.get("task") if isinstance(info, dict) else trajectory.get("task")
    if isinstance(task, dict):
        actions = task.get("actions")
        if isinstance(actions, list):
            return [str(action["name"]) for action in actions if isinstance(action, dict) and "name" in action]
    return []


def _source_trajectory(trajectory: dict[str, Any]) -> str:
    for key in ("trajectory_id", "trace_id", "task_id"):
        value = trajectory.get(key)
        if value is not None:
            return str(value)
    return f"trajectory:{_stable_hash(trajectory)[:12]}"


def _memory_title(trajectory: dict[str, Any], source_id: str) -> str:
    title = trajectory.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return f"Workflow from {source_id}"


def _memory_description(instruction: str) -> str:
    stripped = " ".join(instruction.split())
    if len(stripped) <= 180:
        return stripped
    return f"{stripped[:177]}..."


def _memory_content(sequence: list[str], constraints: dict[str, Any], failure_traps: list[str]) -> str:
    parts = [
        (
            "Actionable procedure: follow the expected workflow, preserve the listed constraints, "
            "and verify current tool evidence before each write."
        ),
        f"Expected workflow: {' -> '.join(sequence)}.",
    ]
    if constraints:
        parts.append(f"Preserve constraints: {json.dumps(constraints, sort_keys=True)}.")
    if failure_traps:
        parts.append(f"Watch for traps: {'; '.join(failure_traps)}.")
    return " ".join(parts)


def _memory_category(
    trajectory: dict[str, Any],
    constraints: dict[str, Any],
) -> Literal["strategy", "recovery", "optimization"]:
    category = trajectory.get("category")
    if category == "strategy":
        return "strategy"
    if category == "recovery":
        return "recovery"
    if category == "optimization":
        return "optimization"
    if constraints.get("optimization"):
        return "optimization"
    return "strategy"


def _intent_action_class(
    instruction: str,
    sequence: list[str],
) -> Literal["book", "update", "cancel", "send_certificate", "handoff", "read_only", "unknown"]:
    sequence_class = _action_class_from_sequence(sequence)
    if sequence_class != "unknown":
        return sequence_class
    lowered = instruction.lower()
    if "send certificate" in lowered:
        return "send_certificate"
    if (
        "compensation" in lowered
        or "compensate" in lowered
        or "voucher" in lowered
        or "delayed flight" in lowered
        or "flight was delayed" in lowered
        or "flight has been delayed" in lowered
    ):
        return "send_certificate"
    if (
        _BOOK_VERB_RE.search(lowered) is not None
        or "new flight" in lowered
        or "want to fly" in lowered
        or "fly from" in lowered
    ):
        return "book"
    if "cancel" in lowered or "cancellation" in lowered:
        return "cancel"
    if "transfer to human" in lowered or "human agent" in lowered:
        return "handoff"
    if (
        "change" in lowered
        or "update" in lowered
        or "modify" in lowered
        or "upgrade" in lowered
        or "downgrade" in lowered
        or "upcoming trip" in lowered
        or "reservation" in lowered
    ):
        return "update"
    return "unknown"


def _action_class_from_sequence(
    sequence: list[str],
) -> Literal["book", "update", "cancel", "send_certificate", "handoff", "read_only", "unknown"]:
    terminal_tools = [tool for tool in sequence if _is_terminal_tool(tool)]
    if not terminal_tools:
        return "read_only" if sequence else "unknown"
    if any(tool == "book_reservation" for tool in terminal_tools):
        return "book"
    if any(tool.startswith("update_reservation_") for tool in terminal_tools):
        return "update"
    if any(tool == "cancel_reservation" for tool in terminal_tools):
        return "cancel"
    if any(tool == "send_certificate" for tool in terminal_tools):
        return "send_certificate"
    if any(tool == "transfer_to_human_agents" for tool in terminal_tools):
        return "handoff"
    return "unknown"


def _is_terminal_tool(tool_name: str) -> bool:
    return tool_name.startswith(("book_", "update_", "cancel_", "send_")) or tool_name == "transfer_to_human_agents"


def _action_class_compatible(query_action_class: str, memory_action_class: str) -> bool:
    """Strict action-class match.

    Refuses to retrieve when either side is ``"unknown"``. Previously this
    failed open in both directions, which let unclassified queries pull in
    every memory and let memories stored with ``intent_action_class="unknown"``
    match every query — the structural source of the cross-class injections
    we saw on tasks 11/13/15/18.
    """
    if query_action_class == "unknown" or memory_action_class == "unknown":
        return False
    return query_action_class == memory_action_class


def _negative_example(trajectory: dict[str, Any]) -> str | None:
    value = trajectory.get("negative_example")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _intent_signature(instruction: str) -> str:
    constraints = _extract_constraints(instruction)
    parts: list[str] = []
    if constraints.get("after_11am"):
        parts.append("after 11am")
    if constraints.get("direct_required") or constraints.get("direct_preferred"):
        parts.append("direct")
    cabin = constraints.get("cabin")
    if isinstance(cabin, str):
        parts.append(cabin)
    if constraints.get("optimization") == "lowest_price":
        parts.append("lowest price")
    if constraints.get("optimization") == "fastest_total_trip_time":
        parts.append("fastest total trip time")
    if constraints.get("one_stop_ok"):
        parts.append("one stop")
    if constraints.get("cancel_intent"):
        parts.append("cancel")
    if constraints.get("delay_complaint"):
        parts.append("delayed flight")
    if constraints.get("compensation_intent"):
        parts.append("compensation")
    if constraints.get("basic_economy"):
        parts.append("basic economy")
    if constraints.get("cabin_upgrade"):
        parts.append("upgrade cabin")
    if constraints.get("passenger_update"):
        parts.append("passenger")
    if constraints.get("baggage_update"):
        parts.append("baggage")
    insurance = constraints.get("insurance")
    if insurance == "yes":
        parts.append("with insurance")
    elif insurance == "no":
        parts.append("no insurance")
    if constraints.get("health_reason"):
        parts.append("health reason")
    if constraints.get("gift_card_preference"):
        parts.append("gift card")
    if constraints.get("payment_preference") == "smallest_balance_gift_card":
        parts.append("smallest gift card")
    if constraints.get("multi_step_ordered"):
        parts.append("ordered sequence")
    if not parts:
        parts = sorted(_token_set(instruction))[:8]
    return " ".join(parts)


def _extract_constraints(text: str) -> dict[str, Any]:
    lowered = text.lower()
    constraints: dict[str, Any] = {}
    if "after 11" in lowered or "not before 11" in lowered:
        constraints["after_11am"] = True
        constraints["departure_time_constraint"] = "after_11am"
    if "economy" in lowered:
        constraints["cabin"] = "economy"
    if "business" in lowered:
        constraints["cabin"] = "business"
    basic_economy_negated = any(
        pattern in lowered
        for pattern in (
            "not basic economy",
            "not in basic economy",
            "not basic",
            "no basic economy",
            "not basic-economy",
        )
    )
    if "basic economy" in lowered and not basic_economy_negated:
        constraints["basic_economy"] = True
        # Preserve legacy behaviour: cabin still resolves to "economy" when
        # the text says "basic economy", so existing intent-signature
        # expectations continue to hold.
        constraints["cabin"] = "economy"
    if "cheapest" in lowered or "lowest price" in lowered or "lowest valid price" in lowered:
        constraints["optimization"] = "lowest_price"
    if "fastest" in lowered or "shortest" in lowered:
        constraints["optimization"] = "fastest_total_trip_time"
    if "nonstop" in lowered or "non-stop" in lowered or "direct only" in lowered:
        constraints["direct_required"] = True
    if "direct" in lowered and "prefer" in lowered:
        constraints["direct_preferred"] = True
    one_stop_ok_patterns = (
        "one stop is fine",
        "one-stop is fine",
        "one stop also fine",
        "one stop is also fine",
        "one-stop also fine",
        "one-stop is also fine",
        "one stop okay",
        "one stop ok",
        "one stop is okay",
        "one stop acceptable",
        "one stop is acceptable",
        "one stop is fine",
        "one stopover also fine",
        "stopover also fine",
        "stopover is fine",
        "stopover acceptable",
    )
    if any(pattern in lowered for pattern in one_stop_ok_patterns):
        constraints["one_stop_ok"] = True
    if "no insurance" in lowered or "do not want insurance" in lowered:
        constraints["insurance"] = "no"
    elif (
        "with insurance" in lowered
        or "have insurance" in lowered
        or "travel insurance" in lowered
        or "insurance was purchased" in lowered
        or "insurance is purchased" in lowered
        or "i am insured" in lowered
        or "since i am insured" in lowered
        or "insured" in lowered
    ):
        constraints["insurance"] = "yes"
    if "no jfk" in lowered or "do not accept jfk" in lowered:
        constraints["forbidden_airports"] = ["JFK"]
    if "cancel" in lowered or "cancellation" in lowered:
        constraints["cancel_intent"] = True
    if (
        "unwell" in lowered
        or "i am sick" in lowered
        or "feel sick" in lowered
        or "health reason" in lowered
        or "health condition" in lowered
        or "medical reason" in lowered
    ):
        constraints["health_reason"] = True
    passenger_patterns = (
        "change passenger",
        "change the passenger",
        "update passenger",
        "update the passenger",
        "remove passenger",
        "remove the passenger",
        "modify passenger",
        "modify the passenger",
        "edit passenger",
        "edit the passenger",
        "add passenger",
        "add a passenger",
        "add an extra passenger",
        "passenger name",
        "passenger to ",
        "passenger change",
        "passenger update",
        "passenger removal",
    )
    if any(pattern in lowered for pattern in passenger_patterns):
        constraints["passenger_update"] = True
    no_baggage_patterns = (
        "no baggage",
        "no bag",
        "no bags",
        "no checked bag",
        "no checked bags",
        "without baggage",
        "without bags",
        "do not want baggage",
        "don't want baggage",
        "do not want checked bags",
        "don't want checked bags",
    )
    remove_baggage_patterns = (
        "remove checked bag",
        "remove checked bags",
        "remove my checked bag",
        "remove my checked bags",
        "remove baggage",
        "remove bag",
        "remove bags",
        "drop checked bag",
        "drop checked bags",
        "refund baggage",
        "refund bag",
    )
    add_baggage_patterns = (
        "checked bag",
        "checked bags",
        "add bag",
        "add bags",
        "add a bag",
        "add a checked bag",
        "add checked bag",
        "add checked bags",
        "baggage",
        "extra bag",
    )
    if any(pattern in lowered for pattern in no_baggage_patterns):
        constraints["baggage"] = "none"
    if any(pattern in lowered for pattern in remove_baggage_patterns):
        constraints["baggage_update"] = True
        constraints["baggage"] = "none"
    elif constraints.get("baggage") != "none" and any(pattern in lowered for pattern in add_baggage_patterns):
        constraints["baggage_update"] = True
    if "gift card" in lowered:
        constraints["gift_card_preference"] = True
    if "gift card" in lowered and "smallest" in lowered and "balance" in lowered:
        constraints["payment_preference"] = "smallest_balance_gift_card"
    if (
        "delayed flight" in lowered
        or "flight was delayed" in lowered
        or "flight has been delayed" in lowered
        or "delay" in lowered
        or "delayed" in lowered
    ):
        constraints["delay_complaint"] = True
    if "compensation" in lowered or "compensate" in lowered or "voucher" in lowered:
        constraints["compensation_intent"] = True
    if "upgrade" in lowered and (
        "cabin" in lowered
        or "to business" in lowered
        or "to economy" in lowered
        or "business class" in lowered
        or "economy class" in lowered
    ):
        constraints["cabin_upgrade"] = True
    if (
        "in order" in lowered
        or "ordered sequence" in lowered
        or "in this exact order" in lowered
        or "mention all" in lowered
        or ("first" in lowered and "then" in lowered)
    ):
        constraints["multi_step_ordered"] = True
    return constraints


def _matched_constraints(query_constraints: dict[str, Any], memory_constraints: dict[str, Any]) -> list[str]:
    matched: list[str] = []
    for key, value in query_constraints.items():
        if memory_constraints.get(key) != value:
            continue
        if key == "cabin":
            matched.append(f"cabin:{value}")
        elif key == "optimization":
            matched.append(f"optimization:{value}")
        elif key == "insurance":
            matched.append(f"insurance:{value}")
        else:
            matched.append(key)
    return matched


def _matched_constraint_keys(query_constraints: dict[str, Any], memory_constraints: dict[str, Any]) -> set[str]:
    return {key for key, value in query_constraints.items() if memory_constraints.get(key) == value}


def _constraint_score(query_constraints: dict[str, Any], matched_keys: set[str]) -> float:
    score = 0.0
    for key in matched_keys:
        score += _constraint_weight(key)
    for key in query_constraints:
        if key not in matched_keys:
            score -= _missing_constraint_penalty(key)
    return score


def _constraint_weight(key: str) -> float:
    if key == "optimization":
        return 0.4
    if key == "cabin":
        return 0.15
    if key in {"after_11am", "forbidden_airports"}:
        return 0.25
    if key in {"compensation_intent", "delay_complaint"}:
        return 0.3
    if key == "direct_required":
        return 0.25
    return 0.2


def _missing_constraint_penalty(key: str) -> float:
    if key == "optimization":
        return 0.45
    if key == "forbidden_airports":
        return 0.3
    return 0.08


def _failure_traps_for_constraints(constraints: dict[str, Any]) -> list[str]:
    traps: list[str] = []
    if constraints.get("optimization") == "lowest_price":
        traps.append("compare all searched options before booking")
    if constraints.get("optimization") == "fastest_total_trip_time":
        traps.append("compare total travel time including stopover time before mutation")
    if constraints.get("payment_preference") == "smallest_balance_gift_card":
        traps.append("choose the smallest sufficient gift card from current user payment evidence")
    if constraints.get("delay_complaint") and constraints.get("compensation_intent"):
        traps.append("delay compensation requires current user/reservation evidence before sending a certificate")
    if constraints.get("direct_preferred") and constraints.get("one_stop_ok"):
        traps.append("direct preference does not override lower-price fallback unless user makes it absolute")
    if constraints.get("direct_required"):
        traps.append("nonstop/direct requirement overrides cheaper connecting options")
    if constraints.get("forbidden_airports"):
        traps.append("valid flight can still violate user airport constraints")
    if constraints.get("basic_economy") and not constraints.get("cancel_intent"):
        traps.append("basic economy cannot be modified; offer cancel if eligible or transfer to human")
    if constraints.get("cancel_intent") and constraints.get("insurance") != "yes":
        traps.append(
            "verify cancel eligibility (24h window, insurance, business cabin, or airline-cancelled) "
            "before calling cancel_reservation"
        )
    if constraints.get("multi_step_ordered"):
        traps.append("apply each mutation tool in the user-stated order; do not bundle or skip steps")
    return traps


def _token_set(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()
