"""Static policy-pack extraction for semantic recovery."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from kairos.models.semantic_recovery import PolicyPack
from kairos.semantic_recovery.memory import compute_prompt_hash, compute_toolset_hash

_BOOK_TOOL_PREFIXES = ("book_", "create_", "submit_")
_UPDATE_TOOL_PREFIXES = ("update_", "cancel_", "send_")


def build_policy_pack(
    *,
    source_name: str,
    system_prompt: str,
    tool_descriptions: list[dict[str, Any]],
    skills_text: str | None = None,
    host_docs_text: str | None = None,
) -> PolicyPack:
    """Build a deterministic static policy pack from host-provided context."""
    tool_names = _tool_names(tool_descriptions)
    source_text = "\n".join(part for part in (system_prompt, skills_text, host_docs_text) if part)
    write_tools = [tool for tool in tool_names if _is_write_tool(tool)]
    required_reads = {tool: _required_reads_for_write_tool(tool, tool_names) for tool in write_tools}
    write_constraints = {tool: _write_constraints_for_tool(tool, source_text) for tool in write_tools}
    expectations = _static_expectations(source_text, write_tools)
    traps = _failure_traps(source_text)
    source_hash = hashlib.sha256(
        json.dumps(
            {
                "system_prompt": system_prompt,
                "tool_descriptions": tool_descriptions,
                "skills_text": skills_text,
                "host_docs_text": host_docs_text,
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()

    return PolicyPack(
        pack_id=f"{source_name}:{source_hash[:12]}",
        source_name=source_name,
        source_hash=source_hash,
        tool_names=tool_names,
        static_expectations=expectations,
        required_read_before_write=required_reads,
        write_tool_constraints=write_constraints,
        failure_traps=traps,
        provenance={
            "prompt_hash": compute_prompt_hash(system_prompt),
            "toolset_hash": compute_toolset_hash(tool_descriptions),
            "system_prompt_chars": len(system_prompt),
            "tool_description_count": len(tool_descriptions),
            "skills_text_chars": len(skills_text or ""),
            "host_docs_text_chars": len(host_docs_text or ""),
        },
    )


def _tool_names(tool_descriptions: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for tool in tool_descriptions:
        name = tool.get("name")
        if not isinstance(name, str):
            function = tool.get("function")
            if isinstance(function, dict):
                name = function.get("name")
        if isinstance(name, str) and name not in names:
            names.append(name)
    return names


def _is_write_tool(tool_name: str) -> bool:
    return tool_name.startswith(_BOOK_TOOL_PREFIXES) or tool_name.startswith(_UPDATE_TOOL_PREFIXES)


def _required_reads_for_write_tool(tool_name: str, tool_names: list[str]) -> list[str]:
    reads: list[str] = []
    if "get_user_details" in tool_names:
        reads.append("get_user_details")
    if tool_name.startswith(("update_", "cancel_")) and "get_reservation_details" in tool_names:
        reads.append("get_reservation_details")
    if tool_name == "send_certificate" and "get_reservation_details" in tool_names:
        reads.append("get_reservation_details")
    return reads


def _write_constraints_for_tool(tool_name: str, source_text: str) -> list[str]:
    lowered = source_text.lower()
    constraints: list[str] = []
    if "explicit user confirmation" in lowered or "obtain explicit" in lowered:
        constraints.append("obtain explicit user confirmation before mutation")
    if tool_name.startswith("book_") and "at most one travel certificate" in lowered:
        constraints.append("at most one travel certificate per reservation")
    if tool_name.startswith("update_") and "basic economy flights cannot be modified" in lowered:
        constraints.append("basic economy cannot be modified")
    if tool_name.startswith("cancel_") and "reason for cancellation" in lowered:
        constraints.append("cancellation requires a grounded reason")
    return constraints


def _static_expectations(source_text: str, write_tools: list[str]) -> list[str]:
    lowered = source_text.lower()
    expectations = ["read trusted state before write tools"] if write_tools else []
    if "cheapest" in lowered or "lowest price" in lowered:
        expectations.append("compare candidate options before optimizing for lowest price")
    if "one certificate" in lowered or "at most one travel certificate" in lowered:
        expectations.append("split bookings or choose one certificate when certificate cardinality is constrained")
    return expectations


def _failure_traps(source_text: str) -> list[str]:
    lowered = source_text.lower()
    traps: list[str] = []
    if "basic economy flights cannot be modified" in lowered:
        traps.append("basic economy cannot be modified")
    if "at most one travel certificate" in lowered:
        traps.append("multiple certificates in one reservation")
    if "cheapest" in lowered or "lowest price" in lowered:
        traps.append("valid option can still be wrong if it is not the lowest valid price")
    return traps
