from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ValidationError

from kairos.models.semantic_recovery import (
    ArgPolicy,
    PolicyPack,
    SemanticDecision,
    ToolPolicy,
    ToolPolicyPack,
    ToolPolicyVerdict,
)

if TYPE_CHECKING:
    from kairos.semantic_recovery.expectation import ExpectationLLMClient

_WRITE_PREFIXES = ("book_", "update_", "cancel_", "send_")


class ToolPolicyAuditor:
    """Optional L2 SLM auditor for one proposed tool call.

    Safety rail: the auditor only blocks when explicitly configured for
    blocking, the SLM returns ``compliant=false`` with high confidence, and
    every violation carries a concrete suggested value that can be applied to
    the proposed kwargs. Otherwise the result is observe-only.
    """

    def __init__(
        self,
        *,
        client: ExpectationLLMClient | None = None,
        blocking: bool = False,
    ) -> None:
        self.client = client
        self.blocking = blocking

    def audit(
        self,
        *,
        tool_name: str,
        kwargs: dict[str, Any],
        tool_policy_pack: ToolPolicyPack | None,
        user_transcript: str,
        evidence: dict[str, Any] | None = None,
    ) -> SemanticDecision | None:
        """Audit one proposed call, returning a decision or None when disabled."""
        if self.client is None or tool_policy_pack is None:
            return None
        policy = tool_policy_pack.tools.get(tool_name)
        if policy is None or policy.side_effect_type == "read":
            return None
        raw = self.client.complete_json(
            system_prompt=_auditor_system_prompt(),
            user_prompt=_auditor_user_prompt(
                tool_policy=policy,
                tool_name=tool_name,
                kwargs=kwargs,
                user_transcript=user_transcript,
                evidence=evidence or {},
            ),
        )
        verdict = _parse_verdict(raw)
        if verdict is None or verdict.compliant:
            return None
        suggested_kwargs = _suggested_kwargs(kwargs, verdict)
        evidence_refs = {
            "tool_policy_pack_id": tool_policy_pack.pack_id,
            "tool_policy": policy.model_dump(exclude={"input_schema"}),
            "verdict": verdict.model_dump(),
        }
        if self.blocking and verdict.confidence == "high" and suggested_kwargs is not None:
            return SemanticDecision(
                verdict="block_with_next_call",
                failure_class="tool_policy_violation",
                reason="Tool kwargs violate the compiled per-tool policy contract.",
                next_tool=tool_name,
                next_kwargs=suggested_kwargs,
                confidence="high",
                evidence_refs=evidence_refs,
                contract_diffs=[f"{violation.field}: {violation.reason}" for violation in verdict.violations],
            )
        return SemanticDecision(
            verdict="observe_only",
            failure_class="tool_policy_violation",
            reason=(
                "Tool policy auditor found a violation, but L2 blocking is disabled "
                "or the verdict lacks a grounded exact replacement."
            ),
            confidence=verdict.confidence,
            evidence_refs=evidence_refs,
            contract_diffs=[f"{violation.field}: {violation.reason}" for violation in verdict.violations],
        )


def build_tool_policy_pack(
    *,
    policy_pack: PolicyPack,
    tool_descriptions: list[dict[str, Any]],
) -> ToolPolicyPack:
    """Compile a detailed per-tool policy memory from prompt + tool schemas."""
    tools: dict[str, ToolPolicy] = {}
    for tool in tool_descriptions:
        name = _tool_name(tool)
        if name is None:
            continue
        schema = _parameter_schema(tool)
        description = _tool_description(tool)
        tools[name] = ToolPolicy(
            tool_name=name,
            purpose=description,
            side_effect_type=_side_effect_type(name),
            required_preconditions=policy_pack.required_read_before_write.get(name, []),
            arg_policies=_arg_policies(name, schema),
            continuity_rules=_continuity_rules(name),
            forbidden_shortcuts=_forbidden_shortcuts(name, policy_pack),
            output_facts_produced=_output_facts_produced(name),
            common_failure_modes=_common_failure_modes(name, policy_pack),
            input_schema=schema,
        )
    source_hash = _stable_hash(
        {
            "policy_pack": policy_pack.model_dump(),
            "tool_descriptions": tool_descriptions,
        }
    )
    return ToolPolicyPack(
        pack_id=f"tool-policy:{source_hash[:12]}",
        source_policy_pack_id=policy_pack.pack_id,
        source_hash=source_hash,
        prompt_hash=_optional_str(policy_pack.provenance.get("prompt_hash")),
        toolset_hash=_optional_str(policy_pack.provenance.get("toolset_hash")),
        tools=tools,
        provenance={
            "tool_count": len(tools),
            "source_name": policy_pack.source_name,
        },
    )


def _auditor_system_prompt() -> str:
    return (
        "You are a strict L2 tool-call policy auditor. Return JSON only with keys "
        "compliant, confidence, violations. A call is compliant iff each kwarg's "
        "source and value are justified by the current transcript, trusted tool "
        "results, and the per-tool policy. Do not invent new IDs or values. When "
        "uncertain, set compliant=false with low confidence."
    )


def _auditor_user_prompt(
    *,
    tool_policy: ToolPolicy,
    tool_name: str,
    kwargs: dict[str, Any],
    user_transcript: str,
    evidence: dict[str, Any],
) -> str:
    return json.dumps(
        {
            "tool": tool_name,
            "schema": tool_policy.input_schema,
            "policy": tool_policy.model_dump(exclude={"input_schema"}),
            "proposed_kwargs": kwargs,
            "user_transcript": user_transcript,
            "recent_tool_results": evidence.get("recent_tool_results", []),
            "evidence_ledger": {key: value for key, value in evidence.items() if key != "recent_tool_results"},
            "instructions": (
                "Return exactly this JSON shape: "
                '{"compliant":bool,"confidence":"low|medium|high",'
                '"violations":[{"field":str,"reason":str,"suggested_value":any|null}]}. '
                "Only use suggested_value when it is grounded in provided evidence."
            ),
        },
        sort_keys=True,
        default=str,
    )


def _parse_verdict(raw: str) -> ToolPolicyVerdict | None:
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return None
        return ToolPolicyVerdict.model_validate(parsed)
    except (json.JSONDecodeError, TypeError, ValidationError, ValueError):
        return None


def _suggested_kwargs(kwargs: dict[str, Any], verdict: ToolPolicyVerdict) -> dict[str, Any] | None:
    if not verdict.violations:
        return None
    suggested = dict(kwargs)
    for violation in verdict.violations:
        if violation.suggested_value is None:
            return None
        if not violation.field or "." in violation.field or "[" in violation.field:
            return None
        suggested[violation.field] = violation.suggested_value
    return suggested


def _arg_policies(tool_name: str, schema: dict[str, Any]) -> dict[str, ArgPolicy]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return {}
    return {field: _arg_policy(tool_name, field) for field in properties if isinstance(field, str)}


def _arg_policy(tool_name: str, field: str) -> ArgPolicy:
    # TODO(refactor): the per-field rules below are tau-bench airline/retail
    # specific — they hard-code field names (passengers, flights,
    # payment_methods, cabin, ...) and tool names (search_direct_flight,
    # get_user_details, ...) into the kairos library. This couples kairos to
    # one host domain. Move this rule library out of kairos into either:
    #   (a) a host-supplied callable injected at build_tool_policy_pack time
    #       (signature: (tool_name, field) -> ArgPolicy | None), or
    #   (b) a JSON/YAML rules file the host provides, parsed once at startup.
    # Default behaviour when the host supplies nothing should fall back to a
    # generic ArgPolicy (the catch-all at the bottom of this function).
    # Tracked for a follow-up commit; for now we ship with airline rules to
    # keep tau-agent's L2 audit working end-to-end.
    if field == "user_id":
        return ArgPolicy(
            field=field,
            allowed_sources=["current_user_instruction", "get_user_details"],
            rule="Must identify the active user for this session; do not switch users from stale context.",
            can_be_inferred=False,
        )
    if field == "reservation_id":
        return ArgPolicy(
            field=field,
            allowed_sources=["user_message", "get_user_details", "get_reservation_details"],
            rule="Must refer to a reservation grounded in the current user session.",
            can_be_inferred=False,
        )
    if field == "passengers":
        return ArgPolicy(
            field=field,
            allowed_sources=["current_user_instruction", "prior_reservation_if_rebooking"],
            rule=(
                "Preserve prior reservation passenger identities/count during cancel-and-rebook "
                "unless the user explicitly changes travelers."
            ),
            can_be_inferred=False,
        )
    if field in {"flights", "flight_number"}:
        return ArgPolicy(
            field=field,
            allowed_sources=[
                "search_direct_flight",
                "search_onestop_flight",
                "prior_reservation_for_unchanged_legs",
            ],
            rule="Flight tuples must come from trusted search results or unchanged reservation legs.",
            can_be_inferred=False,
        )
    if field in {"payment_id", "payment_methods"}:
        return ArgPolicy(
            field=field,
            allowed_sources=["get_user_details", "current_user_payment_preference", "policy_computation"],
            rule="Payment IDs must be grounded in user details and amounts must follow user preference and policy.",
            can_be_inferred=False,
        )
    if field in {"cabin", "insurance", "total_baggages", "nonfree_baggages"}:
        return ArgPolicy(
            field=field,
            allowed_sources=["current_user_instruction", "prior_reservation_when_preserving_defaults"],
            rule="Must match explicit user intent; preserve prior/default values only when user is silent.",
            can_be_inferred=False,
        )
    return ArgPolicy(
        field=field,
        allowed_sources=["tool_schema", "current_user_instruction", "trusted_tool_result"],
        rule="Value must be justified by current user intent or a trusted tool result.",
    )


def _continuity_rules(tool_name: str) -> list[str]:
    if tool_name == "book_reservation":
        return [
            (
                "After cancellation/rebooking, preserve passenger count and identities "
                "unless user explicitly changed them."
            ),
            "Do not copy stale baggage/insurance defaults when the new booking user intent says otherwise.",
        ]
    if tool_name.startswith("update_reservation_"):
        return ["Include only changes grounded in the current user request and trusted reservation state."]
    return []


def _forbidden_shortcuts(tool_name: str, policy_pack: PolicyPack) -> list[str]:
    shortcuts: list[str] = []
    if tool_name == "transfer_to_human_agents":
        shortcuts.append("Do not hand off while a grounded self-service tool path remains available.")
    shortcuts.extend(policy_pack.write_tool_constraints.get(tool_name, []))
    return list(dict.fromkeys(shortcuts))


def _output_facts_produced(tool_name: str) -> list[str]:
    if tool_name == "get_user_details":
        return ["user profile", "payment methods", "reservation ids"]
    if tool_name == "get_reservation_details":
        return ["reservation passengers", "reservation flights", "payment history", "baggage", "insurance"]
    if tool_name.startswith("search_"):
        return ["flight candidates", "availability", "prices", "times"]
    return []


def _common_failure_modes(tool_name: str, policy_pack: PolicyPack) -> list[str]:
    modes = list(policy_pack.failure_traps)
    if tool_name == "book_reservation":
        modes.extend(["passenger drift", "payment amount mismatch", "certificate cardinality violation"])
    if tool_name.startswith("update_reservation_"):
        modes.extend(["wrong valid option", "stale reservation state", "payment preference mismatch"])
    return list(dict.fromkeys(modes))


def _tool_name(tool: dict[str, Any]) -> str | None:
    name = tool.get("name")
    if isinstance(name, str):
        return name
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return str(function["name"])
    return None


def _tool_description(tool: dict[str, Any]) -> str:
    description = tool.get("description")
    if isinstance(description, str):
        return description
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("description"), str):
        return str(function["description"])
    return ""


def _parameter_schema(tool: dict[str, Any]) -> dict[str, Any]:
    parameters = tool.get("parameters")
    if isinstance(parameters, dict):
        return parameters
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("parameters"), dict):
        return dict(function["parameters"])
    return {}


def _side_effect_type(tool_name: str) -> Literal["read", "write", "handoff", "final_response"]:
    if tool_name == "transfer_to_human_agents":
        return "handoff"
    if tool_name == "respond":
        return "final_response"
    if tool_name.startswith(_WRITE_PREFIXES):
        return "write"
    return "read"


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    return None
