from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ToolCallRecord:
    name: str
    arguments: Any
    raw_arguments: str
    tool_call_id: str | None
    output: Any
    raw_output: str | None
    message_index: int
    output_message_index: int | None
    sequence_index: int


@dataclass
class ExpectedActionMatch:
    expected_index: int
    executed_index: int
    name: str
    diffs: list[str]
    expected_args: Any
    actual_args: Any


@dataclass
class DeviationSummary:
    kind: str
    title: str
    detail: str
    highlighted_executed_index: int | None = None
    highlighted_expected_index: int | None = None
    diffs: list[str] | None = None


def _parse_jsonish(value: str | None) -> Any:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _json_text(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)


def _preview_text(value: Any, *, limit: int = 180) -> str:
    text = value if isinstance(value, str) else _json_text(value)
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit].rstrip()}..."


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def find_results_files(results_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in results_dir.glob("*.json")
            if path.is_file()
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def load_results_file(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of run records in {path}")
    return data


def write_results_file(path: Path, records: list[dict[str, Any]]) -> None:
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)


def load_normalized_traces(normalized_dir: Path) -> list[dict[str, Any]]:
    if not normalized_dir.exists():
        return []
    traces: list[dict[str, Any]] = []
    for path in sorted(normalized_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payload["_path"] = str(path)
            traces.append(payload)
    return traces


def find_matching_trace(record: dict[str, Any], traces: list[dict[str, Any]]) -> dict[str, Any] | None:
    info = _as_dict(record.get("info"))
    task = _as_dict(info.get("task"))
    task_instruction = task.get("instruction")
    task_id = record.get("task_id")
    trial = record.get("trial", 0)
    reward = record.get("reward")

    best_trace: dict[str, Any] | None = None
    best_score = -1

    for trace in traces:
        metadata = _as_dict(trace.get("metadata"))
        if task_instruction and trace.get("user_input") != task_instruction:
            continue
        score = 0
        if metadata.get("task_id") == task_id:
            score += 5
        if metadata.get("trial", 0) == trial:
            score += 4
        if trace.get("user_input") == task_instruction and task_instruction:
            score += 4
        if metadata.get("reward") == reward:
            score += 1
        if score > best_score:
            best_score = score
            best_trace = trace

    return best_trace if best_score > 0 else None


def get_annotation(record: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(record.get("debugger_annotation"))


def set_annotation(record: dict[str, Any], *, remarks: str, comments: str) -> None:
    record["debugger_annotation"] = {
        "remarks": remarks,
        "comments": comments,
        "updated_at": _utc_now_text(),
    }


def clear_annotation(record: dict[str, Any]) -> None:
    record.pop("debugger_annotation", None)


def extract_time_context(system_messages: list[dict[str, Any]]) -> str | None:
    for message in system_messages:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("The current time is "):
                return line.removeprefix("The current time is ").strip()
    return None


def extract_expected_actions(record: dict[str, Any]) -> list[dict[str, Any]]:
    info = _as_dict(record.get("info"))
    reward_info = _as_dict(info.get("reward_info"))
    actions = reward_info.get("actions") or []
    return [action for action in actions if isinstance(action, dict)]


def extract_tool_calls(traj: list[dict[str, Any]]) -> list[ToolCallRecord]:
    tool_outputs: dict[str, tuple[int, dict[str, Any]]] = {}
    for idx, message in enumerate(traj):
        if message.get("role") == "tool":
            tool_outputs[str(message.get("tool_call_id"))] = (idx, message)

    calls: list[ToolCallRecord] = []
    sequence_index = 0
    for idx, message in enumerate(traj):
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            raw_arguments = function.get("arguments", "")
            tool_call_id = tool_call.get("id")
            output_info = tool_outputs.get(str(tool_call_id))
            raw_output = output_info[1].get("content") if output_info else None
            calls.append(
                ToolCallRecord(
                    name=str(function.get("name", "")),
                    arguments=_parse_jsonish(raw_arguments),
                    raw_arguments=raw_arguments,
                    tool_call_id=tool_call_id,
                    output=_parse_jsonish(raw_output),
                    raw_output=raw_output,
                    message_index=idx,
                    output_message_index=output_info[0] if output_info else None,
                    sequence_index=sequence_index,
                )
            )
            sequence_index += 1
    return calls


def _compare_values(expected: Any, actual: Any, path: str = "") -> list[str]:
    label = path or "$"
    if type(expected) is not type(actual):
        return [f"{label}: expected {type(expected).__name__}, got {type(actual).__name__}"]

    if isinstance(expected, dict):
        diffs: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            next_path = f"{path}.{key}" if path else key
            if key not in expected:
                diffs.append(f"{next_path}: unexpected key")
                continue
            if key not in actual:
                diffs.append(f"{next_path}: missing key")
                continue
            diffs.extend(_compare_values(expected[key], actual[key], next_path))
        return diffs

    if isinstance(expected, list):
        diffs = []
        if len(expected) != len(actual):
            diffs.append(f"{label}: expected {len(expected)} items, got {len(actual)}")
        for idx, (exp_item, act_item) in enumerate(zip(expected, actual)):
            next_path = f"{path}[{idx}]" if path else f"[{idx}]"
            diffs.extend(_compare_values(exp_item, act_item, next_path))
        return diffs

    if expected != actual:
        return [f"{label}: expected {expected!r}, got {actual!r}"]
    return []


def match_expected_actions(
    expected_actions: list[dict[str, Any]],
    executed_calls: list[ToolCallRecord],
) -> tuple[list[ExpectedActionMatch], DeviationSummary | None]:
    matches: list[ExpectedActionMatch] = []
    search_start = 0

    for expected_index, expected_action in enumerate(expected_actions):
        expected_name = str(expected_action.get("name", ""))
        found_index: int | None = None
        for executed_index in range(search_start, len(executed_calls)):
            if executed_calls[executed_index].name == expected_name:
                found_index = executed_index
                break
        if found_index is None:
            first_extra = next(
                (call for call in executed_calls[search_start:] if call.name != "think"),
                None,
            )
            if first_extra is not None:
                detail = (
                    f"Expected `{expected_name}` next, but the run moved into "
                    f"`{first_extra.name}` instead."
                )
                return matches, DeviationSummary(
                    kind="missing_expected_action",
                    title=f"Expected `{expected_name}` never happened",
                    detail=detail,
                    highlighted_executed_index=first_extra.sequence_index,
                    highlighted_expected_index=expected_index,
                )
            return matches, DeviationSummary(
                kind="missing_expected_action",
                title=f"Expected `{expected_name}` never happened",
                detail="No later executed tool call matched the expected action.",
                highlighted_expected_index=expected_index,
            )

        executed_call = executed_calls[found_index]
        diffs = _compare_values(expected_action.get("kwargs"), executed_call.arguments)
        matches.append(
            ExpectedActionMatch(
                expected_index=expected_index,
                executed_index=found_index,
                name=expected_name,
                diffs=diffs,
                expected_args=expected_action.get("kwargs"),
                actual_args=executed_call.arguments,
            )
        )
        if diffs:
            detail = (
                f"`{expected_name}` ran, but its arguments diverged from the expected outcome."
            )
            return matches, DeviationSummary(
                kind="argument_mismatch",
                title=f"First deviation at `{expected_name}` arguments",
                detail=detail,
                highlighted_executed_index=executed_call.sequence_index,
                highlighted_expected_index=expected_index,
                diffs=diffs[:12],
            )
        search_start = found_index + 1

    return matches, None


def summarize_deviation(record: dict[str, Any]) -> tuple[list[ExpectedActionMatch], DeviationSummary]:
    executed_calls = extract_tool_calls(record.get("traj") or [])
    expected_actions = extract_expected_actions(record)
    matches, deviation = match_expected_actions(expected_actions, executed_calls)
    info = _as_dict(record.get("info"))

    if deviation is not None:
        return matches, deviation

    if info.get("error"):
        return matches, DeviationSummary(
            kind="runtime_error",
            title="Run raised an exception",
            detail=str(info["error"]),
        )

    if record.get("reward") == 1:
        return matches, DeviationSummary(
            kind="passed",
            title="Run matched the expected outcome",
            detail="No semantic deviation detected.",
        )

    if not expected_actions and executed_calls:
        first_real_call = next((call for call in executed_calls if call.name != "think"), None)
        if first_real_call is not None:
            return matches, DeviationSummary(
                kind="unexpected_action",
                title=f"Unexpected tool usage: `{first_real_call.name}`",
                detail="The expected outcome required no tool action, but the run still acted.",
                highlighted_executed_index=first_real_call.sequence_index,
            )

    last_real_call = next((call for call in reversed(executed_calls) if call.name != "think"), None)
    detail = (
        "Tool/action matching looks fine, so this likely failed on conversational policy, "
        "unsupported claims, or response framing. Inspect the highlighted terminal turn."
    )
    return matches, DeviationSummary(
        kind="conversation_mismatch",
        title="No tool mismatch found; likely conversational deviation",
        detail=detail,
        highlighted_executed_index=last_real_call.sequence_index if last_real_call else None,
    )


def build_turns(traj: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[list[tuple[int, dict[str, Any]]]]]:
    system_messages: list[dict[str, Any]] = []
    turns: list[list[tuple[int, dict[str, Any]]]] = []
    current_turn: list[tuple[int, dict[str, Any]]] = []

    for idx, message in enumerate(traj):
        role = message.get("role")
        if role == "system":
            system_messages.append(message)
            continue
        if role == "user":
            if current_turn:
                turns.append(current_turn)
            current_turn = [(idx, message)]
            continue
        if not current_turn:
            current_turn = []
        current_turn.append((idx, message))

    if current_turn:
        turns.append(current_turn)

    return system_messages, turns


def build_task_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        executed_calls = extract_tool_calls(record.get("traj") or [])
        expected_actions = extract_expected_actions(record)
        _, deviation = summarize_deviation(record)
        info = _as_dict(record.get("info"))
        task = _as_dict(info.get("task"))
        annotation = get_annotation(record)
        rows.append(
            {
                "row_index": idx,
                "task_id": record.get("task_id"),
                "reward": record.get("reward"),
                "source": info.get("source", "unknown"),
                "expected_count": len(expected_actions),
                "executed_count": len(executed_calls),
                "expected_tools": [action.get("name", "") for action in expected_actions],
                "executed_tools": [call.name for call in executed_calls if call.name != "think"],
                "deviation_title": deviation.title,
                "instruction": task.get("instruction") or (record.get("traj") or [{}])[0].get("content", ""),
                "annotation_preview": annotation.get("remarks") or annotation.get("comments", ""),
            }
        )
    return rows


def _badge(text: str, kind: str) -> str:
    color_map = {
        "good": "#0f9d58",
        "warn": "#b26a00",
        "bad": "#c23b22",
        "neutral": "#5b6575",
    }
    color = color_map.get(kind, color_map["neutral"])
    return (
        f"<span style='display:inline-block;padding:0.18rem 0.55rem;border-radius:999px;"
        f"background:{color}22;color:{color};font-size:0.82rem;font-weight:600;"
        f"margin-right:0.35rem'>{text}</span>"
    )


def render_app(default_results_dir: Path, default_results_file: str | None = None) -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit(
            "Streamlit is not installed. Run `uv sync` first, then "
            "`uv run streamlit run scripts/trace_debugger.py`."
        ) from exc

    st.set_page_config(page_title="Tau Trace Debugger", layout="wide")
    st.markdown(
        """
        <style>
          .summary-card {
            border: 1px solid rgba(120, 129, 149, 0.28);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            background: linear-gradient(180deg, rgba(248,250,252,0.85), rgba(255,255,255,0.98));
          }
          .turn-card {
            border: 1px solid rgba(120, 129, 149, 0.28);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
            background: #ffffff;
          }
          .turn-card.highlighted {
            border-color: rgba(194, 59, 34, 0.55);
            box-shadow: 0 0 0 2px rgba(194, 59, 34, 0.10);
          }
          .event-label {
            font-size: 0.84rem;
            font-weight: 700;
            color: #4b5563;
            margin-bottom: 0.4rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results-dir", default=str(default_results_dir))
    parser.add_argument("--results-file", default=default_results_file)
    parser.add_argument(
        "--normalized-dir",
        default=str(default_results_dir.parent / "data" / "live" / "normalized"),
    )
    args, _ = parser.parse_known_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    normalized_dir = Path(args.normalized_dir).expanduser().resolve()
    files = find_results_files(results_dir)
    if not files:
        st.error(f"No result JSON files found under `{results_dir}`.")
        return

    default_index = 0
    if args.results_file:
        requested = Path(args.results_file)
        for idx, candidate in enumerate(files):
            if candidate.name == requested.name or candidate == requested.resolve():
                default_index = idx
                break

    st.title("Tau Trace Debugger")
    st.caption(
        "Single-scroll debugging for reward mismatches: conversation turns, tool calls, "
        "tool outputs, and first semantic deviation in one place."
    )
    flash_message = st.session_state.pop("annotation_flash", None)
    if flash_message:
        st.success(str(flash_message))

    with st.sidebar:
        selected_file = st.selectbox(
            "Results file",
            files,
            index=default_index,
            format_func=lambda path: path.name,
        )
        show_failed_only = st.checkbox("Failed runs only", value=True)
        show_system = st.checkbox("Show system prompt", value=False)
        show_think = st.checkbox("Show think tool calls", value=False)
        show_raw_json = st.checkbox("Show full raw record", value=False)

    records = load_results_file(selected_file)
    normalized_traces = load_normalized_traces(normalized_dir)
    rows = build_task_rows(records)
    filtered_rows = [row for row in rows if (row["reward"] != 1 if show_failed_only else True)]
    if not filtered_rows:
        st.warning("No runs match the current filters.")
        return

    selected_row = st.selectbox(
        "Task to inspect",
        filtered_rows,
        format_func=lambda row: (
            f"task {row['task_id']} | reward={row['reward']} | {row['deviation_title']}"
        ),
    )
    record = records[selected_row["row_index"]]
    traj = record.get("traj") or []
    executed_calls = extract_tool_calls(traj)
    expected_actions = extract_expected_actions(record)
    matches, deviation = summarize_deviation(record)
    system_messages, turns = build_turns(traj)
    info = _as_dict(record.get("info"))
    task = _as_dict(info.get("task"))
    annotation = get_annotation(record)
    matched_trace = find_matching_trace(record, normalized_traces)
    trace_metadata = _as_dict(matched_trace.get("metadata")) if matched_trace else {}
    time_context = extract_time_context(system_messages)

    task_instruction = task.get("instruction", "")
    final_assistant = next(
        (
            message.get("content", "")
            for message in reversed(traj)
            if message.get("role") == "assistant" and message.get("content")
        ),
        "",
    )

    expected_names = [action.get("name", "") for action in expected_actions]
    executed_names = [call.name for call in executed_calls if show_think or call.name != "think"]

    status_badge = _badge("passed", "good") if record.get("reward") == 1 else _badge("failed", "bad")
    source_badge = _badge(f"source: {info.get('source', 'unknown')}", "neutral")
    trace_badge = _badge(
        f"trace: {matched_trace.get('trace_id', 'missing') if matched_trace else 'missing'}",
        "warn" if matched_trace else "neutral",
    )

    st.markdown(status_badge + source_badge + trace_badge, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([1.1, 1.1, 1.5, 1.1])
    with col1:
        st.markdown(
            (
                "<div class='summary-card'>"
                f"<div class='event-label'>Expected Outcome</div>"
                f"<div><strong>{len(expected_actions)}</strong> expected decisive action(s)</div>"
                f"<div style='margin-top:0.55rem'>{', '.join(expected_names) or 'None'}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            (
                "<div class='summary-card'>"
                f"<div class='event-label'>Executed Tools</div>"
                f"<div><strong>{len(executed_names)}</strong> tool call(s) shown</div>"
                f"<div style='margin-top:0.55rem'>{', '.join(executed_names) or 'None'}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            (
                "<div class='summary-card'>"
                f"<div class='event-label'>First Deviation</div>"
                f"<div><strong>{deviation.title}</strong></div>"
                f"<div style='margin-top:0.55rem'>{deviation.detail}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            (
                "<div class='summary-card'>"
                f"<div class='event-label'>Trace Context</div>"
                f"<div><strong>{matched_trace.get('trace_id', 'Missing') if matched_trace else 'Missing'}</strong></div>"
                f"<div style='margin-top:0.55rem'>{time_context or (matched_trace.get('started_at') if matched_trace else None) or 'No root time context found'}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    st.subheader("Task Context")
    st.write(task_instruction or "No task instruction found.")

    st.subheader("Annotations")
    with st.form("annotation-form", clear_on_submit=False):
        remarks = st.text_input("Remarks", value=str(annotation.get("remarks", "")))
        comments = st.text_area("Comments", value=str(annotation.get("comments", "")), height=140)
        save_col, clear_col = st.columns(2)
        save_clicked = save_col.form_submit_button("Save Annotation", use_container_width=True)
        clear_clicked = clear_col.form_submit_button("Clear Annotation", use_container_width=True)

    if save_clicked:
        set_annotation(record, remarks=remarks, comments=comments)
        write_results_file(selected_file, records)
        st.session_state["annotation_flash"] = "Annotation saved to the same results JSON file."
        st.rerun()
    elif clear_clicked:
        clear_annotation(record)
        write_results_file(selected_file, records)
        st.session_state["annotation_flash"] = "Annotation cleared from the results JSON file."
        st.rerun()

    if annotation:
        st.caption(f"Last updated: {annotation.get('updated_at', 'unknown')}")

    st.subheader("Trace / Phoenix Context")
    meta_col1, meta_col2, meta_col3 = st.columns(3)
    with meta_col1:
        st.markdown(f"**Trace ID**  \n`{matched_trace.get('trace_id') if matched_trace else 'missing'}`")
        st.markdown(f"**Task ID / Trial**  \n`{record.get('task_id')}` / `{record.get('trial', 0)}`")
        st.markdown(f"**Run Source**  \n`{info.get('source', 'unknown')}`")
    with meta_col2:
        st.markdown(f"**Task Split**  \n`{trace_metadata.get('task_split', 'unknown')}`")
        st.markdown(f"**Agent Strategy**  \n`{trace_metadata.get('agent_strategy', 'unknown')}`")
        st.markdown(f"**Clock Context**  \n`{time_context or 'not found'}`")
    with meta_col3:
        st.markdown(f"**Model**  \n`{trace_metadata.get('model', 'unknown')}`")
        st.markdown(f"**User Model**  \n`{trace_metadata.get('user_model', 'unknown')}`")
        st.markdown(f"**Started / Ended**  \n`{matched_trace.get('started_at', 'unknown') if matched_trace else 'unknown'}`  \n`{matched_trace.get('ended_at', 'unknown') if matched_trace else 'unknown'}`")

    with st.expander("Prompt and root trace data", expanded=False):
        if system_messages:
            st.markdown("**System prompt**")
            for message in system_messages:
                st.code(message.get("content", ""), language="markdown")
        if matched_trace:
            st.markdown("**Trace root user input**")
            st.write(matched_trace.get("user_input") or "None")
            st.markdown("**Trace metadata**")
            st.code(_json_text(trace_metadata), language="json")
            if matched_trace.get("source_metadata") is not None:
                st.markdown("**Source metadata**")
                st.code(_json_text(matched_trace.get("source_metadata")), language="json")
            warnings = _as_list(matched_trace.get("validation_warnings"))
            if warnings:
                st.markdown("**Validation warnings**")
                for warning in warnings:
                    st.code(str(warning), language="text")
        else:
            st.caption(f"No matching normalized trace found under `{normalized_dir}`.")

    if deviation.diffs:
        st.subheader("Argument Diffs")
        for diff in deviation.diffs:
            st.code(diff, language="text")

    if matches:
        with st.expander("Expected vs executed decisive actions", expanded=False):
            for match in matches:
                left, right = st.columns(2)
                with left:
                    st.caption(f"Expected action {match.expected_index + 1}: `{match.name}`")
                    st.code(_json_text(match.expected_args), language="json")
                with right:
                    st.caption(f"Executed call {match.executed_index + 1}: `{match.name}`")
                    st.code(_json_text(match.actual_args), language="json")
                if match.diffs:
                    for diff in match.diffs[:12]:
                        st.code(diff, language="text")

    if show_system and system_messages:
        with st.expander("System prompt", expanded=False):
            for message in system_messages:
                st.code(message.get("content", ""), language="markdown")

    st.subheader("Timeline")
    if final_assistant:
        with st.expander("Final assistant response", expanded=False):
            st.write(final_assistant)

    highlighted_message_indexes = {
        executed_calls[deviation.highlighted_executed_index].message_index
        for _ in [0]
        if deviation.highlighted_executed_index is not None
        and deviation.highlighted_executed_index < len(executed_calls)
    }

    for turn_number, turn in enumerate(turns, start=1):
        is_highlighted = any(message_index in highlighted_message_indexes for message_index, _ in turn)
        card_class = "turn-card highlighted" if is_highlighted else "turn-card"
        st.markdown(f"<div class='{card_class}'>", unsafe_allow_html=True)
        st.markdown(f"### Turn {turn_number}")

        for message_index, message in turn:
            role = message.get("role", "unknown")
            if role == "assistant" and message.get("tool_calls"):
                for tool_call in message.get("tool_calls") or []:
                    function = tool_call.get("function") or {}
                    name = str(function.get("name", ""))
                    if name == "think" and not show_think:
                        continue
                    raw_arguments = function.get("arguments", "")
                    paired_output = next(
                        (
                            call
                            for call in executed_calls
                            if call.tool_call_id == tool_call.get("id")
                        ),
                        None,
                    )
                    is_tool_highlighted = (
                        paired_output is not None
                        and deviation.highlighted_executed_index == paired_output.sequence_index
                    )
                    label = f"Tool call: `{name}`"
                    if is_tool_highlighted:
                        label += "  <- first deviation"
                    st.markdown(f"**{label}**")
                    st.code(_json_text(_parse_jsonish(raw_arguments)), language="json")
                    if paired_output is not None:
                        st.caption("Tool output")
                        st.code(_json_text(paired_output.output), language="json")
                continue

            if role == "tool":
                continue

            content = message.get("content", "")
            label = role.capitalize()
            if role == "assistant" and message_index in highlighted_message_indexes:
                label += "  <- inspect this turn"
            st.markdown(f"**{label}**")
            if content:
                st.write(content)
            else:
                st.caption("No visible text content.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Run Index")
    st.dataframe(
        [
            {
                "task_id": row["task_id"],
                "reward": row["reward"],
                "expected": " -> ".join(row["expected_tools"]) or "None",
                "executed": " -> ".join(row["executed_tools"]) or "None",
                "deviation": row["deviation_title"],
                "annotation": _preview_text(row["annotation_preview"], limit=60),
                "instruction": _preview_text(row["instruction"], limit=120),
            }
            for row in filtered_rows
        ],
        hide_index=True,
        width="stretch",
    )

    if show_raw_json:
        with st.expander("Raw run record", expanded=False):
            st.code(_json_text(record), language="json")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    default_results_dir = repo_root / "results"
    render_app(default_results_dir=default_results_dir)


if __name__ == "__main__":
    main()
