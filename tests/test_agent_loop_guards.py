"""Loop guard, budget warning and force-out summary behaviour (issue #66)."""

from __future__ import annotations

from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from bioagents.graph import (
    _detect_consecutive_duplicate_calls,
    _summarize_agent_progress,
    agent_node,
)
from bioagents.limits import (
    DISCOVERY_TOOL_NAMES,
    MAX_AGENT_TOOL_ROUNDS,
    MAX_CONSECUTIVE_IDENTICAL_DISCOVERY_CALLS,
    MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS,
)

TASK = HumanMessage(content="[SUPERVISOR TASK] find a tool that computes RMSD")


def _call(agent: str, tool: str, args: dict, call_id: str) -> AIMessage:
    msg = AIMessage(content="", name=agent)
    msg.tool_calls = [{"name": tool, "args": args, "id": call_id}]
    return msg


def _repeated(agent: str, tool: str, args: dict, times: int) -> list:
    messages: list = [TASK]
    for i in range(times):
        messages.append(_call(agent, tool, args, f"c{i}"))
        messages.append(ToolMessage(content=f"result {i}", tool_call_id=f"c{i}", name=tool))
    return messages


def test_discovery_tools_cover_the_real_search_tool_names() -> None:
    for name in ("tool_universe_find_tools", "search_custom_tools", "list_custom_tools"):
        assert name in DISCOVERY_TOOL_NAMES


def test_discovery_threshold_is_higher_than_default() -> None:
    assert MAX_CONSECUTIVE_IDENTICAL_DISCOVERY_CALLS > MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS


def test_repeated_identical_discovery_search_is_not_a_loop() -> None:
    messages = _repeated(
        "ToolBuilder",
        "tool_universe_find_tools",
        {"description": "rmsd"},
        MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS,
    )

    is_loop, _ = _detect_consecutive_duplicate_calls(messages, "ToolBuilder")

    assert not is_loop


def test_discovery_search_still_breaks_at_its_own_threshold() -> None:
    messages = _repeated(
        "ToolBuilder",
        "search_custom_tools",
        {"query": "rmsd"},
        MAX_CONSECUTIVE_IDENTICAL_DISCOVERY_CALLS,
    )

    is_loop, desc = _detect_consecutive_duplicate_calls(messages, "ToolBuilder")

    assert is_loop
    assert "search_custom_tools" in desc


def test_non_discovery_identical_calls_still_trip_the_default_guard() -> None:
    messages = _repeated(
        "Shell",
        "write_local_file",
        {"file_path": "a.py"},
        MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS,
    )

    is_loop, _ = _detect_consecutive_duplicate_calls(messages, "Shell")

    assert is_loop


def test_progress_summary_lists_tools_queries_files_and_results() -> None:
    messages = [
        HumanMessage(content="[SUPERVISOR TASK] an older task"),
        _call("ToolBuilder", "write_local_file", {"file_path": "old.py"}, "old"),
        TASK,
        _call("ToolBuilder", "tool_universe_find_tools", {"description": "rmsd calc"}, "a"),
        ToolMessage(content="Found: MDAnalysis_rmsd", tool_call_id="a", name="find"),
        _call("ToolBuilder", "write_local_file", {"file_path": "rmsd.py"}, "b"),
        ToolMessage(content="wrote rmsd.py", tool_call_id="b", name="write_local_file"),
        _call("Research", "write_local_file", {"file_path": "other.py"}, "x"),
    ]

    summary = _summarize_agent_progress(messages, "ToolBuilder")

    assert "[PROGRESS SUMMARY]" in summary
    assert "tool_universe_find_tools x1" in summary
    assert "rmsd calc" in summary
    assert "rmsd.py" in summary
    assert "MDAnalysis_rmsd" in summary
    assert "old.py" not in summary
    assert "other.py" not in summary


def test_progress_summary_is_empty_without_tool_calls() -> None:
    assert _summarize_agent_progress([TASK], "ToolBuilder") == ""


def test_budget_force_out_carries_the_progress_summary() -> None:
    messages: list = [TASK]
    for i in range(MAX_AGENT_TOOL_ROUNDS):
        messages.append(_call("ToolBuilder", "write_local_file", {"file_path": f"f{i}.py"}, str(i)))
        messages.append(ToolMessage(content=f"wrote f{i}.py", tool_call_id=str(i), name="w"))
    agent = Mock()

    result = agent_node({"messages": messages}, agent=agent, name="ToolBuilder")

    agent.assert_not_called()
    content = result["messages"][0].content
    assert "[MAX_TOOL_ROUNDS]" in content
    assert "[PROGRESS SUMMARY]" in content
    assert "f0.py" in content


def test_loop_force_out_carries_the_progress_summary() -> None:
    messages = _repeated(
        "Shell", "write_local_file", {"file_path": "a.py"}, MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS
    )

    result = agent_node({"messages": messages}, agent=Mock(), name="Shell")

    content = result["messages"][0].content
    assert "[LOOP_DETECTED]" in content
    assert "[PROGRESS SUMMARY]" in content


def _rounds(agent: str, n: int) -> list:
    messages: list = [TASK]
    for i in range(n):
        messages.append(_call(agent, "write_local_file", {"file_path": f"f{i}.py"}, str(i)))
        messages.append(ToolMessage(content="ok", tool_call_id=str(i), name="w"))
    return messages


def test_agent_gets_wrap_up_notice_one_round_before_the_cut() -> None:
    agent = Mock(return_value={"messages": [AIMessage(content="findings")]})
    messages = _rounds("DataAcquisition", MAX_AGENT_TOOL_ROUNDS - 1)

    agent_node({"messages": messages}, agent=agent, name="DataAcquisition")

    seen = agent.call_args.args[0]["messages"]
    assert "[TOOL_BUDGET_WARNING]" in seen[-1].content
    # The notice is for this call only; graph state is untouched.
    assert len(messages) == 1 + 2 * (MAX_AGENT_TOOL_ROUNDS - 1)


def test_no_wrap_up_notice_early_in_the_task() -> None:
    agent = Mock(return_value={"messages": [AIMessage(content="findings")]})
    messages = _rounds("DataAcquisition", 1)

    agent_node({"messages": messages}, agent=agent, name="DataAcquisition")

    seen = agent.call_args.args[0]["messages"]
    assert not any("[TOOL_BUDGET_WARNING]" in str(m.content) for m in seen)
