"""The supervisor must not re-delegate work that is already finished (issue #69)."""

from __future__ import annotations

from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from bioagents.agents import supervisor_agent
from bioagents.agents.supervisor_completion import (
    check_completed_work_guard,
    current_turn_start,
    find_completed_subtasks,
    find_duplicate_of_completed,
    has_completion_marker,
)
from bioagents.sandbox.coder_helpers import COMPLETION_MARKER

MEMBERS = ["shell", "analysis", "coder", "report", "research"]
QUERY = HumanMessage(content="Rank the 10 molecules and save final_ranked_results.csv")
TASK = "Consolidate the ranking into final_ranked_results.csv and final_ranked_results.json"


def _task(text: str) -> HumanMessage:
    return HumanMessage(content=f"[SUPERVISOR TASK] {text}", name="Supervisor")


def _tool_round(agent: str, call_id: str) -> list:
    call = AIMessage(content="", name=agent)
    call.tool_calls = [
        {"name": "write_local_file", "args": {"file_path": "final.csv"}, "id": call_id}
    ]
    return [call, ToolMessage(content="ok", tool_call_id=call_id, name="write_local_file")]


def _finished_shell_run() -> list:
    return [
        QUERY,
        _task(TASK),
        *_tool_round("Shell", "1"),
        AIMessage(
            content="Task complete. Wrote final_ranked_results.csv and "
            "final_ranked_results.json (JSON valid, 10 entries).",
            name="Shell",
        ),
    ]


# --- completed-subtask detection -------------------------------------------------


def test_agent_that_reports_completion_finishes_its_subtask() -> None:
    done = find_completed_subtasks(_finished_shell_run())

    assert [(d.task, d.agent) for d in done] == [(TASK, "Shell")]


def test_reply_without_tool_calls_counts_as_completed() -> None:
    messages = [QUERY, _task("Summarise the ranking"), AIMessage(content="Top hit: M3", name="X")]

    assert find_completed_subtasks(messages)


def test_forced_return_is_not_a_completed_subtask() -> None:
    messages = [
        QUERY,
        _task(TASK),
        *_tool_round("Shell", "1"),
        AIMessage(content="[MAX_TOOL_ROUNDS] Agent 'Shell' used its budget", name="Shell"),
    ]

    assert find_completed_subtasks(messages) == []


def test_reported_failure_is_not_a_completed_subtask() -> None:
    messages = [
        QUERY,
        _task(TASK),
        *_tool_round("Shell", "1"),
        AIMessage(content="I was unable to write the file: permission denied.", name="Shell"),
    ]

    assert find_completed_subtasks(messages) == []


def test_tool_work_without_a_completion_claim_is_not_assumed_done() -> None:
    messages = [
        QUERY,
        _task(TASK),
        *_tool_round("Shell", "1"),
        AIMessage(content="Here are intermediate values.", name="Shell"),
    ]

    assert find_completed_subtasks(messages) == []


def test_reworded_task_is_recognised_as_a_duplicate() -> None:
    reworded = TASK.replace("Consolidate", "consolidate") + "."

    assert find_duplicate_of_completed(_finished_shell_run(), reworded) is not None


def test_different_task_is_not_a_duplicate() -> None:
    other = "Dock the top 3 molecules into 4GV1 and report binding energies"

    assert find_duplicate_of_completed(_finished_shell_run(), other) is None


def test_new_user_turn_resets_completion_evidence() -> None:
    messages = [*_finished_shell_run(), HumanMessage(content="Now redo it with 20 molecules")]

    assert current_turn_start(messages) == len(messages) - 1
    assert find_completed_subtasks(messages) == []


# --- explicit completion marker -----------------------------------------------------


def _marked_run() -> list:
    return [
        QUERY,
        _task(TASK),
        AIMessage(content=f"Saved both files.\n\n{COMPLETION_MARKER} done", name="Coder"),
    ]


def test_completion_marker_is_detected_in_the_current_turn() -> None:
    assert has_completion_marker(_marked_run())
    assert not has_completion_marker([*_marked_run(), HumanMessage(content="new request")])


def test_marker_redirects_work_agents_to_a_read_only_report() -> None:
    route, _, task = check_completed_work_guard(_marked_run(), "analysis", "re-check", MEMBERS)

    assert route == "report"
    assert "read-only" in task
    assert "do not re-run" in task


def test_marker_allows_report_once_then_finishes() -> None:
    messages = _marked_run()
    assert check_completed_work_guard(messages, "report", "write up", MEMBERS)[0] is None

    messages.append(AIMessage(content="## Report\nAll done.", name="Report"))
    assert check_completed_work_guard(messages, "report", "write up", MEMBERS)[0] == "FINISH"
    assert check_completed_work_guard(messages, "coder", "redo", MEMBERS)[0] == "FINISH"


def test_marker_without_report_member_finishes() -> None:
    route, _, _ = check_completed_work_guard(_marked_run(), "coder", "x", ["coder"])

    assert route == "FINISH"


def test_guard_leaves_unrelated_routing_alone() -> None:
    messages = [QUERY, _task("Fetch SMILES for the 10 molecules")]

    assert check_completed_work_guard(messages, "research", TASK, MEMBERS) == (None, "", "")
    assert check_completed_work_guard(_finished_shell_run(), "FINISH", "", MEMBERS)[0] is None


# --- supervisor integration ---------------------------------------------------------


def _supervisor(monkeypatch, route: supervisor_agent.RouteResponse):
    chain = Mock()
    chain.invoke.return_value = route
    prompt = Mock()
    prompt.__or__ = Mock(return_value=chain)
    monkeypatch.setattr(supervisor_agent, "get_llm", Mock())
    monkeypatch.setattr(supervisor_agent, "_agent_tool_inventory", lambda: "")
    monkeypatch.setattr(supervisor_agent, "describe_workspace_artifacts", lambda: "")
    with patch.object(supervisor_agent.ChatPromptTemplate, "from_messages", return_value=prompt):
        return supervisor_agent.create_supervisor_agent(MEMBERS)


def test_supervisor_does_not_reissue_a_completed_subtask(monkeypatch) -> None:
    node = _supervisor(
        monkeypatch,
        supervisor_agent.RouteResponse(
            next_agent="analysis", reasoning="double-check", task_for_agent=TASK
        ),
    )

    result = node({"messages": _finished_shell_run()})

    assert result["next"] == "FINISH"
    assert not any("[SUPERVISOR TASK]" in str(m.content) for m in result["messages"])


def test_supervisor_still_delegates_new_work(monkeypatch) -> None:
    node = _supervisor(
        monkeypatch,
        supervisor_agent.RouteResponse(
            next_agent="analysis",
            reasoning="next step",
            task_for_agent="Plot the score distribution of the ranked molecules",
        ),
    )

    result = node({"messages": _finished_shell_run()})

    assert result["next"] == "analysis"
