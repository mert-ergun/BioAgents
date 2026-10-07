"""Guards the per-agent tool-round budget.

Regression context: in benchmark run task1_undirected the DataAcquisition agent looped
`agent -> tools -> agent` 23 times, calling write_local_file 21 times to hand-roll work
it had no execution tool for. It never returned to the supervisor, so none of the
supervisor's loop detection ran and the workflow died at the 50-step cap. The budget
below forces control back so the supervisor can re-route.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from bioagents.graph import AGENT_DISPLAY_NAMES, agent_node, count_agent_tool_rounds
from bioagents.limits import MAX_AGENT_TOOL_ROUNDS


def _tool_round(agent_name: str, call_index: int) -> AIMessage:
    msg = AIMessage(content="", name=agent_name)
    msg.tool_calls = [
        {
            "name": "write_local_file",
            "args": {"file_path": f"f{call_index}.py"},
            "id": str(call_index),
        }
    ]
    return msg


def _history(agent_name: str, rounds: int) -> list:
    messages: list = [
        HumanMessage(content="[SUPERVISOR TASK] download 4gv1 and extract the ligand")
    ]
    messages.extend(_tool_round(agent_name, i) for i in range(rounds))
    return messages


def test_counts_only_the_named_agents_rounds() -> None:
    messages = _history("DataAcquisition", 3)
    messages.append(_tool_round("Research", 99))

    assert count_agent_tool_rounds(messages, "DataAcquisition") == 3
    assert count_agent_tool_rounds(messages, "Research") == 1


def test_budget_resets_on_each_new_supervisor_task() -> None:
    messages = _history("DataAcquisition", 5)
    messages.append(HumanMessage(content="[SUPERVISOR TASK] now summarise what you found"))
    messages.append(_tool_round("DataAcquisition", 100))

    assert count_agent_tool_rounds(messages, "DataAcquisition") == 1


def test_agent_runs_while_within_budget() -> None:
    agent = Mock(return_value={"messages": [AIMessage(content="done")]})
    state = {"messages": _history("DataAcquisition", MAX_AGENT_TOOL_ROUNDS - 1)}

    agent_node(state, agent=agent, name="DataAcquisition")

    agent.assert_called_once()


def test_agent_is_cut_off_once_budget_is_exhausted() -> None:
    agent = Mock(return_value={"messages": [AIMessage(content="should not run")]})
    state = {"messages": _history("DataAcquisition", MAX_AGENT_TOOL_ROUNDS)}

    result = agent_node(state, agent=agent, name="DataAcquisition")

    agent.assert_not_called()
    content = result["messages"][0].content
    assert "[MAX_TOOL_ROUNDS]" in content
    assert "Do not re-delegate the same task" in content


def test_cutoff_message_has_no_dangling_tool_calls() -> None:
    """A cut-off must not leave tool_calls without matching results; APIs reject that."""
    agent = Mock()
    state = {"messages": _history("DataAcquisition", MAX_AGENT_TOOL_ROUNDS + 10)}

    result = agent_node(state, agent=agent, name="DataAcquisition")

    assert not getattr(result["messages"][0], "tool_calls", None)


@pytest.mark.parametrize("node_name,display_name", sorted(AGENT_DISPLAY_NAMES.items()))
def test_display_names_match_what_agent_node_stamps(node_name: str, display_name: str) -> None:
    """The budget matches on display name, so the map must stay in sync with the nodes."""
    agent = Mock(return_value={"messages": [AIMessage(content="ok")]})
    result = agent_node({"messages": []}, agent=agent, name=display_name)

    assert result["messages"][0].name == display_name, node_name
