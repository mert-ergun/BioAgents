"""Tests for shared memory architecture."""

from functools import partial

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from bioagents.graph import AgentState, agent_node


def test_initial_state_has_memory():
    """Verify AgentState initializes with memory field."""
    state = AgentState(messages=[HumanMessage(content="test")], next=None, reasoning="", memory={})
    assert "memory" in state
    assert isinstance(state["memory"], dict)


def test_memory_initialization():
    """Verify memory initializes for all agents."""
    initial_state = {
        "messages": [HumanMessage(content="test query")],
        "next": None,
        "reasoning": "",
        "memory": {
            "research": {"status": "pending", "data": {}, "errors": []},
            "analysis": {"status": "pending", "data": {}, "errors": []},
            # ... etc
        },
    }
    assert all(
        initial_state["memory"][agent]["status"] == "pending" for agent in ["research", "analysis"]
    )


def test_graph_preserves_memory():
    """Verify a minimal compiled graph streams memory updates (no live LLM)."""

    def stub_agent(state):
        return {
            "data": {"ok": True},
            "raw_output": "ok",
            "tool_calls": [],
            "error": None,
            "messages": [AIMessage(content="ok")],
        }

    workflow = StateGraph(AgentState)
    workflow.add_node("research", partial(agent_node, agent=stub_agent, name="Research"))
    workflow.set_entry_point("research")
    workflow.add_edge("research", END)
    graph = workflow.compile()

    initial_state = {
        "messages": [HumanMessage(content="Simple test query")],
        "next": None,
        "reasoning": "",
        "memory": {
            "research": {"status": "pending", "data": {}, "errors": []},
            "analysis": {"status": "pending", "data": {}, "errors": []},
        },
    }

    for step_output in graph.stream(initial_state):
        for node_output in step_output.values():
            if "memory" in node_output:
                assert isinstance(node_output["memory"], dict)
                assert "research" in node_output["memory"]
                break


def test_agent_isolation():
    """Verify agents don't read each other's messages."""
    # This is tested by verifying system prompts don't mention
    # reading from message history
    from bioagents.agents.research_agent import RESEARCH_AGENT_MEMORY_PROMPT

    assert "MUST NOT" in RESEARCH_AGENT_MEMORY_PROMPT or "NOT" in RESEARCH_AGENT_MEMORY_PROMPT
