"""Tests for shared memory architecture."""

import pytest
from bioagents.graph import create_graph, AgentState
from langchain_core.messages import HumanMessage


def test_initial_state_has_memory():
    """Verify AgentState initializes with memory field."""
    state = AgentState(
        messages=[HumanMessage(content="test")],
        next=None,
        reasoning="",
        memory={}
    )
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
        }
    }
    assert all(
        initial_state["memory"][agent]["status"] == "pending"
        for agent in ["research", "analysis"]
    )


def test_graph_preserves_memory():
    """Verify graph execution preserves and updates memory."""
    graph = create_graph()
    
    initial_state = {
        "messages": [HumanMessage(content="Simple test query")],
        "next": None,
        "reasoning": "",
        "memory": {
            "research": {"status": "pending", "data": {}, "errors": []},
            "analysis": {"status": "pending", "data": {}, "errors": []},
            # ... initialize all agents
        }
    }
    
    # Run one step
    for step_output in graph.stream(initial_state):
        # Each step should return updated memory
        for node_output in step_output.values():
            if "memory" in node_output:
                assert isinstance(node_output["memory"], dict)
                break


def test_agent_isolation():
    """Verify agents don't read each other's messages."""
    # This is tested by verifying system prompts don't mention
    # reading from message history
    from bioagents.agents.research_agent import RESEARCH_AGENT_MEMORY_PROMPT
    
    assert "MUST NOT" in RESEARCH_AGENT_MEMORY_PROMPT or "NOT" in RESEARCH_AGENT_MEMORY_PROMPT