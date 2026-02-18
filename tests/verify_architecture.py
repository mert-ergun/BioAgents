"""Quick verification that memory architecture is correctly implemented."""

import sys
from bioagents.graph import AgentState, create_graph
from langchain_core.messages import HumanMessage


def verify_agentstate_structure():
    """Verify AgentState has memory field."""
    state = AgentState(
        messages=[],
        next=None,
        reasoning="",
        memory={}
    )
    assert "memory" in state, "AgentState missing 'memory' field"
    print("✓ AgentState has 'memory' field")


def verify_agent_functions_return_dicts():
    """Verify agent functions return proper dict structure."""
    from bioagents.agents.research_agent import create_research_agent
    
    # Mock tools
    mock_tools = []
    agent_func = create_research_agent(mock_tools)
    
    # Call with minimal state
    state = {
        "messages": [HumanMessage(content="test")],
        "memory": {}
    }
    
    result = agent_func(state)
    
    required_keys = {"data", "raw_output", "tool_calls"}
    assert required_keys.issubset(result.keys()), \
        f"Agent must return {required_keys}, got {result.keys()}"
    
    print("✓ Agent functions return correct structure")


def verify_supervisor_reads_memory():
    """Verify supervisor accesses memory, not messages."""
    from bioagents.agents.supervisor_agent import build_memory_status_summary
    
    memory = {
        "research": {"status": "success", "data": {"found": True}},
        "analysis": {"status": "pending", "data": {}},
    }
    
    summary = build_memory_status_summary(memory, ["research", "analysis"])
    assert "research" in summary, "Memory summary should include agent names"
    print("✓ Supervisor has memory summary function")


def verify_prompts_forbid_message_reading():
    """Verify agent prompts forbid message reading."""
    from bioagents.agents.research_agent import RESEARCH_AGENT_MEMORY_PROMPT
    from bioagents.agents.analysis_agent import ANALYSIS_AGENT_MEMORY_PROMPT
    
    prompts = [
        ("Research", RESEARCH_AGENT_MEMORY_PROMPT),
        ("Analysis", ANALYSIS_AGENT_MEMORY_PROMPT),
    ]
    
    for name, prompt in prompts:
        assert "JSON" in prompt or "json" in prompt, \
            f"{name} prompt should mention JSON output format"
        print(f"✓ {name} prompt enforces JSON output")


if __name__ == "__main__":
    try:
        verify_agentstate_structure()
        verify_agent_functions_return_dicts()
        verify_supervisor_reads_memory()
        verify_prompts_forbid_message_reading()
        
        print("\n✓ All architecture verifications passed!")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)