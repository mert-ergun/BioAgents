"""LangGraph multi-agent workflow definition."""

from functools import partial
from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from bioagents.agents.analysis_agent import create_analysis_agent
from bioagents.agents.report_agent import create_report_agent
from bioagents.agents.research_agent import create_research_agent
from bioagents.agents.supervisor_agent import create_supervisor_agent
from bioagents.tools.analysis_tools import (
    analyze_amino_acid_composition,
    calculate_isoelectric_point,
    calculate_molecular_weight,
)
from bioagents.tools.proteomics_tools import fetch_uniprot_fasta
from bioagents.tools.tool_universe import tool_universe_call_tool, tool_universe_find_tools

from bioagents.tools.pdf_tools import (
    fetch_webpage_as_pdf_text,
    extract_pdf_text_spacy_layout,
)

class AgentState(dict):
    """
    The state object passed between nodes in the graph.

    Attributes:
        messages: List of messages in the conversation
        next: The next agent to route to (set by supervisor)
        reasoning: The reasoning behind supervisor's decision
    """

    messages: Annotated[list[BaseMessage], add_messages]
    next: str
    reasoning: str


def agent_node(state, agent, name):
    """
    Wrapper for agent nodes that adds agent identification.

    Args:
        state: The current state
        agent: The agent function to call
        name: Name of the agent for tracking

    Returns:
        Updated state with agent's response
    """
    result = agent(state)

    # Add agent name to the message metadata for tracking
    if result["messages"]:
        for msg in result["messages"]:
            msg.name = name

    return result


def should_continue_to_tools(state: AgentState) -> Literal["tools", "supervisor"]:
    """
    Conditional edge that checks if tools need to be called.

    Args:
        state: The current agent state

    Returns:
        'tools' if last message has tool calls, 'supervisor' otherwise
    """
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "supervisor"


def route_supervisor(state: AgentState) -> Literal["research", "analysis", "report", "end"]:
    """
    Route based on supervisor's decision.

    Args:
        state: The current agent state

    Returns:
        The next agent to route to, or 'end' if finished
    """
    next_agent: Literal["research", "analysis", "report", "end", "FINISH"] = state.get(
        "next", "FINISH"
    )
    return "end" if next_agent == "FINISH" else next_agent


def create_graph():
    """
    Create and compile the multi-agent LangGraph workflow.

    The workflow uses a supervisor pattern where:
    1. Supervisor routes tasks to specialized agents
    2. Each agent can use tools and return to supervisor
    3. Workflow continues until supervisor says FINISH

    Returns:
        A compiled StateGraph ready for execution
    """
    research_tools = [
        fetch_uniprot_fasta,
        tool_universe_find_tools,
        tool_universe_call_tool,
        fetch_webpage_as_pdf_text,           # <- NEW
        extract_pdf_text_spacy_layout,       # <- NEW
    ]
    analysis_tools = [
        calculate_molecular_weight,
        analyze_amino_acid_composition,
        calculate_isoelectric_point,
    ]

    research_agent = create_research_agent(research_tools)
    analysis_agent = create_analysis_agent(analysis_tools)
    report_agent = create_report_agent()

    members = ["research", "analysis", "report"]
    supervisor_agent = create_supervisor_agent(members)

    research_tool_node = ToolNode(research_tools)
    analysis_tool_node = ToolNode(analysis_tools)

    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("research", partial(agent_node, agent=research_agent, name="Research"))
    workflow.add_node("analysis", partial(agent_node, agent=analysis_agent, name="Analysis"))
    workflow.add_node("report", partial(agent_node, agent=report_agent, name="Report"))

    workflow.add_node("research_tools", research_tool_node)
    workflow.add_node("analysis_tools", analysis_tool_node)

    workflow.set_entry_point("supervisor")

    # Add edges from supervisor to agents
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "research": "research",
            "analysis": "analysis",
            "report": "report",
            "end": END,
        },
    )

    # Research agent can use tools or go back to supervisor
    workflow.add_conditional_edges(
        "research",
        should_continue_to_tools,
        {
            "tools": "research_tools",
            "supervisor": "supervisor",
        },
    )

    # Analysis agent can use tools or go back to supervisor
    workflow.add_conditional_edges(
        "analysis",
        should_continue_to_tools,
        {
            "tools": "analysis_tools",
            "supervisor": "supervisor",
        },
    )

    workflow.add_edge("research_tools", "research")
    workflow.add_edge("analysis_tools", "analysis")

    workflow.add_edge("report", "supervisor")

    return workflow.compile()
