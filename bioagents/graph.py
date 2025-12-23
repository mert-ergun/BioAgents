"""LangGraph multi-agent workflow definition."""

from functools import partial
from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from bioagents.agents.analysis_agent import create_analysis_agent
from bioagents.agents.coder_agent import create_coder_agent, create_coder_node
from bioagents.agents.protein_design_agent import (
    create_protein_design_agent,
    get_all_protein_design_tools,
)
from bioagents.agents.report_agent import create_report_agent
from bioagents.agents.research_agent import create_research_agent
from bioagents.agents.supervisor_agent import create_supervisor_agent
from bioagents.agents.tool_builder_agent import create_tool_builder_agent
from bioagents.tools.analysis_tools import (
    analyze_amino_acid_composition,
    calculate_isoelectric_point,
    calculate_molecular_weight,
)
from bioagents.tools.proteomics_tools import fetch_uniprot_fasta
from bioagents.tools.tool_builder_tools import get_tool_builder_tools
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
        output_dir: Directory path for saving output files (optional)
    """

    messages: Annotated[list[BaseMessage], add_messages]
    next: str
    reasoning: str
    output_dir: str | None = None


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
    if result.get("messages"):
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


def route_supervisor(
    state: AgentState,
) -> Literal["research", "analysis", "coder", "report", "tool_builder", "protein_design", "end"]:
    """
    Route based on supervisor's decision.

    Args:
        state: The current agent state

    Returns:
        The next agent to route to, or 'end' if finished
    """
    next_agent: Literal[
        "research", "analysis", "coder", "report", "tool_builder", "protein_design", "end", "FINISH"
    ] = state.get("next", "FINISH")
    return "end" if next_agent == "FINISH" else next_agent


def create_graph():
    """
    Create and compile the multi-agent LangGraph workflow.

    The workflow uses a supervisor pattern where:
    1. Supervisor routes tasks to specialized agents
    2. Each agent can use tools and return to supervisor
    3. Workflow continues until supervisor says FINISH
    4. Tool Builder can create new tools when existing ones are insufficient

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
    tool_builder_tools = get_tool_builder_tools()
    protein_design_tools = get_all_protein_design_tools()

    research_agent = create_research_agent(research_tools)
    analysis_agent = create_analysis_agent(analysis_tools)
    report_agent = create_report_agent()
    coder_agent = create_coder_agent()
    coder_node_func = create_coder_node(coder_agent)
    tool_builder_agent = create_tool_builder_agent()
    protein_design_agent = create_protein_design_agent()

    members = ["research", "analysis", "coder", "report", "tool_builder", "protein_design"]
    supervisor_agent = create_supervisor_agent(members)

    research_tool_node = ToolNode(research_tools)
    analysis_tool_node = ToolNode(analysis_tools)
    tool_builder_tool_node = ToolNode(tool_builder_tools)
    protein_design_tool_node = ToolNode(protein_design_tools)

    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("research", partial(agent_node, agent=research_agent, name="Research"))
    workflow.add_node("analysis", partial(agent_node, agent=analysis_agent, name="Analysis"))
    workflow.add_node("coder", partial(agent_node, agent=coder_node_func, name="Coder"))
    workflow.add_node("report", partial(agent_node, agent=report_agent, name="Report"))
    workflow.add_node(
        "tool_builder", partial(agent_node, agent=tool_builder_agent, name="ToolBuilder")
    )
    workflow.add_node(
        "protein_design", partial(agent_node, agent=protein_design_agent, name="ProteinDesign")
    )

    workflow.add_node("research_tools", research_tool_node)
    workflow.add_node("analysis_tools", analysis_tool_node)
    workflow.add_node("tool_builder_tools", tool_builder_tool_node)
    workflow.add_node("protein_design_tools", protein_design_tool_node)

    workflow.set_entry_point("supervisor")

    # Add edges from supervisor to agents
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "research": "research",
            "analysis": "analysis",
            "coder": "coder",
            "report": "report",
            "tool_builder": "tool_builder",
            "protein_design": "protein_design",
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

    # Tool Builder agent can use tools or go back to supervisor
    def should_continue_to_tool_builder_tools(
        state: AgentState,
    ) -> Literal["tools", "supervisor"]:
        """Check if tool builder needs to use its tools."""
        messages = state["messages"]
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "supervisor"

    workflow.add_conditional_edges(
        "tool_builder",
        should_continue_to_tool_builder_tools,
        {
            "tools": "tool_builder_tools",
            "supervisor": "supervisor",
        },
    )

    workflow.add_edge("coder", "supervisor")

    # Protein Design agent can use tools or go back to supervisor
    def should_continue_to_protein_design_tools(
        state: AgentState,
    ) -> Literal["tools", "supervisor"]:
        """Check if protein design agent needs to use its tools."""
        messages = state["messages"]
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "supervisor"

    workflow.add_conditional_edges(
        "protein_design",
        should_continue_to_protein_design_tools,
        {
            "tools": "protein_design_tools",
            "supervisor": "supervisor",
        },
    )

    workflow.add_edge("research_tools", "research")
    workflow.add_edge("analysis_tools", "analysis")
    workflow.add_edge("tool_builder_tools", "tool_builder")
    workflow.add_edge("protein_design_tools", "protein_design")

    workflow.add_edge("report", "supervisor")

    return workflow.compile()
