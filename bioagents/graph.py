"""LangGraph multi-agent workflow definition."""

from functools import partial
from typing import Annotated, Any, Literal
from datetime import datetime
import json

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from bioagents.agents.analysis_agent import create_analysis_agent
from bioagents.agents.coder_agent import create_coder_agent, create_coder_node
from bioagents.agents.critic_agent import create_critic_agent
from bioagents.agents.dl_agent import create_dl_agent, create_dl_node
from bioagents.agents.ml_agent import create_ml_agent, create_ml_node
from bioagents.agents.protein_design_agent import create_protein_design_agent
from bioagents.agents.report_agent import create_report_agent
from bioagents.agents.research_agent import create_research_agent
from bioagents.agents.summary_agent import create_summary_agent
from bioagents.agents.supervisor_agent import create_supervisor_agent
from bioagents.agents.tool_builder_agent import create_tool_builder_agent
from bioagents.tools.analysis_tools import (
    analyze_amino_acid_composition,
    calculate_isoelectric_point,
    calculate_molecular_weight,
)
from bioagents.tools.pdf_tools import (
    extract_pdf_text_spacy_layout,
    fetch_webpage_as_pdf_text,
)
from bioagents.tools.protein_design_tools import get_all_protein_design_tools
from bioagents.tools.proteomics_tools import fetch_uniprot_fasta
from bioagents.tools.structural_tools import (
    download_structure_file,
    fetch_alphafold_structure,
    fetch_pdb_structure,
)
from bioagents.tools.tool_builder_tools import get_tool_builder_tools
from bioagents.tools.tool_universe import tool_universe_call_tool, tool_universe_find_tools


class AgentState(dict):
    """
    The state object passed between nodes in the graph.

    Attributes:
        messages: List of orchestration messages (NOT for data passing)
        next: The next agent to route to (set by supervisor)
        reasoning: The reasoning behind supervisor's decision
        output_dir: Directory path for saving output files (optional)
        memory: Shared workspace for agent outputs
            Structure: {
                "agent_name": {
                    "status": "success" | "error" | "pending",
                    "timestamp": ISO timestamp,
                    "data": {...},  # Agent-specific structured output
                    "raw_output": str,  # Full text response
                    "errors": [],  # Any errors encountered
                    "tool_calls": [...]  # Tools used (for audit trail)
                }
            }
    """

    messages: Annotated[list[BaseMessage], add_messages]
    next: str
    reasoning: str
    output_dir: str | None = None
    memory: dict[str, dict[str, Any]] = {}


def agent_node(state, agent, name):
    """
    Wrapper for agent nodes that handles memory-based communication.

    Key changes from old implementation:
    - Agents return structured JSON-like dicts (not just messages)
    - Results are written to state['memory'][agent_name]
    - Messages are used ONLY for orchestration signals
    - Agent name is tracked automatically

    Args:
        state: The current AgentState
        agent: The agent function to call (must return dict with 'data', 'raw_output', 'tool_calls')
        name: Name of the agent for memory tracking (e.g., 'Research', 'Analysis')

    Returns:
        Updated state with memory written to, messages potentially updated
    """
    try:
        # Call the agent
        result = agent(state)

        # Extract what the agent returned
        agent_data = result.get("data", {})
        agent_raw_output = result.get("raw_output", "")
        agent_tool_calls = result.get("tool_calls", [])
        agent_error = result.get("error", None)

        # Normalize agent name for memory key
        memory_key = name.lower()

        # Write to shared memory
        state["memory"][memory_key] = {
            "status": "error" if agent_error else "success",
            "timestamp": datetime.now().isoformat(),
            "data": agent_data,
            "raw_output": agent_raw_output,
            "errors": [agent_error] if agent_error else [],
            "tool_calls": agent_tool_calls,
        }

        # Keep messages for orchestration only
        # Agents may still return an AIMessage for supervisor to see they completed
        # But the actual data is in memory
        if "messages" in result:
            # Add a completion signal message
            from langchain_core.messages import AIMessage
            completion_msg = AIMessage(
                content=f"[COMPLETED] {name} agent has finished. Results written to shared memory.",
                name=memory_key,
            )
            return {
                "messages": [completion_msg],
                "memory": state["memory"],
            }

        return {
            "messages": [],
            "memory": state["memory"],
        }

    except Exception as e:
        # Capture errors in memory
        memory_key = name.lower()
        state["memory"][memory_key] = {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "data": {},
            "raw_output": "",
            "errors": [str(e)],
            "tool_calls": [],
        }

        from langchain_core.messages import SystemMessage
        error_msg = SystemMessage(
            content=f"[ERROR] {name} agent failed: {str(e)}",
            name=memory_key,
        )

        return {
            "messages": [error_msg],
            "memory": state["memory"],
        }


def should_continue_to_tools(state: AgentState) -> Literal["tools", "supervisor"]:
    """
    Conditional edge that checks if tools need to be called.

    In memory-based architecture, we check if the agent returned tool_calls.
    This is tracked in messages temporarily (for compatibility with ToolNode).

    Args:
        state: The current agent state

    Returns:
        'tools' if last message has tool calls, 'supervisor' otherwise
    """
    messages = state["messages"]

    if not messages:
        return "supervisor"

    last_message = messages[-1]

    # Check if last message indicates tool calls are needed
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "supervisor"


def route_supervisor(
    state: AgentState,
) -> Literal[
    "research",
    "analysis",
    "coder",
    "ml",
    "dl",
    "report",
    "tool_builder",
    "protein_design",
    "critic",
    "summary",
]:
    """
    Route based on supervisor's decision.

    Args:
        state: The current agent state

    Returns:
        The next agent to route to, or 'summary' if finished
    """
    next_agent: Literal[
        "research",
        "analysis",
        "coder",
        "ml",
        "dl",
        "report",
        "tool_builder",
        "protein_design",
        "critic",
        "FINISH",
    ] = state.get("next", "FINISH")
    return "summary" if next_agent == "FINISH" else next_agent


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
        fetch_webpage_as_pdf_text,
        extract_pdf_text_spacy_layout,
        fetch_alphafold_structure,
        fetch_pdb_structure,
        download_structure_file,
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
    ml_agent = create_ml_agent()
    ml_node_func = create_ml_node(ml_agent)
    dl_agent = create_dl_agent()
    dl_node_func = create_dl_node(dl_agent)
    tool_builder_agent = create_tool_builder_agent()
    protein_design_agent = create_protein_design_agent(protein_design_tools)
    critic_agent = create_critic_agent()

    members = [
        "research",
        "analysis",
        "coder",
        "ml",
        "dl",
        "report",
        "tool_builder",
        "protein_design",
        "critic",
    ]
    supervisor_agent = create_supervisor_agent(members)
    summary_agent = create_summary_agent()

    research_tool_node = ToolNode(research_tools)
    analysis_tool_node = ToolNode(analysis_tools)
    tool_builder_tool_node = ToolNode(tool_builder_tools)
    protein_design_tool_node = ToolNode(protein_design_tools)

    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("research", partial(agent_node, agent=research_agent, name="Research"))
    workflow.add_node("analysis", partial(agent_node, agent=analysis_agent, name="Analysis"))
    workflow.add_node("coder", partial(agent_node, agent=coder_node_func, name="Coder"))
    workflow.add_node("ml", partial(agent_node, agent=ml_node_func, name="ML"))
    workflow.add_node("dl", partial(agent_node, agent=dl_node_func, name="DL"))
    workflow.add_node("report", partial(agent_node, agent=report_agent, name="Report"))
    workflow.add_node(
        "tool_builder", partial(agent_node, agent=tool_builder_agent, name="tool_builder")
    )
    workflow.add_node(
        "protein_design", partial(agent_node, agent=protein_design_agent, name="protein_design")
    )
    workflow.add_node("critic", partial(agent_node, agent=critic_agent, name="Critic"))
    workflow.add_node("summary", partial(agent_node, agent=summary_agent, name="Summary"))

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
            "ml": "ml",
            "dl": "dl",
            "report": "report",
            "tool_builder": "tool_builder",
            "protein_design": "protein_design",
            "critic": "critic",
            "summary": "summary",
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
    workflow.add_edge("ml", "supervisor")
    workflow.add_edge("dl", "supervisor")

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
    workflow.add_edge("critic", "supervisor")

    # ── FIX: report goes directly to summary instead of back to supervisor ──
    # Previously: report → supervisor → (loops back to report)
    # Now:        report → summary → END
    # This breaks the loop at the graph topology level, making it impossible
    # for supervisor to re-route to report after it completes.
    workflow.add_edge("report", "summary")
    # ────────────────────────────────────────────────────────────────────────

    workflow.add_edge("summary", END)

    return workflow.compile()