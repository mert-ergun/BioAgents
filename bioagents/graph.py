"""LangGraph multi-agent workflow definition."""

import logging
from datetime import datetime
from functools import partial
from typing import Annotated, Any, ClassVar, Literal

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
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
from bioagents.learning.ace_integration import track_agent_execution
from bioagents.references.reference_extractor import extract_references_from_messages
from bioagents.references.reference_manager import ReferenceManager
from bioagents.tools.analysis_tools import (
    analyze_amino_acid_composition,
    calculate_isoelectric_point,
    calculate_molecular_weight,
)
from bioagents.tools.paperqa_wrapper import search_local_papers_with_paperqa
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

logger = logging.getLogger(__name__)


class AgentState(dict):
    """
    The state object passed between nodes in the graph.

    Attributes:
        messages:   Orchestration messages (NOT for data passing between agents).
        next:       The next agent to route to (set by supervisor).
        reasoning:  The reasoning behind the supervisor's routing decision.
        output_dir: Optional directory path for saving output files.
        memory:     Shared workspace for agent outputs.
                    Structure::

                        {
                            "agent_name": {
                                "status":     "success" | "error" | "pending",
                                "timestamp":  ISO-8601 string,
                                "data":       {...},   # agent-specific structured output
                                "raw_output": str,     # full text response
                                "errors":     [...],   # errors encountered
                                "tool_calls": [...],   # audit trail of tool usage
                            }
                        }

        references: Optional ReferenceManager for tracking citations across agents.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    next: str
    reasoning: str
    output_dir: str | None = None
    memory: ClassVar[dict[str, dict[str, Any]]] = {}
    references: ReferenceManager | None = None


def agent_node(state: AgentState, agent, name: str) -> dict:
    """
    Wrapper for agent nodes that:
    - Writes agent output to ``state["memory"][name.lower()]``
    - Extracts and stores citations via ReferenceManager (if enabled)
    - Records execution telemetry via ACE tracking (zero overhead if disabled)
    - Forwards ``messages`` returned by the agent (needed for tool routing)

    Agent callables must return a dict with at least::

        {
            "data":       {...},   # structured output
            "raw_output": str,
            "tool_calls": [...],
            "error":      str | None,
        }

    They may additionally include a ``"messages"`` key whose value is forwarded
    to the graph so that conditional tool edges can fire correctly.

    Args:
        state: The current AgentState.
        agent: The agent callable.
        name:  Human-readable agent name used as the memory key.

    Returns:
        Updated graph state slice.
    """
    try:
        result = agent(state)

        agent_data = result.get("data", {})
        agent_raw_output = result.get("raw_output", "")
        agent_tool_calls = result.get("tool_calls", [])
        agent_error = result.get("error", None)

        memory_key = name.lower()

        # Write structured output to shared memory
        state["memory"][memory_key] = {
            "status": "error" if agent_error else "success",
            "timestamp": datetime.now().isoformat(),
            "data": agent_data,
            "raw_output": agent_raw_output,
            "errors": [agent_error] if agent_error else [],
            "tool_calls": agent_tool_calls,
        }

        # Extract citations if a ReferenceManager is available
        outgoing_messages = result.get("messages", [])
        if state.get("references") is not None and outgoing_messages:
            refs = extract_references_from_messages(outgoing_messages)
            if refs:
                state["references"].add_references(refs)
                logger.info(f"Extracted {len(refs)} references from {name} agent")

        # ACE tracking (no-op if ACE is disabled)
        track_agent_execution(state, result, name)

        # Tag messages with the agent name for supervisor visibility
        for msg in outgoing_messages:
            msg.name = memory_key

        if outgoing_messages:
            # Emit a lightweight completion signal alongside any real messages
            completion_msg = AIMessage(
                content=f"[COMPLETED] {name} agent has finished. Results written to shared memory.",
                name=memory_key,
            )
            return {
                "messages": [*outgoing_messages, completion_msg],
                "memory": state["memory"],
            }

        return {
            "messages": [],
            "memory": state["memory"],
        }

    except Exception as e:
        memory_key = name.lower()
        state["memory"][memory_key] = {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "data": {},
            "raw_output": "",
            "errors": [str(e)],
            "tool_calls": [],
        }

        error_msg = SystemMessage(
            content=f"[ERROR] {name} agent failed: {e!s}",
            name=memory_key,
        )
        logger.exception(f"agent_node caught unhandled error in '{name}': {e}")
        return {
            "messages": [error_msg],
            "memory": state["memory"],
        }


def should_continue_to_tools(state: AgentState) -> Literal["tools", "supervisor"]:
    """
    Conditional edge: checks whether the last message contains tool calls.

    Returns:
        ``"tools"`` if the last message has pending tool calls, else ``"supervisor"``.
    """
    messages = state["messages"]
    if not messages:
        return "supervisor"
    last_message = messages[-1]
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
    Translate the supervisor's ``next`` decision into a graph edge.

    ``"FINISH"`` is remapped to ``"summary"`` so the workflow always
    produces a final summary before reaching ``END``.
    """
    next_agent = state.get("next", "FINISH")
    return "summary" if next_agent == "FINISH" else next_agent


def create_graph(_initialize_references: bool = True):
    """
    Create and compile the multi-agent LangGraph workflow.

    Architecture
    ------------
    - A **supervisor** routes tasks to specialised agents.
    - Each agent writes its output to ``state["memory"]`` (shared memory).
    - The supervisor reads memory to decide next steps.
    - A **tool_builder** agent can create new tools on the fly.
    - ``report → summary → END`` is a hard-wired path that cannot loop.

    Args:
        _initialize_references: Reserved for future use; no-op currently.

    Returns:
        A compiled ``StateGraph`` ready for execution.
    """
    # ── Tool lists ────────────────────────────────────────────────────────────
    research_tools = [
        fetch_uniprot_fasta,
        tool_universe_find_tools,
        tool_universe_call_tool,
        fetch_webpage_as_pdf_text,
        extract_pdf_text_spacy_layout,
        fetch_alphafold_structure,
        fetch_pdb_structure,
        download_structure_file,
        search_local_papers_with_paperqa,  # ← added in shared-memory-paperqa branch
    ]
    analysis_tools = [
        calculate_molecular_weight,
        analyze_amino_acid_composition,
        calculate_isoelectric_point,
    ]
    tool_builder_tools = get_tool_builder_tools()
    protein_design_tools = get_all_protein_design_tools()

    # ── Agent creation ────────────────────────────────────────────────────────
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

    # ── Tool nodes ────────────────────────────────────────────────────────────
    research_tool_node = ToolNode(research_tools)
    analysis_tool_node = ToolNode(analysis_tools)
    tool_builder_tool_node = ToolNode(tool_builder_tools)
    protein_design_tool_node = ToolNode(protein_design_tools)

    # ── Graph construction ────────────────────────────────────────────────────
    workflow = StateGraph(AgentState)

    # Nodes — all wrapped with agent_node for uniform memory writes + ACE tracking
    workflow.add_node("supervisor", partial(agent_node, agent=supervisor_agent, name="Supervisor"))
    workflow.add_node("research", partial(agent_node, agent=research_agent, name="Research"))
    workflow.add_node("analysis", partial(agent_node, agent=analysis_agent, name="Analysis"))
    workflow.add_node("coder", partial(agent_node, agent=coder_node_func, name="Coder"))
    workflow.add_node("ml", partial(agent_node, agent=ml_node_func, name="ML"))
    workflow.add_node("dl", partial(agent_node, agent=dl_node_func, name="DL"))
    workflow.add_node("report", partial(agent_node, agent=report_agent, name="Report"))
    workflow.add_node(
        "tool_builder",
        partial(agent_node, agent=tool_builder_agent, name="tool_builder"),
    )
    workflow.add_node(
        "protein_design",
        partial(agent_node, agent=protein_design_agent, name="protein_design"),
    )
    workflow.add_node("critic", partial(agent_node, agent=critic_agent, name="Critic"))
    workflow.add_node("summary", partial(agent_node, agent=summary_agent, name="Summary"))

    # Tool-executor nodes (LangGraph ToolNode, not agent_node)
    workflow.add_node("research_tools", research_tool_node)
    workflow.add_node("analysis_tools", analysis_tool_node)
    workflow.add_node("tool_builder_tools", tool_builder_tool_node)
    workflow.add_node("protein_design_tools", protein_design_tool_node)

    # Entry point
    workflow.set_entry_point("supervisor")

    # ── Supervisor → agents ───────────────────────────────────────────────────
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

    # ── Agents with tool edges ────────────────────────────────────────────────
    workflow.add_conditional_edges(
        "research",
        should_continue_to_tools,
        {"tools": "research_tools", "supervisor": "supervisor"},
    )

    workflow.add_conditional_edges(
        "analysis",
        should_continue_to_tools,
        {"tools": "analysis_tools", "supervisor": "supervisor"},
    )

    def _should_use_tool_builder_tools(state: AgentState) -> Literal["tools", "supervisor"]:
        messages = state["messages"]
        last = messages[-1] if messages else None
        if last and hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "supervisor"

    workflow.add_conditional_edges(
        "tool_builder",
        _should_use_tool_builder_tools,
        {"tools": "tool_builder_tools", "supervisor": "supervisor"},
    )

    def _should_use_protein_design_tools(state: AgentState) -> Literal["tools", "supervisor"]:
        messages = state["messages"]
        last = messages[-1] if messages else None
        if last and hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "supervisor"

    workflow.add_conditional_edges(
        "protein_design",
        _should_use_protein_design_tools,
        {"tools": "protein_design_tools", "supervisor": "supervisor"},
    )

    # ── Direct edges ──────────────────────────────────────────────────────────
    workflow.add_edge("coder", "supervisor")
    workflow.add_edge("ml", "supervisor")
    workflow.add_edge("dl", "supervisor")
    workflow.add_edge("critic", "supervisor")

    workflow.add_edge("research_tools", "research")
    workflow.add_edge("analysis_tools", "analysis")
    workflow.add_edge("tool_builder_tools", "tool_builder")
    workflow.add_edge("protein_design_tools", "protein_design")

    # ── report → summary → END  (hard-wired; prevents supervisor re-routing) ──
    workflow.add_edge("report", "summary")
    workflow.add_edge("summary", END)

    return workflow.compile()
