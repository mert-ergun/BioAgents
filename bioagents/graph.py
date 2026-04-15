"""LangGraph multi-agent workflow definition."""

import logging
from functools import partial
from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from bioagents.agents.analysis_agent import create_analysis_agent
from bioagents.agents.coder_agent import create_coder_agent, create_coder_node
from bioagents.agents.critic_agent import create_critic_agent
from bioagents.agents.data_acquisition_agent import create_data_acquisition_agent
from bioagents.agents.dl_agent import create_dl_agent, create_dl_node
from bioagents.agents.docking_agent import create_docking_agent
from bioagents.agents.environment_agent import create_environment_agent
from bioagents.agents.genomics_agent import create_genomics_agent
from bioagents.agents.git_agent import create_git_agent
from bioagents.agents.literature_agent import create_literature_agent
from bioagents.agents.ml_agent import create_ml_agent, create_ml_node
from bioagents.agents.paper_replication_agent import create_paper_replication_agent
from bioagents.agents.phylogenetics_agent import create_phylogenetics_agent
from bioagents.agents.planner_agent import create_planner_agent
from bioagents.agents.prompt_optimizer_agent import create_prompt_optimizer_agent
from bioagents.agents.protein_design_agent import create_protein_design_agent
from bioagents.agents.report_agent import create_report_agent
from bioagents.agents.research_agent import create_research_agent
from bioagents.agents.result_checker_agent import create_result_checker_agent
from bioagents.agents.shell_agent import create_shell_agent
from bioagents.agents.structural_biology_agent import create_structural_biology_agent
from bioagents.agents.summary_agent import create_summary_agent
from bioagents.agents.supervisor_agent import create_supervisor_agent
from bioagents.agents.tool_builder_agent import create_tool_builder_agent
from bioagents.agents.tool_discovery_agent import create_tool_discovery_agent
from bioagents.agents.tool_validator_agent import create_tool_validator_agent
from bioagents.agents.transcriptomics_agent import create_transcriptomics_agent
from bioagents.agents.visualization_agent import create_visualization_agent
from bioagents.agents.web_browser_agent import create_web_browser_agent
from bioagents.learning.ace_integration import (
    track_agent_execution,
)
from bioagents.references.reference_extractor import extract_references_from_messages
from bioagents.references.reference_manager import ReferenceManager
from bioagents.tools.analysis_tools import (
    analyze_amino_acid_composition,
    calculate_isoelectric_point,
    calculate_molecular_weight,
)
from bioagents.tools.environment_tools import get_environment_tools
from bioagents.tools.file_tools import get_file_tools
from bioagents.tools.genomics_tools import get_genomics_tools
from bioagents.tools.git_tools import get_git_tools
from bioagents.tools.literature_tools import get_literature_tools
from bioagents.tools.pdf_tools import (
    extract_pdf_text_spacy_layout,
    fetch_webpage_as_pdf_text,
)
from bioagents.tools.protein_design_tools import get_all_protein_design_tools
from bioagents.tools.proteomics_tools import download_uniprot_flat_file, fetch_uniprot_fasta
from bioagents.tools.shell_tools import get_shell_tools
from bioagents.tools.structural_tools import (
    download_structure_file,
    fetch_alphafold_structure,
    fetch_pdb_structure,
    get_structural_tools,
)
from bioagents.tools.tool_builder_tools import get_tool_builder_tools
from bioagents.tools.tool_universe import tool_universe_call_tool, tool_universe_find_tools
from bioagents.tools.transcriptomics_tools import get_transcriptomics_tools
from bioagents.tools.visualization_tools import get_visualization_tools
from bioagents.tools.web_tools import get_web_tools
from bioagents.truncating_tool_node import make_approval_tool_node, make_truncating_tool_node
from bioagents.tools.tool_policy import ToolPolicy, get_default_policy

logger = logging.getLogger(__name__)


AGENT_NAMES = Literal[
    "research",
    "analysis",
    "coder",
    "ml",
    "dl",
    "report",
    "tool_builder",
    "protein_design",
    "critic",
    "literature",
    "web_browser",
    "paper_replication",
    "data_acquisition",
    "genomics",
    "transcriptomics",
    "structural_biology",
    "phylogenetics",
    "docking",
    "planner",
    "tool_validator",
    "tool_discovery",
    "prompt_optimizer",
    "result_checker",
    "shell",
    "git",
    "environment",
    "visualization",
    "summary",
]


class AgentState(dict):
    """The state object passed between nodes in the graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    next: str
    reasoning: str
    output_dir: str | None = None
    references: ReferenceManager | None = None
    failed_agents: set | None = None
    loop_escape_tried: dict[str, list[str]] | None = None
    iteration_count: int = 0
    error_log: list[str] | None = None
    tool_usage_log: list[dict] | None = None


def _count_agent_tool_rounds(messages: list, agent_name: str) -> int:
    """Count tool-call rounds this agent has taken since the last supervisor handoff."""
    from langchain_core.messages import AIMessage, HumanMessage

    rounds = 0
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if "[SUPERVISOR TASK]" in content:
                break
        if (
            isinstance(msg, AIMessage)
            and getattr(msg, "name", "") == agent_name
            and hasattr(msg, "tool_calls")
            and msg.tool_calls
        ):
            rounds += 1
    return rounds


def _detect_consecutive_duplicate_calls(messages: list, agent_name: str) -> tuple[bool, str]:
    """Detect consecutive identical tool calls (same name + same args) by the same agent.

    Returns (is_loop, description) where description explains the loop.
    """
    import hashlib
    import json

    from langchain_core.messages import AIMessage, HumanMessage

    from bioagents.limits import MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS

    call_hashes: list[str] = []
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            break
        if (
            isinstance(msg, AIMessage)
            and getattr(msg, "name", "") == agent_name
            and hasattr(msg, "tool_calls")
            and msg.tool_calls
        ):
            for tc in msg.tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", {})
                args_str = json.dumps(tool_args, sort_keys=True, default=str)
                call_hash = hashlib.md5(f"{tool_name}:{args_str}".encode()).hexdigest()  # noqa: S324
                call_hashes.append(call_hash)

    if not call_hashes:
        return False, ""

    # Check for consecutive identical calls
    consecutive = 1
    for i in range(1, len(call_hashes)):
        if call_hashes[i] == call_hashes[0]:
            consecutive += 1
        else:
            break

    if consecutive >= MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS:
        return True, (
            f"Agent '{agent_name}' made {consecutive} consecutive identical tool calls. "
            f"Breaking loop (limit: {MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS})."
        )

    return False, ""


def _count_tu_tool_calls(messages: list, agent_name: str) -> int:
    """Count tool_universe_call_tool calls by this agent since last supervisor handoff."""
    from langchain_core.messages import AIMessage, HumanMessage

    count = 0
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            break
        if (
            isinstance(msg, AIMessage)
            and getattr(msg, "name", "") == agent_name
            and hasattr(msg, "tool_calls")
            and msg.tool_calls
        ):
            for tc in msg.tool_calls:
                if tc.get("name") == "tool_universe_call_tool":
                    count += 1

    return count


def agent_node(state, agent, name):
    """Wrapper for agent nodes that adds agent identification and ACE tracking."""
    from langchain_core.messages import AIMessage

    from bioagents.limits import MAX_AGENT_TOOL_ROUNDS, MAX_TU_TOOL_CALLS_PER_AGENT

    if MAX_AGENT_TOOL_ROUNDS and MAX_AGENT_TOOL_ROUNDS > 0:
        rounds = _count_agent_tool_rounds(state.get("messages", []), name)
        if rounds >= MAX_AGENT_TOOL_ROUNDS:
            logger.warning(
                "Agent '%s' hit max tool rounds (%d) — forcing return to supervisor.",
                name,
                MAX_AGENT_TOOL_ROUNDS,
            )
            error_msg = AIMessage(
                content=(
                    f"[MAX_TOOL_ROUNDS] Agent '{name}' reached the tool-round limit "
                    f"({MAX_AGENT_TOOL_ROUNDS}). The supervisor should proceed with "
                    f"available results or try a different approach."
                ),
                name=name,
            )
            return {"messages": [error_msg]}

    # Check for consecutive duplicate tool calls (loop detection)
    is_loop, loop_desc = _detect_consecutive_duplicate_calls(
        state.get("messages", []), name
    )
    if is_loop:
        logger.warning("Loop detected for agent '%s': %s", name, loop_desc)
        error_msg = AIMessage(
            content=f"[LOOP_DETECTED] {loop_desc} "
            f"The supervisor should try a different approach.",
            name=name,
        )
        return {"messages": [error_msg]}

    # Check for excessive tool_universe_call_tool usage
    tu_count = _count_tu_tool_calls(state.get("messages", []), name)
    if tu_count >= MAX_TU_TOOL_CALLS_PER_AGENT:
        logger.warning(
            "Agent '%s' hit TU tool call limit (%d) — forcing return to supervisor.",
            name,
            MAX_TU_TOOL_CALLS_PER_AGENT,
        )
        error_msg = AIMessage(
            content=(
                f"[MAX_TU_CALLS] Agent '{name}' exceeded the ToolUniverse "
                f"call limit ({MAX_TU_TOOL_CALLS_PER_AGENT}). The supervisor "
                f"should proceed with available results or try a different approach."
            ),
            name=name,
        )
        return {"messages": [error_msg]}

    try:
        result = agent(state)
    except (TimeoutError, Exception) as exc:
        is_timeout = isinstance(exc, TimeoutError) or "timeout" in str(exc).lower()
        if is_timeout:
            logger.warning("Agent '%s' hit LLM timeout — returning error to supervisor.", name)
            error_msg = AIMessage(
                content=f"[TIMEOUT] Agent '{name}' could not complete: "
                f"LLM call exceeded time limit. The supervisor should "
                f"try a different agent or finish with available results.",
                name=name,
            )
            return {"messages": [error_msg], "next": "supervisor"}
        raise

    if result.get("messages"):
        for msg in result["messages"]:
            msg.name = name

    if state.get("references") is not None and result.get("messages"):
        refs = extract_references_from_messages(result["messages"])
        if refs:
            state["references"].add_references(refs)
            logger.info(f"Extracted {len(refs)} references from {name} agent")

    track_agent_execution(state, result, name)

    return result


def should_continue_to_tools(state: AgentState) -> Literal["tools", "supervisor"]:
    """Conditional edge: route to tools if last message has tool calls."""
    from bioagents.llms.timeout_llm import _workflow_deadline

    if _workflow_deadline is not None:
        import time

        if time.monotonic() >= _workflow_deadline:
            logger.warning("Workflow deadline reached — skipping tool execution, returning to supervisor.")
            return "supervisor"

    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "supervisor"


def route_supervisor(state: AgentState) -> AGENT_NAMES:
    """Route based on supervisor's decision."""
    next_agent = state.get("next", "FINISH")
    return "summary" if next_agent == "FINISH" else next_agent


ALL_MEMBERS = [
    "research",
    "analysis",
    "coder",
    "ml",
    "dl",
    "report",
    "tool_builder",
    "protein_design",
    "critic",
    "literature",
    "web_browser",
    "paper_replication",
    "data_acquisition",
    "genomics",
    "transcriptomics",
    "structural_biology",
    "phylogenetics",
    "docking",
    "planner",
    "tool_validator",
    "tool_discovery",
    "prompt_optimizer",
    "result_checker",
    "shell",
    "git",
    "environment",
    "visualization",
]


def create_graph(_initialize_references: bool = True, checkpointer=None, policy: ToolPolicy | None = None):
    """Create and compile the multi-agent LangGraph workflow.

    The workflow uses a supervisor pattern where:
    1. Supervisor routes tasks to specialized agents
    2. Each agent can use tools and return to supervisor
    3. Workflow continues until supervisor says FINISH
    4. 30 specialized agents covering research, computation, domain science,
       infrastructure, meta-cognition, and quality assurance

    Args:
        _initialize_references: Whether to initialize reference tracking.
        checkpointer: Optional LangGraph checkpointer for mid-execution state updates.
        policy: Optional ToolPolicy instance. If provided, tool nodes use the
            approval gate. If None, default policy is used (auto-approve everything).
    """
    # ---- existing tool lists ----
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
    analysis_tools_list = [
        calculate_molecular_weight,
        analyze_amino_acid_composition,
        calculate_isoelectric_point,
    ]
    tb_tools = get_tool_builder_tools()
    pd_tools = get_all_protein_design_tools()

    # ---- new tool lists ----
    _tu_tools = [tool_universe_find_tools, tool_universe_call_tool]
    lit_tools = get_literature_tools() + _tu_tools
    web_tools = get_web_tools()
    paper_rep_tools = get_web_tools() + get_git_tools()
    data_acq_tools = get_web_tools() + get_file_tools() + [download_uniprot_flat_file]
    gen_tools = get_genomics_tools() + _tu_tools
    trans_tools = get_transcriptomics_tools() + _tu_tools
    struct_tools = get_structural_tools()
    phylo_tools = get_genomics_tools()
    td_tools = get_tool_builder_tools()
    sh_tools = get_shell_tools()
    git_tools_list = get_git_tools()
    env_tools = get_environment_tools()
    viz_tools = get_visualization_tools()

    # ---- create agents ----
    research_agent = create_research_agent(research_tools)
    analysis_agent = create_analysis_agent(analysis_tools_list)
    report_agent = create_report_agent()
    coder_agent = create_coder_agent()
    coder_node_func = create_coder_node(coder_agent)
    ml_agent = create_ml_agent()
    ml_node_func = create_ml_node(ml_agent)
    dl_agent = create_dl_agent()
    dl_node_func = create_dl_node(dl_agent)
    tool_builder_agent = create_tool_builder_agent()
    protein_design_agent = create_protein_design_agent()
    critic_agent = create_critic_agent()
    summary_agent = create_summary_agent()

    literature_agent = create_literature_agent(extra_tools=_tu_tools)
    web_browser_agent = create_web_browser_agent()
    paper_replication_agent = create_paper_replication_agent()
    data_acquisition_agent = create_data_acquisition_agent()
    genomics_agent = create_genomics_agent(extra_tools=_tu_tools)
    transcriptomics_agent = create_transcriptomics_agent(extra_tools=_tu_tools)
    structural_biology_agent = create_structural_biology_agent()
    phylogenetics_agent = create_phylogenetics_agent()
    docking_agent = create_docking_agent()
    planner_agent = create_planner_agent()
    tool_validator_agent = create_tool_validator_agent()
    tool_discovery_agent = create_tool_discovery_agent()
    prompt_optimizer_agent = create_prompt_optimizer_agent()
    result_checker_agent = create_result_checker_agent()
    shell_agent = create_shell_agent()
    git_agent_inst = create_git_agent()
    environment_agent = create_environment_agent()
    visualization_agent = create_visualization_agent()

    supervisor_agent = create_supervisor_agent(ALL_MEMBERS)

    # ---- tool nodes (with approval gate when policy is provided) ----
    active_policy = policy or get_default_policy()
    research_tool_node = make_approval_tool_node(research_tools, policy=active_policy)
    analysis_tool_node = make_approval_tool_node(analysis_tools_list, policy=active_policy)
    tool_builder_tool_node = make_approval_tool_node(tb_tools, policy=active_policy)
    protein_design_tool_node = make_approval_tool_node(pd_tools, policy=active_policy)
    literature_tool_node = make_approval_tool_node(lit_tools, policy=active_policy)
    web_browser_tool_node = make_approval_tool_node(web_tools, policy=active_policy)
    paper_replication_tool_node = make_approval_tool_node(paper_rep_tools, policy=active_policy)
    data_acquisition_tool_node = make_approval_tool_node(data_acq_tools, policy=active_policy)
    genomics_tool_node = make_approval_tool_node(gen_tools, policy=active_policy)
    transcriptomics_tool_node = make_approval_tool_node(trans_tools, policy=active_policy)
    structural_biology_tool_node = make_approval_tool_node(struct_tools, policy=active_policy)
    phylogenetics_tool_node = make_approval_tool_node(phylo_tools, policy=active_policy)
    tool_discovery_tool_node = make_approval_tool_node(td_tools, policy=active_policy)
    shell_tool_node = make_approval_tool_node(sh_tools, policy=active_policy)
    git_tool_node = make_approval_tool_node(git_tools_list, policy=active_policy)
    environment_tool_node = make_approval_tool_node(env_tools, policy=active_policy)
    visualization_tool_node = make_approval_tool_node(viz_tools, policy=active_policy)

    # ---- build graph ----
    workflow = StateGraph(AgentState)

    # add all agent nodes
    workflow.add_node("supervisor", partial(agent_node, agent=supervisor_agent, name="Supervisor"))
    workflow.add_node("research", partial(agent_node, agent=research_agent, name="Research"))
    workflow.add_node("analysis", partial(agent_node, agent=analysis_agent, name="Analysis"))
    workflow.add_node("coder", partial(agent_node, agent=coder_node_func, name="Coder"))
    workflow.add_node("ml", partial(agent_node, agent=ml_node_func, name="ML"))
    workflow.add_node("dl", partial(agent_node, agent=dl_node_func, name="DL"))
    workflow.add_node("report", partial(agent_node, agent=report_agent, name="Report"))
    workflow.add_node(
        "tool_builder", partial(agent_node, agent=tool_builder_agent, name="ToolBuilder")
    )
    workflow.add_node(
        "protein_design", partial(agent_node, agent=protein_design_agent, name="ProteinDesign")
    )
    workflow.add_node("critic", partial(agent_node, agent=critic_agent, name="Critic"))
    workflow.add_node("summary", partial(agent_node, agent=summary_agent, name="Summary"))
    workflow.add_node("literature", partial(agent_node, agent=literature_agent, name="Literature"))
    workflow.add_node(
        "web_browser", partial(agent_node, agent=web_browser_agent, name="WebBrowser")
    )
    workflow.add_node(
        "paper_replication",
        partial(agent_node, agent=paper_replication_agent, name="PaperReplication"),
    )
    workflow.add_node(
        "data_acquisition",
        partial(agent_node, agent=data_acquisition_agent, name="DataAcquisition"),
    )
    workflow.add_node("genomics", partial(agent_node, agent=genomics_agent, name="Genomics"))
    workflow.add_node(
        "transcriptomics", partial(agent_node, agent=transcriptomics_agent, name="Transcriptomics")
    )
    workflow.add_node(
        "structural_biology",
        partial(agent_node, agent=structural_biology_agent, name="StructuralBiology"),
    )
    workflow.add_node(
        "phylogenetics", partial(agent_node, agent=phylogenetics_agent, name="Phylogenetics")
    )
    workflow.add_node("docking", partial(agent_node, agent=docking_agent, name="Docking"))
    workflow.add_node("planner", partial(agent_node, agent=planner_agent, name="Planner"))
    workflow.add_node(
        "tool_validator", partial(agent_node, agent=tool_validator_agent, name="ToolValidator")
    )
    workflow.add_node(
        "tool_discovery", partial(agent_node, agent=tool_discovery_agent, name="ToolDiscovery")
    )
    workflow.add_node(
        "prompt_optimizer",
        partial(agent_node, agent=prompt_optimizer_agent, name="PromptOptimizer"),
    )
    workflow.add_node(
        "result_checker", partial(agent_node, agent=result_checker_agent, name="ResultChecker")
    )
    workflow.add_node("shell", partial(agent_node, agent=shell_agent, name="Shell"))
    workflow.add_node("git", partial(agent_node, agent=git_agent_inst, name="Git"))
    workflow.add_node(
        "environment", partial(agent_node, agent=environment_agent, name="Environment")
    )
    workflow.add_node(
        "visualization", partial(agent_node, agent=visualization_agent, name="Visualization")
    )

    # add tool nodes
    workflow.add_node("research_tools", research_tool_node)
    workflow.add_node("analysis_tools", analysis_tool_node)
    workflow.add_node("tool_builder_tools", tool_builder_tool_node)
    workflow.add_node("protein_design_tools", protein_design_tool_node)
    workflow.add_node("literature_tools", literature_tool_node)
    workflow.add_node("web_browser_tools", web_browser_tool_node)
    workflow.add_node("paper_replication_tools", paper_replication_tool_node)
    workflow.add_node("data_acquisition_tools", data_acquisition_tool_node)
    workflow.add_node("genomics_tools", genomics_tool_node)
    workflow.add_node("transcriptomics_tools", transcriptomics_tool_node)
    workflow.add_node("structural_biology_tools", structural_biology_tool_node)
    workflow.add_node("phylogenetics_tools", phylogenetics_tool_node)
    workflow.add_node("tool_discovery_tools", tool_discovery_tool_node)
    workflow.add_node("shell_tools", shell_tool_node)
    workflow.add_node("git_tools", git_tool_node)
    workflow.add_node("environment_tools", environment_tool_node)
    workflow.add_node("visualization_tools", visualization_tool_node)

    # ---- entry point ----
    workflow.set_entry_point("supervisor")

    # ---- supervisor routing ----
    routing_map = {m: m for m in ALL_MEMBERS}
    routing_map["summary"] = "summary"
    workflow.add_conditional_edges("supervisor", route_supervisor, routing_map)

    # ---- tool-using agents: conditional edges to their tool nodes ----
    tool_agent_pairs = [
        ("research", "research_tools"),
        ("analysis", "analysis_tools"),
        ("tool_builder", "tool_builder_tools"),
        ("protein_design", "protein_design_tools"),
        ("literature", "literature_tools"),
        ("web_browser", "web_browser_tools"),
        ("paper_replication", "paper_replication_tools"),
        ("data_acquisition", "data_acquisition_tools"),
        ("genomics", "genomics_tools"),
        ("transcriptomics", "transcriptomics_tools"),
        ("structural_biology", "structural_biology_tools"),
        ("phylogenetics", "phylogenetics_tools"),
        ("tool_discovery", "tool_discovery_tools"),
        ("shell", "shell_tools"),
        ("git", "git_tools"),
        ("environment", "environment_tools"),
        ("visualization", "visualization_tools"),
    ]

    for agent_name, tool_node_name in tool_agent_pairs:
        workflow.add_conditional_edges(
            agent_name,
            should_continue_to_tools,
            {"tools": tool_node_name, "supervisor": "supervisor"},
        )
        workflow.add_edge(tool_node_name, agent_name)

    # ---- non-tool agents: direct edges back to supervisor ----
    non_tool_agents = [
        "coder",
        "ml",
        "dl",
        "critic",
        "report",
        "docking",
        "planner",
        "tool_validator",
        "prompt_optimizer",
        "result_checker",
    ]
    for agent_name in non_tool_agents:
        workflow.add_edge(agent_name, "supervisor")

    # ---- summary is terminal ----
    workflow.add_edge("summary", END)

    return workflow.compile(checkpointer=checkpointer)
