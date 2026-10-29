"""LangGraph multi-agent workflow definition."""

import logging
from functools import partial
from typing import Annotated, Literal

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

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
from bioagents.tools.agent_manifest import build_agent_tool_map
from bioagents.tools.smol_tool_wrappers import (
    ToolUniverseExecuteTool,
    ToolUniverseSearchTool,
    get_esm_smol_tools,
)
from bioagents.tools.tool_policy import ToolPolicy, get_default_policy
from bioagents.truncating_tool_node import make_approval_tool_node

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


def _detect_consecutive_duplicate_calls(messages: list, agent_name: str) -> tuple[bool, str]:
    """Detect consecutive identical tool calls (same name + same args) by the same agent.

    Read-only discovery tools (``DISCOVERY_TOOL_NAMES``) get the higher
    ``MAX_CONSECUTIVE_IDENTICAL_DISCOVERY_CALLS`` threshold: repeating a search over a
    large tool index is normal search behaviour, not a loop.

    Returns (is_loop, description) where description explains the loop.
    """
    import hashlib
    import json

    from langchain_core.messages import AIMessage, HumanMessage

    from bioagents.limits import (
        DISCOVERY_TOOL_NAMES,
        MAX_CONSECUTIVE_IDENTICAL_DISCOVERY_CALLS,
        MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS,
    )

    calls: list[tuple[str, str]] = []
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
                call_hash = hashlib.md5(
                    f"{tool_name}:{args_str}".encode(), usedforsecurity=False
                ).hexdigest()
                calls.append((call_hash, tool_name))

    if not calls:
        return False, ""

    first_hash, first_tool = calls[0]
    consecutive = 1
    for call_hash, _ in calls[1:]:
        if call_hash == first_hash:
            consecutive += 1
        else:
            break

    limit = (
        MAX_CONSECUTIVE_IDENTICAL_DISCOVERY_CALLS
        if first_tool in DISCOVERY_TOOL_NAMES
        else MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS
    )
    if consecutive >= limit:
        return True, (
            f"Agent '{agent_name}' made {consecutive} consecutive identical tool calls "
            f"('{first_tool}'). Breaking loop (limit: {limit})."
        )

    return False, ""


_FILE_ARG_KEYS = ("file_path", "path", "filename", "output_path", "output_file")
_WRITE_TOOL_HINTS = ("write", "save", "create", "register", "generate", "download")
_PROGRESS_EXCERPT_CHARS = 400
_PROGRESS_MAX_ITEMS = 10


def _task_start_index(messages: list) -> int:
    """Index of the most recent [SUPERVISOR TASK] handoff (0 when there is none)."""
    from langchain_core.messages import HumanMessage

    from bioagents.agents.helpers import get_message_content

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, HumanMessage) and "[SUPERVISOR TASK]" in get_message_content(msg):
            return i
    return 0


def _first_text_arg(args: dict) -> str:
    return next((v for v in args.values() if isinstance(v, str) and v.strip()), "")


def _summarize_agent_progress(messages: list, agent_name: str) -> str:
    """Compact digest of what ``agent_name`` did since the last supervisor handoff.

    Appended to forced-return messages so the supervisor (and whichever agent it picks
    next) inherits the discovered tools, written files and latest results instead of
    restarting the subtask from zero.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    from bioagents.agents.helpers import get_message_content
    from bioagents.limits import DISCOVERY_TOOL_NAMES

    task_messages = messages[_task_start_index(messages) :]
    tool_counts: dict[str, int] = {}
    discovery_queries: list[str] = []
    files_written: list[str] = []
    own_call_ids: set[str] = set()
    results: list[str] = []

    for msg in task_messages:
        if isinstance(msg, AIMessage) and getattr(msg, "name", "") == agent_name:
            for tc in getattr(msg, "tool_calls", None) or []:
                tool_name = tc.get("name", "")
                args = tc.get("args") or {}
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
                if tc.get("id"):
                    own_call_ids.add(tc["id"])
                if not isinstance(args, dict):
                    continue
                query = _first_text_arg(args)[:120]
                if tool_name in DISCOVERY_TOOL_NAMES and query and query not in discovery_queries:
                    discovery_queries.append(query)
                if any(h in tool_name.lower() for h in _WRITE_TOOL_HINTS):
                    for key in _FILE_ARG_KEYS:
                        value = args.get(key)
                        if isinstance(value, str) and value and value not in files_written:
                            files_written.append(value)
        elif isinstance(msg, ToolMessage) and getattr(msg, "tool_call_id", "") in own_call_ids:
            text = get_message_content(msg).strip()
            if text:
                label = getattr(msg, "name", "") or "tool"
                results.append(f"{label}: {text[:_PROGRESS_EXCERPT_CHARS]}")

    if not tool_counts:
        return ""

    lines = ["[PROGRESS SUMMARY] Work done before the cut-off (reuse it, do not redo it):"]
    lines.append("- Tools called: " + ", ".join(f"{n} x{c}" for n, c in tool_counts.items()))
    if discovery_queries:
        lines.append(
            "- Discovery searches run: " + "; ".join(discovery_queries[:_PROGRESS_MAX_ITEMS])
        )
    if files_written:
        lines.append("- Files written: " + ", ".join(files_written[:_PROGRESS_MAX_ITEMS]))
    if results:
        lines.append("- Latest results:")
        lines.extend(f"  * {r}" for r in results[-2:])
    notes = _extract_best_agent_content(task_messages, agent_name)
    if notes:
        lines.append(f"- Last agent notes: {notes[:_PROGRESS_EXCERPT_CHARS]}")
    return "\n".join(lines)


def _with_progress(text: str, messages: list, agent_name: str) -> str:
    """Append the agent's progress summary to a forced-return message, when there is one."""
    summary = _summarize_agent_progress(messages, agent_name)
    return f"{text}\n\n{summary}" if summary else text


def _wrap_up_notice(agent_name: str, rounds_used: int, budget: int) -> str:
    remaining = budget - rounds_used
    return (
        f"[TOOL_BUDGET_WARNING] Agent '{agent_name}' has used {rounds_used} of {budget} "
        f"tool-calling rounds for this task; {remaining} remaining. Wrap up now: if you "
        "already have enough, reply WITHOUT tool calls and report your findings (tools "
        "found, files written, results). Otherwise make only the single most important "
        "remaining call, then report. Work not reported before the limit is lost."
    )


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


def _extract_best_agent_content(messages: list, agent_name: str) -> str:
    """Extract the most substantive content from an agent's previous AIMessages."""
    from langchain_core.messages import AIMessage

    best = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "name", "") == agent_name:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content and len(content) > len(best) and len(content) > 50:
                best = content
    return best[:4000] if best else ""


#: State channels an agent node may write. Anything else LangGraph discards in silence.
STATE_CHANNELS: frozenset[str] = frozenset(
    {
        "messages",
        "next",
        "reasoning",
        "output_dir",
        "references",
        "failed_agents",
        "loop_escape_tried",
        "iteration_count",
        "error_log",
        "tool_usage_log",
    }
)


def ensure_result_is_visible(result, name):
    """Guarantee an agent's work reaches the supervisor as a message.

    LangGraph applies only the keys that are declared state channels and drops the rest
    without raising. A node returning just ``data``/``raw_output``/``error`` therefore
    vanished completely: the supervisor saw no reply, assumed the agent had not run, and
    re-delegated the same task until the workflow timed out — losing results the agent
    had already computed. Anything carried on a non-channel key is rebuilt into a
    message here so a finished task can never read as a missing one.
    """
    from langchain_core.messages import AIMessage

    if not isinstance(result, dict):
        return result
    if result.get("messages"):
        return result

    carried = {k: v for k, v in result.items() if k not in STATE_CHANNELS and v}
    error = result.get("error")
    raw = result.get("raw_output")

    text = ""
    if isinstance(raw, str) and raw.strip():
        text = raw.strip()
    elif error:
        text = f"[AGENT_ERROR] {name} failed: {error}"
    elif carried:
        text = f"[AGENT_RESULT] {name} returned data without a summary: {carried}"

    if not text:
        if any(k in STATE_CHANNELS for k in result):
            return result
        text = (
            f"[NO_OUTPUT] Agent '{name}' returned no usable result. "
            f"Do not re-delegate the same task to '{name}'."
        )

    logger.error(
        "Agent '%s' returned no messages; its output would have been dropped. Keys: %s",
        name,
        sorted(result),
    )
    return {**result, "messages": [AIMessage(content=text[:8000], name=name)]}


def agent_node(state, agent, name):
    """Wrapper for agent nodes that adds agent identification and ACE tracking."""
    from langchain_core.messages import AIMessage

    from bioagents.limits import (
        AGENT_TOOL_ROUNDS_WARNING_MARGIN,
        MAX_AGENT_TOOL_ROUNDS,
        MAX_TU_TOOL_CALLS_PER_AGENT,
    )

    messages = state.get("messages", [])

    is_loop, loop_desc = _detect_consecutive_duplicate_calls(messages, name)
    if is_loop:
        logger.warning("Loop detected for agent '%s': %s", name, loop_desc)
        error_msg = AIMessage(
            content=_with_progress(
                f"[LOOP_DETECTED] {loop_desc} The supervisor should try a different approach.",
                messages,
                name,
            ),
            name=name,
        )
        return {"messages": [error_msg]}

    # Checked here, not on the conditional edge, so the agent stops with a clean message
    # instead of pending tool_calls that have no matching results.
    tool_rounds = count_agent_tool_rounds(messages, name)
    if tool_rounds >= MAX_AGENT_TOOL_ROUNDS:
        logger.warning(
            "Agent '%s' hit its tool-round budget (%d) for this task — "
            "forcing return to supervisor.",
            name,
            MAX_AGENT_TOOL_ROUNDS,
        )
        error_msg = AIMessage(
            content=_with_progress(
                f"[MAX_TOOL_ROUNDS] Agent '{name}' used its full budget of "
                f"{MAX_AGENT_TOOL_ROUNDS} tool-calling rounds for this task without "
                f"finishing it. Control is returning to the supervisor. The supervisor "
                f"should either delegate the remaining work to a better-suited agent "
                f"(for example, the Coder agent for computation rather than an agent "
                f"that can only read and write files), or finish with the results "
                f"already obtained. Do not re-delegate the same task to '{name}'.",
                messages,
                name,
            ),
            name=name,
        )
        return {"messages": [error_msg]}

    tu_count = _count_tu_tool_calls(messages, name)
    if tu_count >= MAX_TU_TOOL_CALLS_PER_AGENT:
        logger.warning(
            "Agent '%s' hit TU tool call limit (%d) — forcing return to supervisor.",
            name,
            MAX_TU_TOOL_CALLS_PER_AGENT,
        )
        error_msg = AIMessage(
            content=_with_progress(
                f"[MAX_TU_CALLS] Agent '{name}' exceeded the ToolUniverse "
                f"call limit ({MAX_TU_TOOL_CALLS_PER_AGENT}). The supervisor "
                f"should proceed with available results or try a different approach.",
                messages,
                name,
            ),
            name=name,
        )
        return {"messages": [error_msg]}

    agent_state = state
    warn_at = max(1, MAX_AGENT_TOOL_ROUNDS - AGENT_TOOL_ROUNDS_WARNING_MARGIN)
    if AGENT_TOOL_ROUNDS_WARNING_MARGIN and tool_rounds >= warn_at:
        # Shown to the agent for this call only. Persisting it as a HumanMessage would
        # reset the duplicate-call scan, which stops at the latest HumanMessage.
        logger.info("Agent '%s' is near its tool-round budget; asking it to wrap up.", name)
        notice = HumanMessage(content=_wrap_up_notice(name, tool_rounds, MAX_AGENT_TOOL_ROUNDS))
        agent_state = {**state, "messages": [*messages, notice]}

    try:
        result = agent(agent_state)
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
            if name == "Supervisor":
                return {"messages": [error_msg], "next": "FINISH"}
            return {"messages": [error_msg], "next": "supervisor"}
        raise

    result = ensure_result_is_visible(result, name)

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


# Loop detection and tool-round budgets match on the display name, so these must agree.
AGENT_DISPLAY_NAMES: dict[str, str] = {
    "supervisor": "Supervisor",
    "research": "Research",
    "analysis": "Analysis",
    "coder": "Coder",
    "ml": "ML",
    "dl": "DL",
    "report": "Report",
    "tool_builder": "ToolBuilder",
    "protein_design": "ProteinDesign",
    "critic": "Critic",
    "summary": "Summary",
    "literature": "Literature",
    "web_browser": "WebBrowser",
    "paper_replication": "PaperReplication",
    "data_acquisition": "DataAcquisition",
    "genomics": "Genomics",
    "transcriptomics": "Transcriptomics",
    "structural_biology": "StructuralBiology",
    "phylogenetics": "Phylogenetics",
    "docking": "Docking",
    "planner": "Planner",
    "tool_validator": "ToolValidator",
    "tool_discovery": "ToolDiscovery",
    "prompt_optimizer": "PromptOptimizer",
    "result_checker": "ResultChecker",
    "shell": "Shell",
    "git": "Git",
    "environment": "Environment",
    "visualization": "Visualization",
}


def count_agent_tool_rounds(messages: list, agent_display_name: str) -> int:
    """Count how many tool-calling rounds ``agent_display_name`` has run for this task.

    A "task" begins at the most recent supervisor handoff ([SUPERVISOR TASK] message),
    so the budget resets every time the supervisor delegates again.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    from bioagents.agents.helpers import get_message_content

    rounds = 0
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and "[SUPERVISOR TASK]" in get_message_content(msg):
            break
        if (
            isinstance(msg, AIMessage)
            and getattr(msg, "name", "") == agent_display_name
            and getattr(msg, "tool_calls", None)
        ):
            rounds += 1
    return rounds


def should_continue_to_tools(state: AgentState) -> Literal["tools", "supervisor"]:
    """Conditional edge: route to tools if the last message has tool calls.

    The per-agent tool-round budget is enforced in ``agent_node`` rather than here:
    cutting an agent off at this edge would leave its pending ``tool_calls`` without
    matching tool results, which several LLM APIs reject on the next request.
    """
    from bioagents.llms.timeout_llm import _workflow_deadline

    if _workflow_deadline is not None:
        import time

        if time.monotonic() >= _workflow_deadline:
            logger.warning(
                "Workflow deadline reached — skipping tool execution, returning to supervisor."
            )
            return "supervisor"

    messages = state["messages"]
    last_message = messages[-1]

    if not (hasattr(last_message, "tool_calls") and last_message.tool_calls):
        return "supervisor"

    return "tools"


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
    "user_input",
]


def create_graph(
    _initialize_references: bool = True,
    checkpointer=None,
    policy: ToolPolicy | None = None,
    session_context_tool=None,
):
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
        session_context_tool: Optional LangChain tool for retrieving previous session
            context. When provided, it is appended to every tool-using agent's tool
            list and corresponding tool node so agents can pull prior-turn details.
    """
    # One manifest keeps the wiring, the supervisor's view and the UI catalogue in step.
    agent_tools = build_agent_tool_map(session_context_tool)

    research_tools = agent_tools["research"]
    analysis_tools_list = agent_tools["analysis"]
    tb_tools = agent_tools["tool_builder"]
    pd_tools = agent_tools["protein_design"]
    lit_tools = agent_tools["literature"]
    web_tools = agent_tools["web_browser"]
    paper_rep_tools = agent_tools["paper_replication"]
    data_acq_tools = agent_tools["data_acquisition"]
    gen_tools = agent_tools["genomics"]
    trans_tools = agent_tools["transcriptomics"]
    struct_tools = agent_tools["structural_biology"]
    phylo_tools = agent_tools["phylogenetics"]
    td_tools = agent_tools["tool_discovery"]
    sh_tools = agent_tools["shell"]
    git_tools_list = agent_tools["git"]
    env_tools = agent_tools["environment"]
    viz_tools = agent_tools["visualization"]
    docking_tools = agent_tools["docking"]
    from bioagents.tools.tool_universe import tool_universe_call_tool, tool_universe_find_tools

    _tu_tools = [tool_universe_find_tools, tool_universe_call_tool]

    research_agent = create_research_agent(research_tools)
    analysis_agent = create_analysis_agent(analysis_tools_list)
    report_agent = create_report_agent()
    coder_agent = create_coder_agent()
    coder_node_func = create_coder_node(coder_agent)
    # Code-writing agents get the ESM tools as smolagents Tools so they can score
    # mutations with a real model instead of hand-rolling (or faking) the computation.
    _code_agent_tools = [ToolUniverseSearchTool(), ToolUniverseExecuteTool(), *get_esm_smol_tools()]
    ml_agent = create_ml_agent(tools=list(_code_agent_tools))
    ml_node_func = create_ml_node(ml_agent)
    dl_agent = create_dl_agent(tools=list(_code_agent_tools))
    dl_node_func = create_dl_node(dl_agent)
    tool_builder_agent = create_tool_builder_agent()
    protein_design_agent = create_protein_design_agent(pd_tools)
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
    docking_tool_node = make_approval_tool_node(docking_tools, policy=active_policy)

    workflow = StateGraph(AgentState)

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
    workflow.add_node("docking_tools", docking_tool_node)

    workflow.set_entry_point("supervisor")

    def user_input_node(state):
        import json
        import os

        messages = state.get("messages", [])
        engagement_data = None

        for msg in reversed(messages[-5:]):
            content = getattr(msg, "content", "")
            if not isinstance(content, str):
                continue
            marker = "[ENGAGEMENT_PENDING] "
            idx = content.find(marker)
            if idx != -1:
                try:
                    engagement_data = json.loads(content[idx + len(marker) :])
                except json.JSONDecodeError:
                    engagement_data = {
                        "type": "clarification",
                        "question": content[idx + len(marker) :],
                    }
                break

        # No marker: build a question from the last task and the original query.
        if engagement_data is None:
            context_hint = ""
            original_query = ""
            for msg in messages:
                if isinstance(msg, HumanMessage) and "[SUPERVISOR TASK]" not in str(msg.content):
                    original_query = str(msg.content)[:300]
                    break

            for msg in reversed(messages[-10:]):
                content = getattr(msg, "content", "")
                if isinstance(msg, HumanMessage) and "[SUPERVISOR TASK]" in str(content):
                    task = str(content).replace("[SUPERVISOR TASK]", "").strip()
                    context_hint = task[:300]
                    break

            if context_hint:
                question = f"I need clarification: {context_hint}. How would you like to proceed?"
            elif original_query:
                question = (
                    f"Regarding your request ({original_query[:100]}...), "
                    "how would you like to proceed?"
                )
            else:
                question = "How would you like to proceed?"

            engagement_data = {
                "type": "clarification",
                "question": question,
                "options": [],
                "context": context_hint or original_query,
                "agent": "Supervisor",
            }

        payload = engagement_data

        response = interrupt(payload)

        if payload.get("type") == "api_key_request":
            api_key = response.get("content", "") if isinstance(response, dict) else response
            env_var_name = payload.get("env_var", "API_KEY")

            # 1. Inject into the system programmatically (LLM cannot read this)
            os.environ[env_var_name] = str(api_key).strip()

            # 2. Return a system message to the LLM that the key was added, NOT the key itself
            return {
                "messages": [
                    HumanMessage(
                        content=f"[SYSTEM RESPONSE] {env_var_name} has been securely added to the system. You can now retry the interrupted operation."
                    )
                ]
            }

        response_text = ""
        if isinstance(response, dict):
            response_text = response.get("content", "")
            selected_option = response.get("selected_option")
            if selected_option:
                response_text = f"{selected_option}. {response_text}".strip()
        elif isinstance(response, str):
            response_text = response

        if not response_text:
            response_text = "Proceed with your best judgment."

        return {"messages": [HumanMessage(content=f"[USER RESPONSE] {response_text}")]}

    workflow.add_node("user_input", user_input_node)
    workflow.add_edge("user_input", "supervisor")

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
        ("docking", "docking_tools"),
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
