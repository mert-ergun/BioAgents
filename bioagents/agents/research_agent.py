"""Research Agent for fetching biological data - Tool-executing implementation."""

import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from bioagents.agents.agent_executor import execute_agent_with_tools, safe_json_output
from bioagents.agents.helpers import (
    create_retry_response,
    extract_task_from_messages,
    get_content_text,
    resolve_tool_name,
    filter_allowed_tools,          
    inject_tools_to_system_prompt  
)
from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

DYNAMIC_BASE_PROMPT = load_prompt("research")

DYNAMIC_SYSTEM_PROMPT = """
You are the Research Agent. Your task is to fetch biological data, sequences, and analyze scientific literature.

INSTRUCTIONS:
1. Use your available tools to fetch FASTA sequences and literature data
2. Call tools like fetch_uniprot_fasta to get protein sequences
3. When asked to review, summarize, or answer questions based on local PDF papers, MUST use the `search_local_papers_with_paperqa` tool.
4. Integrate the tool's findings into your final summary.
5. Once you have gathered the data, provide a final summary.

FINAL OUTPUT FORMAT:
After gathering data, respond with ONLY valid JSON (no markdown, no explanations):

{
    "fetched_sequences": ["seq1", "seq2"],
    "literature_findings": "Detailed summary of findings from papers or online sources",
    "data_sources": ["source1", "source2"],
    "completeness": "full" | "partial" | "failed",
    "next_steps": "what should be done next",
    "status": "success" | "error",
    "error": null or "error message"
}

If you cannot fetch data, return the JSON with completeness="failed" and status="error".
ALWAYS end with valid JSON output.
"""

RESEARCH_AGENT_MEMORY_PROMPT = (
    "CRITICAL: You MUST NOT read other agents' outputs from messages. "
    "Use only shared memory and your tools.\n\n" + DYNAMIC_SYSTEM_PROMPT
)

# Maximum number of tool-result rounds a sub-agent is allowed before being
# forced to synthesise with what it has.
MAX_SUB_AGENT_TOOL_ROUNDS = 1


def _tool_call_signature(tc: dict) -> str:
    """Create a hashable signature for a tool call to detect duplicates."""
    name = tc.get("name", "")
    args = tc.get("args", {})
    args_str = json.dumps(args, sort_keys=True, default=str)
    return hashlib.md5(f"{name}:{args_str}".encode(), usedforsecurity=False).hexdigest()


def _extract_fetched_data_summary(messages: list) -> str:
    """Build a brief summary of data already fetched, to share across sub-agents."""
    seen_data: list[str] = []
    for msg in reversed(messages[-40:]):
        if not isinstance(msg, ToolMessage):
            continue
        content = get_content_text(msg.content)
        # UniProt FASTA
        if content.startswith(">sp|") or content.startswith(">tr|"):
            first_line = content.split("\n")[0]
            if first_line not in seen_data:
                seen_data.append(f"FASTA already fetched: {first_line}")
        # AlphaFold structure download
        if '"status": "success"' in content and "model_v" in content and ".pdb" in content:
            try:
                data = json.loads(content)
                path = data.get("file_path", "")
                if path and path not in str(seen_data):
                    seen_data.append(f"Structure already downloaded: {path}")
            except (json.JSONDecodeError, KeyError):
                pass
    return "\n".join(seen_data)


def parse_sub_tasks(content) -> list[str]:
    """
    Parse a numbered list of sub-tasks from string or content-block list.

    Args:
        content: String or list of content blocks (Gemini format)

    Returns:
        List of task descriptions
    """
    tasks = []
    if not isinstance(content, str):
        content = get_content_text(content)
    if not content:
        return []
    lines = content.strip().split("\n")
    for line in lines:
        match = re.match(r"^\d+[\.\)\s]+(.*)", line.strip())
        if match:
            task = match.group(1).strip()
            if task:
                tasks.append(task)

    if not tasks and content.strip():
        potential_tasks = [
            line.strip() for line in lines if line.strip() and len(line.strip()) > 10
        ]
        tasks = potential_tasks or [content.strip()]

    return tasks[:4]


def create_research_agent(tools: list, allowed_tool_names: list[str] | None = None):
    """
    Create the Research Agent with tool-executing loop and parallel sub-agent support.
    """
    active_tools = filter_allowed_tools(tools, allowed_tool_names)
    tool_names: list[str] = [resolve_tool_name(t) for t in active_tools]

    llm = get_llm(prompt_name="research")
    llm_with_tools = llm.bind_tools(active_tools)

    system_prompt = inject_tools_to_system_prompt(DYNAMIC_SYSTEM_PROMPT, tool_names)
    base_prompt = inject_tools_to_system_prompt(DYNAMIC_BASE_PROMPT, tool_names)

    # Build tools_dict for execute_agent_with_tools using only active tools
    tools_dict = {}
    for tool in active_tools:
        if isinstance(tool, BaseTool):
            tools_dict[tool.name] = tool.func if hasattr(tool, "func") else tool
        else:
            tools_dict[resolve_tool_name(tool)] = tool

    logger.info(f"Research agent active tools: {list(tools_dict.keys())}")

    # Track how many parallel tool-result rounds we've done (per invocation).
    _parallel_round = 0

    def research_node(state: dict) -> dict:
        """
        Research agent node.

        Supports two execution paths:
        - Parallel sub-agent decomposition (from develop) for complex tasks
        - Direct tool-execution loop (from shared-memory branch) as fallback

        Always returns shared-memory-compatible dict:
            {"data": {...}, "raw_output": str, "tool_calls": [...], "error": str|None}
        """
        try:
            messages = state.get("messages", [])

            if not messages:
                return {
                    "data": {
                        "fetched_sequences": [],
                        "literature_findings": "",
                        "data_sources": [],
                        "completeness": "partial",
                        "next_steps": "",
                        "status": "success",
                        "error": None,
                    },
                    "raw_output": "",
                    "tool_calls": [],
                    "error": None,
                    "messages": [AIMessage(content="No messages to process")],
                }

            # ── Parallel sub-agent path (from develop) ──────────────────────
            # If the last message is a ToolMessage we are mid-execution;
            # delegate to the parallel continuation handler.
            last_message = messages[-1] if messages else None

            if isinstance(last_message, ToolMessage):
                result = _handle_tool_message(messages, llm_with_tools, tool_names)
                # Wrap parallel-path AIMessage result into shared-memory format
                return _wrap_parallel_result(result)

            if (
                isinstance(last_message, AIMessage)
                and getattr(last_message, "name", None) == "Research"
                and not getattr(last_message, "tool_calls", None)
            ):
                # Check if this is a continuation after tool execution (not a true loop).
                # If the second-to-last message is a ToolMessage, we're in a
                # post-tool-return state where the agent previously produced a
                # final merged result — forward it rather than discarding it.
                prior_message = messages[-2] if len(messages) >= 2 else None
                if isinstance(prior_message, ToolMessage):
                    logger.info("Research Agent: Returning merged result after tool execution.")
                    return {
                        "data": {},
                        "raw_output": get_content_text(last_message.content),
                        "tool_calls": [],
                        "error": None,
                        "messages": [last_message],
                    }

                logger.warning(
                    "Research Agent: Detected possible loop. "
                    "Last message is already from Research. Returning empty."
                )
                return {
                    "data": {},
                    "raw_output": "",
                    "tool_calls": [],
                    "error": None,
                }

            # ── Decompose & parallel phase ───────────────────────────────────
            original_request = _extract_original_request(messages)

            # Reset round counter for fresh invocations.
            _parallel_round = 0

            logger.info("Research Agent: Entering Decompose & Parallel Phase")
            decomposer_llm = get_llm(prompt_name="research_decomposer")
            decomposer_prompt = load_prompt("research_decomposer")

            decomposer_messages = [
                SystemMessage(content=decomposer_prompt),
                HumanMessage(content=f"Please decompose this research task: {original_request}"),
            ]
            decomposition_response = decomposer_llm.invoke(decomposer_messages)
            sub_tasks = parse_sub_tasks(decomposition_response.content)

            if not sub_tasks:
                logger.warning(
                    "Research Agent: Decomposition failed. Falling back to direct tool-loop."
                )
                return _direct_tool_loop(
                    messages, tools_dict, llm_with_tools, system_prompt
                )

            logger.info(f"Research Agent: Running {len(sub_tasks)} sub-tasks in parallel")

            def run_sub_agent_initial(sub_task):
                # Build a data context note so sub-agents don't re-fetch
                already_fetched = _extract_fetched_data_summary(messages)
                context_note = ""
                if already_fetched:
                    context_note = (
                        f"\n\nIMPORTANT - Data already fetched by other sub-agents "
                        f"(do NOT re-fetch these):\n{already_fetched}"
                    )
                sub_agent_messages = [
                    SystemMessage(content=base_prompt),
                    HumanMessage(
                        content=(
                            f"Perform this sub-task: {sub_task}\n\n"
                            f"Context of original request: {original_request}"
                            f"{context_note}"
                        )
                    ),
                ]
                res = create_retry_response(
                    agent_name="Research Sub-agent",
                    messages_with_system=sub_agent_messages,
                    tool_names=tool_names,
                    llm_with_tools=llm_with_tools,
                    task_extractor=lambda _: sub_task,
                )
                return res["messages"][0], sub_task

            with ThreadPoolExecutor(max_workers=len(sub_tasks)) as executor:
                results = list(executor.map(run_sub_agent_initial, sub_tasks))

            parallel_result = _aggregate_and_return(results, messages, original_request)
            return _wrap_parallel_result(parallel_result)

        except Exception as e:
            logger.error(f"Research agent error: {e}", exc_info=True)
            return {
                "data": {
                    "fetched_sequences": [],
                    "literature_findings": "",
                    "data_sources": [],
                    "completeness": "failed",
                    "next_steps": "error recovery needed",
                    "status": "error",
                    "error": str(e),
                },
                "raw_output": str(e),
                "tool_calls": [],
                "error": str(e),
            }

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _extract_original_request(messages: list) -> str:
        """Pull the first non-supervisor HumanMessage as the original request."""
        for msg in messages:
            if isinstance(msg, HumanMessage) and "[SUPERVISOR TASK]" not in str(msg.content):
                return get_content_text(msg.content)
        return "the requested research topic"

    def _direct_tool_loop(messages, tools_dict, llm_with_tools, system_prompt) -> dict:
        """
        Fallback: run a single-agent tool-execution loop and return
        a shared-memory-compatible dict.
        """
        user_message = None
        for m in messages:
            if isinstance(m, HumanMessage):
                user_message = m
                break
        if user_message is None:
            user_message = HumanMessage(content="Fetch biological data.")

        raw_output, tool_calls_used, last_ai, exec_err = execute_agent_with_tools(
            llm_with_tools=llm_with_tools,
            system_prompt=system_prompt,
            user_message=user_message,
            tools_dict=tools_dict,
            max_iterations=5,
            full_message_history=messages,
        )

        default_json: dict[str, Any] = {
            "fetched_sequences": [],
            "literature_findings": raw_output[:500] if raw_output else "",
            "data_sources": [],
            "completeness": "partial",
            "next_steps": "data retrieval incomplete",
            "status": "success" if raw_output else "error",
            "error": None,
        }
        structured_data = safe_json_output(raw_output, default_json)
        out: dict[str, Any] = {
            "data": structured_data,
            "raw_output": raw_output,
            "tool_calls": tool_calls_used,
            "error": exec_err,
        }
        if last_ai is not None:
            out["messages"] = [last_ai]
        return out

    def _wrap_parallel_result(parallel_result: dict) -> dict:
        """
        Convert a parallel-path {"messages": [...]} result into the
        shared-memory format {"data": {...}, "raw_output": ..., ...}.
        """
        msgs = parallel_result.get("messages", [])
        raw_text = ""
        tool_calls_used = []

        for msg in msgs:
            if hasattr(msg, "content"):
                raw_text += get_content_text(msg.content) + "\n"
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    if isinstance(tc, dict):
                        tool_calls_used.append(tc.get("name", "unknown"))
                    else:
                        tool_calls_used.append(str(tc))

        raw_text = raw_text.strip()

        default_json: dict[str, Any] = {
            "fetched_sequences": [],
            "literature_findings": raw_text[:1000] if raw_text else "",
            "data_sources": [],
            "completeness": "full" if raw_text else "partial",
            "next_steps": "",
            "status": "success" if raw_text else "error",
            "error": None,
        }
        structured_data = safe_json_output(raw_text, default_json)
        return {
            "data": structured_data,
            "raw_output": raw_text,
            "tool_calls": tool_calls_used,
            "error": None,
            # Keep messages so agent_node can forward them for tool routing
            "messages": msgs,
        }

    def _handle_tool_message(messages, llm_with_tools, tool_names):
        """Handle mid-execution ToolMessage — continue parallel sub-agents."""
        initiator = None
        initiator_idx = -1
        for i, msg in enumerate(reversed(messages)):
            if (
                isinstance(msg, AIMessage)
                and getattr(msg, "name", None) == "Research"
                and msg.tool_calls
            ):
                initiator = msg
                initiator_idx = len(messages) - 1 - i
                break

        if initiator and "Research Phase - Sub-agent findings" in str(initiator.content):
            logger.info("Research Agent: Continuing parallel sub-agents")
            initiator_content_str = get_content_text(initiator.content)
            sub_tasks_raw = re.findall(r"--- Sub-task \d+: (.*?) ---", initiator_content_str)
            if not sub_tasks_raw:
                logger.warning("Research Agent: Could not parse sub-tasks. Falling back to merger.")
                return _handle_merge(
                    messages, initiator_content_str, _extract_original_request(messages)
                )

            tool_results_per_sub_agent: dict[int, list[ToolMessage]] = {
                i: [] for i in range(len(sub_tasks_raw))
            }
            for i in range(len(sub_tasks_raw)):
                suffix = f"_{i}"
                for msg in messages[initiator_idx + 1 :]:
                    if not isinstance(msg, ToolMessage):
                        continue
                    tcid = msg.tool_call_id
                    if tcid and tcid.endswith(suffix):
                        msg_copy = msg.model_copy()
                        msg_copy.tool_call_id = tcid[: -len(suffix)]
                        tool_results_per_sub_agent[i].append(msg_copy)

            original_request = _extract_original_request(messages)

            # Build data context for sub-agent continuation
            already_fetched = _extract_fetched_data_summary(messages)

            def run_sub_agent_continue(i, sub_task):
                suffix = f"_{i}"
                my_tool_calls = [
                    tc.copy()
                    for tc in initiator.tool_calls
                    if (tid := tc.get("id")) and isinstance(tid, str) and tid.endswith(suffix)
                ]
                if not my_tool_calls:
                    finding_pattern = (
                        rf"--- Sub-task {i + 1}: {re.escape(sub_task)} ---\n"
                        r"(.*?)(?=\n--- Sub-task|\n\(Sub-agent|\n\Z)"
                    )
                    match = re.search(finding_pattern, initiator_content_str, re.DOTALL)
                    finding = match.group(1).strip() if match else "Task completed."
                    return AIMessage(content=finding), sub_task

                for tc in my_tool_calls:
                    oid = tc.get("id")
                    if isinstance(oid, str):
                        tc["id"] = oid[: -len(suffix)]

                context_note = ""
                if already_fetched:
                    context_note = (
                        f"\n\nIMPORTANT - Data already fetched "
                        f"(do NOT re-fetch these):\n{already_fetched}"
                    )
                sub_agent_messages = [
                    SystemMessage(content=base_prompt),
                    HumanMessage(
                        content=(
                            f"Perform this sub-task: {sub_task}\n\n"
                            f"Context of original request: {original_request}"
                            f"{context_note}"
                        )
                    ),
                    AIMessage(content="", tool_calls=my_tool_calls),
                    *tool_results_per_sub_agent[i],
                ]
                res = create_retry_response(
                    agent_name=f"Research Sub-agent {i + 1} (Continue)",
                    messages_with_system=sub_agent_messages,
                    tool_names=tool_names,
                    llm_with_tools=llm_with_tools,
                    task_extractor=lambda _: sub_task,
                )
                return res["messages"][0], sub_task

            with ThreadPoolExecutor(max_workers=len(sub_tasks_raw)) as executor:
                results = list(
                    executor.map(
                        lambda x: run_sub_agent_continue(*x),
                        enumerate(sub_tasks_raw),
                    )
                )

            return _aggregate_and_return(results, messages, original_request)
        else:
            logger.info("Research Agent: ToolMessage received for single-agent mode. Continuing.")
            msgs_with_system = [
                SystemMessage(content=base_prompt),
                *messages,
            ]
            return create_retry_response(
                agent_name="Research agent (continue)",
                messages_with_system=msgs_with_system,
                tool_names=tool_names,
                llm_with_tools=llm_with_tools,
                task_extractor=extract_task_from_messages,
            )

    def _aggregate_and_return(results, messages, original_request):
        nonlocal _parallel_round
        _parallel_round += 1

        all_tool_calls = []
        combined_content = "Research Phase - Sub-agent findings:\n"
        needs_more_tools = False

        # Deduplicate tool calls across sub-agents
        seen_signatures: set[str] = set()

        # Also track signatures already executed in prior rounds (from message history)
        for msg in reversed(messages[-60:]):
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    seen_signatures.add(_tool_call_signature(tc))

        # Count recent ToolUniverse failures to avoid retry loops
        tu_recent_failures = 0
        for msg in reversed(messages[-30:]):
            if isinstance(msg, ToolMessage):
                tc = get_content_text(msg.content).lower()
                if "tooluniverse" in tc and ("failed" in tc or "error" in tc):
                    tu_recent_failures += 1

        for i, (resp, sub_task) in enumerate(results):
            combined_content += f"\n--- Sub-task {i + 1}: {sub_task} ---\n"
            content_text = get_content_text(resp.content).strip()
            sub_agent_tools = getattr(resp, "tool_calls", [])

            if sub_agent_tools:
                # Filter out ToolUniverse calls if they've been consistently failing
                filtered_tools = []
                for tc in sub_agent_tools:
                    tc_name = tc.get("name", "")
                    if "tool_universe" in tc_name and tu_recent_failures >= 2:
                        logger.info(
                            "Research Agent: Skipping ToolUniverse call '%s' "
                            "after %d recent failures.",
                            tc_name,
                            tu_recent_failures,
                        )
                        continue

                    # Deduplicate: skip tool calls already seen
                    sig = _tool_call_signature(tc)
                    if sig in seen_signatures:
                        logger.info(
                            "Research Agent: Skipping duplicate tool call '%s'.",
                            tc_name,
                        )
                        continue
                    seen_signatures.add(sig)

                    tc_copy = tc.copy()
                    if tc_copy.get("id"):
                        tc_copy["id"] = f"{tc_copy['id']}_{i}"
                    filtered_tools.append(tc_copy)

                if filtered_tools:
                    needs_more_tools = True
                    combined_content += (
                        f"(Sub-agent {i + 1} is fetching data using "
                        f"{len(filtered_tools)} tools...)\n"
                    )
                    if content_text and len(content_text) > 150:
                        combined_content += f"{content_text}\n"
                    all_tool_calls.extend(filtered_tools)
                else:
                    # All tool calls were filtered or deduplicated
                    if content_text:
                        combined_content += f"{content_text}\n"
                    else:
                        combined_content += "(Sub-agent tools unavailable; used existing data)\n"
            elif content_text:
                combined_content += f"{content_text}\n"
            else:
                combined_content += "(Sub-agent completed its part of the task)\n"

        # Force merge if we've exceeded the round cap
        if _parallel_round > MAX_SUB_AGENT_TOOL_ROUNDS:
            logger.info(
                "Research Agent: Reached max parallel rounds (%d). Forcing merge.",
                MAX_SUB_AGENT_TOOL_ROUNDS,
            )
            return _handle_merge(messages, combined_content, original_request)

        if not needs_more_tools:
            logger.info("Research Agent: All sub-agents finished. Proceeding to merge.")
            return _handle_merge(messages, combined_content, original_request)

        return {
            "messages": [
                AIMessage(
                    content=combined_content,
                    tool_calls=all_tool_calls,
                )
            ]
        }

    def _handle_merge(messages, findings_summary, original_request):
        logger.info("Research Agent: Entering Merge Phase")
        merger_llm = get_llm(prompt_name="research_merger")
        merger_prompt = load_prompt("research_merger")

        if not findings_summary:
            for msg in reversed(messages):
                content = get_content_text(msg.content)
                if isinstance(msg, AIMessage) and "Research Phase - Sub-agent findings" in content:
                    findings_summary = content
                    break

        merge_instruction = (
            f"You have completed the parallel research phase for the topic: "
            f"'{original_request}'.\n\n"
            f"Below are the findings from individual sub-agents and the results "
            f"of the tools they called.\n"
            f"Please synthesize all this information into a final, coherent "
            f"research report.\n\n"
            f"SUB-AGENT FINDINGS:\n{findings_summary}\n\n"
            f"TOOL RESULTS: (Available in the conversation history as ToolMessages)\n\n"
            f"Please provide the final synthesis now."
        )

        messages_for_merger = [
            SystemMessage(content=merger_prompt),
            *messages,
            HumanMessage(content=merge_instruction),
        ]

        return create_retry_response(
            agent_name="Research Merger",
            messages_with_system=messages_for_merger,
            tool_names=[],
            llm_with_tools=merger_llm,
            task_extractor=lambda _: "merge research results",
        )

    return research_node
