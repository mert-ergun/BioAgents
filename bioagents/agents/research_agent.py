"""Research Agent for fetching biological data - Tool-executing implementation."""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from bioagents.agents.agent_executor import execute_agent_with_tools, safe_json_output
from bioagents.agents.helpers import (
    create_retry_response,
    extract_task_from_messages,
    get_content_text,
)
from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

RESEARCH_AGENT_PROMPT = load_prompt("research")

RESEARCH_AGENT_SYSTEM_PROMPT = """
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


def parse_sub_tasks(content: str) -> list[str]:
    """
    Parse a numbered list of sub-tasks from string.

    Args:
        content: String containing a numbered list of tasks

    Returns:
        List of task descriptions
    """
    tasks = []
    if not isinstance(content, str):
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


def create_research_agent(tools: list):
    """
    Create the Research Agent with tool-executing loop and parallel sub-agent support.

    Args:
        tools: List of tool objects

    Returns:
        Agent node function that returns shared-memory-compatible dict
    """
    llm = get_llm(prompt_name="research")
    llm_with_tools = llm.bind_tools(tools)

    tool_names = [
        getattr(t, "name", None) or getattr(t, "__name__", str(t)) for t in tools
    ]

    # Build tools_dict for execute_agent_with_tools (shared memory path)
    tools_dict = {}
    for tool in tools:
        if isinstance(tool, BaseTool):
            tools_dict[tool.name] = tool.func if hasattr(tool, "func") else tool
        elif hasattr(tool, "__call__"):
            tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", "unknown")
            tools_dict[tool_name] = tool
        else:
            tools_dict[str(tool)] = tool

    logger.info(f"Research agent tools: {list(tools_dict.keys())}")

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

            # ── Parallel sub-agent path (from develop) ──────────────────────
            # If the last message is a ToolMessage we are mid-execution;
            # delegate to the parallel continuation handler.
            last_message = messages[-1] if messages else None

            if isinstance(last_message, ToolMessage):
                result = _handle_tool_message(
                    messages, last_message, llm_with_tools, tool_names, llm
                )
                # Wrap parallel-path AIMessage result into shared-memory format
                return _wrap_parallel_result(result)

            if (
                isinstance(last_message, AIMessage)
                and getattr(last_message, "name", None) == "Research"
                and not getattr(last_message, "tool_calls", None)
            ):
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

            logger.info("Research Agent: Entering Decompose & Parallel Phase")
            decomposer_llm = get_llm(prompt_name="research_decomposer")
            decomposer_prompt = load_prompt("research_decomposer")

            decomposer_messages = [
                SystemMessage(content=decomposer_prompt),
                HumanMessage(
                    content=f"Please decompose this research task: {original_request}"
                ),
            ]
            decomposition_response = decomposer_llm.invoke(decomposer_messages)
            sub_tasks = parse_sub_tasks(decomposition_response.content)

            if not sub_tasks:
                logger.warning(
                    "Research Agent: Decomposition failed. Falling back to direct tool-loop."
                )
                return _direct_tool_loop(
                    messages, tools_dict, llm_with_tools,
                    RESEARCH_AGENT_SYSTEM_PROMPT, original_request
                )

            logger.info(f"Research Agent: Running {len(sub_tasks)} sub-tasks in parallel")

            def run_sub_agent_initial(sub_task):
                sub_agent_messages = [
                    SystemMessage(content=RESEARCH_AGENT_PROMPT),
                    HumanMessage(
                        content=(
                            f"Perform this sub-task: {sub_task}\n\n"
                            f"Context of original request: {original_request}"
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

            parallel_result = _aggregate_and_return(
                results, messages, original_request, llm_with_tools, tool_names, llm
            )
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
                return msg.content
        return "the requested research topic"

    def _direct_tool_loop(
        messages, tools_dict, llm_with_tools, system_prompt, original_request
    ) -> dict:
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

        raw_output, tool_calls_used = execute_agent_with_tools(
            llm_with_tools=llm_with_tools,
            system_prompt=system_prompt,
            user_message=user_message,
            tools_dict=tools_dict,
            max_iterations=5,
        )

        default_json = {
            "fetched_sequences": [],
            "literature_findings": raw_output[:500] if raw_output else "",
            "data_sources": [],
            "completeness": "partial",
            "next_steps": "data retrieval incomplete",
            "status": "success" if raw_output else "error",
            "error": None,
        }
        structured_data = safe_json_output(raw_output, default_json)
        return {
            "data": structured_data,
            "raw_output": raw_output,
            "tool_calls": tool_calls_used,
            "error": None,
        }

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
                tool_calls_used.extend(msg.tool_calls)

        raw_text = raw_text.strip()

        default_json = {
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

    def _handle_tool_message(messages, last_message, llm_with_tools, tool_names, llm):
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
            sub_tasks_raw = re.findall(
                r"--- Sub-task \d+: (.*?) ---", initiator.content
            )
            if not sub_tasks_raw:
                logger.warning(
                    "Research Agent: Could not parse sub-tasks. Falling back to merger."
                )
                return _handle_merge(messages, initiator.content, _extract_original_request(messages), llm)

            tool_results_per_sub_agent: dict[int, list[ToolMessage]] = {
                i: [] for i in range(len(sub_tasks_raw))
            }
            for i in range(len(sub_tasks_raw)):
                suffix = f"_{i}"
                for msg in messages[initiator_idx + 1:]:
                    if isinstance(msg, ToolMessage) and msg.tool_call_id.endswith(suffix):
                        msg_copy = msg.model_copy()
                        msg_copy.tool_call_id = msg.tool_call_id[: -len(suffix)]
                        tool_results_per_sub_agent[i].append(msg_copy)

            original_request = _extract_original_request(messages)

            def run_sub_agent_continue(i, sub_task):
                suffix = f"_{i}"
                my_tool_calls = [
                    tc.copy()
                    for tc in initiator.tool_calls
                    if tc["id"].endswith(suffix)
                ]
                if not my_tool_calls:
                    finding_pattern = (
                        rf"--- Sub-task {i + 1}: {re.escape(sub_task)} ---\n"
                        r"(.*?)(?=\n--- Sub-task|\n\(Sub-agent|\n\Z)"
                    )
                    match = re.search(finding_pattern, initiator.content, re.DOTALL)
                    finding = match.group(1).strip() if match else "Task completed."
                    return AIMessage(content=finding), sub_task

                for tc in my_tool_calls:
                    tc["id"] = tc["id"][: -len(suffix)]

                sub_agent_messages = [
                    SystemMessage(content=RESEARCH_AGENT_PROMPT),
                    HumanMessage(
                        content=(
                            f"Perform this sub-task: {sub_task}\n\n"
                            f"Context of original request: {original_request}"
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

            return _aggregate_and_return(
                results, messages, original_request, llm_with_tools, tool_names, llm
            )
        else:
            logger.info(
                "Research Agent: ToolMessage received for single-agent mode. Continuing."
            )
            msgs_with_system = [
                SystemMessage(content=RESEARCH_AGENT_PROMPT),
                *messages,
            ]
            return create_retry_response(
                agent_name="Research agent (continue)",
                messages_with_system=msgs_with_system,
                tool_names=tool_names,
                llm_with_tools=llm_with_tools,
                task_extractor=extract_task_from_messages,
            )

    def _aggregate_and_return(results, messages, original_request, llm_with_tools, tool_names, llm):
        all_tool_calls = []
        combined_content = "Research Phase - Sub-agent findings:\n"
        needs_more_tools = False

        for i, (resp, sub_task) in enumerate(results):
            combined_content += f"\n--- Sub-task {i + 1}: {sub_task} ---\n"
            content_text = get_content_text(resp.content).strip()
            sub_agent_tools = getattr(resp, "tool_calls", [])

            if sub_agent_tools:
                needs_more_tools = True
                combined_content += (
                    f"(Sub-agent {i + 1} is fetching data using "
                    f"{len(sub_agent_tools)} tools...)\n"
                )
                if content_text and len(content_text) > 150:
                    combined_content += f"{content_text}\n"
                for tc in sub_agent_tools:
                    tc_copy = tc.copy()
                    if tc_copy.get("id"):
                        tc_copy["id"] = f"{tc_copy['id']}_{i}"
                    all_tool_calls.append(tc_copy)
            elif content_text:
                combined_content += f"{content_text}\n"
            else:
                combined_content += "(Sub-agent completed its part of the task)\n"

        if not needs_more_tools:
            logger.info("Research Agent: All sub-agents finished. Proceeding to merge.")
            return _handle_merge(messages, combined_content, original_request, llm)

        return {
            "messages": [
                AIMessage(
                    content=combined_content,
                    tool_calls=all_tool_calls,
                    additional_kwargs={"show_ui": False},
                )
            ]
        }

    def _handle_merge(messages, findings_summary, original_request, llm):
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