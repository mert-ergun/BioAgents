"""Research Agent for fetching biological data and conducting literature searches."""

import logging
import re
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from bioagents.agents.helpers import (
    create_retry_response,
    extract_task_from_messages,
    get_content_text,
)
from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

RESEARCH_AGENT_PROMPT = load_prompt("research")


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
    # Match lines starting with numbers like "1. ", "2) ", etc.
    lines = content.strip().split("\n")
    for line in lines:
        # Match "1. Task", "1) Task", "1 Task"
        match = re.match(r"^\d+[\.\)\s]+(.*)", line.strip())
        if match:
            task = match.group(1).strip()
            if task:
                tasks.append(task)

    # If no numbered list found, but we have multiple lines, treat each line as a task
    if not tasks and content.strip():
        potential_tasks = [
            line.strip() for line in lines if line.strip() and len(line.strip()) > 10
        ]
        tasks = potential_tasks or [content.strip()]

    return tasks[:4]


def create_research_agent(tools: list):
    """
    Create the Research Agent node function.

    Args:
        tools: List of tools available to the agent

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm(prompt_name="research")
    llm_with_tools = llm.bind_tools(tools)

    # Get tool names for helpful error messages
    tool_names = [t.name for t in tools]

    def agent_node(state):
        """
        The research agent node function.

        Args:
            state: The current AgentState

        Returns:
            A dict with the 'messages' key containing the agent's response
        """
        messages = state["messages"]
        if not messages:
            return {"messages": []}

        last_message = messages[-1]

        original_request = ""
        for msg in messages:
            if isinstance(msg, HumanMessage) and "[SUPERVISOR TASK]" not in str(msg.content):
                original_request = msg.content
                break
        if not original_request:
            original_request = "the requested research topic"

        if isinstance(last_message, ToolMessage):
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

                sub_tasks_raw = re.findall(r"--- Sub-task \d+: (.*?) ---", initiator.content)
                if not sub_tasks_raw:
                    logger.warning(
                        "Research Agent: Could not parse sub-tasks from initiator. Falling back to merger."
                    )
                    return handle_merge(messages, initiator.content, original_request)

                tool_results_per_sub_agent: dict[int, list[ToolMessage]] = {}
                for i in range(len(sub_tasks_raw)):
                    tool_results_per_sub_agent[i] = []
                    suffix = f"_{i}"
                    for msg in messages[initiator_idx + 1 :]:
                        if isinstance(msg, ToolMessage) and msg.tool_call_id.endswith(suffix):
                            msg_copy = msg.model_copy()
                            msg_copy.tool_call_id = msg.tool_call_id[: -len(suffix)]
                            tool_results_per_sub_agent[i].append(msg_copy)

                def run_sub_agent_continue(i, sub_task):
                    suffix = f"_{i}"
                    my_tool_calls = [
                        tc.copy() for tc in initiator.tool_calls if tc["id"].endswith(suffix)
                    ]
                    if not my_tool_calls:
                        finding_pattern = rf"--- Sub-task {i + 1}: {re.escape(sub_task)} ---\n(.*?)(?=\n--- Sub-task|\n\(Sub-agent|\n\Z)"
                        match = re.search(finding_pattern, initiator.content, re.DOTALL)
                        finding = match.group(1).strip() if match else "Task completed."
                        return AIMessage(content=finding), sub_task

                    for tc in my_tool_calls:
                        tc["id"] = tc["id"][: -len(suffix)]

                    sub_agent_messages = [
                        SystemMessage(content=RESEARCH_AGENT_PROMPT),
                        HumanMessage(
                            content=f"Perform this sub-task: {sub_task}\n\nContext of original request: {original_request}"
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
                        executor.map(lambda x: run_sub_agent_continue(*x), enumerate(sub_tasks_raw))
                    )

                return aggregate_and_return(results, messages, original_request)
            else:
                logger.info(
                    "Research Agent: Tool result received for single-agent mode. Continuing."
                )
                messages_with_system = [
                    SystemMessage(content=RESEARCH_AGENT_PROMPT),
                    *messages,
                ]
                return create_retry_response(
                    agent_name="Research agent (continue)",
                    messages_with_system=messages_with_system,
                    tool_names=tool_names,
                    llm_with_tools=llm_with_tools,
                    task_extractor=extract_task_from_messages,
                )

        if (
            isinstance(last_message, AIMessage)
            and getattr(last_message, "name", None) == "Research"
            and not getattr(last_message, "tool_calls", None)
        ):
            logger.warning(
                "Research Agent: Detected possible loop. Last message is already from Research. Returning to supervisor."
            )
            return {"messages": []}

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
                "Research Agent: Decomposition failed. Falling back to single-agent mode."
            )
            messages_with_system = [
                SystemMessage(content=RESEARCH_AGENT_PROMPT),
                *messages,
            ]
            return create_retry_response(
                agent_name="Research agent (fallback)",
                messages_with_system=messages_with_system,
                tool_names=tool_names,
                llm_with_tools=llm_with_tools,
                task_extractor=extract_task_from_messages,
            )

        logger.info(f"Research Agent: Running {len(sub_tasks)} sub-tasks in parallel")

        def run_sub_agent_initial(sub_task):
            sub_agent_messages = [
                SystemMessage(content=RESEARCH_AGENT_PROMPT),
                HumanMessage(
                    content=f"Perform this sub-task: {sub_task}\n\nContext of original request: {original_request}"
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

        # Execute sub-agents in parallel
        with ThreadPoolExecutor(max_workers=len(sub_tasks)) as executor:
            results = list(executor.map(run_sub_agent_initial, sub_tasks))

        return aggregate_and_return(results, messages, original_request)

    def aggregate_and_return(results, messages, original_request):
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
                    f"(Sub-agent {i + 1} is fetching data using {len(sub_agent_tools)} tools...)\n"
                )
                # If there's substantial text beyond just "I will use tools...", include it
                if content_text and len(content_text) > 150:
                    combined_content += f"{content_text}\n"

                for tc in sub_agent_tools:
                    # Make tool call ID unique to this sub-agent to prevent conflicts
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
            return handle_merge(messages, combined_content, original_request)

        # Return aggregated tool calls
        return {
            "messages": [
                AIMessage(
                    content=combined_content,
                    tool_calls=all_tool_calls,
                    additional_kwargs={"show_ui": False},
                )
            ]
        }

    def handle_merge(messages, findings_summary, original_request):
        logger.info("Research Agent: Entering Merge Phase")
        merger_llm = get_llm(prompt_name="research_merger")
        merger_prompt = load_prompt("research_merger")

        # If findings_summary is empty, try to find the last aggregated findings in history
        if not findings_summary:
            for msg in reversed(messages):
                content = get_content_text(msg.content)
                if isinstance(msg, AIMessage) and "Research Phase - Sub-agent findings" in content:
                    findings_summary = content
                    break

        # Construct a more focused prompt for the merger
        merge_instruction = (
            f"You have completed the parallel research phase for the topic: '{original_request}'.\n\n"
            f"Below are the findings from individual sub-agents and the results of the tools they called.\n"
            f"Please synthesize all this information into a final, coherent research report.\n\n"
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
            tool_names=[],  # No tools for merger
            llm_with_tools=merger_llm,
            task_extractor=lambda _: "merge research results",
        )

    return agent_node
