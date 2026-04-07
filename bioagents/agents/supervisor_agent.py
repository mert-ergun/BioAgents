"""Supervisor Agent for routing tasks to specialized agents."""

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from bioagents.agents.helpers import get_message_content
from bioagents.agents.supervisor_helpers import (
    check_for_empty_response_loop,
    check_for_missing_tool,
    check_for_repeated_routing,
    check_tool_builder_execution_success,
    check_tool_builder_success,
    extract_original_query,
    get_all_created_tools,
)
from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt

logger = logging.getLogger(__name__)


class RouteResponse(BaseModel):
    """Response from supervisor for routing."""

    next_agent: Literal[
        "research",
        "analysis",
        "coder",
        "ml",
        "dl",
        "report",
        "tool_builder",
        "protein_design",
        "rdkit_validator",
        "critic",
        "FINISH",
    ]
    reasoning: str
    task_for_agent: str = Field(
        default="",
        description=(
            "A clear, specific instruction for the next agent explaining what task they should "
            "perform. This should be actionable and explicit about what data to fetch, analyze, "
            "or produce."
        ),
    )


SUPERVISOR_PROMPT = load_prompt("supervisor")


def check_report_complete_in_memory(memory: dict) -> bool:
    """
    Return True when the report agent has already written a successful result
    to shared memory.  This is the authoritative signal that report is done.

    Args:
        memory: The shared memory dict from AgentState

    Returns:
        True if report memory shows status=success with data present
    """
    report_mem = memory.get("report", {})
    is_complete = report_mem.get("status") == "success" and bool(report_mem.get("data"))
    if is_complete:
        logger.info(
            "Supervisor: report memory shows status=success with data present. "
            "Report is done — routing to FINISH."
        )
    return is_complete


def create_supervisor_agent(members: list[str]):
    """
    Create the Supervisor Agent that routes based on shared memory state.

    Routing priority (highest → lowest):
    1. Shared memory check  — if report is done, FINISH immediately
    2. Tool builder execution success — route to report to summarise results
    3. Empty-response loop detection — terminate
    4. Repeated-routing loop detection — break loop or escalate
    5. Missing-tool detection — route to tool_builder
    6. LLM-based routing with memory-status context

    Args:
        members: List of available agent names

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm(prompt_name="supervisor")

    options = [*members, "FINISH"]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUPERVISOR_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above and the shared memory state below, "
                "who should act next? "
                f"Choose from: {', '.join(options)!s}\n\n"
                "SHARED MEMORY STATUS:\n"
                "{memory_status}"
                "\n\nCRITICAL ROUTING RULE: If the 'report' agent already has "
                "status=success in shared memory, you MUST choose FINISH. "
                "Do NOT route to 'report' again under any circumstances.",
            ),
        ]
    )

    supervisor_chain = prompt | llm.with_structured_output(RouteResponse)

    def supervisor_node(state: dict) -> dict:
        """
        The supervisor node function (memory-aware).

        Args:
            state: The current AgentState

        Returns:
            A dict with next_agent, reasoning, messages, and memory
        """
        messages = state["messages"]
        memory = state.get("memory", {})

        # ── 1. Shared-memory check (most reliable signal) ────────────────────
        if check_report_complete_in_memory(memory):
            return {
                "next": "FINISH",
                "reasoning": (
                    "Report agent completed successfully (confirmed via shared memory). Finishing."
                ),
                "messages": [],
                "memory": memory,
            }

        # ── 2. Tool builder execution-success check ───────────────────────────
        # Only trigger once; guard against re-processing the same signal.
        execution_already_handled = any(
            "[EXECUTION_SUCCESS]" in get_message_content(msg)
            for msg in messages[-10:]
            if isinstance(msg, SystemMessage)
        )

        if not execution_already_handled:
            execution_success, exec_tool_name = check_tool_builder_execution_success(messages)
            if execution_success:
                logger.info(
                    f"ToolBuilder successfully executed tool '{exec_tool_name}' "
                    "and returned results. Routing to report."
                )
                success_marker = SystemMessage(
                    content=(
                        f"[EXECUTION_SUCCESS] ToolBuilder successfully executed "
                        f"tool '{exec_tool_name}' and task is complete."
                    )
                )
                return {
                    "next": "report",
                    "reasoning": (
                        f"ToolBuilder successfully executed tool '{exec_tool_name}' "
                        "and completed the task. Routing to report to summarise the results."
                    ),
                    "messages": [success_marker],
                    "memory": memory,
                }

        # ── 3. Empty-response loop detection ─────────────────────────────────
        is_empty_loop, empty_agent = check_for_empty_response_loop(messages)
        if is_empty_loop:
            logger.error(
                f"Supervisor: Detected empty response loop from agent '{empty_agent}'. "
                "Terminating to prevent infinite loop."
            )
            error_msg = SystemMessage(
                content=(
                    f"[SYSTEM] Workflow terminated: Agent '{empty_agent}' failed to produce "
                    "a response after multiple attempts."
                )
            )
            return {
                "next": "FINISH",
                "reasoning": (
                    f"Loop detected: agent '{empty_agent}' returned empty responses repeatedly."
                ),
                "messages": [error_msg],
                "memory": memory,
            }

        # ── 4. Repeated-routing loop detection ───────────────────────────────
        is_routing_loop, looping_agent = check_for_repeated_routing(messages)
        if is_routing_loop:
            logger.warning(
                f"Supervisor: Detected repeated routing to agent '{looping_agent}'. "
                "Attempting to break the loop."
            )

            # Special case: ToolBuilder may legitimately need multiple calls
            if looping_agent == "ToolBuilder":
                # Check execution success first
                exec_ok, exec_tool_name = check_tool_builder_execution_success(messages)
                if exec_ok:
                    logger.info(f"ToolBuilder executed tool '{exec_tool_name}'. Routing to report.")
                    success_marker = SystemMessage(
                        content=(
                            f"[EXECUTION_SUCCESS] ToolBuilder successfully executed "
                            f"tool '{exec_tool_name}' and task is complete."
                        )
                    )
                    return {
                        "next": "report",
                        "reasoning": (
                            f"ToolBuilder successfully executed tool '{exec_tool_name}'. "
                            "Routing to report to summarise results."
                        ),
                        "messages": [success_marker],
                        "memory": memory,
                    }

                # Check tool creation success
                tool_success, tool_name = check_tool_builder_success(messages)
                if tool_success:
                    created_tools = get_all_created_tools(messages)
                    original_query = extract_original_query(messages)

                    if created_tools:
                        tools_list = ", ".join([f"`{t}`" for t in created_tools])
                        task_instruction = (
                            f"Use the newly created tools {tools_list} to complete the "
                            f"original task: {original_query}. "
                            f"First, search for these tools using search_custom_tools, "
                            f"then execute them with execute_custom_tool."
                        )
                    else:
                        task_instruction = (
                            f"Use the newly created tool `{tool_name}` to complete the "
                            f"original task. Search for it using search_custom_tools, then "
                            f"execute it with execute_custom_tool."
                        )
                        if original_query:
                            task_instruction += f" Original task: {original_query}"

                    target_agent = "research" if "research" in members else "analysis"
                    handoff_msg = HumanMessage(
                        content=f"[SUPERVISOR TASK] {task_instruction}",
                        name="Supervisor",
                    )
                    logger.info(
                        f"ToolBuilder created tool(s): {created_tools or [tool_name]}. "
                        f"Routing to {target_agent} to use the tool(s)."
                    )
                    return {
                        "next": target_agent,
                        "reasoning": (
                            f"ToolBuilder successfully created tool(s): "
                            f"{created_tools or [tool_name]}. "
                            f"Routing to {target_agent} to execute and complete the task."
                        ),
                        "messages": [handoff_msg],
                        "memory": memory,
                    }

            # Standard loop handling for other agents
            if looping_agent == "report":
                logger.info("Supervisor: Report appears to be looping. Finishing workflow.")
                return {
                    "next": "FINISH",
                    "reasoning": "Report agent completed. Breaking loop by finishing workflow.",
                    "messages": [],
                    "memory": memory,
                }

            if looping_agent == "protein_design":
                logger.info("Supervisor: Protein design appears to be looping. Finishing workflow.")
                return {
                    "next": "FINISH",
                    "reasoning": (
                        "Protein design agent has gathered sufficient data. "
                        "Breaking loop by finishing workflow."
                    ),
                    "messages": [],
                    "memory": memory,
                }

            if looping_agent != "report" and "report" in members:
                return {
                    "next": "report",
                    "reasoning": (
                        f"Loop detected: agent '{looping_agent}' was called repeatedly "
                        "without progress. Escalating to report agent to summarise current state."
                    ),
                    "messages": [],
                    "memory": memory,
                }

            error_msg = SystemMessage(
                content=(
                    f"[SYSTEM] Workflow terminated: Detected repeated routing to agent "
                    f"'{looping_agent}' without progress. "
                    "The task may require manual intervention."
                )
            )
            return {
                "next": "FINISH",
                "reasoning": (
                    f"Loop detected: agent '{looping_agent}' was called repeatedly. Terminating."
                ),
                "messages": [error_msg],
                "memory": memory,
            }

        # ── 5. Missing-tool detection ─────────────────────────────────────────
        if "tool_builder" in members:
            should_route_to_builder, reason = check_for_missing_tool(messages)
            if should_route_to_builder:
                logger.info(f"Supervisor: Overriding to route to tool_builder — {reason}")
                return {
                    "next": "tool_builder",
                    "reasoning": (
                        f"Programmatic detection: {reason}. "
                        "Routing to tool_builder to create the missing tool."
                    ),
                    "messages": [],
                    "memory": memory,
                }

        # ── 6. LLM-based routing ──────────────────────────────────────────────
        result = supervisor_chain.invoke(
            {
                "messages": messages,
                "memory_status": build_memory_status_summary(memory, members),
            }
        )

        # Post-LLM safety net: prevent re-routing to report if already done
        if result.next_agent == "report" and check_report_complete_in_memory(memory):
            logger.warning(
                "Supervisor: LLM tried to re-route to report despite it being complete. "
                "Overriding to FINISH."
            )
            return {
                "next": "FINISH",
                "reasoning": "Report already complete. Overriding LLM re-route to FINISH.",
                "messages": [],
                "memory": memory,
            }

        handoff_messages = []
        if result.next_agent != "FINISH" and result.task_for_agent:
            handoff_msg = HumanMessage(
                content=f"[SUPERVISOR TASK] {result.task_for_agent}",
                name="Supervisor",
            )
            handoff_messages.append(handoff_msg)
            logger.info(f"Supervisor handoff to {result.next_agent}: {result.task_for_agent}")

        return {
            "next": result.next_agent,
            "reasoning": result.reasoning,
            "messages": handoff_messages,
            "memory": memory,
        }

    return supervisor_node


def build_memory_status_summary(memory: dict, members: list[str]) -> str:
    """
    Build a human-readable summary of shared memory state for the LLM prompt.

    Args:
        memory: The shared memory dict
        members: List of agent names

    Returns:
        Formatted string of memory status
    """
    summary_lines = []

    for agent_name in members:
        agent_mem = memory.get(agent_name, {})
        status = agent_mem.get("status", "pending")
        has_data = bool(agent_mem.get("data", {}))
        completeness = agent_mem.get("data", {}).get("completeness", "unknown")
        errors = agent_mem.get("errors", [])

        error_str = f", errors: {errors}" if errors else ""
        summary_lines.append(
            f"  {agent_name}: status={status}, has_data={has_data}, "
            f"completeness={completeness}{error_str}"
        )

    return "\n".join(summary_lines)
