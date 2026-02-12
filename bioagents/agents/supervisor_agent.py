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
        "critic",
        "FINISH",
    ]
    reasoning: str
    task_for_agent: str = Field(
        default="",
        description="A clear, specific instruction for the next agent explaining what task they should perform. "
        "This should be actionable and explicit about what data to fetch, analyze, or produce.",
    )


SUPERVISOR_PROMPT = load_prompt("supervisor")


def create_supervisor_agent(members: list[str]):
    """
    Create the Supervisor Agent for routing.

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
                "Given the conversation above, who should act next? "
                f"Choose from: {', '.join(options)!s}",
            ),
        ]
    )

    supervisor_chain = prompt | llm.with_structured_output(RouteResponse)

    def supervisor_node(state):
        """
        The supervisor node function.

        Args:
            state: The current AgentState

        Returns:
            A dict with next_agent and reasoning for routing
        """
        messages = state["messages"]

        # FIRST: Check for tool builder execution success (only if not already handled)
        # Skip if SystemMessage already exists (already handled, let normal LLM routing decide next step)
        execution_already_handled = any(
            isinstance(msg, SystemMessage)
            or (hasattr(msg, "__class__") and msg.__class__.__name__ == "SystemMessage")
            for msg in messages[-10:]
            if "[EXECUTION_SUCCESS]" in get_message_content(msg)
        )

        if not execution_already_handled:
            # Only check for execution success if we haven't already handled it
            execution_success, exec_tool_name = check_tool_builder_execution_success(messages)
            if execution_success:
                logger.info(
                    f"ToolBuilder successfully executed tool '{exec_tool_name}' and returned results. "
                    "Routing to report to summarize findings."
                )
                # Mark execution success to prevent re-detection
                success_marker = SystemMessage(
                    content=f"[EXECUTION_SUCCESS] ToolBuilder successfully executed tool '{exec_tool_name}' and task is complete."
                )
                return {
                    "next": "report",
                    "reasoning": f"ToolBuilder successfully executed tool '{exec_tool_name}' and completed the task. "
                    "Routing to report to summarize the results.",
                    "messages": [success_marker],
                }

        is_empty_loop, empty_agent = check_for_empty_response_loop(messages)
        if is_empty_loop:
            logger.error(
                f"Supervisor: Detected empty response loop from agent '{empty_agent}'. "
                "Terminating to prevent infinite loop."
            )
            error_msg = SystemMessage(
                content=f"[SYSTEM] Workflow terminated: Agent '{empty_agent}' failed to produce "
                "a response after multiple attempts. This may indicate a configuration issue "
                "or the agent lacks the necessary tools to complete the task."
            )
            return {
                "next": "FINISH",
                "reasoning": f"Loop detected: agent '{empty_agent}' returned empty responses repeatedly. "
                "Terminating to prevent infinite loop.",
                "messages": [error_msg],
            }

        is_routing_loop, looping_agent = check_for_repeated_routing(messages)
        if is_routing_loop:
            logger.warning(
                f"Supervisor: Detected repeated routing to agent '{looping_agent}'. "
                "Attempting to break the loop."
            )

            # Special case: ToolBuilder might need multiple calls to complete
            if looping_agent == "ToolBuilder":
                # First check if ToolBuilder successfully executed a tool (task complete)
                execution_success, exec_tool_name = check_tool_builder_execution_success(messages)
                if execution_success:
                    logger.info(
                        f"ToolBuilder successfully executed tool '{exec_tool_name}' and returned results. "
                        "Routing to report to summarize findings."
                    )
                    # Mark execution success to prevent re-detection
                    success_marker = SystemMessage(
                        content=f"[EXECUTION_SUCCESS] ToolBuilder successfully executed tool '{exec_tool_name}' and task is complete."
                    )
                    return {
                        "next": "report",
                        "reasoning": f"ToolBuilder successfully executed tool '{exec_tool_name}' and completed the task. "
                        "Routing to report to summarize the results.",
                        "messages": [success_marker],
                    }

                # Then check if ToolBuilder successfully created a tool (needs to be used)
                tool_success, tool_name = check_tool_builder_success(messages)
                if tool_success:
                    # Get all created tools (might be multiple)
                    created_tools = get_all_created_tools(messages)
                    # Get original query to provide context
                    original_query = extract_original_query(messages)

                    # Create a clear task instruction for research agent
                    if created_tools:
                        tools_list = ", ".join([f"`{t}`" for t in created_tools])
                        if original_query:
                            task_instruction = (
                                f"Use the newly created tools {tools_list} to complete the original task: {original_query}. "
                                f"First, search for these tools using search_custom_tools, then execute them with execute_custom_tool."
                            )
                        else:
                            task_instruction = (
                                f"Use the newly created tools {tools_list} to complete the task. "
                                f"First, search for these tools using search_custom_tools, then execute them with execute_custom_tool."
                            )
                    else:
                        # Fallback if we can't find all tools
                        task_instruction = (
                            f"Use the newly created tool `{tool_name}` to complete the original task. "
                            f"Search for it using search_custom_tools, then execute it with execute_custom_tool."
                        )
                        if original_query:
                            task_instruction = f"{task_instruction} Original task: {original_query}"

                    # Route to research or analysis agent based on tool category
                    # For now, default to research - can be enhanced based on tool category
                    target_agent = "research" if "research" in members else "analysis"
                    handoff_msg = HumanMessage(
                        content=f"[SUPERVISOR TASK] {task_instruction}", name="Supervisor"
                    )
                    logger.info(
                        f"ToolBuilder created tool(s): {created_tools or [tool_name]}. "
                        f"Routing to {target_agent} to use the tool(s)."
                    )
                    return {
                        "next": target_agent,
                        "reasoning": f"ToolBuilder successfully created tool(s): {created_tools or [tool_name]}. "
                        f"Routing to {target_agent} to execute the tool(s) and complete the task.",
                        "messages": [handoff_msg],
                    }

            if looping_agent != "report" and "report" in members:
                return {
                    "next": "report",
                    "reasoning": f"Loop detected: agent '{looping_agent}' was called repeatedly without progress. "
                    "Escalating to report agent to summarize current state.",
                    "messages": [],
                }
            else:
                error_msg = SystemMessage(
                    content=f"[SYSTEM] Workflow terminated: Detected repeated routing to agent "
                    f"'{looping_agent}' without progress. The task may require manual intervention."
                )
                return {
                    "next": "FINISH",
                    "reasoning": f"Loop detected: agent '{looping_agent}' was called repeatedly. Terminating.",
                    "messages": [error_msg],
                }

        if "tool_builder" in members:
            should_route_to_builder, reason = check_for_missing_tool(messages)
            if should_route_to_builder:
                logger.info(f"Supervisor: Overriding to route to tool_builder - {reason}")
                return {
                    "next": "tool_builder",
                    "reasoning": f"Programmatic detection: {reason}. Routing to tool_builder to create the missing tool.",
                    "messages": [],
                }

        # Normal LLM-based routing
        result = supervisor_chain.invoke({"messages": messages})

        handoff_messages = []
        if result.next_agent != "FINISH" and result.task_for_agent:
            handoff_msg = HumanMessage(
                content=f"[SUPERVISOR TASK] {result.task_for_agent}", name="Supervisor"
            )
            handoff_messages.append(handoff_msg)
            logger.info(f"Supervisor handoff to {result.next_agent}: {result.task_for_agent}")

        return {
            "next": result.next_agent,
            "reasoning": result.reasoning,
            "messages": handoff_messages,
        }

    return supervisor_node
