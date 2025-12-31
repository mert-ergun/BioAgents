"""Supervisor Agent for routing tasks to specialized agents."""

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from bioagents.agents.supervisor_helpers import (
    check_for_empty_response_loop,
    check_for_missing_tool,
    check_for_repeated_routing,
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
