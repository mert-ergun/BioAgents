"""Supervisor Agent for routing tasks to specialized agents."""

import logging
import re
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel

from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt

logger = logging.getLogger(__name__)


class RouteResponse(BaseModel):
    """Response from supervisor for routing."""

    next_agent: Literal[
        "research",
        "analysis",
        "coder",
        "report",
        "tool_builder",
        "protein_design",
        "critic",
        "FINISH",
    ]
    reasoning: str


SUPERVISOR_PROMPT = load_prompt("supervisor")

TOOL_MISSING_PATTERNS = [
    r"no suitable tool found",
    r"no tool found",
    r"tool not available",
    r"cannot find a tool",
    r"no tools found",
    r"failed to find.*tool",
    r"could not find a suitable",
    r"no suitable.*found",
    r"cannot.*search.*by gene name",
    r"missing required parameters",
    r"tool.*not.*exist",
    r"no.*capability",
    r"lacks.*capability",
]


def _check_for_missing_tool(messages) -> tuple[bool, str]:
    """
    Check recent messages for patterns indicating a tool is missing.

    Args:
        messages: List of conversation messages

    Returns:
        Tuple of (should_route_to_tool_builder, reason)
    """
    # Check last 3 messages for tool missing patterns
    recent_messages = messages[-3:] if len(messages) >= 3 else messages

    for msg in recent_messages:
        content = ""
        if hasattr(msg, "content"):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
        elif isinstance(msg, dict):
            content = msg.get("content", "")

        content_lower = content.lower()

        for pattern in TOOL_MISSING_PATTERNS:
            if re.search(pattern, content_lower):
                logger.info(f"Detected missing tool pattern: '{pattern}' in message")
                return True, f"Detected missing tool: pattern '{pattern}' matched"

    return False, ""


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

        if "tool_builder" in members:
            should_route_to_builder, reason = _check_for_missing_tool(messages)
            if should_route_to_builder:
                logger.info(f"Supervisor: Overriding to route to tool_builder - {reason}")
                return {
                    "next": "tool_builder",
                    "reasoning": f"Programmatic detection: {reason}. Routing to tool_builder to create the missing tool.",
                    "messages": [],
                }

        # Normal LLM-based routing
        result = supervisor_chain.invoke({"messages": messages})

        return {"next": result.next_agent, "reasoning": result.reasoning, "messages": []}

    return supervisor_node
