"""Helper functions for the Supervisor Agent."""

import logging
import re

from langchain_core.messages import AIMessage

from bioagents.agents.helpers import get_message_content

logger = logging.getLogger(__name__)

MAX_EMPTY_RESPONSES = 2

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


def check_for_empty_response_loop(messages) -> tuple[bool, str]:
    """
    Check if the last agent returned an empty response, indicating a potential loop.

    Args:
        messages: List of conversation messages

    Returns:
        Tuple of (is_loop_detected, agent_name_that_failed)
    """
    if len(messages) < 2:
        return False, ""

    consecutive_empty = 0
    last_agent = ""

    for msg in reversed(messages[-6:]):
        if isinstance(msg, AIMessage) or (
            hasattr(msg, "__class__") and msg.__class__.__name__ == "AIMessage"
        ):
            content = get_message_content(msg)
            agent_name = getattr(msg, "name", "unknown")

            has_tool_calls = hasattr(msg, "tool_calls") and msg.tool_calls
            is_empty = not content.strip() and not has_tool_calls

            if is_empty:
                consecutive_empty += 1
                if not last_agent:
                    last_agent = agent_name
            else:
                break

    if consecutive_empty >= MAX_EMPTY_RESPONSES:
        logger.warning(
            f"Loop detected: {consecutive_empty} consecutive empty responses from agent '{last_agent}'"
        )
        return True, last_agent

    return False, ""


def check_for_repeated_routing(messages) -> tuple[bool, str]:
    """
    Check message history for signs of repeated routing to the same agent.

    Looks for patterns like the same agent being called multiple times without progress.

    Args:
        messages: List of conversation messages

    Returns:
        Tuple of (is_loop_detected, agent_name)
    """
    if len(messages) < 4:
        return False, ""

    agent_counts: dict[str, int] = {}
    for msg in messages[-10:]:
        agent_name = getattr(msg, "name", None)
        if agent_name:
            agent_counts[agent_name] = agent_counts.get(agent_name, 0) + 1

    for agent, count in agent_counts.items():
        if count >= 4:
            logger.warning(
                f"Potential loop: agent '{agent}' appeared {count} times in last 10 messages"
            )
            return True, agent

    return False, ""


def check_for_missing_tool(messages) -> tuple[bool, str]:
    """
    Check recent messages for patterns indicating a tool is missing.

    Args:
        messages: List of conversation messages

    Returns:
        Tuple of (should_route_to_tool_builder, reason)
    """
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
