"""Helper functions for the Supervisor Agent."""

import json
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

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
    # Removed overly broad patterns:
    # r"no.*capability",  # Too broad - catches "I have no capability to calculate X" when agent should route to another agent
    # r"lacks.*capability",  # Too broad - similar issue
    r"no.*tool.*capability",  # More specific - only matches when explicitly about tool capability
    r"lacks.*tool.*capability",  # More specific
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


def check_tool_builder_execution_success(messages) -> tuple[bool, str]:
    """
    Check if tool_builder has successfully executed a tool and returned results.

    This is different from tool creation - this means ToolBuilder executed an
    existing tool and got results, indicating the task is complete.

    Args:
        messages: List of conversation messages

    Returns:
        Tuple of (execution_success, tool_name)
    """
    # First check if execution success was already marked (to avoid re-detection)
    for msg in reversed(messages[-5:]):
        if isinstance(msg, SystemMessage) or (
            hasattr(msg, "__class__") and msg.__class__.__name__ == "SystemMessage"
        ):
            content = get_message_content(msg)
            if "[EXECUTION_SUCCESS]" in content or "ToolBuilder successfully executed" in content:
                # Extract tool name from the message if available
                tool_match = re.search(r"tool '([\w_]+)'", content)
                if tool_match:
                    tool_name = tool_match.group(1)
                    logger.info(f"Execution success already marked for tool: {tool_name}")
                    return True, tool_name
                return True, ""

    # Check for ToolMessage with execution success format
    # Format: {"status": "success", "tool": "...", "result": {...}}
    for msg in reversed(messages[-10:]):
        if isinstance(msg, ToolMessage) or (
            hasattr(msg, "__class__") and msg.__class__.__name__ == "ToolMessage"
        ):
            content = get_message_content(msg)
            try:
                result = json.loads(content)
                # Check for execution success format (has "tool" and "result" keys)
                if result.get("status") == "success" and "tool" in result and "result" in result:
                    tool_name = result.get("tool")
                    logger.info(f"Detected tool execution success from ToolMessage: {tool_name}")
                    return True, tool_name
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass

    # Also check AIMessage from ToolBuilder for execution success patterns
    for msg in reversed(messages[-5:]):
        agent_name = getattr(msg, "name", "")
        if agent_name == "ToolBuilder":
            content = get_message_content(msg)
            content_lower = content.lower()

            # Patterns indicating successful tool execution (not creation)
            execution_patterns = [
                r"successfully performed.*analysis",
                r"successfully.*executed.*tool",
                r"successfully.*completed.*analysis",
                r"i have successfully.*performed",
                r"analysis.*completed.*successfully",
                r"found.*significant.*pathways",
                r"enrichment analysis.*completed",
                r"successfully.*performed.*pathway",
            ]

            for pattern in execution_patterns:
                if re.search(pattern, content_lower):
                    # Try to extract tool name
                    tool_match = re.search(
                        r"tool.*['\"`]([\w_]+)['\"`]|['\"`]([\w_]+)['\"`].*tool",
                        content_lower,
                    )
                    if tool_match:
                        tool_name = tool_match.group(1) or tool_match.group(2)
                        logger.info(
                            f"Detected tool execution success from pattern '{pattern}': {tool_name}"
                        )
                        return True, tool_name
                    # If pattern matches but no tool name, still return success
                    logger.info(f"Detected tool execution success from pattern '{pattern}'")
                    return True, ""

    return False, ""


def check_tool_builder_success(messages) -> tuple[bool, str]:
    """
    Check if tool_builder has successfully created a tool.

    Uses a lightweight LLM call to determine if ToolBuilder successfully created a tool.
    This is more reliable than pattern matching and handles various message formats.

    Args:
        messages: List of conversation messages

    Returns:
        Tuple of (tool_created, tool_name)
    """
    # First, quick check for structured JSON responses (no LLM call needed)
    for msg in reversed(messages[-10:]):
        # Check if it's a ToolMessage (tool call result)
        if isinstance(msg, ToolMessage) or (
            hasattr(msg, "__class__") and msg.__class__.__name__ == "ToolMessage"
        ):
            content = get_message_content(msg)
            try:
                # Try to parse as JSON (tool responses are JSON)
                result = json.loads(content)
                if result.get("status") == "success" and "tool_name" in result:
                    tool_name = result.get("tool_name")
                    logger.info(f"Detected tool success from ToolMessage: {tool_name}")
                    return True, tool_name
            except (json.JSONDecodeError, AttributeError, TypeError):
                # Not JSON, continue checking
                pass

    # If no structured response found, use lightweight LLM call
    # Only check last 5 messages to minimize tokens
    recent_messages = messages[-5:]
    if not recent_messages:
        return False, ""

    try:
        from langchain_core.prompts import ChatPromptTemplate
        from pydantic import BaseModel, Field

        from bioagents.llms.llm_provider import get_llm

        # Lightweight LLM model (use same as supervisor for consistency)
        llm = get_llm(prompt_name="supervisor")

        # Create a simple prompt - only ask about ToolBuilder success
        class ToolBuilderSuccessResponse(BaseModel):
            """Response indicating if ToolBuilder successfully created a tool."""

            success: bool = Field(description="True if ToolBuilder successfully created a tool")
            tool_name: str = Field(
                default="", description="Name of the tool if created, empty string otherwise"
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are analyzing conversation messages to determine if the ToolBuilder agent "
                    "successfully created a tool. Look for indicators like 'tool registered successfully', "
                    "'tool validated successfully', 'created a new tool', or similar success messages. "
                    "Respond with whether a tool was successfully created and its name if available.",
                ),
                (
                    "human",
                    "Did the ToolBuilder agent successfully create a tool in these messages?\n\n{context}",
                ),
            ]
        )

        # Extract only ToolBuilder-related messages to minimize tokens
        tool_builder_context = []
        for msg in recent_messages:
            agent_name = getattr(msg, "name", "")
            if agent_name == "ToolBuilder":
                content = get_message_content(msg)
                # Truncate long messages to save tokens (keep first 500 chars)
                if len(content) > 500:
                    content = content[:500] + "..."
                tool_builder_context.append(f"[ToolBuilder]: {content}")

        if not tool_builder_context:
            return False, ""

        context_text = "\n".join(tool_builder_context)

        # Make lightweight LLM call
        chain = prompt | llm.with_structured_output(ToolBuilderSuccessResponse)
        result = chain.invoke({"context": context_text})

        if result.success:
            logger.info(f"LLM detected tool success: tool_name={result.tool_name or 'unknown'}")
            return True, result.tool_name or "unknown"

    except Exception as e:
        logger.warning(
            f"LLM-based tool success check failed: {e}, falling back to pattern matching"
        )
        # Fallback to pattern matching if LLM call fails
        return _check_tool_builder_success_patterns(messages)

    return False, ""


def _check_tool_builder_success_patterns(messages) -> tuple[bool, str]:
    """
    Fallback pattern matching for tool builder success detection.

    Args:
        messages: List of conversation messages

    Returns:
        Tuple of (tool_created, tool_name)
    """
    success_patterns = [
        r"successfully created.*tool",
        r"tool.*registered successfully",
        r"tool.*validated successfully",
        r"created.*registered.*validated",
        r"i have created.*tool",
        r"created a new tool",
        r"tool.*has been.*created",
        r"tool.*created and.*validated",
    ]

    for msg in reversed(messages[-10:]):
        content = get_message_content(msg)
        agent_name = getattr(msg, "name", "")

        if agent_name == "ToolBuilder":
            # Check for JSON in content
            if '"status": "success"' in content and '"tool_name"' in content:
                try:
                    json_match = re.search(
                        r'\{[^{}]*"status"\s*:\s*"success"[^{}]*"tool_name"\s*:\s*"([^"]+)"[^{}]*\}',
                        content,
                    )
                    if json_match:
                        tool_name = json_match.group(1)
                        logger.info(f"Detected tool success from AIMessage JSON: {tool_name}")
                        return True, tool_name
                except Exception:  # nosec B110
                    pass

            # Check for success patterns
            for pattern in success_patterns:
                if re.search(pattern, content.lower()):
                    tool_match = re.search(r"tool.*['\"`]([\w_]+)['\"`]", content.lower())
                    if tool_match:
                        tool_name = tool_match.group(1)
                        logger.info(f"Detected tool success from pattern '{pattern}': {tool_name}")
                        return True, tool_name

    return False, ""


def get_all_created_tools(messages) -> list[str]:
    """
    Extract all successfully created tool names from ToolMessage responses.

    Args:
        messages: List of conversation messages

    Returns:
        List of tool names that were successfully created
    """
    created_tools = []

    for msg in reversed(messages[-20:]):  # Check last 20 messages
        if isinstance(msg, ToolMessage) or (
            hasattr(msg, "__class__") and msg.__class__.__name__ == "ToolMessage"
        ):
            content = get_message_content(msg)
            try:
                result = json.loads(content)
                if result.get("status") == "success" and "tool_name" in result:
                    tool_name = result.get("tool_name")
                    if tool_name and tool_name not in created_tools:
                        created_tools.append(tool_name)
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass

    return created_tools


def extract_original_query(messages) -> str | None:
    """
    Extract the original user query from messages.

    Args:
        messages: List of messages in the conversation

    Returns:
        The original user query string, or None if not found
    """
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content
            return str(content) if content is not None else None
    return None


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


def check_for_existing_tool(messages) -> tuple[bool, str]:
    """
    Check if tool_builder found an existing tool that can be used.

    Args:
        messages: List of conversation messages

    Returns:
        Tuple of (tool_found, tool_name)
    """
    # Patterns indicating an existing tool was found
    existing_tool_patterns = [
        r"tool.*already exists",
        r"tool.*already.*found",
        r"existing tool",
        r"tool.*found.*can",
        r"tool named.*already",
        r"tool.*exists.*can.*use",
        r"already exists.*can.*convert",
        r"already exists.*can.*perform",
        r"tool.*already exists.*can",
        r"found.*existing tool",  # Added for "found an existing tool"
    ]

    # Check last few messages from ToolBuilder
    for msg in reversed(messages[-5:]):
        agent_name = getattr(msg, "name", "")
        if agent_name == "ToolBuilder":
            content = get_message_content(msg)
            content_lower = content.lower()

            # Check for existing tool patterns
            for pattern in existing_tool_patterns:
                if re.search(pattern, content_lower):
                    # Try multiple extraction patterns to catch different formats
                    # Pattern 1: "tool named 'test_tool'" or "tool named `test_tool`"
                    tool_match = re.search(
                        r"tool.*named.*['\"`]([\w_]+)['\"`]",
                        content_lower,
                    )
                    if tool_match:
                        tool_name = tool_match.group(1)
                        logger.info(f"Detected existing tool from ToolBuilder: {tool_name}")
                        return True, tool_name

                    # Pattern 2: "'test_tool' already exists" or "`test_tool` already exists"
                    tool_match = re.search(
                        r"['\"`]([\w_]+)['\"`].*already exists",
                        content_lower,
                    )
                    if tool_match:
                        tool_name = tool_match.group(1)
                        logger.info(f"Detected existing tool from ToolBuilder: {tool_name}")
                        return True, tool_name

                    # Pattern 3: "existing tool 'test_tool'" or "existing tool `test_tool`"
                    tool_match = re.search(
                        r"existing tool.*['\"`]([\w_]+)['\"`]",
                        content_lower,
                    )
                    if tool_match:
                        tool_name = tool_match.group(1)
                        logger.info(f"Detected existing tool from ToolBuilder: {tool_name}")
                        return True, tool_name

                    # Pattern 4: "found an existing tool `sam_to_bam`" or "found an existing tool 'test_tool'"
                    tool_match = re.search(
                        r"found.*existing tool.*['\"`]([\w_]+)['\"`]",
                        content_lower,
                    )
                    if tool_match:
                        tool_name = tool_match.group(1)
                        logger.info(f"Detected existing tool from ToolBuilder: {tool_name}")
                        return True, tool_name

                    # Fallback: look for tool name in backticks/quotes near "already" or "exists"
                    tool_match = re.search(
                        r"['\"`]([\w_]+)['\"`].*already|already.*['\"`]([\w_]+)['\"`]",
                        content_lower,
                    )
                    if tool_match:
                        tool_name = tool_match.group(1) or tool_match.group(2)
                        logger.info(f"Detected existing tool (fallback): {tool_name}")
                        return True, tool_name

                    # If pattern matches but no tool name found, still return True
                    logger.info("Detected existing tool pattern but couldn't extract name")
                    return True, ""

    return False, ""
