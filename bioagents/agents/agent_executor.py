"""Reusable agent executor that handles tool execution loops and JSON output."""

import json
import logging
import re
import uuid
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)


def execute_agent_with_tools(
    llm_with_tools,
    system_prompt: str,
    user_message: HumanMessage,
    tools_dict: dict[str, Callable],
    max_iterations: int = 5,
) -> tuple[str, list[str]]:
    """
    Execute an agent with tool-calling loop until it produces final output.

    Args:
        llm_with_tools: LLM with tools bound via .bind_tools()
        system_prompt: System instruction for the agent
        user_message: The user's request as HumanMessage
        tools_dict: Dict mapping tool names to callable functions
        max_iterations: Max iterations to prevent infinite loops

    Returns:
        Tuple of (final_text_output, list_of_tool_names_used)
    """
    messages: list[Any] = [SystemMessage(content=system_prompt), user_message]
    tool_calls_used: list[str] = []
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        try:
            # Call LLM with tools
            response = llm_with_tools.invoke(messages)

            # Track any tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tc in response.tool_calls:
                    tool_calls_used.append(tc.get("name", "unknown"))

            # If no tool calls, this is the final response
            if not hasattr(response, "tool_calls") or not response.tool_calls:
                final_content: str = (
                    response.content
                    if hasattr(response, "content") and isinstance(response.content, str)
                    else str(response)
                )
                return (final_content, tool_calls_used)

            # Add AI response to messages
            messages.append(response)

            # Execute tool calls
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})

                # Get tool_call_id with multiple fallback strategies
                tool_call_id: str = (
                    tool_call.get("id") or tool_call.get("tool_call_id") or str(uuid.uuid4())
                )

                # Find and execute the tool
                if tool_name and tool_name in tools_dict:
                    try:
                        logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                        tool_result = tools_dict[tool_name](**tool_args)
                        tool_output = (
                            str(tool_result)
                            if tool_result is not None
                            else "Tool executed successfully"
                        )
                        logger.info(f"Tool {tool_name} result: {tool_output[:100]}")
                    except Exception as e:
                        logger.warning(f"Tool {tool_name} failed: {e}", exc_info=True)
                        tool_output = f"Error executing {tool_name}: {e!s}"
                else:
                    tool_output = f"Tool '{tool_name}' not found in available tools"
                    logger.warning(f"Tool not found: {tool_name}")

                # Add tool result to messages
                try:
                    messages.append(
                        ToolMessage(
                            tool_call_id=tool_call_id,
                            content=tool_output,
                            name=tool_name,
                        )
                    )
                except Exception as e:
                    logger.error(f"Failed to create ToolMessage for {tool_name}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Agent iteration {iteration} failed: {e}", exc_info=True)
            if len(messages) > 2:
                for msg in reversed(messages):
                    if (
                        hasattr(msg, "content")
                        and isinstance(msg.content, str)
                        and msg.content.strip()
                    ):
                        return (msg.content, tool_calls_used)
            return ("", tool_calls_used)

    # If we hit max iterations, try to get the last response
    logger.warning(f"Agent reached max iterations ({max_iterations})")
    for msg in reversed(messages):
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
            return (msg.content, tool_calls_used)

    return ("", tool_calls_used)


def safe_json_output(
    text: str,
    default_structure: dict[str, Any],
) -> dict[str, Any]:
    """
    Parse JSON from text with multiple fallback strategies.

    Args:
        text: Text that should contain JSON
        default_structure: Fallback structure if parsing fails

    Returns:
        Parsed JSON dict or default_structure
    """
    if not text or not isinstance(text, str):
        return default_structure

    text = text.strip()

    # Strategy 1: Direct JSON parse
    try:
        result: dict[str, Any] = json.loads(text)
        return result
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract JSON from text (handles markdown, explanations, etc.)
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            return result
        except json.JSONDecodeError:
            pass

    # Strategy 3: Try to fix common JSON issues
    try:
        fixed_text = text.replace("'", '"').replace("\n", " ")
        result = json.loads(fixed_text)
        return result
    except json.JSONDecodeError:
        pass

    # Strategy 4: If text is non-empty, wrap it in the default structure
    if text and len(text.strip()) > 10:
        default_structure["raw_text_fallback"] = text[:500]
        return default_structure

    logger.warning(f"Could not parse JSON from text: {text[:200]}")
    return default_structure
