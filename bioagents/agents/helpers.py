"""Shared helper functions for agent modules."""

import logging
import re

from langchain_core.messages import AIMessage, SystemMessage

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


def get_content_text(content) -> str:
    """Extract text from various content formats.

    Args:
        content: Content that can be a string, list of blocks, or other format

    Returns:
        Extracted text as a string
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Handle list of content blocks (e.g., [{"type": "text", "text": "..."}])
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return " ".join(text_parts)
    return str(content) if content else ""


def is_empty_response(response) -> bool:
    """Check if an LLM response is effectively empty.

    Args:
        response: The LLM response message

    Returns:
        True if the response has no content and no tool calls
    """
    has_content = bool(get_content_text(response.content).strip())
    has_tool_calls = hasattr(response, "tool_calls") and bool(response.tool_calls)
    return not has_content and not has_tool_calls


def get_message_content(msg) -> str:
    """Extract text content from a message, handling various formats.

    Args:
        msg: A message object or dict

    Returns:
        The text content of the message
    """
    if hasattr(msg, "content"):
        content = msg.content
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Handle list of content blocks (e.g., [{"type": "text", "text": "..."}])
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            return " ".join(text_parts)
        return str(content)
    elif isinstance(msg, dict):
        return str(msg.get("content", ""))
    return ""


def extract_task_from_messages(messages) -> str:
    """Try to extract the main task from recent messages.

    Args:
        messages: List of conversation messages

    Returns:
        A hint string describing the detected task
    """
    for msg in reversed(messages[-5:]):
        content = get_content_text(getattr(msg, "content", ""))

        # Look for UniProt ID patterns
        uniprot_match = re.search(r"\b([A-Z][A-Z0-9]{5}|[A-Z]\d{5})\b", content)
        if uniprot_match:
            return f"fetch_uniprot_fasta for protein ID: {uniprot_match.group(1)}"

        # Look for FASTA requests
        if "fasta" in content.lower() or "sequence" in content.lower():
            return "fetch protein sequence using fetch_uniprot_fasta tool"

        # Look for search requests
        if "search" in content.lower() or "find" in content.lower():
            return "search using tool_universe_find_tools and tool_universe_call_tool"

    return "complete the research task using available tools"


def create_retry_response(
    agent_name: str,
    messages_with_system: list,
    tool_names: list[str],
    llm_with_tools,
    task_extractor=None,
) -> dict:
    """Handle retry logic for empty LLM responses.

    Args:
        agent_name: Name of the agent for logging
        messages_with_system: Messages including system prompt
        tool_names: List of available tool names
        llm_with_tools: LLM instance with tools bound
        task_extractor: Optional function to extract task hint from messages

    Returns:
        Dict with 'messages' key containing the response
    """
    for attempt in range(MAX_RETRIES + 1):
        response = llm_with_tools.invoke(messages_with_system)

        if not is_empty_response(response):
            return {"messages": [response]}

        if attempt < MAX_RETRIES:
            logger.warning(
                f"{agent_name} received empty response from LLM "
                f"(attempt {attempt + 1}/{MAX_RETRIES + 1}). Retrying..."
            )
            if task_extractor:
                original_messages = messages_with_system[1:]
                task_hint = task_extractor(original_messages)
                hint_content = (
                    f"Your previous response was empty. Please use your tools to {task_hint}. "
                    f"Available tools: {', '.join(tool_names)}"
                )
            else:
                hint_content = (
                    f"Your previous response was empty. Please analyze the data or use your tools. "
                    f"Available tools: {', '.join(tool_names)}"
                )
            hint_message = SystemMessage(content=hint_content)
            messages_with_system.append(hint_message)
        else:
            logger.error(
                f"{agent_name} received {MAX_RETRIES + 1} consecutive empty responses. "
                "Returning error message."
            )

    if task_extractor:
        original_messages = messages_with_system[1:]
        task_hint = task_extractor(original_messages)
        error_content = (
            f"I apologize, but I was unable to process this request. "
            f"The task appears to be: {task_hint}. "
            f"Available tools: {', '.join(tool_names)}. "
            f"Please try rephrasing your request or check if there are any configuration issues."
        )
    else:
        error_content = (
            "I apologize, but I was unable to process this analysis request. "
            f"Available tools: {', '.join(tool_names)}. "
            "Please try rephrasing your request or provide more specific data to analyze."
        )

    error_response = AIMessage(content=error_content)
    return {"messages": [error_response]}
