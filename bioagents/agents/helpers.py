"""Shared helper functions for agent modules."""

import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


def resolve_tool_name(tool: Any) -> str:
    """Return a stable tool name for dict registration (handles unittest.mock.Mock)."""
    if isinstance(tool, BaseTool):
        return str(tool.name)
    mock_name = getattr(tool, "_mock_name", None)
    if isinstance(mock_name, str) and mock_name:
        return mock_name
    n = getattr(tool, "name", None)
    if isinstance(n, str) and n:
        return n
    if callable(tool):
        fn = getattr(tool, "__name__", None)
        if isinstance(fn, str) and fn:
            return fn
    return str(tool)


MAX_RETRIES = 2

MAX_TOOL_RESULT_LENGTH = 2000
MAX_AI_CONTENT_LENGTH = 3000
MAX_CONTEXT_MESSAGES = 30


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
        logger.info(
            "Invoking LLM for %s (attempt %d/%d)",
            agent_name,
            attempt + 1,
            MAX_RETRIES + 1,
        )
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


def _strip_signature_blobs(content):
    """Strip base64 signature blobs and other non-essential metadata from content."""
    if isinstance(content, str):
        return re.sub(
            r"'extras':\s*\{[^}]*'signature':\s*'[A-Za-z0-9+/=]{100,}'[^}]*\}",
            "'extras': {<signature stripped>}",
            content,
        )
    if isinstance(content, list):
        cleaned = []
        for block in content:
            if isinstance(block, dict):
                block = dict(block)
                if block.get("extras") and isinstance(block["extras"], dict):
                    sig = block["extras"].get("signature", "")
                    if isinstance(sig, str) and len(sig) > 100:
                        block["extras"] = {"signature": "<stripped>"}
                if block.get("type") == "text":
                    text = block.get("text", "")
                    block["text"] = _strip_signature_blobs(text)
                cleaned.append(block)
            elif isinstance(block, str):
                cleaned.append(_strip_signature_blobs(block))
            else:
                cleaned.append(block)
        return cleaned
    return content


def _truncate_content(content, max_length: int) -> str:
    """Truncate content to max_length, preserving structure."""
    text = get_content_text(content)
    if len(text) <= max_length:
        return text
    return text[:max_length] + "\n... [truncated]"


def prepare_messages_for_agent(
    messages: list,
    agent_name: str,
    max_messages: int = MAX_CONTEXT_MESSAGES,
    max_tool_result_len: int = MAX_TOOL_RESULT_LENGTH,
    max_ai_content_len: int = MAX_AI_CONTENT_LENGTH,
    summary_mode: bool = False,
) -> list:
    """Prepare messages for an agent by windowing and truncating.

    Keeps the original user query, supervisor handoff messages, and recent
    context while truncating large tool results and AI content blobs.

    Args:
        messages: Full message list from state
        agent_name: Name of the target agent (for logging)
        max_messages: Maximum number of messages to keep
        max_tool_result_len: Max chars for tool result content
        max_ai_content_len: Max chars for AI message content
        summary_mode: If True, keep only final agent outputs (for report/summary)

    Returns:
        Windowed and truncated message list
    """
    if not messages:
        return messages

    first_human = None
    supervisor_tasks = []
    recent = []

    for msg in messages:
        if isinstance(msg, HumanMessage) and first_human is None:
            first_human = msg
        if isinstance(msg, HumanMessage) and "[SUPERVISOR TASK]" in get_message_content(msg):
            supervisor_tasks.append(msg)

    if summary_mode:
        kept = []
        if first_human:
            kept.append(first_human)
        for msg in messages:
            if isinstance(msg, AIMessage):
                content = get_message_content(msg)
                if content.strip() and len(content.strip()) > 50:
                    name = getattr(msg, "name", "")
                    has_tool_calls = hasattr(msg, "tool_calls") and msg.tool_calls
                    if not has_tool_calls and name not in ("Supervisor",):
                        cleaned_content = _strip_signature_blobs(content)
                        truncated = _truncate_content(cleaned_content, max_ai_content_len)
                        kept.append(AIMessage(content=truncated, name=name))
            elif isinstance(msg, SystemMessage):
                content = get_message_content(msg)
                if "[SYSTEM]" in content or "[EXECUTION_SUCCESS]" in content:
                    kept.append(msg)
        for task_msg in supervisor_tasks:
            if task_msg not in kept:
                kept.append(task_msg)
        return kept[-max_messages:]

    window_start = max(0, len(messages) - max_messages)
    windowed = messages[window_start:]

    if first_human and first_human not in windowed:
        windowed = [first_human, *windowed]

    for task_msg in supervisor_tasks:
        if task_msg not in windowed:
            windowed.insert(1 if first_human else 0, task_msg)

    result = []
    for msg in windowed:
        if isinstance(msg, ToolMessage):
            content = get_message_content(msg)
            if len(content) > max_tool_result_len:
                truncated = content[:max_tool_result_len] + "\n... [truncated]"
                result.append(
                    ToolMessage(
                        content=truncated,
                        tool_call_id=getattr(msg, "tool_call_id", ""),
                        name=getattr(msg, "name", ""),
                    )
                )
            else:
                result.append(msg)
        elif isinstance(msg, AIMessage):
            content = msg.content
            cleaned = _strip_signature_blobs(content)
            text = get_content_text(cleaned)
            has_tool_calls = hasattr(msg, "tool_calls") and msg.tool_calls
            if len(text) > max_ai_content_len and not has_tool_calls:
                truncated = text[:max_ai_content_len] + "\n... [truncated]"
                new_msg = AIMessage(content=truncated, name=getattr(msg, "name", ""))
                if has_tool_calls:
                    new_msg.tool_calls = msg.tool_calls
                result.append(new_msg)
            elif cleaned != content:
                new_msg = AIMessage(
                    content=cleaned if isinstance(cleaned, str) else text,
                    name=getattr(msg, "name", ""),
                )
                if has_tool_calls:
                    new_msg.tool_calls = msg.tool_calls
                result.append(new_msg)
            else:
                result.append(msg)
        else:
            result.append(msg)

    return result


def invoke_with_retry(
    agent_name: str,
    llm,
    messages_with_system: list,
    max_retries: int = MAX_RETRIES,
) -> dict:
    """Invoke an LLM with retry logic for empty responses.

    Works for agents that do NOT use tool binding (report, summary, critic).

    Args:
        agent_name: Name of the agent for logging
        llm: The LLM instance (without tools)
        messages_with_system: Messages including system prompt
        max_retries: Maximum retry attempts

    Returns:
        Dict with 'messages' key containing the response
    """
    for attempt in range(max_retries + 1):
        response = llm.invoke(messages_with_system)

        if not is_empty_response(response):
            return {"messages": [response]}

        if attempt < max_retries:
            logger.warning(
                f"{agent_name} received empty response from LLM "
                f"(attempt {attempt + 1}/{max_retries + 1}). Retrying..."
            )
            hint = SystemMessage(
                content=(
                    "Your previous response was empty. You MUST provide a substantive response. "
                    "Analyze the conversation history and provide your analysis, findings, or summary."
                )
            )
            messages_with_system.append(hint)
        else:
            logger.error(f"{agent_name} received {max_retries + 1} consecutive empty responses.")

    best = extract_best_content(messages_with_system)
    if best:
        fallback = (
            f"[Auto-generated summary due to {agent_name} failure]\n\n"
            f"The following content was produced during this workflow:\n\n{best}"
        )
    else:
        fallback = (
            f"I was unable to generate a response for this request. "
            f"The {agent_name} agent encountered repeated empty responses. "
            f"Please try again or rephrase the request."
        )

    return {"messages": [AIMessage(content=fallback)]}


def extract_best_content(messages: list) -> str:
    """Extract the most substantive content from message history.

    Scans all AIMessages for the longest, most meaningful content
    to use as a fallback when agents fail.

    Args:
        messages: Full message list

    Returns:
        The best content string, or empty string if nothing found
    """
    best_content = ""
    best_length = 0

    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        content = get_message_content(msg)
        content = _strip_signature_blobs(content) if isinstance(content, str) else content
        text = get_content_text(content) if not isinstance(content, str) else content
        text = text.strip()

        if not text:
            continue
        has_tool_calls = hasattr(msg, "tool_calls") and msg.tool_calls
        if has_tool_calls and not text:
            continue

        if len(text) > best_length:
            best_content = text
            best_length = len(text)

    if len(best_content) > 4000:
        best_content = best_content[:4000] + "\n... [truncated]"

    return best_content
