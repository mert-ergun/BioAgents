"""ACE integration utilities for graph nodes."""

import logging
from typing import TYPE_CHECKING, Optional

from langchain_core.messages import BaseMessage, HumanMessage

if TYPE_CHECKING:
    from bioagents.learning.ace_wrapper import BioAgentsACE

logger = logging.getLogger(__name__)

# ACE wrapper cache (lazy initialization)
_ace_wrappers: dict[str, Optional["BioAgentsACE"]] = {}


def clear_ace_cache() -> None:
    """Clear ACE wrapper cache (useful for development/reloading)."""
    global _ace_wrappers
    _ace_wrappers.clear()


def get_ace_wrapper(agent_name: str) -> Optional["BioAgentsACE"]:
    """
    Get or create ACE wrapper for an agent (lazy initialization).

    Args:
        agent_name: Name of the agent (e.g., 'supervisor', 'coder')

    Returns:
        ACE wrapper instance if enabled, None otherwise
    """
    # Check if ACE is enabled first (re-check on each call to handle env var changes)
    try:
        from bioagents.learning import is_ace_enabled

        if not is_ace_enabled():
            # Clear cache if ACE was disabled
            if agent_name in _ace_wrappers:
                _ace_wrappers[agent_name] = None
            return None
    except ImportError:
        # ACE not available
        return None

    # Only create wrapper if not in cache or if it was None (ACE was disabled before)
    if agent_name not in _ace_wrappers or _ace_wrappers[agent_name] is None:
        try:
            from bioagents.learning import create_ace_wrapper_if_enabled
            from bioagents.llms.llm_provider import get_llm

            llm = get_llm()
            _ace_wrappers[agent_name] = create_ace_wrapper_if_enabled(
                agent_name=agent_name,
                llm_provider=llm,
            )
        except Exception:
            # If ACE is not available or fails to initialize, disable it
            _ace_wrappers[agent_name] = None

    return _ace_wrappers.get(agent_name)


def extract_task_from_messages(messages: list[BaseMessage]) -> str:
    """
    Extract the original task/query from messages.

    Args:
        messages: List of messages in the conversation

    Returns:
        Task string (truncated to 500 chars)
    """
    # Get the first HumanMessage (original query)
    for msg in messages:
        if isinstance(msg, HumanMessage) and hasattr(msg, "content"):
            content = msg.content

            # Handle content that might be a list (multi-modal messages)
            if isinstance(content, list):
                # Extract text from list of content parts
                text_parts = []
                for part in content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                content = " ".join(text_parts)
            elif not isinstance(content, str):
                content = str(content) if content else ""

            if content:
                return content[:500]
    return ""


def extract_output_from_result(result: dict) -> str:
    """
    Extract output from agent result.

    Args:
        result: Agent result dictionary

    Returns:
        Output string (truncated to 500 chars)
    """
    if result.get("messages"):
        last_msg = result["messages"][-1]
        if hasattr(last_msg, "content"):
            content = last_msg.content

            # Handle content that might be a list (multi-modal messages)
            if isinstance(content, list):
                # Extract text from list of content parts
                text_parts = []
                for part in content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                content = " ".join(text_parts)
            elif not isinstance(content, str):
                content = str(content) if content else ""

            if content:
                return content[:500]
    return ""


def map_agent_name_to_ace_name(name: str) -> str:
    """
    Map graph agent name to ACE agent name.

    Args:
        name: Graph agent name (e.g., "Research", "Coder")

    Returns:
        ACE agent name (e.g., "research", "coder")
    """
    name_mapping = {
        "Research": "research",
        "Analysis": "analysis",
        "Coder": "coder",
        "ML": "ml",
        "DL": "dl",
        "Report": "report",
        "ToolBuilder": "tool_builder",
        "ProteinDesign": "protein_design",
        "Critic": "critic",
        "Supervisor": "supervisor",
    }
    return name_mapping.get(name, name.lower())


def track_agent_execution(
    state: dict,
    result: dict,
    agent_name: str,
) -> None:
    """
    Track agent execution with ACE (if enabled).

    Args:
        state: Current agent state
        result: Agent result dictionary
        agent_name: Name of the agent for tracking
    """
    # Get ACE wrapper for this agent
    ace_agent_name = map_agent_name_to_ace_name(agent_name)
    ace_wrapper = get_ace_wrapper(ace_agent_name)

    if ace_wrapper is None:
        return

    if not hasattr(ace_wrapper, "is_enabled") or not ace_wrapper.is_enabled():
        return

    try:
        # Extract task and output
        task = extract_task_from_messages(state.get("messages", []))
        output = extract_output_from_result(result)

        # Determine success (simplified - could be enhanced)
        # For now, assume success if no error messages
        success = True
        error_message = None

        # Check for error indicators in output
        if output:
            error_indicators = [
                "error",
                "exception",
                "failed",
                "cannot",
                "unable",
            ]
            output_lower = output.lower()
            if any(indicator in output_lower for indicator in error_indicators):
                # Not necessarily a failure, but could be
                # More sophisticated error detection could be added
                pass

        # Track execution
        ace_wrapper.track_execution(
            task=task,
            output=output,
            success=success,
            error_message=error_message,
        )

        # Run evolve cycle (periodic - curator runs every N executions)
        ace_wrapper.evolve_cycle(
            task=task,
            agent_output=output,
            success=success,
        )
    except Exception as e:
        # Don't let ACE errors break the normal flow
        # Log but continue
        logger.warning(f"ACE tracking failed for {agent_name}: {e}")
