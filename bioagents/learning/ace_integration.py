"""ACE integration utilities for graph nodes."""

import logging
from contextvars import ContextVar
from typing import TYPE_CHECKING, Optional

from langchain_core.messages import BaseMessage, HumanMessage

if TYPE_CHECKING:
    from bioagents.learning.ace_wrapper import BioAgentsACE

logger = logging.getLogger(__name__)

#: Identifier of the run currently being executed. A benchmark process runs many
#: unrelated use cases back to back; without this scope every use case would keep
#: evolving the *same* in-memory wrapper (and its ExecutionTracker), so one task's
#: trajectory would shape the next task's learning. Rebinding the run id gives the
#: next run a fresh wrapper.
_current_run_id: ContextVar[str] = ContextVar("ace_run_id", default="default")

# ACE wrapper cache, keyed by (run_id, agent_name) (lazy initialization).
_ace_wrappers: dict[tuple[str, str], Optional["BioAgentsACE"]] = {}


def set_ace_run_scope(run_id: str | None) -> None:
    """Bind ACE wrapper state to a run.

    Call this at the start of every run (use case, benchmark sample, session) so
    wrappers created for a previous run are not reused for this one.

    Args:
        run_id: Identifier of the run; ``None`` resets to the default scope.
    """
    _current_run_id.set(run_id or "default")


def get_ace_run_scope() -> str:
    """Return the run id that ACE wrapper state is currently bound to."""
    return _current_run_id.get()


def clear_ace_cache(run_id: str | None = None) -> None:
    """Clear ACE wrapper cache.

    Args:
        run_id: Clear only this run's wrappers. ``None`` clears every run.
    """
    if run_id is None:
        _ace_wrappers.clear()
        return
    for key in [k for k in _ace_wrappers if k[0] == run_id]:
        del _ace_wrappers[key]


def get_ace_wrapper(agent_name: str) -> Optional["BioAgentsACE"]:
    """
    Get or create ACE wrapper for an agent (lazy initialization).

    Wrappers are cached per ``(run scope, agent)`` so learning state does not
    carry across unrelated runs in the same process.

    Args:
        agent_name: Name of the agent (e.g., 'supervisor', 'coder')

    Returns:
        ACE wrapper instance if enabled, None otherwise
    """
    cache_key = (get_ace_run_scope(), agent_name)
    try:
        from bioagents.learning import is_ace_enabled

        if not is_ace_enabled():
            if cache_key in _ace_wrappers:
                _ace_wrappers[cache_key] = None
            return None
    except ImportError:
        return None

    # Only create wrapper if not in cache or if it was None (ACE was disabled before)
    if cache_key not in _ace_wrappers or _ace_wrappers[cache_key] is None:
        try:
            from bioagents.learning import create_ace_wrapper_if_enabled
            from bioagents.llms.llm_provider import get_llm

            llm = get_llm()
            _ace_wrappers[cache_key] = create_ace_wrapper_if_enabled(
                agent_name=agent_name,
                llm_provider=llm,
            )
        except Exception:
            # If ACE is not available or fails to initialize, disable it
            _ace_wrappers[cache_key] = None

    return _ace_wrappers.get(cache_key)


def extract_task_from_messages(messages: list[BaseMessage]) -> str:
    """
    Extract the original task/query from messages.

    Args:
        messages: List of messages in the conversation

    Returns:
        Task string (truncated to 500 chars)
    """
    for msg in messages:
        if isinstance(msg, HumanMessage) and hasattr(msg, "content"):
            content = msg.content

            if isinstance(content, list):
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

            if isinstance(content, list):
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
    ace_agent_name = map_agent_name_to_ace_name(agent_name)
    ace_wrapper = get_ace_wrapper(ace_agent_name)

    if ace_wrapper is None:
        return

    if not hasattr(ace_wrapper, "is_enabled") or not ace_wrapper.is_enabled():
        return

    try:
        task = extract_task_from_messages(state.get("messages", []))
        output = extract_output_from_result(result)

        # Determine success (simplified - could be enhanced)
        # For now, assume success if no error messages
        success = True
        error_message = None

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

        ace_wrapper.track_execution(
            task=task,
            output=output,
            success=success,
            error_message=error_message,
        )

        ace_wrapper.evolve_cycle(
            task=task,
            agent_output=output,
            success=success,
        )
    except Exception as e:
        # Don't let ACE errors break the normal flow
        logger.warning(f"ACE tracking failed for {agent_name}: {e}")
