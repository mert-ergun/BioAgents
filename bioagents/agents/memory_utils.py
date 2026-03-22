"""Utilities for working with shared memory in multi-agent system."""

import json
from datetime import datetime
from typing import Any


def write_to_memory(
    state: dict,
    name: str,
    data: dict[str, Any],
    raw_output: str = "",
    tool_calls: list[str] | None = None,
    error: str | None = None,
) -> None:
    """
    Write agent results to shared memory.

    Args:
        state: The AgentState dict
        name: Name of the agent writing
        data: Structured JSON-like results
        raw_output: Full text output
        tool_calls: List of tool names used
        error: Any error encountered
    """
    memory_key = name.lower()
    state["memory"][memory_key] = {
        "status": "error" if error else "success",
        "timestamp": datetime.now().isoformat(),
        "data": data,
        "raw_output": raw_output,
        "errors": [error] if error else [],
        "tool_calls": tool_calls or [],
    }


def read_from_memory(
    state: dict,
    agent_name: str,
    default: Any = None,
) -> Any:
    """
    Read structured data from shared memory.

    Args:
        state: The AgentState dict
        agent_name: Name of the agent to read from
        default: Default value if not found

    Returns:
        Agent's data from memory, or default
    """
    memory_key = agent_name.lower()
    agent_mem = state.get("memory", {}).get(memory_key, {})
    return agent_mem.get("data", default)


def get_memory_summary(state: dict) -> str:
    """
    Get a human-readable summary of memory state.

    Args:
        state: The AgentState dict

    Returns:
        Formatted summary string
    """
    memory = state.get("memory", {})
    lines = []

    for agent_name, agent_data in sorted(memory.items()):
        status = agent_data.get("status", "unknown")
        has_data = bool(agent_data.get("data", {}))
        lines.append(f"{agent_name}: status={status}, has_data={has_data}")

    return "\n".join(lines)


def get_completed_agents(state: dict) -> list[str]:
    """
    Get list of agents that completed successfully.

    Args:
        state: The AgentState dict

    Returns:
        List of agent names
    """
    memory = state.get("memory", {})
    return [name for name, data in memory.items() if data.get("status") == "success"]
