"""Track agent executions for reflection cycle."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ExecutionRecord:
    """Record of an agent execution for reflection."""

    agent_name: str
    task: str
    output: str
    success: bool
    error_message: str | None = None
    instructions_used: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class ExecutionTracker:
    """Track agent executions for ACE reflection cycle."""

    def __init__(self, max_records: int = 100):
        """
        Initialize execution tracker.

        Args:
            max_records: Maximum number of records to keep in memory
        """
        self.max_records = max_records
        self.executions: list[ExecutionRecord] = []

    def track_execution(
        self,
        agent_name: str,
        task: str,
        output: str,
        success: bool,
        error_message: str | None = None,
        instructions_used: list[str] | None = None,
    ) -> None:
        """
        Record an agent execution.

        Args:
            agent_name: Name of the agent that executed
            task: The task that was executed
            output: The output from the agent
            success: Whether the execution was successful
            error_message: Error message if execution failed
            instructions_used: List of instruction IDs used (if available)
        """
        record = ExecutionRecord(
            agent_name=agent_name,
            task=task,
            output=output,
            success=success,
            error_message=error_message,
            instructions_used=instructions_used or [],
            timestamp=datetime.now(),
        )

        self.executions.append(record)

        # Keep only recent records
        if len(self.executions) > self.max_records:
            self.executions = self.executions[-self.max_records :]

    def get_recent_executions(
        self, agent_name: str | None = None, limit: int = 10
    ) -> list[ExecutionRecord]:
        """
        Get recent executions, optionally filtered by agent.

        Args:
            agent_name: Filter by agent name (None for all agents)
            limit: Maximum number of records to return

        Returns:
            List of recent execution records
        """
        records = self.executions

        if agent_name:
            records = [r for r in records if r.agent_name == agent_name]

        # Return most recent first
        return list(reversed(records[-limit:]))

    def clear(self) -> None:
        """Clear all execution records."""
        self.executions.clear()
