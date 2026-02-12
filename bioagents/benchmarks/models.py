"""Data models for benchmark framework."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class STELLATask:
    """Represents a STELLA benchmark task."""

    task_name: str
    task_type: str
    task_description: str
    input: dict[str, Any]
    output: dict[str, Any] | None = None
    success_criteria: list[str] = field(default_factory=list)
    source: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "STELLATask":
        """Create STELLATask from dictionary."""
        # Handle task_name which might be a string directly
        task_name = data.get("task_name", "")
        if isinstance(task_name, dict):
            # If it's a dict, try to extract the value
            task_name = next(iter(task_name.values())) if task_name else ""
        elif not isinstance(task_name, str):
            task_name = str(task_name)

        # Handle task_type which might be a string directly
        task_type = data.get("task_type", "")
        if isinstance(task_type, dict):
            task_type = next(iter(task_type.values())) if task_type else ""
        elif not isinstance(task_type, str):
            task_type = str(task_type)

        # Handle task_description
        task_description = data.get("task_description", "")
        if isinstance(task_description, dict):
            task_description = next(iter(task_description.values())) if task_description else ""
        elif not isinstance(task_description, str):
            task_description = str(task_description)

        return cls(
            task_name=task_name,
            task_type=task_type,
            task_description=task_description,
            input=data.get("input", {}),
            output=data.get("output"),
            success_criteria=data.get("success_criteria", []),
            source=data.get("source"),
        )

    def to_query_string(self) -> str:
        """Convert task input to a query string for BioAgents."""
        # If input has a 'query' key, use it directly
        if "query" in self.input:
            query = self.input["query"]
            if isinstance(query, str) and query.strip():
                return query
            if isinstance(query, list) and len(query) > 0:
                query_str = str(query[0])
                if query_str.strip():
                    return query_str

        # Use task description if available and non-empty
        if self.task_description and self.task_description.strip():
            return self.task_description

        # Fall back to task_type if description is empty
        if self.task_type and self.task_type.strip():
            return self.task_type

        # Last resort: use task name
        if self.task_name:
            # Convert task name to a more readable query
            # e.g., "Non_cutter_Restriction_Enzyme_Identification_pCMV_PE6c" ->
            # "Identify all restriction enzymes that do NOT cut within the pCMV-PE6c plasmid sequence"
            name = self.task_name.replace("_", " ")
            return name

        # Should not reach here, but provide a fallback
        return "Execute the task"


@dataclass
class STELLAEvaluationScore:
    """Evaluation scores for a single task across STELLA dimensions."""

    technical_accuracy: float = 0.0  # 0-5.0, 25% weight
    domain_knowledge: float = 0.0  # 0-5.0, 20% weight
    analytical_quality: float = 0.0  # 0-5.0, 20% weight
    innovation_impact: float = 0.0  # 0-5.0, 15% weight
    communication_quality: float = 0.0  # 0-5.0, 20% weight

    def overall_score(self) -> float:
        """Calculate weighted overall score."""
        return (
            self.technical_accuracy * 0.25
            + self.domain_knowledge * 0.20
            + self.analytical_quality * 0.20
            + self.innovation_impact * 0.15
            + self.communication_quality * 0.20
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "technical_accuracy": self.technical_accuracy,
            "domain_knowledge": self.domain_knowledge,
            "analytical_quality": self.analytical_quality,
            "innovation_impact": self.innovation_impact,
            "communication_quality": self.communication_quality,
            "overall_score": self.overall_score(),
        }


@dataclass
class BenchmarkResult:
    """Extended result for benchmark evaluation with STELLA scores."""

    task: STELLATask
    query: str
    execution_time: float
    total_steps: int
    workflow_completed: bool
    final_message_count: int
    error_message: str | None
    agent_flow: list[str]
    final_messages: list[str]
    workflow_path: str
    stella_scores: STELLAEvaluationScore | None = None
    raw_output: str | None = None  # Full output from workflow

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "task_name": self.task.task_name,
            "task_type": self.task.task_type,
            "query": self.query,
            "execution_time": self.execution_time,
            "total_steps": self.total_steps,
            "workflow_completed": self.workflow_completed,
            "final_message_count": self.final_message_count,
            "error_message": self.error_message,
            "agent_flow": self.agent_flow,
            "final_messages": self.final_messages,
            "workflow_path": self.workflow_path,
            "raw_output": self.raw_output,
        }
        if self.stella_scores:
            result["stella_scores"] = self.stella_scores.to_dict()
        return result
