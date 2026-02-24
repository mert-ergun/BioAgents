"""Data models for the modular use-case and experiment system."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime


class FailureMode(str, Enum):
    """Failure mode classification for a use-case run."""

    COMPLETED = "completed"
    TIMEOUT = "timeout"
    MAX_STEPS = "max_steps"
    EXCEPTION = "exception"
    INCOMPLETE = "incomplete"


@dataclass
class UseCase:
    """
    A single use-case defining a prompt and optional expected behavior.

    Use cases are the atomic unit of evaluation. They can be defined in YAML
    files and loaded via use_case_loader.py.
    """

    id: str
    name: str
    prompt: str
    description: str = ""
    # Optional expectations for the judge / strict checks
    expected_agents: list[str] = field(default_factory=list)
    expected_tools: list[str] = field(default_factory=list)
    expected_output_contains: list[str] = field(default_factory=list)
    reference_output: str | None = None
    evaluation_criteria: str | None = None
    # Metadata for filtering / display
    tags: list[str] = field(default_factory=list)
    category: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UseCase:
        return cls(
            id=str(data.get("id", uuid.uuid4().hex[:8])),
            name=str(data.get("name", "")),
            prompt=str(data.get("prompt", "")),
            description=str(data.get("description", "")),
            expected_agents=list(data.get("expected_agents", [])),
            expected_tools=list(data.get("expected_tools", [])),
            expected_output_contains=list(data.get("expected_output_contains", [])),
            reference_output=data.get("reference_output"),
            evaluation_criteria=data.get("evaluation_criteria"),
            tags=list(data.get("tags", [])),
            category=data.get("category"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "prompt": self.prompt,
            "description": self.description,
            "expected_agents": self.expected_agents,
            "expected_tools": self.expected_tools,
            "expected_output_contains": self.expected_output_contains,
            "reference_output": self.reference_output,
            "evaluation_criteria": self.evaluation_criteria,
            "tags": self.tags,
            "category": self.category,
        }


@dataclass
class ExperimentConfig:
    """
    Configuration for an experiment run.

    The same set of use cases can be run with different configs to compare
    baseline vs. tweaked prompts, different models, etc.
    """

    name: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str = ""
    max_steps: int = 50
    timeout: int = 180
    # Mapping of prompt_name -> override text (replaces the XML-loaded prompt)
    system_prompt_overrides: dict[str, str] = field(default_factory=dict)
    # Optional model override: {"provider": "openai", "model": "gpt-4o"}
    model_override: dict[str, str] = field(default_factory=dict)
    temperature: float = 0.0
    # Extra free-form parameters for future use
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        return cls(
            id=str(data.get("id", uuid.uuid4().hex[:8])),
            name=str(data.get("name", "default")),
            description=str(data.get("description", "")),
            max_steps=int(data.get("max_steps", 50)),
            timeout=int(data.get("timeout", 180)),
            system_prompt_overrides=dict(data.get("system_prompt_overrides", {})),
            model_override=dict(data.get("model_override", {})),
            temperature=float(data.get("temperature", 0.0)),
            extra=dict(data.get("extra", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "max_steps": self.max_steps,
            "timeout": self.timeout,
            "system_prompt_overrides": self.system_prompt_overrides,
            "model_override": self.model_override,
            "temperature": self.temperature,
            "extra": self.extra,
        }


@dataclass
class ToolCallRecord:
    """Record of a single tool call made during a run."""

    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    result_preview: str = ""
    tool_call_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "args": self.args,
            "result_preview": self.result_preview,
            "tool_call_id": self.tool_call_id,
        }


@dataclass
class TokenUsage:
    """Token usage for a single run (if available from LLM metadata)."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class JudgeBreakdown:
    """Optional per-dimension breakdown from the judge."""

    correctness: float | None = None
    tool_use: float | None = None
    clarity: float | None = None
    completeness: float | None = None
    # Raw per-dimension dict for extensibility (scores, reasoning, or notes)
    dimensions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "correctness": self.correctness,
            "tool_use": self.tool_use,
            "clarity": self.clarity,
            "completeness": self.completeness,
            "dimensions": self.dimensions,
        }


@dataclass
class RunResult:
    """
    Result of running a single UseCase through BioAgents.

    Extends the information captured by BenchmarkResult to include
    tool_calls, token_usage, judge scores, and a structured failure_mode.
    """

    use_case_id: str
    use_case_name: str
    prompt: str
    execution_time: float
    total_steps: int
    workflow_completed: bool
    agent_flow: list[str]
    tool_calls: list[ToolCallRecord]
    final_messages: list[str]
    raw_output: str | None = None
    error_message: str | None = None
    failure_mode: FailureMode = FailureMode.COMPLETED
    token_usage: TokenUsage | None = None
    judge_score: float | None = None
    judge_reasoning: str | None = None
    judge_breakdown: JudgeBreakdown | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "use_case_id": self.use_case_id,
            "use_case_name": self.use_case_name,
            "prompt": self.prompt,
            "execution_time": self.execution_time,
            "total_steps": self.total_steps,
            "workflow_completed": self.workflow_completed,
            "agent_flow": self.agent_flow,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "final_messages": self.final_messages,
            "raw_output": self.raw_output,
            "error_message": self.error_message,
            "failure_mode": self.failure_mode.value,
            "token_usage": self.token_usage.to_dict() if self.token_usage else None,
            "judge_score": self.judge_score,
            "judge_reasoning": self.judge_reasoning,
            "judge_breakdown": self.judge_breakdown.to_dict() if self.judge_breakdown else None,
        }


@dataclass
class ExperimentRun:
    """
    A complete experiment run: one config applied to N use cases.

    Contains per-case RunResults plus aggregate metrics.
    """

    run_id: str
    experiment_config: ExperimentConfig
    use_cases: list[UseCase]
    results: list[RunResult]
    started_at: datetime
    finished_at: datetime | None = None

    # ---- aggregate helpers ----

    def mean_score(self) -> float | None:
        scored = [r.judge_score for r in self.results if r.judge_score is not None]
        return sum(scored) / len(scored) if scored else None

    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.workflow_completed) / len(self.results)

    def total_execution_time(self) -> float:
        return sum(r.execution_time for r in self.results)

    def total_tokens(self) -> int | None:
        totals = [
            r.token_usage.total_tokens
            for r in self.results
            if r.token_usage and r.token_usage.total_tokens is not None
        ]
        return sum(totals) if totals else None

    def failure_mode_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            key = r.failure_mode.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_config": self.experiment_config.to_dict(),
            "use_cases": [uc.to_dict() for uc in self.use_cases],
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "results": [r.to_dict() for r in self.results],
            "aggregates": {
                "mean_score": self.mean_score(),
                "success_rate": self.success_rate(),
                "total_execution_time": self.total_execution_time(),
                "total_tokens": self.total_tokens(),
                "failure_mode_counts": self.failure_mode_counts(),
                "total_use_cases": len(self.use_cases),
            },
        }
