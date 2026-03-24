"""Generic LLM-as-a-judge for BioAgents use-case evaluation."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from bioagents.benchmarks.use_case_models import JudgeBreakdown, RunResult, UseCase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output
# ---------------------------------------------------------------------------


class DimensionScore(BaseModel):
    """Score and reasoning for a single evaluation dimension."""

    score: float = Field(
        description=(
            "Score between 0.0 and 5.0 at 0.25 precision. "
            "5.0 = exceptional, 4.0 = good, 3.0 = adequate, "
            "2.0 = poor, 1.0 = very poor, 0.0 = completely failed."
        )
    )
    reasoning: str = Field(description="One or two sentences explaining the score.")


class JudgeEvaluation(BaseModel):
    """Complete evaluation output from the LLM judge."""

    overall_score: float = Field(
        description=(
            "Overall quality score between 0.0 and 5.0 at 0.25 precision. "
            "Weighted holistic assessment across all dimensions."
        )
    )
    overall_reasoning: str = Field(
        description=(
            "2-4 sentence explanation of the overall score, highlighting key strengths "
            "and the most important weaknesses."
        )
    )
    correctness: DimensionScore = Field(
        description="Did the agent answer the question correctly and accurately?"
    )
    tool_use: DimensionScore = Field(
        description="Did the agent use appropriate tools and use them effectively?"
    )
    clarity: DimensionScore = Field(
        description="Is the response well-structured, clear, and easy to understand?"
    )
    completeness: DimensionScore = Field(
        description="Did the response fully address the prompt, including all sub-questions?"
    )


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator for a multi-agent bioinformatics AI system called BioAgents.

Your task is to evaluate the agent's response to a user prompt across four dimensions:
1. **Correctness** - factual accuracy of the scientific content.
2. **Tool Use** - whether the right tools were called with appropriate inputs.
3. **Clarity** - how well-structured, readable, and professional the response is.
4. **Completeness** - whether all aspects of the prompt were addressed.

Score each dimension and provide an overall score, all on a 0.0 - 5.0 scale
(0.25 increment precision). Be strict but fair: 5.0 should be exceptional.

{criteria_section}
"""

_JUDGE_HUMAN_PROMPT = """\
## User Prompt
{prompt}

{reference_section}

## Agent Execution Summary
- Workflow completed: {workflow_completed}
- Steps taken: {total_steps}
- Execution time: {execution_time:.1f}s
- Agent flow: {agent_flow}
- Tools called: {tools_summary}
{error_section}

## Agent Response
{raw_output}

Evaluate the response across all four dimensions and provide an overall score.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def judge_run_result(
    use_case: UseCase,
    result: RunResult,
    llm: Any,
) -> tuple[float, str, JudgeBreakdown]:
    """
    Evaluate a RunResult against its UseCase using an LLM judge.

    Args:
        use_case: The use case that was run (provides expectations and criteria).
        result: The RunResult to evaluate.
        llm: A LangChain LLM instance (must support structured output).

    Returns:
        Tuple of (overall_score, overall_reasoning, JudgeBreakdown).

    Raises:
        Exception: If the LLM call fails (caller should handle and fall back).
    """
    criteria_section = _build_criteria_section(use_case)
    reference_section = _build_reference_section(use_case)
    error_section = _build_error_section(result)
    tools_summary = _build_tools_summary(result)
    agent_flow_str = " → ".join(result.agent_flow[-20:]) if result.agent_flow else "(none)"

    raw_output = result.raw_output or ""
    if not raw_output and result.final_messages:
        raw_output = result.final_messages[-1]
    if len(raw_output) > 8000:
        raw_output = raw_output[:8000] + "\n…[truncated]"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _JUDGE_SYSTEM_PROMPT),
            ("human", _JUDGE_HUMAN_PROMPT),
        ]
    )

    chain = prompt | llm.with_structured_output(JudgeEvaluation)

    evaluation: JudgeEvaluation = chain.invoke(
        {
            "criteria_section": criteria_section,
            "prompt": use_case.prompt.strip(),
            "reference_section": reference_section,
            "workflow_completed": "Yes" if result.workflow_completed else "No",
            "total_steps": result.total_steps,
            "execution_time": result.execution_time,
            "agent_flow": agent_flow_str,
            "tools_summary": tools_summary,
            "error_section": error_section,
            "raw_output": raw_output if raw_output else "(No output captured)",
        }
    )

    overall_score = max(0.0, min(5.0, evaluation.overall_score))

    breakdown = JudgeBreakdown(
        correctness=max(0.0, min(5.0, evaluation.correctness.score)),
        tool_use=max(0.0, min(5.0, evaluation.tool_use.score)),
        clarity=max(0.0, min(5.0, evaluation.clarity.score)),
        completeness=max(0.0, min(5.0, evaluation.completeness.score)),
        dimensions={
            "correctness_reasoning": evaluation.correctness.reasoning,
            "tool_use_reasoning": evaluation.tool_use.reasoning,
            "clarity_reasoning": evaluation.clarity.reasoning,
            "completeness_reasoning": evaluation.completeness.reasoning,
        },
    )

    logger.info(
        "Judge score for '%s': %.2f (correctness=%.2f, tool_use=%.2f, clarity=%.2f, completeness=%.2f)",
        use_case.name,
        overall_score,
        evaluation.correctness.score,
        evaluation.tool_use.score,
        evaluation.clarity.score,
        evaluation.completeness.score,
    )

    return overall_score, evaluation.overall_reasoning, breakdown


def heuristic_score(use_case: UseCase, result: RunResult) -> tuple[float, str, JudgeBreakdown]:
    """
    Fast heuristic fallback when an LLM judge is not available.

    Produces a rough score based on:
    - Workflow completion
    - Presence of expected agents
    - Presence of expected tools
    - Output length
    - Absence of errors
    """
    score = 0.0
    notes: list[str] = []

    # Base score for completion
    if result.workflow_completed:
        score += 2.0
        notes.append("Workflow completed.")
    else:
        notes.append(f"Workflow did NOT complete ({result.failure_mode.value}).")

    # Expected agents
    if use_case.expected_agents:
        found = [a for a in use_case.expected_agents if a in result.agent_flow]
        ratio = len(found) / len(use_case.expected_agents)
        score += ratio * 1.0
        notes.append(
            f"Agent coverage: {len(found)}/{len(use_case.expected_agents)} expected agents found."
        )

    # Expected tools
    actual_tools = {tc.tool_name for tc in result.tool_calls}
    if use_case.expected_tools:
        found_tools = [t for t in use_case.expected_tools if t in actual_tools]
        ratio = len(found_tools) / len(use_case.expected_tools)
        score += ratio * 1.0
        notes.append(
            f"Tool coverage: {len(found_tools)}/{len(use_case.expected_tools)} expected tools used."
        )

    # Output quality proxy (length)
    output_len = len(result.raw_output or "")
    if output_len > 500:
        score += 1.0
        notes.append("Substantial output produced.")
    elif output_len > 100:
        score += 0.5
        notes.append("Brief output produced.")
    else:
        notes.append("Output is very short or empty.")

    score = min(5.0, score)

    breakdown = JudgeBreakdown(
        dimensions={"method": "heuristic", "notes": "; ".join(notes)},
    )
    reasoning = " ".join(notes)
    return score, reasoning, breakdown


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_criteria_section(use_case: UseCase) -> str:
    parts: list[str] = []

    if use_case.evaluation_criteria:
        parts.append(f"## Evaluation Criteria\n{use_case.evaluation_criteria}")

    if use_case.expected_agents:
        parts.append(f"Expected agents: {', '.join(use_case.expected_agents)}")

    if use_case.expected_tools:
        parts.append(f"Expected tools: {', '.join(use_case.expected_tools)}")

    if use_case.expected_output_contains:
        items = "\n".join(f"- {s}" for s in use_case.expected_output_contains)
        parts.append(f"Response must contain:\n{items}")

    return "\n\n".join(parts) if parts else ""


def _build_reference_section(use_case: UseCase) -> str:
    if use_case.reference_output:
        return f"## Reference / Expected Output\n{use_case.reference_output}"
    return ""


def _build_error_section(result: RunResult) -> str:
    if result.error_message:
        return f"- Error: {result.error_message}\n- Failure mode: {result.failure_mode.value}"
    return ""


def _build_tools_summary(result: RunResult) -> str:
    if not result.tool_calls:
        return "(no tools called)"
    names = [tc.tool_name for tc in result.tool_calls]
    unique = list(dict.fromkeys(names))  # preserve order, deduplicate
    return f"{len(result.tool_calls)} calls — {', '.join(unique)}"
