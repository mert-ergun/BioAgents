"""STELLA evaluation criteria implementation."""

import logging
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from bioagents.benchmarks.models import BenchmarkResult, STELLAEvaluationScore

logger = logging.getLogger(__name__)


def load_evaluation_criteria(criteria_file: str | Path) -> dict[str, Any]:
    """
    Load evaluation criteria from markdown file.

    Args:
        criteria_file: Path to evaluation_criteria.md

    Returns:
        Dictionary with evaluation criteria information
    """
    criteria_file = Path(criteria_file)
    if not criteria_file.exists():
        logger.warning(f"Evaluation criteria file not found: {criteria_file}")
        return {}

    with criteria_file.open("r", encoding="utf-8") as f:
        content = f.read()

    # Parse markdown to extract criteria (basic implementation)
    # In a full implementation, this would parse the structured markdown
    return {"content": content}


def evaluate_task_stella(
    result: BenchmarkResult,
    criteria: dict[str, Any] | None = None,
    use_llm_scoring: bool = True,  # Default to True for LLM-based scoring
    llm: Any = None,
) -> STELLAEvaluationScore:
    """
    Evaluate a benchmark task result using STELLA criteria.

    Args:
        result: BenchmarkResult to evaluate
        criteria: Evaluation criteria dictionary (optional)
        use_llm_scoring: Whether to use LLM for scoring (default: True)
        llm: LLM instance for scoring (if use_llm_scoring is True)

    Returns:
        STELLAEvaluationScore with dimension scores

    Note:
        Uses LLM-based scoring by default for accurate evaluation.
        Falls back to heuristic-based scoring if LLM is not available.
    """
    scores = STELLAEvaluationScore()

    # Use LLM-based scoring if requested and available
    if use_llm_scoring and llm:
        try:
            scores = _evaluate_with_llm(result, llm, criteria)
            logger.info("LLM-based evaluation completed successfully")
            return scores
        except Exception as e:
            logger.warning(f"LLM evaluation failed, falling back to heuristic scores: {e}")

    # Fallback to heuristic-based scoring

    # Technical Accuracy (0-5.0, 25% weight)
    # Based on workflow completion and error absence
    if result.workflow_completed and not result.error_message:
        scores.technical_accuracy = 4.0  # Base score for successful completion
        if result.final_message_count > 0:
            scores.technical_accuracy += 0.5  # Bonus for meaningful output
        if result.total_steps > 0:
            scores.technical_accuracy += 0.5  # Bonus for successful execution
    elif result.workflow_completed:
        scores.technical_accuracy = 3.0  # Completed but with errors
    else:
        scores.technical_accuracy = 1.0  # Failed to complete

    # Cap at 5.0
    scores.technical_accuracy = min(5.0, scores.technical_accuracy)

    # Domain Knowledge (0-5.0, 20% weight)
    # Based on output quality (heuristic: longer output suggests more domain knowledge)
    if result.raw_output:
        output_length = len(result.raw_output)
        if output_length > 500:
            scores.domain_knowledge = 4.0
        elif output_length > 200:
            scores.domain_knowledge = 3.0
        elif output_length > 50:
            scores.domain_knowledge = 2.0
        else:
            scores.domain_knowledge = 1.0
    else:
        scores.domain_knowledge = 0.5

    # Analytical Quality (0-5.0, 20% weight)
    # Based on workflow complexity (number of steps) and completion
    if result.total_steps > 10 and result.workflow_completed:
        scores.analytical_quality = 4.0
    elif result.total_steps > 5 and result.workflow_completed:
        scores.analytical_quality = 3.0
    elif result.workflow_completed:
        scores.analytical_quality = 2.0
    else:
        scores.analytical_quality = 1.0

    # Innovation Impact (0-5.0, 15% weight)
    # Based on tool usage (if ToolBuilder was used, suggests innovation)
    if "tool_builder" in result.agent_flow:
        scores.innovation_impact = 4.0
    elif result.workflow_completed:
        scores.innovation_impact = 3.0
    else:
        scores.innovation_impact = 2.0

    # Communication Quality (0-5.0, 20% weight)
    # Based on message count and output presence
    if result.final_message_count > 3 and result.raw_output:
        scores.communication_quality = 4.0
    elif result.final_message_count > 1:
        scores.communication_quality = 3.0
    elif result.final_message_count > 0:
        scores.communication_quality = 2.0
    else:
        scores.communication_quality = 1.0

    return scores


class STELLADimensionScore(BaseModel):
    """Score for a single STELLA evaluation dimension."""

    score: float = Field(
        description="Score from 0.0 to 5.0, evaluated to 0.25 point precision (e.g., 4.0, 4.25, 4.5, etc.)"
    )
    reasoning: str = Field(
        description="Brief explanation of the score, highlighting key strengths and weaknesses"
    )


class STELLACompleteEvaluation(BaseModel):
    """Complete STELLA evaluation scores for all dimensions."""

    technical_accuracy: STELLADimensionScore = Field(
        description="Technical Accuracy score (0-5.0, 25% weight). Measures correctness and precision of scientific content and methodology."
    )
    domain_knowledge: STELLADimensionScore = Field(
        description="Domain Knowledge score (0-5.0, 20% weight). Evaluates understanding of bioengineering concepts and principles."
    )
    analytical_quality: STELLADimensionScore = Field(
        description="Analytical Quality score (0-5.0, 20% weight). Assesses depth and sophistication of analysis."
    )
    innovation_impact: STELLADimensionScore = Field(
        description="Innovation Impact score (0-5.0, 15% weight). Evaluates creative problem-solving and practical applicability."
    )
    communication_quality: STELLADimensionScore = Field(
        description="Communication Quality score (0-5.0, 20% weight). Assesses clarity, organization, and professionalism of output."
    )


def _evaluate_with_llm(
    result: BenchmarkResult, llm: Any, criteria: dict[str, Any] | None = None
) -> STELLAEvaluationScore:
    """
    Use LLM to evaluate task results based on STELLA criteria.

    This uses structured prompts to evaluate each dimension according to
    the STELLA evaluation criteria.

    Args:
        result: BenchmarkResult to evaluate
        llm: LLM instance for scoring
        criteria: Evaluation criteria dictionary (optional)

    Returns:
        STELLAEvaluationScore with LLM-generated scores
    """
    # Build evaluation context
    task_description = result.task.task_description or result.task.task_name
    task_type = result.task.task_type
    raw_output = (
        result.raw_output or "\n".join(result.final_messages[-3:])
        if result.final_messages
        else "No output"
    )
    workflow_path = result.workflow_path
    error_message = result.error_message
    workflow_completed = result.workflow_completed

    # Load criteria content if available
    criteria_content = ""
    if criteria and "content" in criteria:
        criteria_content = criteria["content"]
    else:
        # Try to load from default location (relative to project root)
        # Path: bioagents/benchmarks/stella_evaluator.py -> benchmarks/STELLA/evaluation_criteria.md
        criteria_file = (
            Path(__file__).parent.parent.parent / "benchmarks" / "STELLA" / "evaluation_criteria.md"
        )
        if criteria_file.exists():
            with criteria_file.open("r", encoding="utf-8") as f:
                criteria_content = f.read()
        else:
            logger.warning(f"Evaluation criteria file not found at {criteria_file}")

    # Build comprehensive evaluation prompt
    evaluation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert evaluator for the STELLA (Scientific Tool Evaluation and Learning Assessment) benchmark.

Your task is to evaluate a bioinformatics agent's response to a scientific task using the STELLA evaluation criteria.

EVALUATION CRITERIA:
{criteria}

SCORING GUIDELINES:
- Scores must be between 0.0 and 5.0
- Scores should be evaluated to 0.25 point precision (e.g., 4.0, 4.25, 4.5, 4.75, 5.0)
- Be strict but fair: 5.0 should be reserved for exceptional performance
- Consider both what was done well and what was missing or incorrect
- For each dimension, provide a score and brief reasoning

DIMENSION DESCRIPTIONS:

1. Technical Accuracy (25% weight):
   - Measures correctness and precision of scientific content and methodology
   - Key points: Data accuracy, methodology correctness, calculation precision, protocol accuracy, database query accuracy, tool specificity, parameter validation, error handling
   - 5.0: Perfect accuracy with comprehensive detail
   - 4.0-4.9: Very accurate with minor errors
   - 3.0-3.9: Generally accurate with some errors
   - <3.0: Major errors or omissions

2. Domain Knowledge (20% weight):
   - Evaluates understanding of bioengineering concepts and principles
   - Key points: Biological concept understanding, technical terminology, cross-disciplinary integration, current research awareness, field-specific best practices
   - 5.0: Expert-level understanding with cross-domain integration
   - 4.0-4.9: Strong understanding with good context
   - 3.0-3.9: Good understanding with some context
   - <3.0: Superficial or incorrect understanding

3. Analytical Quality (20% weight):
   - Assesses depth and sophistication of analysis
   - Key points: Analysis depth, logical reasoning, data interpretation, statistical rigor, critical evaluation, alternative methods, result validation, limitation discussion
   - 5.0: Comprehensive multi-level analysis with novel insights
   - 4.0-4.9: Thorough analysis with good reasoning
   - 3.0-3.9: Good analysis with some depth
   - <3.0: Superficial or flawed analysis

4. Innovation Impact (15% weight):
   - Evaluates creative problem-solving and practical applicability
   - Key points: Solution creativity, practical applicability, future implications, alternative approaches, improvement suggestions, resource optimization
   - 5.0: Groundbreaking insights with immediate practical value
   - 4.0-4.9: Creative solutions with good utility
   - 3.0-3.9: Useful insights with some creativity
   - <3.0: Minimal innovation or impractical solutions

5. Communication Quality (20% weight):
   - Assesses clarity, organization, and professionalism of output
   - Key points: Information structure, clarity of expression, professional formatting, completeness, accessibility, documentation quality
   - 5.0: Exceptional clarity with perfect organization
   - 4.0-4.9: Very clear with good organization
   - 3.0-3.9: Clear with adequate organization
   - <3.0: Unclear or poorly organized

Evaluate the response and provide scores for all five dimensions.""",
            ),
            (
                "human",
                """TASK INFORMATION:
Task Name: {task_name}
Task Type: {task_type}
Task Description: {task_description}

WORKFLOW EXECUTION:
Workflow Completed: {workflow_completed}
Workflow Path: {workflow_path}
Total Steps: {total_steps}
Execution Time: {execution_time:.2f}s
{error_section}

AGENT RESPONSE:
{raw_output}

Please evaluate this response according to the STELLA criteria and provide scores for all five dimensions.""",
            ),
        ]
    )

    # Prepare error section
    error_section = ""
    if error_message:
        error_section = f"Error Message: {error_message}\n"
    if not workflow_completed:
        error_section += "Note: Workflow did not complete successfully.\n"

    # Limit content lengths to avoid token limits
    criteria_text = (
        criteria_content[:5000] if criteria_content else "See STELLA evaluation criteria document."
    )
    output_text = raw_output[:8000]  # Limit output length

    # Get LLM evaluation
    try:
        chain = evaluation_prompt | llm.with_structured_output(STELLACompleteEvaluation)
        evaluation = chain.invoke(
            {
                "criteria": criteria_text,
                "task_name": result.task.task_name,
                "task_type": task_type,
                "task_description": task_description,
                "workflow_completed": "Yes" if workflow_completed else "No",
                "workflow_path": workflow_path,
                "total_steps": result.total_steps,
                "execution_time": result.execution_time,
                "error_section": error_section,
                "raw_output": output_text,
            }
        )

        # Convert to STELLAEvaluationScore
        scores = STELLAEvaluationScore(
            technical_accuracy=max(0.0, min(5.0, evaluation.technical_accuracy.score)),
            domain_knowledge=max(0.0, min(5.0, evaluation.domain_knowledge.score)),
            analytical_quality=max(0.0, min(5.0, evaluation.analytical_quality.score)),
            innovation_impact=max(0.0, min(5.0, evaluation.innovation_impact.score)),
            communication_quality=max(0.0, min(5.0, evaluation.communication_quality.score)),
        )

        logger.info(
            f"LLM evaluation scores: Technical={scores.technical_accuracy:.2f}, "
            f"Domain={scores.domain_knowledge:.2f}, Analytical={scores.analytical_quality:.2f}, "
            f"Innovation={scores.innovation_impact:.2f}, Communication={scores.communication_quality:.2f}"
        )

        return scores

    except Exception as e:
        logger.error(f"LLM evaluation failed: {e}")
        raise
