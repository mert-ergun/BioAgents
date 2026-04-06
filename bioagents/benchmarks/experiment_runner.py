"""Experiment runner: orchestrates use-case runs, judging, and persistence."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bioagents.benchmarks.judge import heuristic_score, judge_run_result
from bioagents.benchmarks.runner import run_use_case
from bioagents.benchmarks.use_case_models import (
    ExperimentConfig,
    ExperimentRun,
    RunResult,
    UseCase,
)
from bioagents.graph import create_graph
from bioagents.prompts.prompt_loader import set_experiment_prompt_overrides

logger = logging.getLogger(__name__)

# Default directory for persisting experiment run JSON files
DEFAULT_RUNS_DIR = Path(__file__).parent.parent.parent / "experiment_runs"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_experiment(
    use_cases: list[UseCase],
    config: ExperimentConfig | None = None,
    *,
    skip_judge: bool = False,
    judge_llm: Any = None,
    show_trace: bool = False,
    runs_dir: str | Path | None = None,
    save_results: bool = True,
    run_id: str | None = None,
) -> ExperimentRun:
    """
    Run a set of use cases with a given experiment config.

    For each use case:
    1. Apply any system_prompt_overrides from *config* via the context variable.
    2. Run the BioAgents graph (reusing a single compiled graph per experiment).
    3. Optionally judge the result with an LLM (falls back to heuristic scoring).
    4. Persist the full ExperimentRun to JSON.

    Args:
        use_cases: Use cases to evaluate.
        config: Experiment configuration; uses defaults if None.
        skip_judge: Skip LLM and heuristic judging (scores will be None).
        judge_llm: LangChain LLM for judging. If None, heuristic scoring is used.
        show_trace: Print per-step execution trace to stdout.
        runs_dir: Directory for JSON persistence. Defaults to ``experiment_runs/``.
        save_results: Whether to persist the run to disk.
        run_id: Optional pre-assigned run ID. Generated if not provided.

    Returns:
        ExperimentRun with per-case RunResults and aggregate metrics.
    """
    if config is None:
        config = ExperimentConfig(name="default")

    if run_id is None:
        run_id = uuid.uuid4().hex
    started_at = datetime.now(tz=UTC)

    logger.info(
        "Starting experiment run %s — config '%s' — %d use cases",
        run_id,
        config.name,
        len(use_cases),
    )

    # Apply prompt overrides for this run (context-var approach, Option A)
    if config.system_prompt_overrides:
        set_experiment_prompt_overrides(config.system_prompt_overrides)
        logger.info("Prompt overrides active: %s", list(config.system_prompt_overrides.keys()))
    else:
        set_experiment_prompt_overrides(None)

    # Cap per-LLM-call timeout so a single invoke cannot exceed the benchmark
    # wall-clock budget.  This must happen BEFORE create_graph() because each
    # TimeoutBoundLLM captures the limit at init time.
    import bioagents.limits as _limits

    _original_llm_timeout = _limits.AGENT_LLM_INVOKE_TIMEOUT_SEC
    _original_tool_rounds = _limits.MAX_AGENT_TOOL_ROUNDS
    if config.timeout and config.timeout > 0:
        capped = min(_original_llm_timeout, float(config.timeout))
        _limits.AGENT_LLM_INVOKE_TIMEOUT_SEC = capped
        _limits.MAX_AGENT_TOOL_ROUNDS = min(_original_tool_rounds, 3)
        logger.info(
            "LLM invoke timeout capped to %.0fs, tool rounds capped to %d "
            "(experiment timeout=%ds)",
            capped,
            _limits.MAX_AGENT_TOOL_ROUNDS,
            config.timeout,
        )

    # Build graph once for the entire experiment (re-use across use cases)
    graph = create_graph()

    results: list[RunResult] = []

    for idx, use_case in enumerate(use_cases, start=1):
        logger.info("[%d/%d] Running use case: %s", idx, len(use_cases), use_case.name)
        try:
            result = run_use_case(
                use_case=use_case,
                config=config,
                graph=graph,
                show_trace=show_trace,
            )
        except Exception as exc:
            logger.error("Unexpected error for use case '%s': %s", use_case.name, exc)
            # Build a minimal failed result so the experiment can continue
            from bioagents.benchmarks.use_case_models import FailureMode

            result = RunResult(
                use_case_id=use_case.id,
                use_case_name=use_case.name,
                prompt=use_case.prompt,
                execution_time=0.0,
                total_steps=0,
                workflow_completed=False,
                agent_flow=[],
                tool_calls=[],
                final_messages=[],
                error_message=str(exc),
                failure_mode=FailureMode.EXCEPTION,
            )

        # Judge
        if not skip_judge:
            try:
                if judge_llm is not None:
                    score, reasoning, breakdown = judge_run_result(use_case, result, judge_llm)
                else:
                    score, reasoning, breakdown = heuristic_score(use_case, result)
                result.judge_score = score
                result.judge_reasoning = reasoning
                result.judge_breakdown = breakdown
            except Exception as exc:
                logger.warning(
                    "Judging failed for '%s' (falling back to heuristic): %s",
                    use_case.name,
                    exc,
                )
                try:
                    score, reasoning, breakdown = heuristic_score(use_case, result)
                    result.judge_score = score
                    result.judge_reasoning = reasoning
                    result.judge_breakdown = breakdown
                except Exception as exc2:
                    logger.error("Heuristic scoring also failed: %s", exc2)

        results.append(result)
        logger.info(
            "  Completed '%s' in %.1fs — score=%s — failure_mode=%s",
            use_case.name,
            result.execution_time,
            f"{result.judge_score:.2f}" if result.judge_score is not None else "N/A",
            result.failure_mode.value,
        )

    # Clear prompt overrides after run
    set_experiment_prompt_overrides(None)

    # Restore original limits
    _limits.AGENT_LLM_INVOKE_TIMEOUT_SEC = _original_llm_timeout
    _limits.MAX_AGENT_TOOL_ROUNDS = _original_tool_rounds

    finished_at = datetime.now(tz=UTC)

    experiment_run = ExperimentRun(
        run_id=run_id,
        experiment_config=config,
        use_cases=use_cases,
        results=results,
        started_at=started_at,
        finished_at=finished_at,
    )

    if save_results:
        _persist_run(experiment_run, runs_dir)

    logger.info(
        "Experiment run %s finished — mean_score=%s — success_rate=%.0f%%",
        run_id,
        f"{experiment_run.mean_score():.2f}" if experiment_run.mean_score() is not None else "N/A",
        experiment_run.success_rate() * 100,
    )

    return experiment_run


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _persist_run(run: ExperimentRun, runs_dir: str | Path | None) -> Path:
    """Save an ExperimentRun as a JSON file and return the file path."""
    directory = Path(runs_dir) if runs_dir else DEFAULT_RUNS_DIR
    directory.mkdir(parents=True, exist_ok=True)

    timestamp = run.started_at.strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{run.run_id}.json"
    filepath = directory / filename

    with filepath.open("w", encoding="utf-8") as f:
        json.dump(run.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info("Experiment run saved to %s", filepath)
    return filepath


def load_run(run_id: str, runs_dir: str | Path | None = None) -> dict[str, Any] | None:
    """
    Load a persisted ExperimentRun dict by run_id.

    Args:
        run_id: The 32-character hex run_id.
        runs_dir: Directory to search. Defaults to ``experiment_runs/``.

    Returns:
        Raw dict (as saved by to_dict()) or None if not found.
    """
    directory = Path(runs_dir) if runs_dir else DEFAULT_RUNS_DIR
    if not directory.exists():
        return None

    for filepath in sorted(directory.glob(f"*_{run_id}.json")):
        try:
            with filepath.open("r", encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)
                return data
        except Exception as exc:
            logger.warning("Failed to load run file %s: %s", filepath, exc)

    return None


def list_runs(runs_dir: str | Path | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """
    List recent persisted ExperimentRuns (summary only).

    Returns dicts with run_id, config name, started_at, aggregates, etc.
    """
    directory = Path(runs_dir) if runs_dir else DEFAULT_RUNS_DIR
    if not directory.exists():
        return []

    summaries: list[dict[str, Any]] = []
    files = sorted(directory.glob("*.json"), reverse=True)[:limit]

    for filepath in files:
        try:
            with filepath.open("r", encoding="utf-8") as f:
                data = json.load(f)
            summaries.append(
                {
                    "run_id": data.get("run_id"),
                    "config_name": data.get("experiment_config", {}).get("name", ""),
                    "config_id": data.get("experiment_config", {}).get("id", ""),
                    "started_at": data.get("started_at"),
                    "finished_at": data.get("finished_at"),
                    "total_use_cases": data.get("aggregates", {}).get("total_use_cases", 0),
                    "mean_score": data.get("aggregates", {}).get("mean_score"),
                    "success_rate": data.get("aggregates", {}).get("success_rate"),
                    "total_execution_time": data.get("aggregates", {}).get("total_execution_time"),
                    "total_tokens": data.get("aggregates", {}).get("total_tokens"),
                    "failure_mode_counts": data.get("aggregates", {}).get(
                        "failure_mode_counts", {}
                    ),
                }
            )
        except Exception as exc:
            logger.warning("Failed to read summary from %s: %s", filepath, exc)

    return summaries
