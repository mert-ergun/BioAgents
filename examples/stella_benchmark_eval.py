"""STELLA Benchmark Evaluation Script.

Runs the STELLA Tool Creation Benchmark tasks through the BioAgents workflow
and evaluates results using STELLA evaluation criteria.

Usage:
    python stella_benchmark_eval.py              # Run all tasks
    python stella_benchmark_eval.py --task 1 5 10  # Run specific tasks
    python stella_benchmark_eval.py --type pathway_to_gene_mapping  # Run tasks by type
    python stella_benchmark_eval.py --output results/  # Specify output directory
"""

import argparse
import json
import logging
import textwrap
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from bioagents.benchmarks.loader import get_tasks_by_type, iter_tasks, load_stella_tasks
from bioagents.benchmarks.models import BenchmarkResult, STELLATask
from bioagents.benchmarks.runner import run_benchmark_task
from bioagents.benchmarks.stella_evaluator import (
    evaluate_task_stella,
    load_evaluation_criteria,
)
from bioagents.graph import create_graph
from bioagents.llms.llm_provider import get_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for beautiful terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Agent colors
    SUPERVISOR = "\033[95m"  # Magenta
    RESEARCH = "\033[94m"  # Blue
    ANALYSIS = "\033[96m"  # Cyan
    REPORT = "\033[93m"  # Yellow
    TOOL_BUILDER = "\033[92m"  # Green
    PROTEIN_DESIGN = "\033[96m"  # Cyan
    CRITIC = "\033[91m"  # Red
    CODER = "\033[93m"  # Yellow

    # Status colors
    SUCCESS = "\033[92m"  # Green
    WARNING = "\033[93m"  # Yellow
    ERROR = "\033[91m"  # Red
    INFO = "\033[94m"  # Blue

    # Node type colors
    TOOLS = "\033[90m"  # Dark gray
    END = "\033[92m"  # Green


def get_agent_color(node_name: str) -> str:
    """Get color code for an agent node."""
    node_lower = node_name.lower()
    if "supervisor" in node_lower:
        return Colors.SUPERVISOR
    elif "research" in node_lower:
        return Colors.RESEARCH
    elif "analysis" in node_lower:
        return Colors.ANALYSIS
    elif "report" in node_lower:
        return Colors.REPORT
    elif "tool_builder" in node_lower or "toolbuilder" in node_lower:
        return Colors.TOOL_BUILDER
    elif "protein_design" in node_lower or "proteindesign" in node_lower:
        return Colors.PROTEIN_DESIGN
    elif "critic" in node_lower:
        return Colors.CRITIC
    elif "coder" in node_lower:
        return Colors.CODER
    elif "tools" in node_lower:
        return Colors.TOOLS
    elif "end" in node_lower:
        return Colors.END
    else:
        return Colors.INFO


def format_node_name(node_name: str) -> str:
    """Format node name with color."""
    color = get_agent_color(node_name)
    return f"{color}{Colors.BOLD}{node_name.upper()}{Colors.RESET}"


def print_banner(text: str):
    """Print a banner."""
    print(f"\n{Colors.BOLD}{'=' * 100}{Colors.RESET}")
    print(f"{Colors.BOLD}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 100}{Colors.RESET}")


def print_task_header(task_num: int, total_tasks: int, task: STELLATask):
    """Print task header."""
    print(f"\n{Colors.BOLD}{'─' * 100}{Colors.RESET}")
    print(f"{Colors.BOLD}TASK {task_num}/{total_tasks}: {task.task_name}{Colors.RESET}")
    print(f"{Colors.DIM}Type: {task.task_type}{Colors.RESET}")
    print(
        f"{Colors.DIM}Description: {textwrap.shorten(task.task_description, width=90)}{Colors.RESET}"
    )
    print(f"{Colors.BOLD}{'─' * 100}{Colors.RESET}")


def print_task_result(result: BenchmarkResult):
    """Print task execution result."""
    status_icon = (
        f"{Colors.SUCCESS}✓{Colors.RESET}"
        if result.workflow_completed
        else f"{Colors.ERROR}✗{Colors.RESET}"
    )
    status_text = (
        f"{Colors.SUCCESS}Completed{Colors.RESET}"
        if result.workflow_completed
        else f"{Colors.ERROR}Failed{Colors.RESET}"
    )
    print(f"\n{Colors.BOLD}Execution Result:{Colors.RESET} {status_icon} {status_text}")
    print(f"Execution Time: {Colors.INFO}{result.execution_time:.2f}s{Colors.RESET}")
    print(f"Steps: {Colors.INFO}{result.total_steps}{Colors.RESET}")
    if result.error_message:
        print(f"{Colors.ERROR}Error: {result.error_message}{Colors.RESET}")

    # Show final workflow path
    if result.agent_flow:
        flow_colored = " → ".join([format_node_name(node) for node in result.agent_flow[-10:]])
        print(f"\n{Colors.DIM}Workflow Path: {flow_colored}{Colors.RESET}")

    if result.stella_scores:
        scores = result.stella_scores
        print(f"\n{Colors.BOLD}STELLA Evaluation Scores:{Colors.RESET}")
        print(
            f"  Technical Accuracy: {Colors.INFO}{scores.technical_accuracy:.2f}/5.0{Colors.RESET} (25% weight)"
        )
        print(
            f"  Domain Knowledge: {Colors.INFO}{scores.domain_knowledge:.2f}/5.0{Colors.RESET} (20% weight)"
        )
        print(
            f"  Analytical Quality: {Colors.INFO}{scores.analytical_quality:.2f}/5.0{Colors.RESET} (20% weight)"
        )
        print(
            f"  Innovation Impact: {Colors.INFO}{scores.innovation_impact:.2f}/5.0{Colors.RESET} (15% weight)"
        )
        print(
            f"  Communication Quality: {Colors.INFO}{scores.communication_quality:.2f}/5.0{Colors.RESET} (20% weight)"
        )
        print(
            f"  {Colors.BOLD}Overall Score: {Colors.SUCCESS}{scores.overall_score():.2f}/5.0{Colors.RESET}"
        )


def save_results(results: list[BenchmarkResult], output_dir: Path, timestamp: str):
    """Save results to JSON and text files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON results
    json_file = output_dir / f"stella_benchmark_results_{timestamp}.json"
    results_dict = [result.to_dict() for result in results]
    with json_file.open("w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    print(f"\n{Colors.INFO}Results saved to: {json_file}{Colors.RESET}")

    # Save text summary
    txt_file = output_dir / f"stella_benchmark_summary_{timestamp}.txt"
    with txt_file.open("w", encoding="utf-8") as f:
        f.write("STELLA Benchmark Evaluation Summary\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 100}\n\n")

        total_tasks = len(results)
        completed = sum(1 for r in results if r.workflow_completed)
        failed = total_tasks - completed

        f.write(f"Total Tasks: {total_tasks}\n")
        f.write(f"Completed: {completed}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Success Rate: {(completed / total_tasks * 100):.1f}%\n\n")

        # Calculate average scores
        if any(r.stella_scores for r in results):
            avg_scores = {
                "technical_accuracy": 0.0,
                "domain_knowledge": 0.0,
                "analytical_quality": 0.0,
                "innovation_impact": 0.0,
                "communication_quality": 0.0,
            }
            scored_count = 0
            for result in results:
                if result.stella_scores:
                    scores = result.stella_scores
                    avg_scores["technical_accuracy"] += scores.technical_accuracy
                    avg_scores["domain_knowledge"] += scores.domain_knowledge
                    avg_scores["analytical_quality"] += scores.analytical_quality
                    avg_scores["innovation_impact"] += scores.innovation_impact
                    avg_scores["communication_quality"] += scores.communication_quality
                    scored_count += 1

            if scored_count > 0:
                f.write(f"\nAverage STELLA Scores (over {scored_count} tasks):\n")
                for key, value in avg_scores.items():
                    avg = value / scored_count
                    f.write(f"  {key.replace('_', ' ').title()}: {avg:.2f}/5.0\n")

                # Overall average
                overall_avg = sum(avg_scores.values()) / (scored_count * 5.0) * 5.0
                f.write(f"\n  Overall Average: {overall_avg:.2f}/5.0\n")

        f.write(f"\n{'=' * 100}\n\n")

        # Per-task details
        for i, result in enumerate(results, 1):
            f.write(f"\nTask {i}: {result.task.task_name}\n")
            f.write(f"  Type: {result.task.task_type}\n")
            f.write(f"  Completed: {result.workflow_completed}\n")
            f.write(f"  Execution Time: {result.execution_time:.2f}s\n")
            f.write(f"  Steps: {result.total_steps}\n")
            if result.error_message:
                f.write(f"  Error: {result.error_message}\n")
            if result.stella_scores:
                scores = result.stella_scores
                f.write(f"  Overall Score: {scores.overall_score():.2f}/5.0\n")
            f.write("\n")

    print(f"{Colors.INFO}Summary saved to: {txt_file}{Colors.RESET}")


def main():
    """Run STELLA benchmark evaluation."""
    parser = argparse.ArgumentParser(
        description="Run STELLA Tool Creation Benchmark evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stella_benchmark_eval.py                    # Run all tasks
  python stella_benchmark_eval.py --task 1 5 10      # Run specific tasks
  python stella_benchmark_eval.py --type pathway_to_gene_mapping  # Run tasks by type
  python stella_benchmark_eval.py --output results/  # Specify output directory
        """,
    )
    parser.add_argument(
        "-t",
        "--task",
        type=int,
        nargs="+",
        metavar="N",
        help="Task number(s) to run (1-based). If not specified, runs all tasks.",
    )
    parser.add_argument(
        "--type",
        type=str,
        help="Filter tasks by type (e.g., 'pathway_to_gene_mapping')",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="outputs",
        help="Output directory for results (default: outputs/)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum workflow steps per task (default: 50)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per task in seconds (default: 300)",
    )

    args = parser.parse_args()

    # Load .env file
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)

    # Load tasks
    benchmarks_dir = Path(__file__).parent.parent / "benchmarks" / "STELLA"
    tasks_file = benchmarks_dir / "tasks.yaml"
    criteria_file = benchmarks_dir / "evaluation_criteria.md"

    if not tasks_file.exists():
        print(f"{Colors.ERROR}Error: Tasks file not found: {tasks_file}{Colors.RESET}")
        return

    print_banner("STELLA BENCHMARK EVALUATION")
    print(f"Loading tasks from: {tasks_file}")

    try:
        tasks = load_stella_tasks(tasks_file)
        if len(tasks) == 0:
            print(f"{Colors.ERROR}Error: No tasks loaded from YAML file{Colors.RESET}")
            print(
                f"{Colors.WARNING}Note: The YAML file may have formatting issues. Check logs for details.{Colors.RESET}"
            )
            return
        print(f"{Colors.SUCCESS}Loaded {len(tasks)} tasks{Colors.RESET}")
        if len(tasks) < 46:
            print(
                f"{Colors.WARNING}Note: Expected 46 tasks, but only {len(tasks)} loaded. Some tasks may have YAML formatting issues.{Colors.RESET}"
            )
    except Exception as e:
        print(f"{Colors.ERROR}Error loading tasks: {e}{Colors.RESET}")
        logger.exception("Failed to load tasks")
        return

    # Filter tasks
    if args.type:
        tasks = get_tasks_by_type(tasks, args.type)
        print(f"{Colors.INFO}Filtered to {len(tasks)} tasks of type '{args.type}'{Colors.RESET}")

    if args.task:
        # Convert to 0-based indices
        task_indices = [t - 1 for t in args.task if 1 <= t <= len(tasks)]
        tasks = [tasks[i] for i in task_indices if 0 <= i < len(tasks)]
        if not tasks:
            print(f"{Colors.ERROR}Error: Invalid task numbers{Colors.RESET}")
            return
        print(f"{Colors.INFO}Running {len(tasks)} selected tasks{Colors.RESET}")

    print(f"Date: {Colors.DIM}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
    print_banner("STARTING EVALUATION")

    # Load evaluation criteria
    criteria = load_evaluation_criteria(criteria_file) if criteria_file.exists() else None

    # Create LLM for evaluation (use same provider as supervisor for consistency)
    logger.info("Creating LLM for evaluation...")
    try:
        eval_llm = get_llm(prompt_name="supervisor")
        logger.info("LLM created successfully for evaluation")
    except Exception as e:
        logger.warning(
            f"Failed to create LLM for evaluation: {e}. Will use heuristic-based scoring."
        )
        eval_llm = None

    # Create graph (shared across tasks)
    logger.info("Creating workflow graph...")
    graph = create_graph()

    # Run tasks
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output)

    for task_num, task in iter_tasks(tasks):
        print_task_header(task_num, len(tasks), task)

        try:
            result = run_benchmark_task(
                task,
                graph=graph,
                max_steps=args.max_steps,
                timeout=args.timeout,
                show_trace=True,  # Enable detailed execution trace
            )

            # Evaluate using STELLA criteria with LLM-based scoring
            stella_scores = evaluate_task_stella(
                result,
                criteria=criteria,
                use_llm_scoring=True,  # Use LLM-based scoring
                llm=eval_llm,
            )
            result.stella_scores = stella_scores

            results.append(result)
            print_task_result(result)

        except KeyboardInterrupt:
            print(f"\n{Colors.WARNING}Evaluation interrupted by user{Colors.RESET}")
            break
        except Exception as e:
            print(f"{Colors.ERROR}Error executing task: {e}{Colors.RESET}")
            logger.exception(f"Failed to execute task {task_num}")
            # Create a failed result
            failed_result = BenchmarkResult(
                task=task,
                query=task.to_query_string(),
                execution_time=0.0,
                total_steps=0,
                workflow_completed=False,
                final_message_count=0,
                error_message=str(e),
                agent_flow=[],
                final_messages=[],
                workflow_path="",
            )
            results.append(failed_result)

    # Save results
    if results:
        save_results(results, output_dir, timestamp)

        # Print summary
        print_banner("EVALUATION SUMMARY")
        total = len(results)
        completed = sum(1 for r in results if r.workflow_completed)
        print(f"Total Tasks: {total}")
        print(f"Completed: {Colors.SUCCESS}{completed}{Colors.RESET}")
        print(f"Failed: {Colors.ERROR}{total - completed}{Colors.RESET}")
        print(f"Success Rate: {Colors.INFO}{(completed / total * 100):.1f}%{Colors.RESET}")

        if any(r.stella_scores for r in results):
            scored_results = [r for r in results if r.stella_scores]
            avg_overall = sum(r.stella_scores.overall_score() for r in scored_results) / len(
                scored_results
            )
            print(f"\nAverage Overall Score: {Colors.SUCCESS}{avg_overall:.2f}/5.0{Colors.RESET}")


if __name__ == "__main__":
    main()
