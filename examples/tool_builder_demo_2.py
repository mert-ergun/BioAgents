"""
Evaluation script for BioAgents workflow.
Tests bioinformatic queries including tool fallback scenarios.

Standard queries test basic functionality.
Tool fallback queries test scenarios where:
- Exact tool for use case doesn't exist
- Specialized tools are missing (samtools, BLAST, Scanpy, etc.)
- Tools might fail or need custom workflows

Usage:
    python tool_builder_demo_2.py              # Run all queries
    python tool_builder_demo_2.py --query 1    # Run only query 1
    python tool_builder_demo_2.py -q 3         # Run only query 3
    python tool_builder_demo_2.py --query 6 7 8  # Run tool fallback test cases
"""

import argparse
import json
import logging
import os
import textwrap
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage

from bioagents.graph import create_graph

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


# Evaluation queries: standard tasks + tool fallback test cases
BIOINFORMATIC_QUERIES = [
    # Standard queries (existing)
    "Calculate the molecular weight of protein P04637 (p53).",
    "What is the isoelectric point of protein P04637?",
    "Analyze the amino acid composition of protein P04637.",
    "Fetch the AlphaFold structure for protein P04637 and analyze its key structural features.",
    "What is the function of protein P04637 (p53)?",
    # Tool fallback test cases
    # Case 1: Missing tool - specialized software (samtools)
    "Convert a SAM file to BAM format and index it using samtools. Use test data if no file is provided.",
    # Case 2: Missing tool - specialized analysis (pathway enrichment)
    "Perform pathway enrichment analysis on a list of differentially expressed genes from RNA-seq data. Use test data if no gene list is provided (e.g., sample genes: ['TP53', 'BRCA1', 'MYC', 'EGFR', 'KRAS'] with human organism and KEGG/GO databases).",
    # Case 3: Missing tool - specialized database query
    "Search for all protein-protein interactions involving BRCA1 in the STRING database and visualize the interaction network. If visualization tools are not available, provide the interaction data in a structured format.",
    # Case 4: Missing tool - specialized ML/AI tool
    "Use ESM (Evolutionary Scale Modeling) to predict the secondary structure of protein sequence MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL",
    # Case 5: Missing tool - specialized analysis (single-cell)
    "Perform single-cell RNA-seq analysis using Scanpy to identify cell types and create a UMAP visualization. Use test data if no input file is provided.",
    # Case 6: No exact tool - custom workflow
    "Create a workflow to analyze ChIP-seq data: align reads, call peaks, and annotate peaks with nearby genes. Use test data if no input file is provided.",
    # Case 7: Tool might fail - complex query
    "Run BLAST search to find homologs of human p53 protein in all available species and create a phylogenetic tree.",
    # Case 8: Missing tool - specialized database (Ensembl)
    "Retrieve all transcript variants for gene TP53 from Ensembl and determine which variant is the canonical one.",
]


@dataclass
class EvaluationResult:
    """Results for a single query evaluation."""

    query: str
    execution_time: float
    total_steps: int
    workflow_completed: bool
    final_message_count: int
    error_message: str | None
    agent_flow: list[str]
    final_messages: list[str]
    workflow_path: str


def run_query(
    graph, query: str, max_steps: int = 50, timeout: int = 180, show_trace: bool = True
) -> EvaluationResult:
    """Run a single query through the workflow."""
    initial_state = {"messages": [HumanMessage(content=query)]}

    start_time = time.time()
    total_steps = 0
    workflow_completed = False
    error_message = None
    agent_flow = []

    if show_trace:
        print(f"\n{Colors.BOLD}{'─' * 100}{Colors.RESET}")
        print(f"{Colors.BOLD}EXECUTION TRACE{Colors.RESET}")
        print(f"{Colors.BOLD}{'─' * 100}{Colors.RESET}\n")

    try:
        final_state = None
        timeout_reached = False
        last_state_with_messages = None  # Keep track of last state that had messages

        # Stream through workflow to track progress and accumulate state
        last_node_time = start_time
        for step_output in graph.stream(initial_state, {"recursion_limit": max_steps}):
            # Check timeout first
            elapsed = time.time() - start_time
            if elapsed > timeout:
                timeout_reached = True
                error_message = f"Timeout after {timeout}s (elapsed: {elapsed:.1f}s)"
                workflow_completed = False
                logger.warning(f"Workflow timeout at step {total_steps}")
                break

            # Check if we're stuck on a single node (no progress for >60s)
            node_elapsed = time.time() - last_node_time
            if node_elapsed > 60 and total_steps > 0:
                logger.warning(
                    f"No progress for {node_elapsed:.1f}s after step {total_steps}. Possible hang."
                )
                if show_trace:
                    print(
                        f"{Colors.WARNING}⚠️  No progress for {node_elapsed:.1f}s - workflow may be stuck{Colors.RESET}"
                    )

            total_steps += 1
            node_name = next(iter(step_output))
            last_node_time = time.time()  # Update last node time
            node_state = step_output[node_name]

            # LangGraph's add_messages reducer already accumulates messages
            # Each step's state contains all accumulated messages up to that point
            # Save any state that has messages (the last one will be our final state)
            if node_state.get("messages"):
                # Deep copy to avoid reference issues
                last_state_with_messages = {
                    k: v.copy() if isinstance(v, list) else v for k, v in node_state.items()
                }
                # Ensure messages list is properly copied
                if "messages" in last_state_with_messages:
                    last_state_with_messages["messages"] = list(node_state["messages"])

            agent_flow.append(node_name)

            # Print colored step trace
            if show_trace:
                color = get_agent_color(node_name)
                print(
                    f"  {Colors.BOLD}Step {total_steps}:{Colors.RESET} {color}{Colors.BOLD}{node_name.upper()}{Colors.RESET}"
                )

                # Show supervisor routing decisions
                if node_name == "supervisor" and "next" in node_state:
                    next_agent = node_state.get("next", "FINISH")
                    reasoning = node_state.get("reasoning", "")
                    next_color = get_agent_color(next_agent)
                    print(
                        f"    {Colors.SUPERVISOR}→ Routing to: {next_color}{next_agent}{Colors.RESET}"
                    )
                    if reasoning:
                        reasoning_wrapped = textwrap.fill(
                            reasoning,
                            width=90,
                            initial_indent="    Reasoning: ",
                            subsequent_indent="               ",
                        )
                        print(f"{Colors.DIM}{reasoning_wrapped}{Colors.RESET}")

                # Show tool execution
                elif node_name.endswith("_tools"):
                    print(f"    {Colors.TOOLS}Executing tools...{Colors.RESET}")
                    # Show tool calls and results if available
                    if "messages" in node_state:
                        # Find tool calls and their corresponding results
                        tool_calls_found = []
                        tool_results_found = []

                        for msg in node_state["messages"]:
                            # Find AIMessage with tool_calls
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for call in msg.tool_calls:
                                    tool_name = call.get("name", "unknown")
                                    tool_id = call.get("id", "unknown")
                                    tool_args = call.get("args", {})
                                    tool_calls_found.append(
                                        {"name": tool_name, "id": tool_id, "args": tool_args}
                                    )

                            # Find ToolMessage (tool results)
                            if isinstance(msg, ToolMessage):
                                tool_result_id = getattr(msg, "tool_call_id", None)
                                tool_result_content = getattr(msg, "content", "")
                                tool_results_found.append(
                                    {"id": tool_result_id, "content": tool_result_content}
                                )

                        # Show tool calls with arguments
                        for tool_call in tool_calls_found[-3:]:  # Show last 3 tool calls
                            tool_name = tool_call["name"]
                            tool_args = tool_call["args"]

                            # Format arguments (truncate if too long)
                            args_str = str(tool_args)
                            if len(args_str) > 150:
                                args_str = args_str[:150] + "..."

                            print(f"      {Colors.TOOLS}• {Colors.BOLD}{tool_name}{Colors.RESET}")
                            if tool_args:
                                print(f"        {Colors.DIM}Args: {args_str}{Colors.RESET}")

                            # Find corresponding result
                            tool_id = tool_call.get("id")
                            if tool_id:
                                for result in tool_results_found:
                                    if result["id"] == tool_id:
                                        result_content = str(result["content"])
                                        # Truncate result if too long
                                        if len(result_content) > 200:
                                            result_content = result_content[:200] + "..."
                                        print(
                                            f"        {Colors.DIM}Result: {result_content}{Colors.RESET}"
                                        )
                                        break

                        # If no tool calls found but we have messages, show what we can
                        if not tool_calls_found and node_state["messages"]:
                            # Check if there are any ToolMessage results
                            for msg in node_state["messages"][-3:]:
                                if isinstance(msg, ToolMessage):
                                    result_content = str(getattr(msg, "content", ""))
                                    if len(result_content) > 200:
                                        result_content = result_content[:200] + "..."
                                    print(
                                        f"      {Colors.TOOLS}• Tool result: {result_content}{Colors.RESET}"
                                    )

                # Show agent processing
                elif node_name != "supervisor" and not node_name.endswith("_tools"):
                    agent_color = get_agent_color(node_name)
                    print(f"    {agent_color}Processing...{Colors.RESET}")
                    # Show message preview if available
                    if node_state.get("messages"):
                        last_msg = node_state["messages"][-1]
                        if hasattr(last_msg, "content"):
                            content = str(last_msg.content)
                            if len(content) > 0:
                                preview = content[:100] + "..." if len(content) > 100 else content
                                print(f"    {Colors.DIM}{preview}{Colors.RESET}")

                    # If ToolBuilder found an existing tool, show a hint about what should happen next
                    if node_name == "tool_builder" and "messages" in node_state:
                        last_msg_content = ""
                        if node_state["messages"]:
                            last_msg = node_state["messages"][-1]
                            if hasattr(last_msg, "content"):
                                last_msg_content = str(last_msg.content).lower()

                        if "existing" in last_msg_content or "found" in last_msg_content:
                            print(
                                f"    {Colors.DIM}i ToolBuilder found existing tool - should route to supervisor next...{Colors.RESET}"
                            )

            # Log progress (less verbose now that we have colored trace)
            msg_count = len(node_state.get("messages", [])) if "messages" in node_state else 0
            if total_steps % 10 == 0:
                logger.info(
                    f"Query progress: {total_steps} steps, {elapsed:.1f}s elapsed, {msg_count} messages"
                )

            # Safety: Stop if too many steps
            if total_steps >= max_steps:
                error_message = f"Stopped after {max_steps} steps (likely stuck)"
                if show_trace:
                    print(
                        f"\n{Colors.WARNING}⚠ Stopped after {max_steps} steps (likely stuck){Colors.RESET}"
                    )
                break

        # Mark as completed if we didn't hit timeout/max_steps
        if not timeout_reached and error_message is None:
            workflow_completed = True

        # Use the last state that had messages, or fall back to initial state
        if last_state_with_messages:
            final_state = last_state_with_messages
        else:
            # Fallback: use initial state if no messages were found
            final_state = initial_state.copy()
            logger.warning("No messages found in any step, using initial state")

        final_message_count = len(final_state.get("messages", [])) if final_state else 0

        # Debug: Log message count and types
        if final_state and "messages" in final_state:
            messages = final_state["messages"]
            logger.info(f"Total messages in final state: {len(messages)}")
            # Log message types
            msg_types = {}
            for msg in messages:
                msg_type = type(msg).__name__
                msg_types[msg_type] = msg_types.get(msg_type, 0) + 1
            logger.info(f"Message types: {msg_types}")
        else:
            messages = []
            logger.warning("No messages found in final state")

        # Extract qualitative data - get all meaningful messages
        final_messages = []
        if messages:
            # First pass: Extract Report agent messages (always include these)
            report_messages = []
            for msg in messages:
                agent_name = getattr(msg, "name", "")
                if agent_name == "Report":
                    # Handle different message types
                    content = msg.content if hasattr(msg, "content") else str(msg)

                    # Handle dict content (e.g., {'type': 'text', 'text': '...'})
                    if isinstance(content, dict):
                        # Extract text from dict if available
                        if "text" in content:
                            content = content["text"]
                        elif "content" in content:
                            content = content["content"]
                        else:
                            # Fallback: convert dict to string
                            content = str(content)

                    # Handle list content (some messages have list content)
                    if isinstance(content, list):
                        # Extract text from list items if they're dicts
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict):
                                if "text" in item:
                                    text_parts.append(str(item["text"]))
                                elif "content" in item:
                                    text_parts.append(str(item["content"]))
                                else:
                                    text_parts.append(str(item))
                            else:
                                text_parts.append(str(item))
                        content = " ".join(text_parts) if text_parts else str(content)

                    if (
                        isinstance(content, str)
                        and content.strip()
                        and len(content.strip()) > 10
                        and "[EXECUTION_SUCCESS]" not in content
                    ):
                        report_messages.append(content)

            # Add Report messages first
            final_messages.extend(report_messages)

            # If Report agent returned empty, prioritize ToolBuilder's last successful messages
            if not report_messages:
                tool_builder_messages = []
                for msg in reversed(messages[-30:]):  # Check last 30 messages
                    agent_name = getattr(msg, "name", "")
                    if agent_name == "ToolBuilder":
                        content = msg.content if hasattr(msg, "content") else str(msg)

                        # Handle dict/list content
                        if isinstance(content, dict):
                            content = content.get("text", content.get("content", str(content)))
                        elif isinstance(content, list):
                            text_parts = []
                            for item in content:
                                if isinstance(item, dict):
                                    text_parts.append(
                                        str(item.get("text", item.get("content", item)))
                                    )
                                else:
                                    text_parts.append(str(item))
                            content = " ".join(text_parts) if text_parts else str(content)

                        if (
                            isinstance(content, str)
                            and content.strip()
                            and len(content.strip()) > 20
                        ):
                            # Look for success indicators
                            content_lower = content.lower()
                            if any(
                                pattern in content_lower
                                for pattern in [
                                    "successfully",
                                    "created",
                                    "tool",
                                    "result",
                                    "analysis",
                                    "completed",
                                ]
                            ) and not any(
                                skip in content_lower
                                for skip in [
                                    "tool call",
                                    "function call",
                                    "[system]",
                                    "supervisor handoff",
                                ]
                            ):
                                if content not in tool_builder_messages:
                                    tool_builder_messages.insert(0, content)
                                if len(tool_builder_messages) >= 3:
                                    break

                if tool_builder_messages:
                    final_messages.extend(tool_builder_messages)

            # Second pass: Extract other meaningful messages
            for msg in messages:
                agent_name = getattr(msg, "name", "")
                # Skip Report messages (already added)
                if agent_name == "Report":
                    continue

                # Handle different message types
                content = msg.content if hasattr(msg, "content") else str(msg)

                # Handle dict content (e.g., {'type': 'text', 'text': '...'})
                if isinstance(content, dict):
                    # Extract text from dict if available
                    if "text" in content:
                        content = content["text"]
                    elif "content" in content:
                        content = content["content"]
                    else:
                        # Fallback: convert dict to string
                        content = str(content)

                # Handle list content (some messages have list content)
                if isinstance(content, list):
                    # Extract text from list items if they're dicts
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict):
                            if "text" in item:
                                text_parts.append(str(item["text"]))
                            elif "content" in item:
                                text_parts.append(str(item["content"]))
                            else:
                                text_parts.append(str(item))
                        else:
                            text_parts.append(str(item))
                    content = " ".join(text_parts) if text_parts else str(content)

                if isinstance(content, str) and content.strip():
                    # Skip system messages and very short messages
                    if len(content.strip()) < 10:
                        continue

                    # Filter out execution success markers and other technical markers
                    if "[EXECUTION_SUCCESS]" in content:
                        # Remove the marker but keep the rest of the content
                        content = content.replace("[EXECUTION_SUCCESS]", "").strip()
                        if len(content.strip()) < 10:
                            continue  # Skip if only marker was present

                    # Skip messages that are only execution markers
                    if content.strip() in [
                        "[EXECUTION_SUCCESS]",
                        "[EXECUTION_SUCCESS] ToolBuilder successfully executed",
                    ]:
                        continue

                    # Look for final_answer patterns (case insensitive)
                    content_lower = content.lower()
                    # Always include full content for final answers (no truncation)
                    if (
                        any(
                            pattern in content_lower
                            for pattern in [
                                "final answer",
                                "answer:",
                                "result:",
                                "molecular weight",
                                "isoelectric point",
                                "amino acid composition",
                                "analysis",
                                "completed successfully",
                                "task is complete",
                                "successfully created",
                                "successfully executed",
                                "enrichment",
                                "pathway",
                            ]
                        )
                        or (
                            len(final_messages) < 5
                            and not any(
                                skip in content_lower
                                for skip in [
                                    "tool call",
                                    "function call",
                                    "[system]",
                                    "workflow terminated",
                                    "[execution_success]",
                                    "supervisor handoff",
                                ]
                            )
                        )
                    ) and content not in final_messages:
                        final_messages.append(content)  # Full content, no truncation

            # If still no messages, get last 10 non-empty messages
            if not final_messages:
                for msg in reversed(messages[-20:]):
                    content = msg.content if hasattr(msg, "content") else str(msg)

                    # Handle dict content
                    if isinstance(content, dict):
                        if "text" in content:
                            content = content["text"]
                        elif "content" in content:
                            content = content["content"]
                        else:
                            content = str(content)

                    if isinstance(content, list):
                        # Extract text from list items if they're dicts
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict):
                                if "text" in item:
                                    text_parts.append(str(item["text"]))
                                elif "content" in item:
                                    text_parts.append(str(item["content"]))
                                else:
                                    text_parts.append(str(item))
                            else:
                                text_parts.append(str(item))
                        content = " ".join(text_parts) if text_parts else str(content)

                    if isinstance(content, str) and content.strip() and len(content.strip()) > 10:
                        # Filter out execution success markers
                        if "[EXECUTION_SUCCESS]" in content:
                            content = content.replace("[EXECUTION_SUCCESS]", "").strip()
                            if len(content.strip()) < 10:
                                continue

                        # Skip system/error messages and execution markers
                        if not any(
                            skip in content.lower()
                            for skip in [
                                "[system]",
                                "workflow terminated",
                                "error",
                                "[execution_success]",
                            ]
                        ):
                            # Clean up content before adding
                            clean_content = content[:300] + "..." if len(content) > 300 else content
                            final_messages.insert(0, clean_content)
                            if len(final_messages) >= 5:
                                break

        # Create workflow path summary
        workflow_path = " → ".join(agent_flow[-10:])  # Last 10 agents

    except Exception as e:
        workflow_completed = False
        error_message = str(e)
        final_message_count = 0
        final_messages = []
        workflow_path = " → ".join(agent_flow) if agent_flow else "Error"
        logger.error(f"Error executing query: {e}")

    execution_time = time.time() - start_time

    return EvaluationResult(
        query=query,
        execution_time=execution_time,
        total_steps=total_steps,
        workflow_completed=workflow_completed,
        final_message_count=final_message_count,
        error_message=error_message,
        agent_flow=agent_flow,
        final_messages=final_messages,
        workflow_path=workflow_path,
    )


def print_evaluation_summary(results: list[EvaluationResult]):
    """Print evaluation summary."""
    print(f"\n{Colors.BOLD}{'=' * 100}{Colors.RESET}")
    print(f"{Colors.BOLD}EVALUATION SUMMARY{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 100}{Colors.RESET}")

    total_queries = len(results)

    if total_queries == 0:
        print(f"\n{Colors.WARNING}No queries were completed.{Colors.RESET}")
        print(f"{Colors.BOLD}{'=' * 100}{Colors.RESET}")
        return

    completed = sum(1 for r in results if r.workflow_completed)
    failed = total_queries - completed

    print(f"\nTotal Queries: {total_queries}")
    status_color = Colors.SUCCESS if completed > 0 else Colors.ERROR
    print(
        f"Completed: {status_color}{completed} ({completed / total_queries * 100:.1f}%){Colors.RESET}"
    )
    print(
        f"Failed: {Colors.ERROR if failed > 0 else Colors.SUCCESS}{failed} ({failed / total_queries * 100:.1f}%){Colors.RESET}"
    )

    if completed > 0:
        avg_time = sum(r.execution_time for r in results if r.workflow_completed) / completed
        avg_steps = sum(r.total_steps for r in results if r.workflow_completed) / completed
        print(f"\nAverage Execution Time (completed): {Colors.INFO}{avg_time:.2f}s{Colors.RESET}")
        print(f"Average Steps (completed): {Colors.INFO}{avg_steps:.1f}{Colors.RESET}")

    print(f"\n{Colors.BOLD}{'-' * 100}{Colors.RESET}")
    print(f"{Colors.BOLD}QUERY DETAILS{Colors.RESET}")
    print(f"{Colors.BOLD}{'-' * 100}{Colors.RESET}")

    for i, result in enumerate(results, 1):
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
        print(f"\n{i}. {result.query}")
        print(f"   Status: {status_icon} {status_text}")
        print(f"   Time: {Colors.INFO}{result.execution_time:.2f}s{Colors.RESET}")
        print(f"   Steps: {Colors.INFO}{result.total_steps}{Colors.RESET}")

        # Show agent flow with colors
        if result.agent_flow:
            flow_colored = " → ".join([format_node_name(node) for node in result.agent_flow[-15:]])
            print(f"   Flow: {flow_colored}")

        if result.error_message:
            print(f"   {Colors.ERROR}Error: {result.error_message}{Colors.RESET}")
        if result.final_messages:
            print(f"   Final Messages: {Colors.INFO}{len(result.final_messages)}{Colors.RESET}")
            for j, msg in enumerate(result.final_messages[:3], 1):  # Show up to 3 messages
                # Show full message, but wrap long messages nicely
                if len(msg) > 200:
                    print(f"     {j}. {msg[:200]}...")
                    print(
                        f"        {Colors.DIM}(Full message: {len(msg)} characters){Colors.RESET}"
                    )
                else:
                    print(f"     {j}. {msg}")

    print(f"\n{Colors.BOLD}{'=' * 100}{Colors.RESET}")


def main():
    """Run evaluation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run BioAgents workflow evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tool_builder_demo_2.py              # Run all queries
  python tool_builder_demo_2.py --query 1    # Run only query 1
  python tool_builder_demo_2.py -q 3         # Run only query 3
  python tool_builder_demo_2.py --query 2 5  # Run queries 2 and 5
        """,
    )
    parser.add_argument(
        "-q",
        "--query",
        type=int,
        nargs="+",
        metavar="N",
        help="Query number(s) to run (1-5). If not specified, runs all queries.",
        choices=list(range(1, len(BIOINFORMATIC_QUERIES) + 1)),
    )

    args = parser.parse_args()

    # Load .env file from BioAgents root directory
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)

    # Debug: Print loaded GEMINI_MODEL
    gemini_model = os.getenv("GEMINI_MODEL")
    if gemini_model:
        logger.info(f"Loaded GEMINI_MODEL from .env: {gemini_model}")
    else:
        logger.warning("GEMINI_MODEL not found in environment variables")

    # Determine which queries to run
    if args.query:
        # Convert to 0-based indices and filter valid queries
        query_indices = [q - 1 for q in args.query if 1 <= q <= len(BIOINFORMATIC_QUERIES)]
        queries_to_run = [(i + 1, BIOINFORMATIC_QUERIES[i]) for i in query_indices]
        if not queries_to_run:
            print("Error: Invalid query numbers. Must be between 1 and 5.")
            return
    else:
        # Run all queries
        queries_to_run = [(i + 1, query) for i, query in enumerate(BIOINFORMATIC_QUERIES)]

    print(f"{Colors.BOLD}{'=' * 100}{Colors.RESET}")
    print(f"{Colors.BOLD}BIOAGENTS WORKFLOW EVALUATION{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 100}{Colors.RESET}")
    print(
        f"Queries to run: {Colors.INFO}{len(queries_to_run)}/{len(BIOINFORMATIC_QUERIES)}{Colors.RESET}"
    )
    if args.query:
        print(f"Selected queries: {Colors.INFO}{', '.join(map(str, args.query))}{Colors.RESET}")
    print(f"Date: {Colors.DIM}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 100}{Colors.RESET}")

    # Create graph
    logger.info("Creating workflow graph...")
    graph = create_graph()

    # Run queries
    results = []
    for query_num, query in queries_to_run:
        print(f"\n{Colors.BOLD}{'=' * 100}{Colors.RESET}")
        print(f"{Colors.BOLD}QUERY {query_num}/{len(BIOINFORMATIC_QUERIES)}: {query}{Colors.RESET}")
        print(f"{Colors.BOLD}{'=' * 100}{Colors.RESET}")

        try:
            result = run_query(graph, query, max_steps=50, timeout=180, show_trace=True)
            results.append(result)

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
            print(f"\n{Colors.BOLD}Result:{Colors.RESET} {status_icon} {status_text}")
            print(f"Execution Time: {Colors.INFO}{result.execution_time:.2f}s{Colors.RESET}")
            print(f"Steps: {Colors.INFO}{result.total_steps}{Colors.RESET}")
            if result.error_message:
                print(f"{Colors.ERROR}Error: {result.error_message}{Colors.RESET}")

            # Show final workflow path
            if result.agent_flow:
                flow_colored = " → ".join(
                    [format_node_name(node) for node in result.agent_flow[-10:]]
                )
                print(f"\n{Colors.DIM}Workflow Path: {flow_colored}{Colors.RESET}")

        except KeyboardInterrupt:
            logger.warning("Evaluation interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error running query: {e}")
            results.append(
                EvaluationResult(
                    query=query,
                    execution_time=0,
                    total_steps=0,
                    workflow_completed=False,
                    final_message_count=0,
                    error_message=str(e),
                    agent_flow=[],
                    final_messages=[],
                    workflow_path="Error",
                )
            )

    # Print summary
    print_evaluation_summary(results)

    # Create outputs directory if it doesn't exist
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results to JSON
    output_file = output_dir / "tool_builder_demo_2_results.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "total_queries": len(queries_to_run),
                    "queries_run": [q[0] for q in queries_to_run]
                    if args.query
                    else list(range(1, len(BIOINFORMATIC_QUERIES) + 1)),
                    "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                "results": [asdict(r) for r in results],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n{Colors.SUCCESS}Results saved to: {output_file.absolute()}{Colors.RESET}")

    # Save full messages to separate text file
    messages_file = output_dir / "tool_builder_demo_2_full_messages.txt"
    with messages_file.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("FULL MESSAGES FROM EVALUATION\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Queries: {len(results)}\n\n")

        for i, result in enumerate(results, 1):
            f.write("-" * 80 + "\n")
            f.write(f"QUERY {i}: {result.query}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Status: {'✓ Completed' if result.workflow_completed else '✗ Failed'}\n")
            f.write(f"Execution Time: {result.execution_time:.2f}s\n")
            f.write(f"Steps: {result.total_steps}\n")
            if result.error_message:
                f.write(f"Error: {result.error_message}\n")
            f.write(f"\nTotal Messages: {result.final_message_count}\n")
            f.write(f"Final Messages: {len(result.final_messages)}\n\n")

            if result.final_messages:
                for j, msg in enumerate(result.final_messages, 1):
                    f.write(f"\n--- Message {j} ---\n")
                    f.write(f"Length: {len(msg)} characters\n")
                    f.write(f"Content:\n{msg}\n")
                    f.write("\n" + "-" * 80 + "\n\n")
            else:
                f.write("\nNo final messages found.\n\n")

            f.write("\n" + "=" * 80 + "\n\n")

    print(f"{Colors.SUCCESS}Full messages saved to: {messages_file.absolute()}{Colors.RESET}")


if __name__ == "__main__":
    main()
