"""Generic benchmark runner for BioAgents workflow."""

import logging
import textwrap
import time
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage

from bioagents.benchmarks.models import BenchmarkResult, STELLATask
from bioagents.graph import create_graph

logger = logging.getLogger(__name__)


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""

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


def _get_agent_color(node_name: str) -> str:
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
    elif "ml" in node_lower:
        return Colors.INFO
    elif "tools" in node_lower:
        return Colors.TOOLS
    elif "end" in node_lower:
        return Colors.END
    else:
        return Colors.INFO


logger = logging.getLogger(__name__)


def run_benchmark_task(
    task: STELLATask,
    graph: Any = None,
    max_steps: int = 50,
    timeout: int = 180,
    show_trace: bool = False,
) -> BenchmarkResult:
    """
    Run a single benchmark task through the BioAgents workflow.

    Args:
        task: STELLATask to execute
        graph: LangGraph workflow (if None, creates a new one)
        max_steps: Maximum number of workflow steps
        timeout: Timeout in seconds
        show_trace: Whether to show execution trace

    Returns:
        BenchmarkResult with execution metrics
    """
    if graph is None:
        graph = create_graph()

    # Convert task to query string
    query = task.to_query_string()
    initial_state = {"messages": [HumanMessage(content=query)]}

    start_time = time.time()
    total_steps = 0
    workflow_completed = False
    error_message = None
    agent_flow = []
    final_messages = []
    raw_output = None

    try:
        last_state_with_messages = None
        last_node_time = start_time

        if show_trace:
            print(f"\n{Colors.BOLD}{'─' * 100}{Colors.RESET}")
            print(f"{Colors.BOLD}EXECUTION TRACE{Colors.RESET}")
            print(f"{Colors.BOLD}{'─' * 100}{Colors.RESET}\n")

        # Stream through workflow to track progress and accumulate state
        for step_output in graph.stream(initial_state, {"recursion_limit": max_steps}):
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
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
            last_node_time = time.time()
            node_state = step_output[node_name]

            # Save state that has messages
            if node_state.get("messages"):
                last_state_with_messages = {
                    k: (v.copy() if isinstance(v, list) else v) for k, v in node_state.items()
                }
                if "messages" in last_state_with_messages:
                    last_state_with_messages["messages"] = list(node_state["messages"])

            agent_flow.append(node_name)

            # Print colored step trace
            if show_trace:
                color = _get_agent_color(node_name)
                print(
                    f"  {Colors.BOLD}Step {total_steps}:{Colors.RESET} {color}{Colors.BOLD}{node_name.upper()}{Colors.RESET}"
                )

                # Show supervisor routing decisions
                if node_name == "supervisor" and "next" in node_state:
                    next_agent = node_state.get("next", "FINISH")
                    reasoning = node_state.get("reasoning", "")
                    next_color = _get_agent_color(next_agent)
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

                # Show agent processing
                elif node_name != "supervisor" and not node_name.endswith("_tools"):
                    agent_color = _get_agent_color(node_name)
                    print(f"    {agent_color}Processing...{Colors.RESET}")
                    # Special note for coder agent (it can take a long time)
                    if node_name == "coder":
                        print(
                            f"    {Colors.DIM}Note: Coder agent executes code in Jupyter kernel - this may take a while...{Colors.RESET}"
                        )
                    # Show message preview if available
                    if node_state.get("messages"):
                        last_msg = node_state["messages"][-1]
                        if hasattr(last_msg, "content"):
                            content = str(last_msg.content)
                            if len(content) > 0:
                                preview = content[:100] + "..." if len(content) > 100 else content
                                print(f"    {Colors.DIM}{preview}{Colors.RESET}")

            # Log progress periodically
            msg_count = len(node_state.get("messages", [])) if "messages" in node_state else 0
            if total_steps % 10 == 0:
                logger.info(
                    f"Task progress: {total_steps} steps, {elapsed:.1f}s elapsed, {msg_count} messages"
                )

            # Safety: Stop if too many steps
            if total_steps >= max_steps:
                error_message = f"Stopped after {max_steps} steps (likely stuck)"
                if show_trace:
                    print(
                        f"\n{Colors.WARNING}⚠ Stopped after {max_steps} steps (likely stuck){Colors.RESET}"
                    )
                break

        execution_time = time.time() - start_time

        # Extract final messages
        if last_state_with_messages and "messages" in last_state_with_messages:
            messages = last_state_with_messages["messages"]

            # Check if workflow completed (reached END)
            if agent_flow and agent_flow[-1] == "__end__":
                workflow_completed = True
            elif total_steps >= max_steps:
                workflow_completed = False
                error_message = f"Reached max_steps limit ({max_steps})"
            else:
                workflow_completed = True

            # Extract text content from messages
            for msg in messages[-10:]:  # Last 10 messages
                content = getattr(msg, "content", "")
                if content:
                    if isinstance(content, str):
                        final_messages.append(content)
                    elif isinstance(content, list):
                        # Handle list of dicts (e.g., [{"type": "text", "text": "..."}])
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                final_messages.append(item["text"])
                            elif isinstance(item, str):
                                final_messages.append(item)

            # Get raw output (last message content)
            if messages:
                last_msg = messages[-1]
                content = getattr(last_msg, "content", "")
                if isinstance(content, str):
                    raw_output = content
                elif isinstance(content, list):
                    # Extract text from list
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            text_parts.append(item["text"])
                        elif isinstance(item, str):
                            text_parts.append(item)
                    raw_output = "\n".join(text_parts)

        # Determine if workflow completed
        # If we reached max_steps without naturally ending, consider it incomplete
        if total_steps >= max_steps:
            workflow_completed = False
            if not error_message:
                error_message = f"Reached max_steps limit ({max_steps})"
        elif not error_message and total_steps > 0:
            # If we have steps and no error, consider it completed
            workflow_completed = True

    except Exception as e:
        execution_time = time.time() - start_time
        error_message = str(e)
        workflow_completed = False
        logger.error(f"Error executing task {task.task_name}: {e}", exc_info=True)

    # Build workflow path string
    workflow_path = " → ".join(agent_flow[-15:]) if agent_flow else ""

    return BenchmarkResult(
        task=task,
        query=query,
        execution_time=execution_time,
        total_steps=total_steps,
        workflow_completed=workflow_completed,
        final_message_count=len(final_messages),
        error_message=error_message,
        agent_flow=agent_flow,
        final_messages=final_messages,
        workflow_path=workflow_path,
        raw_output=raw_output,
    )
