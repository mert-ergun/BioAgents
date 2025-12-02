"""
Demonstration of the BioAgents multi-agent workflow with comprehensive logging.

This example shows how the supervisor pattern works with specialized agents:
1. Supervisor routes tasks to appropriate agents
2. Research agent fetches protein data
3. Analysis agent analyzes the data
4. Report agent synthesizes findings

The enhanced logging system captures all intercommunications between agents,
including messages, tool calls, supervisor decisions, and agent reasoning.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from bioagents.graph import create_graph
from bioagents.llms.langsmith_config import (
    print_langsmith_status,
    setup_langsmith_environment,
)


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for beautiful terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Agent colors
    SUPERVISOR = "\033[95m"  # Magenta
    RESEARCH = "\033[94m"  # Blue
    ANALYSIS = "\033[92m"  # Green
    REPORT = "\033[93m"  # Yellow

    # Message type colors
    HUMAN = "\033[96m"  # Cyan
    AI = "\033[97m"  # White
    TOOL = "\033[91m"  # Red
    SYSTEM = "\033[90m"  # Gray

    # Status colors
    SUCCESS = "\033[92m"  # Green
    WARNING = "\033[93m"  # Yellow
    ERROR = "\033[91m"  # Red
    INFO = "\033[94m"  # Blue


class AgentLogger:
    """
    Comprehensive logging system for multi-agent workflows.

    Captures and beautifully displays:
    - Agent intercommunications
    - Tool calls and responses
    - Supervisor routing decisions
    - Message flows and content
    - Execution timing
    """

    def __init__(self, log_to_file: bool = True, verbose: bool = True):
        """
        Initialize the agent logger.

        Args:
            log_to_file: Whether to log to a file in addition to console
            verbose: Whether to show detailed message content
        """
        self.verbose = verbose
        self.step_counter = 0
        self.start_time = datetime.now()

        # Create logs directory if it doesn't exist
        if log_to_file:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)

            timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"agent_workflow_{timestamp}.log"

            # Configure file logging
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)

            # Get root logger and add file handler
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)

            print(f"{Colors.INFO}📝 Logging to file: {log_file}{Colors.RESET}\n")

    def get_agent_color(self, agent_name: str) -> str:
        """Get the color code for a specific agent."""
        color_map = {
            "supervisor": Colors.SUPERVISOR,
            "research": Colors.RESEARCH,
            "analysis": Colors.ANALYSIS,
            "report": Colors.REPORT,
        }
        return color_map.get(agent_name.lower(), Colors.SYSTEM)

    def print_header(self, text: str, level: int = 1):
        """Print a formatted header."""
        if level == 1:
            print(f"\n{Colors.BOLD}{'=' * 100}{Colors.RESET}")
            print(f"{Colors.BOLD}{text:^100}{Colors.RESET}")
            print(f"{Colors.BOLD}{'=' * 100}{Colors.RESET}\n")
        elif level == 2:
            print(f"\n{Colors.BOLD}{'-' * 100}{Colors.RESET}")
            print(f"{Colors.BOLD}{text}{Colors.RESET}")
            print(f"{Colors.BOLD}{'-' * 100}{Colors.RESET}")
        else:
            print(f"\n{Colors.DIM}{'·' * 100}{Colors.RESET}")
            print(f"{text}")

    def log_step(self, node_name: str, description: str):
        """Log a workflow step."""
        self.step_counter += 1
        color = self.get_agent_color(node_name)

        elapsed = (datetime.now() - self.start_time).total_seconds()

        print(
            f"\n{Colors.BOLD}[Step {self.step_counter}]{Colors.RESET} "
            f"{color}● {node_name.upper()}{Colors.RESET} "
            f"{Colors.DIM}({elapsed:.2f}s){Colors.RESET}"
        )
        print(f"  {description}")

    def log_supervisor_decision(self, next_agent: str, reasoning: str):
        """Log a supervisor routing decision."""
        color = Colors.SUPERVISOR
        print(f"\n  {color}🎯 SUPERVISOR DECISION:{Colors.RESET}")
        print(f"  {color}├─ Next Agent: {Colors.BOLD}{next_agent.upper()}{Colors.RESET}")

        if reasoning:
            print(f"  {color}└─ Reasoning:{Colors.RESET}")
            for line in reasoning.split("\n"):
                if line.strip():
                    print(f"     {Colors.DIM}{line.strip()}{Colors.RESET}")

    def log_message(self, message: Any):
        """Log a message with detailed information."""
        if not self.verbose:
            return

        msg_type = type(message).__name__

        # Determine color based on message type
        if isinstance(message, HumanMessage):
            color = Colors.HUMAN
            icon = "👤"
        elif isinstance(message, AIMessage):
            color = Colors.AI
            icon = "🤖"
        elif isinstance(message, ToolMessage):
            color = Colors.TOOL
            icon = "🔧"
        else:
            color = Colors.SYSTEM
            icon = "📨"

        print(f"\n  {color}{icon} MESSAGE [{msg_type}]{Colors.RESET}")

        # Log message metadata
        if hasattr(message, "name") and message.name:
            print(f"  {color}├─ From: {message.name}{Colors.RESET}")

        # Log message content
        if hasattr(message, "content") and message.content:
            content = str(message.content)
            if len(content) > 500 and not self.verbose:
                content = content[:500] + "..."

            print(f"  {color}├─ Content:{Colors.RESET}")
            for line in content.split("\n"):
                if line.strip():
                    print(f"  {color}│  {Colors.DIM}{line.strip()}{Colors.RESET}")

        # Log tool calls if present
        if hasattr(message, "tool_calls") and message.tool_calls:
            print(f"  {color}├─ Tool Calls: {len(message.tool_calls)}{Colors.RESET}")
            for i, tool_call in enumerate(message.tool_calls, 1):
                print(
                    f"  {color}│  [{i}] {Colors.BOLD}{tool_call.get('name', 'unknown')}{Colors.RESET}"
                )
                if "args" in tool_call:
                    print(f"  {color}│      Args: {Colors.DIM}{tool_call['args']}{Colors.RESET}")

        # Log tool response if it's a ToolMessage
        if isinstance(message, ToolMessage):
            print(f"  {color}├─ Tool: {getattr(message, 'name', 'unknown')}{Colors.RESET}")
            if hasattr(message, "tool_call_id"):
                print(f"  {color}├─ Call ID: {message.tool_call_id}{Colors.RESET}")

        print(f"  {color}└─{Colors.RESET}")

    def log_state_update(self, node_name: str, state_output: dict):
        """Log a state update from a node."""
        if not self.verbose:
            return

        color = self.get_agent_color(node_name)

        print(f"\n  {color}📊 STATE UPDATE:{Colors.RESET}")

        # Log next agent if present
        if "next" in state_output:
            next_agent = state_output["next"]
            if next_agent:
                print(f"  {color}├─ Next: {Colors.BOLD}{next_agent}{Colors.RESET}")

        # Log messages
        if state_output.get("messages"):
            print(f"  {color}├─ New Messages: {len(state_output['messages'])}{Colors.RESET}")
            for msg in state_output["messages"]:
                self.log_message(msg)

        print(f"  {color}└─{Colors.RESET}")

    def log_tool_execution(self, tool_name: str, agent_name: str):
        """Log tool execution."""
        color = self.get_agent_color(agent_name)
        print(f"\n  {color}🔧 EXECUTING TOOL: {Colors.BOLD}{tool_name}{Colors.RESET}")

    def log_error(self, error: Exception):
        """Log an error."""
        print(f"\n{Colors.ERROR}❌ ERROR:{Colors.RESET}")
        print(f"{Colors.ERROR}{error!s}{Colors.RESET}")

    def log_summary(self):
        """Log execution summary."""
        elapsed = (datetime.now() - self.start_time).total_seconds()

        self.print_header("Workflow Summary", level=2)
        print(
            f"  {Colors.SUCCESS}✓{Colors.RESET} Total Steps: {Colors.BOLD}{self.step_counter}{Colors.RESET}"
        )
        print(
            f"  {Colors.SUCCESS}✓{Colors.RESET} Total Time: {Colors.BOLD}{elapsed:.2f}s{Colors.RESET}"
        )
        print(
            f"  {Colors.SUCCESS}✓{Colors.RESET} Average Step Time: {Colors.BOLD}{elapsed / self.step_counter:.2f}s{Colors.RESET}"
        )


def run_multi_agent_demo(verbose: bool = True, log_to_file: bool = True):
    """
    Run a demonstration of the multi-agent workflow with comprehensive logging.

    Args:
        verbose: Show detailed message content and intercommunications
        log_to_file: Save logs to a file in the logs/ directory
    """
    load_dotenv()

    # Set up LangSmith monitoring if enabled
    try:
        setup_langsmith_environment()
        print_langsmith_status()
    except ValueError as e:
        print(f"\n{Colors.WARNING}⚠ Warning: {e}{Colors.RESET}")

    logger = AgentLogger(log_to_file=log_to_file, verbose=verbose)

    logger.print_header("BioAgents Multi-Agent Workflow Demo", level=1)

    print(f"{Colors.INFO}🤖 Agent System Architecture:{Colors.RESET}")
    print(f"  {Colors.SUPERVISOR}● Supervisor{Colors.RESET} - Routes tasks to specialized agents")
    print(f"  {Colors.RESEARCH}● Research{Colors.RESET} - Fetches protein data from databases")
    print(f"  {Colors.ANALYSIS}● Analysis{Colors.RESET} - Analyzes protein properties")
    print(f"  {Colors.REPORT}● Report{Colors.RESET} - Synthesizes findings into reports")
    print()

    graph = create_graph()

    queries = [
        {
            "title": "Simple Data Fetch",
            "query": "Fetch the protein sequence for human p53 (P04637)",
        },
        {
            "title": "Comprehensive Analysis",
            "query": """
            Please provide a comprehensive analysis of human insulin (P01308):
            1. Fetch the sequence
            2. Calculate its molecular weight
            3. Analyze amino acid composition
            4. Estimate isoelectric point
            5. Provide a summary report
            """,
        },
    ]

    for i, example in enumerate(queries, 1):
        logger.print_header(f"Example {i}: {example['title']}", level=1)

        print(f"{Colors.BOLD}📋 Query:{Colors.RESET}")
        print(f"{Colors.DIM}{example['query'].strip()}{Colors.RESET}")

        logger.print_header("Workflow Execution", level=2)

        # Reset step counter for each example
        logger.step_counter = 0
        logger.start_time = datetime.now()

        initial_state = {"messages": [HumanMessage(content=example["query"])]}

        # Log initial message
        print(f"\n{Colors.HUMAN}👤 USER INPUT:{Colors.RESET}")
        print(f"{Colors.DIM}{example['query'].strip()}{Colors.RESET}")

        try:
            # Stream to show detailed progression
            for step_output in graph.stream(initial_state):
                for node_name, node_output in step_output.items():
                    if node_name == "__end__":
                        continue

                    # Log the step
                    if node_name == "supervisor":
                        logger.log_step(
                            node_name, "Analyzing conversation and routing to next agent"
                        )

                        # Extract supervisor decision with reasoning
                        if node_output.get("next"):
                            next_agent = node_output["next"]
                            reasoning = node_output.get("reasoning", "")
                            logger.log_supervisor_decision(next_agent, reasoning)

                    elif node_name.endswith("_tools"):
                        agent_name = node_name.replace("_tools", "")
                        logger.log_step(node_name, f"Executing {agent_name} tools")

                        # Log tool messages
                        if "messages" in node_output:
                            for msg in node_output["messages"]:
                                logger.log_message(msg)

                    else:
                        logger.log_step(node_name, f"{node_name.capitalize()} agent processing")

                        # Log state update
                        logger.log_state_update(node_name, node_output)

            # Get final result
            final_result = graph.invoke(initial_state)
            last_message = final_result["messages"][-1]

            logger.print_header("Final Response", level=2)

            if hasattr(last_message, "content"):
                print(f"\n{Colors.SUCCESS}{last_message.content}{Colors.RESET}\n")

            # Log summary
            logger.log_summary()

        except Exception as e:
            logger.log_error(e)
            import traceback

            print(f"\n{Colors.ERROR}{traceback.format_exc()}{Colors.RESET}")

        if i < len(queries):
            print(f"\n{Colors.DIM}{'─' * 100}{Colors.RESET}")
            input(f"\n{Colors.BOLD}Press Enter to continue to next example...{Colors.RESET}")

    logger.print_header("Demo Completed Successfully!", level=1)
    print(f"{Colors.SUCCESS}✓ All examples executed{Colors.RESET}\n")


if __name__ == "__main__":
    # Parse command line arguments for different verbosity levels
    import argparse

    parser = argparse.ArgumentParser(description="BioAgents Multi-Agent Workflow Demo")
    parser.add_argument(
        "--quiet", action="store_true", help="Reduce verbosity (hide detailed message content)"
    )
    parser.add_argument(
        "--no-file", action="store_true", help="Don't log to file, only console output"
    )

    args = parser.parse_args()

    run_multi_agent_demo(verbose=not args.quiet, log_to_file=not args.no_file)
