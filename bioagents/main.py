"""Main entry point for the BioAgents multi-agent system."""

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from bioagents.graph import create_graph
from bioagents.graph_streaming import iter_graph_stream
from bioagents.limits import GRAPH_RECURSION_LIMIT
from bioagents.llms.langsmith_config import (
    get_langsmith_config,
    print_langsmith_status,
    setup_langsmith_environment,
)


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_message_details(message, index):
    """Print detailed information about a message."""
    print(f"\n[Message {index}]")
    print(f"Type: {message.__class__.__name__}")

    if hasattr(message, "name") and message.name:
        print(f"Agent: {message.name}")

    print("-" * 80)

    if hasattr(message, "content") and message.content:
        print(message.content)

    if hasattr(message, "tool_calls") and message.tool_calls:
        print("\nTool Calls:")
        for tool_call in message.tool_calls:
            print(f"  - {tool_call['name']}")
            print(f"    Args: {tool_call['args']}")


def main():
    """Execute a demonstration query through the multi-agent workflow."""
    load_dotenv()

    # Set up LangSmith monitoring if enabled
    try:
        setup_langsmith_environment()
    except ValueError as e:
        print(f"\n⚠ Warning: {e}")
        print("   Continuing without LangSmith monitoring...\n")

    # Print LangSmith status
    print_langsmith_status()

    # Create the multi-agent graph
    print("\nInitializing BioAgents Multi-Agent Workflow...")
    print("   - Agents: Supervisor, Research, Analysis, Report, Summary")
    graph = create_graph()

    # Get LangSmith config for tracing
    try:
        langsmith_config = get_langsmith_config()
    except ValueError as e:
        langsmith_config = None
        print(f"\nWarning: {e!s}")
        print("   Continuing without LangSmith monitoring...\n")

    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = """
    I need a comprehensive analysis of the human tumor suppressor protein p53 (P04637).
    Please fetch the sequence, analyze its properties, and provide a detailed report.
    """

    print("\nUser Query:")
    print_separator("-")
    print(query.strip())
    print_separator("-")

    initial_state = {"messages": [HumanMessage(content=query)]}

    print("\nMulti-Agent Workflow Starting...\n")

    try:
        # Stream the execution to see each step
        # LangSmith will automatically trace if environment variables are set
        stream_config = langsmith_config
        for step_num, step_output in enumerate(
            iter_graph_stream(graph, initial_state, config=stream_config or None), 1
        ):
            print(f"\n{'=' * 80}")
            print(f"STEP {step_num}: {next(iter(step_output.keys())).upper()}")
            print(f"{'=' * 80}")

            for node_name, node_output in step_output.items():
                if node_name == "supervisor":
                    if "next" in node_output:
                        print(f"Supervisor Decision: Route to '{node_output['next']}'")
                elif node_name.endswith("_tools"):
                    print("Executing tools...")
                else:
                    print(f"Agent '{node_name}' working...")

        print(f"\n{'=' * 80}")
        print("FINAL RESULTS")
        print_separator()

        # Invoke with LangSmith config for final execution trace
        invoke_cfg = {**(langsmith_config or {}), "recursion_limit": GRAPH_RECURSION_LIMIT}
        final_result = graph.invoke(initial_state, config=invoke_cfg)

        # Print all messages with better formatting
        for i, message in enumerate(final_result["messages"], 1):
            print_message_details(message, i)

        print(f"\n{('=' * 80)}")
        print("Multi-Agent Workflow Completed Successfully!")
        print(f"{'=' * 80}\n")

    except Exception as e:
        print(f"\nError during workflow execution: {e!s}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
