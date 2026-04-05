"""Main entry point for the BioAgents multi-agent system."""

import json
import sys
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from bioagents.graph import create_graph
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

    initial_state: dict[str, Any] = {
        "messages": [HumanMessage(content=query)],
        "next": None,
        "reasoning": "",
        "memory": {
            # Pre-initialize memory structure for all possible agents
            "research": {
                "status": "pending",
                "timestamp": None,
                "data": {},
                "raw_output": "",
                "errors": [],
                "tool_calls": [],
            },
            "analysis": {
                "status": "pending",
                "timestamp": None,
                "data": {},
                "raw_output": "",
                "errors": [],
                "tool_calls": [],
            },
            "coder": {
                "status": "pending",
                "timestamp": None,
                "data": {},
                "raw_output": "",
                "errors": [],
                "tool_calls": [],
            },
            "ml": {
                "status": "pending",
                "timestamp": None,
                "data": {},
                "raw_output": "",
                "errors": [],
                "tool_calls": [],
            },
            "dl": {
                "status": "pending",
                "timestamp": None,
                "data": {},
                "raw_output": "",
                "errors": [],
                "tool_calls": [],
            },
            "protein_design": {
                "status": "pending",
                "timestamp": None,
                "data": {},
                "raw_output": "",
                "errors": [],
                "tool_calls": [],
            },
            "tool_builder": {
                "status": "pending",
                "timestamp": None,
                "data": {},
                "raw_output": "",
                "errors": [],
                "tool_calls": [],
            },
            "report": {
                "status": "pending",
                "timestamp": None,
                "data": {},
                "raw_output": "",
                "errors": [],
                "tool_calls": [],
            },
            "critic": {
                "status": "pending",
                "timestamp": None,
                "data": {},
                "raw_output": "",
                "errors": [],
                "tool_calls": [],
            },
        },
    }

    print("\nMulti-Agent Workflow Starting...\n")

    try:
        stream_config = langsmith_config
        for step_num, step_output in enumerate(
            graph.stream(initial_state, config=stream_config), 1
        ):
            print(f"\n{'=' * 80}")
            print(f"STEP {step_num}: Nodes executing: {list(step_output.keys())}")
            print(f"{'=' * 80}")

            for node_name, node_output in step_output.items():
                print(f"\n[{node_name.upper()}]")
                if node_name == "supervisor":
                    if "next" in node_output:
                        print(f"  -> Routing to: '{node_output['next']}'")
                    if "reasoning" in node_output:
                        print(f"  -> Reasoning: {node_output['reasoning'][:200]}")
                elif node_name.endswith("_tools"):
                    print("  -> Executing tools...")
                else:
                    print(f"  -> Agent working...")
                    if "memory" in node_output and node_name in node_output["memory"]:
                        agent_mem = node_output["memory"][node_name]
                        print(f"     Status: {agent_mem.get('status', 'unknown')}")


        print(f"\n{'=' * 80}")
        print("FINAL RESULTS (FROM SHARED MEMORY)")
        print_separator()

        final_result = graph.invoke(initial_state, config=langsmith_config)

        # Print memory contents instead of messages
        memory = final_result.get("memory", {})

        for agent_name in [
            "research",
            "analysis",
            "coder",
            "ml",
            "dl",
            "protein_design",
            "report",
            "critic",
            "summary",
        ]:
            agent_mem = memory.get(agent_name, {})
            if agent_mem.get("status") == "success":
                print(f"\n[{agent_name.upper()}]")
                print(f"Status: {agent_mem['status']}")
                if agent_mem.get("data"):
                    print("Data:")
                    print(json.dumps(agent_mem["data"], indent=2))
                if agent_mem.get("raw_output"):
                    print("Output:")
                    print(agent_mem["raw_output"][:500])

        # The final user-facing output lives in summary memory when available,
        # otherwise fall back to report memory.
        summary_mem = memory.get("summary", {})
        report_mem = memory.get("report", {})

        final_output = summary_mem.get("raw_output") or report_mem.get("raw_output") or ""

        if final_output:
            print(f"\n{'=' * 80}")
            print("USER-FACING SUMMARY")
            print("=" * 80)
            print(final_output)

        print(f"\n{'=' * 80}")
        print("Multi-Agent Workflow Completed Successfully!")
        print(f"{'=' * 80}\n")

    except Exception as e:
        print(f"\nError during workflow execution: {e!s}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
