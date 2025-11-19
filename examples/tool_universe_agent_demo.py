"""End-to-end demo that exercises the ToolUniverse-enabled multi-agent graph.

This script:
1. Loads the default BioAgents LangGraph workflow (Supervisor → Research → Analysis → Report).
2. Issues a user request that explicitly requires ToolUniverse search and execution.
3. Streams intermediate steps so you can observe tool routing and agent responses.

Usage:
    uv run python examples/tool_universe_agent_demo.py
"""

from __future__ import annotations

import textwrap

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from bioagents.graph import create_graph
from bioagents.tools.tool_universe import DEFAULT_WRAPPER


def print_banner(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def pretty_print_message(message, index):
    print(f"\n[Message {index}] {message.__class__.__name__}")

    if getattr(message, "name", None):
        print(f" Agent: {message.name}")

    content = getattr(message, "content", None)
    if content:
        print(textwrap.fill(_stringify_content(content), width=100))

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        print(" Tool Calls:")
        for call in tool_calls:
            print(f"   - {call['name']} | args={call['args']}")


def _stringify_content(content) -> str:
    """Convert common LangChain message content structures into readable text."""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content)


def main():
    load_dotenv()

    if DEFAULT_WRAPPER.client_available:
        print("ToolUniverse SDK detected: live tool execution is enabled.\n")
    else:
        print(
            "ToolUniverse SDK not installed; the demo will run in catalogue mode.\n"
            "Install the 'tooluniverse' package (see README) to enable live tool calls.\n"
        )

    graph = create_graph()

    demo_query = textwrap.dedent(
        """
        You are assisting with a hypertension drug discovery briefing.

        1. Use ToolUniverse to discover relevant OpenTargets or UniProt tools that list
           disease-target associations for hypertension (EFO_0000537).
        2. Call the best tool to fetch structured data (live call when SDK is installed;
           otherwise note that only catalogue information was available).
        3. Summarize the most promising targets and outline recommended follow-up wet lab
           experiments.
        """
    ).strip()

    initial_state = {"messages": [HumanMessage(content=demo_query)]}

    print_banner("USER QUERY")
    print(demo_query)

    print_banner("STREAMING EXECUTION")
    for step in graph.stream(initial_state):
        node_name = next(iter(step))
        print(f"\n→ Step: {node_name}")
        if node_name == "supervisor" and "next" in step[node_name]:
            print(f"  Supervisor routed to: {step[node_name]['next']}")

    print_banner("FINAL OUTPUT")
    final_state = graph.invoke(initial_state)
    for idx, message in enumerate(final_state["messages"], start=1):
        pretty_print_message(message, idx)


if __name__ == "__main__":
    main()
