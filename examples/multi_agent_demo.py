"""
Demonstration of the BioAgents multi-agent workflow.

This example shows how the supervisor pattern works with specialized agents:
1. Supervisor routes tasks to appropriate agents
2. Research agent fetches protein data
3. Analysis agent analyzes the data
4. Report agent synthesizes findings
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from bioagents.graph import create_graph


def run_multi_agent_demo():
    """Run a demonstration of the multi-agent workflow."""
    load_dotenv()

    # Ensure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set in environment")
        print("Please set it in .env file or environment variables")
        return

    print("=" * 80)
    print("BioAgents Multi-Agent Workflow Demo")
    print("=" * 80)
    print()
    print("This demo showcases a supervisor-based multi-agent system where:")
    print("  Supervisor - Routes tasks to specialized agents")
    print("  Research Agent - Fetches protein data from databases")
    print("  Analysis Agent - Analyzes protein properties")
    print("  Report Agent - Synthesizes findings into reports")
    print()
    print("=" * 80)

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
        print(f"\n{'=' * 80}")
        print(f"Example {i}: {example['title']}")
        print(f"{'=' * 80}")
        print(f"\nQuery: {example['query'].strip()}")
        print(f"\n{'-' * 80}")
        print("Workflow Execution:")
        print("-" * 80)

        initial_state = {"messages": [HumanMessage(content=example["query"])]}

        try:
            # Stream to show progression
            for step_num, step_output in enumerate(graph.stream(initial_state), 1):
                for node_name, node_output in step_output.items():
                    if node_name == "supervisor":
                        if node_output.get("next"):
                            next_agent = node_output["next"]
                            print(f"  Step {step_num}: Supervisor → {next_agent}")
                    elif node_name.endswith("_tools"):
                        agent_name = node_name.replace("_tools", "")
                        print(f"  Step {step_num}: {agent_name} agent using tools...")
                    elif node_name != "__end__":
                        print(f"  Step {step_num}: {node_name} agent working...")

            final_result = graph.invoke(initial_state)
            last_message = final_result["messages"][-1]

            print(f"\n{'-' * 80}")
            print("Final Response:")
            print("-" * 80)
            if hasattr(last_message, "content"):
                print(last_message.content)

        except Exception as e:
            print(f"Error: {e!s}")

        if i < len(queries):
            input("\nPress Enter to continue to next example...")

    print(f"\n{'=' * 80}")
    print("Demo completed!")
    print("=" * 80)


if __name__ == "__main__":
    run_multi_agent_demo()
