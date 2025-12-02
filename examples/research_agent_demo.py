"""Research Agent Demo: Literature Search and Data Retrieval Examples.

This demo showcases the Research Agent's capabilities for:
1. Literature searches using ToolUniverse (PubMed, ArXiv, BioRxiv, etc.)
2. Multi-source literature synthesis
3. Protein data retrieval from UniProt
4. Targeted database queries (OpenTargets, ChEMBL, etc.)
5. Combined literature and data workflows

Usage:
    # Run all examples
    uv run python examples/research_agent_demo.py

    # Run specific example
    uv run python examples/research_agent_demo.py --example literature_search
    uv run python examples/research_agent_demo.py --example multi_source
    uv run python examples/research_agent_demo.py --example protein_data
    uv run python examples/research_agent_demo.py --example combined

Requirements:
    - Set up API keys in .env file
"""

from __future__ import annotations

import argparse
import textwrap
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from bioagents.graph import create_graph
from bioagents.llms.langsmith_config import (
    print_langsmith_status,
    setup_langsmith_environment,
)
from bioagents.tools.tool_universe import DEFAULT_WRAPPER


def print_banner(title: str, char: str = "="):
    """Print a formatted banner."""
    width = 100
    print("\n" + char * width)
    print(title.center(width))
    print(char * width)


def print_section(title: str):
    """Print a section header."""
    print("\n" + "-" * 100)
    print(f"  {title}")
    print("-" * 100)


def pretty_print_message(message: Any, index: int | None = None):
    """Pretty print a message with formatting."""
    if index is not None:
        print(f"\n[Message {index}] {message.__class__.__name__}")
    else:
        print(f"\n{message.__class__.__name__}")

    if hasattr(message, "name") and message.name:
        print(f"  Agent: {message.name}")

    content = getattr(message, "content", None)
    if content:
        formatted_content = textwrap.fill(
            _stringify_content(content), width=100, initial_indent="  ", subsequent_indent="  "
        )
        print(formatted_content)

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        print("  Tool Calls:")
        for call in tool_calls:
            print(f"    - {call['name']}")
            args_str = str(call["args"])[:100]
            print(f"      args: {args_str}")


def _stringify_content(content: Any) -> str:
    """Convert message content to string."""
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


def run_example(title: str, description: str, query: str):
    """Run a single example query through the graph."""
    print_banner(title)
    print(f"\n{description}\n")

    graph = create_graph()
    initial_state = {"messages": [HumanMessage(content=query)]}

    print_section("USER QUERY")
    print(f"\n{query}\n")

    print_section("EXECUTION TRACE")

    for step_count, step in enumerate(graph.stream(initial_state)):
        node_name = next(iter(step))
        print(f"\n  Step {step_count}: {node_name}")

        if node_name == "supervisor" and "next" in step[node_name]:
            next_agent = step[node_name].get("next")
            reasoning = step[node_name].get("reasoning", "")
            print(f"    → Routing to: {next_agent}")
            if reasoning:
                reasoning_wrapped = textwrap.fill(
                    reasoning,
                    width=90,
                    initial_indent="    Reasoning: ",
                    subsequent_indent="               ",
                )
                print(reasoning_wrapped)

    print_section("FINAL RESULT")
    final_state = graph.invoke(initial_state)

    # Print only the last few relevant messages
    final_messages = final_state["messages"][-5:]
    for idx, message in enumerate(final_messages, start=len(final_state["messages"]) - 4):
        pretty_print_message(message, idx)

    print("\n")


def example_literature_search():
    """Example 1: Basic literature search using PubMed."""
    title = "Example 1: Literature Search - CRISPR Gene Editing"
    description = textwrap.dedent(
        """
        This example demonstrates a basic literature search workflow:
        - Research agent discovers PubMed search tools via ToolUniverse
        - Executes a search for CRISPR gene editing papers
        - Returns structured results with paper titles, authors, and relevance
        """
    ).strip()

    query = textwrap.dedent(
        """
        Search PubMed for recent papers on CRISPR-Cas9 gene editing applications in human
        therapeutics. Focus on papers from the last 3 years. Return the top 15 most relevant
        papers with titles, authors, publication dates, and brief summaries. Do not use composite tools, use PubMed_search_articles tool.
        """
    ).strip()

    run_example(title, description, query)


def example_multi_source_search():
    """Example 2: Multi-source literature search."""
    title = "Example 2: Multi-Source Literature Search - Protein Folding ML"
    description = textwrap.dedent(
        """
        This example demonstrates searching multiple literature databases:
        - Research agent searches both ArXiv (for ML papers) and BioRxiv (for biology)
        - Synthesizes findings from multiple sources
        - Identifies key papers across disciplines
        """
    ).strip()

    query = textwrap.dedent(
        """
        Conduct a comprehensive literature search on machine learning approaches for protein
        structure prediction. Search both ArXiv for machine learning papers and BioRxiv for
        biology preprints. Focus on deep learning methods like AlphaFold. Return top papers
        from each source and synthesize the key findings.
        """
    ).strip()

    run_example(title, description, query)


def example_protein_data_retrieval():
    """Example 3: Protein data retrieval from UniProt."""
    title = "Example 3: Protein Data Retrieval - p53 Tumor Suppressor"
    description = textwrap.dedent(
        """
        This example demonstrates direct protein data retrieval:
        - Research agent fetches protein sequence from UniProt
        - Returns FASTA format data
        - Can be combined with literature search for context
        """
    ).strip()

    query = textwrap.dedent(
        """
        Fetch the protein sequence for human p53 tumor suppressor (P53_HUMAN or P04637)
        from UniProt. Return the sequence in FASTA format.
        """
    ).strip()

    run_example(title, description, query)


def example_targeted_database_query():
    """Example 4: Targeted database query for disease-target associations."""
    title = "Example 4: Targeted Database Query - Alzheimer's Disease Targets"
    description = textwrap.dedent(
        """
        This example demonstrates querying specialized biological databases:
        - Research agent discovers OpenTargets tools
        - Queries disease-target associations
        - Returns structured data about therapeutic targets
        """
    ).strip()

    query = textwrap.dedent(
        """
        Use ToolUniverse to find and query OpenTargets for disease-target associations
        related to Alzheimer's disease (EFO_0000249). Return the top 10 most promising
        therapeutic targets with their association scores and supporting evidence.
        """
    ).strip()

    run_example(title, description, query)


def example_combined_workflow():
    """Example 5: Combined literature search and data retrieval."""
    title = "Example 5: Combined Workflow - p53 Research and Data"
    description = textwrap.dedent(
        """
        This example demonstrates a combined workflow:
        - Retrieves p53 protein sequence from UniProt
        - Searches literature for p53 in cancer research
        - Provides comprehensive context (data + literature)
        """
    ).strip()

    query = textwrap.dedent(
        """
        I'm researching the p53 tumor suppressor protein. Please:
        1. Fetch the human p53 protein sequence from UniProt (P53_HUMAN)
        2. Search PubMed for recent papers (last 2 years) on p53 mutations in cancer
        3. Search for any structural studies of p53 in BioRxiv or ArXiv

        Provide a comprehensive summary that combines the protein data with key findings
        from the literature.
        """
    ).strip()

    run_example(title, description, query)


def example_preprint_search():
    """Example 6: Preprint server search (BioRxiv/MedRxiv)."""
    title = "Example 6: Preprint Search - COVID-19 Immunology"
    description = textwrap.dedent(
        """
        This example demonstrates searching preprint servers:
        - Research agent discovers BioRxiv/MedRxiv search tools
        - Searches for cutting-edge preprints
        - Returns latest findings before peer review
        """
    ).strip()

    query = textwrap.dedent(
        """
        Search BioRxiv and MedRxiv for recent preprints on COVID-19 immunology and
        T-cell responses. Focus on papers from the last 6 months. Return the top 20
        most relevant preprints with their key findings.
        """
    ).strip()

    run_example(title, description, query)


def example_comprehensive_research():
    """Example 7: Comprehensive research project."""
    title = "Example 7: Comprehensive Research - Drug Discovery for Hypertension"
    description = textwrap.dedent(
        """
        This example demonstrates a complete research workflow:
        - Multi-database literature search
        - Disease-target association queries
        - Clinical trial searches
        - Comprehensive synthesis of findings
        """
    ).strip()

    query = textwrap.dedent(
        """
        I'm conducting a drug discovery project for hypertension. Please:

        1. Search PubMed for recent papers on hypertension drug targets
        2. Use ToolUniverse to find and query OpenTargets for disease-target associations
           for hypertension (EFO_0000537)
        3. Search ClinicalTrials.gov for ongoing clinical trials for hypertension treatments
        4. Search for any relevant chemical compounds in PubChem or ChEMBL

        Synthesize all findings into a comprehensive research briefing that identifies:
        - Most promising therapeutic targets
        - Key research trends
        - Current clinical development status
        - Potential drug candidates
        - Recommended next steps for wet lab validation
        """
    ).strip()

    run_example(title, description, query)


def main():
    """Main function to run examples."""
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Research Agent Demo - Literature Search and Data Retrieval Examples"
    )
    parser.add_argument(
        "--example",
        type=str,
        choices=[
            "literature_search",
            "multi_source",
            "protein_data",
            "targeted_db",
            "combined",
            "preprint",
            "comprehensive",
            "all",
        ],
        default="all",
        help="Specific example to run (default: all)",
    )
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Setup LangSmith monitoring
    try:
        setup_langsmith_environment()
        print_langsmith_status()
    except ValueError as e:
        print(f"\n⚠ Warning: {e}\n")

    # Check ToolUniverse availability
    print_banner("RESEARCH AGENT DEMO", "=")
    print()
    if DEFAULT_WRAPPER.client_available:
        print("✓ ToolUniverse SDK detected: Live tool execution is enabled.")
        print("  Research agent will make real API calls to external databases.\n")
    else:
        print("⚠ ToolUniverse SDK not installed: Running in catalog mode.")
        print("  Research agent will use the local tool catalog for demonstration.")
        print("  Install 'tooluniverse' package for live tool execution.\n")

    print("Note: These examples may take several minutes to run, especially with live API calls.")
    print("      Press Ctrl+C to interrupt at any time.\n")

    # Define all examples
    examples = {
        "literature_search": example_literature_search,
        "multi_source": example_multi_source_search,
        "protein_data": example_protein_data_retrieval,
        "targeted_db": example_targeted_database_query,
        "combined": example_combined_workflow,
        "preprint": example_preprint_search,
        "comprehensive": example_comprehensive_research,
    }

    # Run selected example(s)
    if args.example == "all":
        print("Running all examples...\n")
        for example_func in examples.values():
            try:
                example_func()
            except KeyboardInterrupt:
                print("\n\n⚠ Interrupted by user. Stopping examples.\n")
                break
            except Exception as e:
                print(f"\n\n❌ Error running example: {e}\n")
                print("Continuing to next example...\n")
    else:
        print(f"Running example: {args.example}\n")
        examples[args.example]()

    print_banner("DEMO COMPLETE", "=")
    print("\nFor more information, see:")
    print("  - README.md")
    print("  - docs/PROMPT_ENGINEERING.md")
    print("  - bioagents/tools/tool_universe.md")
    print()


if __name__ == "__main__":
    main()
