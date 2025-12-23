#!/usr/bin/env python3
"""Protein Binder Design Demo: De Novo Binder Generation Workflow.

This demo showcases the Protein Design Agent's capabilities for:
1. Retrieving target structures from PDB/AlphaFold DB
2. Identifying binding interfaces and hotspot residues
3. Designing binder candidates using RFdiffusion + ProteinMPNN
4. Predicting complex structures with AlphaFold-Multimer
5. Computing interface quality metrics (iPTM, ipSAE)
6. Ranking designs by binding quality

USE CASE:
    Using the provided UniProt accession for the target protein and access to PDB/AlphaFold DB,
    first identify and retrieve its 3D structure along with any experimentally resolved
    protein-protein complexes to define the native binding interface and extract key
    interaction motifs from known binders. Next, design binder candidates by generating
    fully de novo protein binders using a deep learning pipeline (RFdiffusion for backbone
    generation followed by ProteinMPNN sequence design and AlphaFold2 structure prediction),
    yielding a library of at least 100 unique binder sequences. Finally, re-model each
    target-binder pair with AlphaFold-Multimer and compute interface quality metrics
    (iPTM and ipSAE) for all designs.

Usage:
    # Run the full demo
    uv run python examples/protein_binder_design_demo.py

    # Run specific example
    uv run python examples/protein_binder_design_demo.py --example structure_retrieval
    uv run python examples/protein_binder_design_demo.py --example interface_analysis
    uv run python examples/protein_binder_design_demo.py --example binder_design
    uv run python examples/protein_binder_design_demo.py --example full_pipeline

Requirements:
    - Set up API keys in .env file (OPENAI_API_KEY)
    - Optional: Install RFdiffusion, ProteinMPNN, ColabFold for local execution
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


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for beautiful terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Agent colors
    SUPERVISOR = "\033[95m"  # Magenta
    PROTEIN_DESIGN = "\033[96m"  # Cyan
    RESEARCH = "\033[94m"  # Blue

    # Status colors
    SUCCESS = "\033[92m"  # Green
    WARNING = "\033[93m"  # Yellow
    ERROR = "\033[91m"  # Red
    INFO = "\033[94m"  # Blue


def print_banner(title: str, char: str = "="):
    """Print a formatted banner."""
    width = 100
    print(f"\n{Colors.BOLD}{char * width}{Colors.RESET}")
    print(f"{Colors.BOLD}{title:^{width}}{Colors.RESET}")
    print(f"{Colors.BOLD}{char * width}{Colors.RESET}\n")


def print_section(title: str):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{'-' * 100}{Colors.RESET}")
    print(f"  {Colors.PROTEIN_DESIGN}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{'-' * 100}{Colors.RESET}")


def pretty_print_message(message: Any, index: int | None = None):
    """Pretty print a message with formatting."""
    if index is not None:
        print(f"\n[Message {index}] {message.__class__.__name__}")
    else:
        print(f"\n{message.__class__.__name__}")

    if hasattr(message, "name") and message.name:
        print(f"  {Colors.PROTEIN_DESIGN}Agent: {message.name}{Colors.RESET}")

    content = getattr(message, "content", None)
    if content:
        content_str = str(content) if not isinstance(content, str) else content
        # Truncate very long content
        if len(content_str) > 2000:
            content_str = content_str[:2000] + "\n... [truncated]"
        formatted_content = textwrap.fill(
            content_str, width=100, initial_indent="  ", subsequent_indent="  "
        )
        print(formatted_content)

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        print(f"  {Colors.INFO}Tool Calls:{Colors.RESET}")
        for call in tool_calls:
            print(f"    - {Colors.BOLD}{call['name']}{Colors.RESET}")
            args_str = str(call["args"])[:150]
            print(f"      args: {Colors.DIM}{args_str}{Colors.RESET}")


def run_example(title: str, description: str, query: str, show_all_messages: bool = False):
    """Run a single example query through the graph."""
    print_banner(title)
    print(f"{description}\n")

    graph = create_graph()
    initial_state = {"messages": [HumanMessage(content=query)]}

    print_section("USER QUERY")
    print(f"\n{Colors.DIM}{query}{Colors.RESET}\n")

    print_section("EXECUTION TRACE")

    # Use stream and collect final state (don't call invoke separately)
    final_state = None

    for step_count, step in enumerate(
        graph.stream(initial_state, {"recursion_limit": 50}), start=1
    ):
        node_name = next(iter(step))
        node_output = step[node_name]

        # Track state for final output
        if node_output.get("messages"):
            if final_state is None:
                final_state = {"messages": list(initial_state["messages"])}
            final_state["messages"].extend(node_output["messages"])

        print(f"\n  {Colors.BOLD}Step {step_count}: {node_name.upper()}{Colors.RESET}")

        if node_name == "supervisor" and "next" in node_output:
            next_agent = node_output.get("next")
            reasoning = node_output.get("reasoning", "")
            print(f"    {Colors.SUPERVISOR}→ Routing to: {next_agent}{Colors.RESET}")
            if reasoning:
                reasoning_wrapped = textwrap.fill(
                    reasoning,
                    width=90,
                    initial_indent="    Reasoning: ",
                    subsequent_indent="               ",
                )
                print(f"{Colors.DIM}{reasoning_wrapped}{Colors.RESET}")

        elif node_name == "protein_design":
            print(f"    {Colors.PROTEIN_DESIGN}Protein Design Agent processing...{Colors.RESET}")

        elif node_name.endswith("_tools"):
            print(f"    {Colors.INFO}Executing tools...{Colors.RESET}")

    print_section("FINAL RESULT")

    if final_state is None:
        final_state = initial_state

    # Print final messages
    if show_all_messages:
        for idx, message in enumerate(final_state["messages"], start=1):
            pretty_print_message(message, idx)
    else:
        # Just show the last few relevant messages
        final_messages = final_state["messages"][-5:]
        start_idx = max(1, len(final_state["messages"]) - 4)
        for idx, message in enumerate(final_messages, start=start_idx):
            pretty_print_message(message, idx)

    print("\n")
    return final_state


def example_structure_retrieval():
    """Example 1: Retrieve target structure from AlphaFold DB."""
    title = "Example 1: Structure Retrieval - AlphaFold DB"
    description = textwrap.dedent(
        """
        This example demonstrates retrieving a protein structure from AlphaFold DB
        using a UniProt accession. The agent fetches:
        - PDB/CIF structure file URLs
        - Confidence metrics (pLDDT)
        - Predicted Aligned Error (PAE) data
        """
    ).strip()

    query = textwrap.dedent(
        """
        Retrieve the AlphaFold predicted structure for human p53 tumor suppressor
        (UniProt: P04637). Also search for any experimentally resolved PDB complexes
        containing p53 bound to other proteins.
        """
    ).strip()

    return run_example(title, description, query)


def example_interface_analysis():
    """Example 2: Analyze protein-protein interface."""
    title = "Example 2: Interface Analysis - p53-MDM2 Complex"
    description = textwrap.dedent(
        """
        This example demonstrates analyzing a protein-protein interface:
        - Retrieving a known complex structure from PDB
        - Identifying interface residues
        - Extracting binding hotspots for binder design
        """
    ).strip()

    query = textwrap.dedent(
        """
        Analyze the interface between p53 and MDM2 in PDB structure 1YCR.
        First fetch the structure, then identify all polymer entities (chains),
        and finally analyze the interface contacts between the two proteins.
        Extract the key hotspot residues that could be targeted in binder design.
        """
    ).strip()

    return run_example(title, description, query)


def example_binder_design_setup():
    """Example 3: Set up binder design pipeline."""
    title = "Example 3: Binder Design Pipeline Setup"
    description = textwrap.dedent(
        """
        This example demonstrates setting up a de novo binder design pipeline:
        - Configuring RFdiffusion for backbone generation
        - Setting up ProteinMPNN for sequence design
        - Preparing AlphaFold-Multimer for structure prediction

        Note: Actual execution may require local tool installations.
        """
    ).strip()

    query = textwrap.dedent(
        """
        Set up a binder design pipeline for targeting human p53 (UniProt P04637).
        The binders should target the DNA-binding domain hotspot residues.
        Configure the pipeline to generate 100 unique binder designs with
        lengths between 50-80 residues. Use the default quality thresholds
        (iPTM >= 0.7, ipSAE <= 10).
        """
    ).strip()

    return run_example(title, description, query)


def example_full_pipeline():
    """Example 4: Complete binder design workflow."""
    title = "Example 4: Complete Binder Design Workflow"
    description = textwrap.dedent(
        """
        This example demonstrates the complete protein binder design workflow:

        1. Retrieve target structure from AlphaFold/PDB
        2. Search for known complexes to identify binding interfaces
        3. Extract interface hotspots from known binders
        4. Set up RFdiffusion + ProteinMPNN + AlphaFold pipeline
        5. Compute iPTM and ipSAE metrics
        6. Rank designs and report the maximum iPTM among qualifying binders

        This addresses the use case:
        "What is the maximum iPTM score (rounded to 3 decimal points) observed
        for any binder whose ipSAE meets the quality threshold?"
        """
    ).strip()

    query = textwrap.dedent(
        """
        Using the provided UniProt accession for the target protein (P04637 - human p53)
        and access to PDB/AlphaFold DB, please:

        1. First identify and retrieve its 3D structure along with any experimentally
           resolved protein-protein complexes to define the native binding interface
           and extract key interaction motifs from known binders.

        2. Next, design binder candidates using a deep learning pipeline:
           - RFdiffusion for backbone generation
           - ProteinMPNN for sequence design
           - AlphaFold2/AlphaFold-Multimer for structure prediction
           Generate at least 100 unique binder sequences.

           Use the p53-MDM2 binding interface hotspots: residues 17,19,23,26
           (the well-known F19, W23, L26 binding motif).

        3. Finally, compute interface quality metrics (iPTM and ipSAE) for all designs.

        Question: Among the designed binders, what is the maximum iPTM score
        (rounded to 3 decimal points) observed for any binder whose ipSAE meets
        the quality threshold (ipSAE below 10)?

        Note: If local tools (RFdiffusion, ProteinMPNN, ColabFold) are not available,
        provide the setup configuration and instructions for running them externally.
        """
    ).strip()

    return run_example(title, description, query, show_all_messages=False)


def example_metrics_ranking():
    """Example 5: Rank binder designs by quality metrics."""
    title = "Example 5: Ranking Binder Designs"
    description = textwrap.dedent(
        """
        This example demonstrates how to rank and filter binder designs:
        - Computing binding quality metrics
        - Filtering by iPTM and ipSAE thresholds
        - Finding the best designs that meet quality criteria
        """
    ).strip()

    query = textwrap.dedent(
        """
        Given a hypothetical set of binder design results, demonstrate how to:
        1. Compute binding quality metrics (iPTM, ipSAE) for a complex
        2. Rank all designs by iPTM score
        3. Filter to only include designs with ipSAE below 10
        4. Report the maximum iPTM among the qualifying designs

        Use the compute_binding_metrics and rank_binder_designs tools to show
        the workflow, even if no actual results are available yet.
        """
    ).strip()

    return run_example(title, description, query)


def main():
    """Main function to run examples."""
    parser = argparse.ArgumentParser(
        description="Protein Binder Design Demo - De Novo Binder Generation Workflow"
    )
    parser.add_argument(
        "--example",
        type=str,
        choices=[
            "structure_retrieval",
            "interface_analysis",
            "binder_design",
            "full_pipeline",
            "metrics_ranking",
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
        print(f"\n{Colors.WARNING}⚠ Warning: {e}{Colors.RESET}\n")

    # Print introduction
    print_banner("PROTEIN BINDER DESIGN DEMO", "=")

    print(f"""
{Colors.BOLD}This demo showcases the complete de novo protein binder design workflow:{Colors.RESET}

{Colors.PROTEIN_DESIGN}Pipeline Steps:{Colors.RESET}
  1. {Colors.INFO}Structure Retrieval{Colors.RESET} - Fetch target from PDB/AlphaFold DB
  2. {Colors.INFO}Interface Analysis{Colors.RESET} - Identify binding hotspots from known complexes
  3. {Colors.INFO}Backbone Generation{Colors.RESET} - Generate binder backbones with RFdiffusion
  4. {Colors.INFO}Sequence Design{Colors.RESET} - Design sequences with ProteinMPNN
  5. {Colors.INFO}Structure Prediction{Colors.RESET} - Predict complexes with AlphaFold-Multimer
  6. {Colors.INFO}Quality Assessment{Colors.RESET} - Compute iPTM and ipSAE metrics

{Colors.BOLD}Quality Thresholds:{Colors.RESET}
  • iPTM ≥ 0.7 (high confidence binding)
  • ipSAE ≤ 10 (good interface quality)

{Colors.WARNING}Note:{Colors.RESET} Some design steps require local tool installations
      (RFdiffusion, ProteinMPNN, ColabFold). When unavailable, the agent
      will provide alternative instructions.
""")

    # Define examples
    examples = {
        "structure_retrieval": example_structure_retrieval,
        "interface_analysis": example_interface_analysis,
        "binder_design": example_binder_design_setup,
        "full_pipeline": example_full_pipeline,
        "metrics_ranking": example_metrics_ranking,
    }

    # Run selected example(s)
    if args.example == "all":
        print(f"{Colors.INFO}Running all examples...{Colors.RESET}\n")
        for name, example_func in examples.items():
            try:
                print(f"\n{Colors.BOLD}{'─' * 100}{Colors.RESET}")
                example_func()
            except KeyboardInterrupt:
                print(
                    f"\n\n{Colors.WARNING}⚠ Interrupted by user. Stopping examples.{Colors.RESET}\n"
                )
                break
            except Exception as e:
                print(f"\n\n{Colors.ERROR}❌ Error running example '{name}': {e}{Colors.RESET}\n")
                import traceback

                traceback.print_exc()
                print("Continuing to next example...\n")
    else:
        print(f"{Colors.INFO}Running example: {args.example}{Colors.RESET}\n")
        try:
            examples[args.example]()
        except Exception as e:
            print(f"\n{Colors.ERROR}❌ Error: {e}{Colors.RESET}\n")
            import traceback

            traceback.print_exc()

    print_banner("DEMO COMPLETE", "=")

    print(f"""
{Colors.BOLD}Summary:{Colors.RESET}
  The Protein Design Agent can handle the complete binder design workflow,
  from structure retrieval through design to quality assessment.

{Colors.BOLD}Key Tools Used:{Colors.RESET}
  • fetch_alphafold_structure - Get AlphaFold predictions
  • search_pdb_complexes - Find known binding partners
  • analyze_interface_contacts - Identify hotspot residues
  • design_binders_bindcraft - Configure design pipeline
  • generate_binder_backbones - RFdiffusion backbone generation
  • design_binder_sequences - ProteinMPNN sequence design
  • predict_complex_structure - AlphaFold-Multimer prediction
  • compute_binding_metrics - Calculate iPTM, ipSAE
  • rank_binder_designs - Rank and filter by quality

{Colors.BOLD}For More Information:{Colors.RESET}
  • bioagents/agents/protein_design_agent.py
  • bioagents/tools/structural_tools.py
  • bioagents/tools/protein_design_tools.py
  • bioagents/prompts/protein_design.xml
""")


if __name__ == "__main__":
    main()
