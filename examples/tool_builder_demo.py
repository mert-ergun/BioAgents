"""Demo script showing how to use the Tool Builder Agent.

This example demonstrates:
1. Literature mining to extract tool mentions
2. Creating custom tool wrappers
3. Registering tools for use by other agents
4. Searching and executing custom tools
"""

import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def demo_tool_registry():
    """Demonstrate the Tool Registry functionality."""
    from bioagents.tools.tool_registry import (
        ToolDefinition,
        ToolParameter,
        get_registry,
    )

    print("\n" + "=" * 60)
    print("DEMO: Tool Registry")
    print("=" * 60)

    registry = get_registry()

    # Create a sample tool definition
    tool_def = ToolDefinition(
        name="demo_protein_length",
        description="Calculate the length of a protein sequence",
        category="proteomics",
        parameters=[
            ToolParameter(
                name="sequence",
                type="string",
                description="Protein sequence in single-letter amino acid code",
                required=True,
            ),
        ],
        return_type="integer",
        return_description="The number of amino acids in the sequence",
        code='''"""Demo tool for calculating protein length."""

def demo_protein_length(sequence: str) -> int:
    """Calculate the length of a protein sequence.

    Args:
        sequence: Protein sequence in single-letter code

    Returns:
        Number of amino acids
    """
    # Remove whitespace and newlines
    clean_seq = "".join(sequence.split())
    return len(clean_seq)


if __name__ == "__main__":
    # Test
    test_seq = "MKWVTFISLLFLFSSAYSRGVFRR"
    print(f"Length: {demo_protein_length(test_seq)}")
''',
        dependencies=[],
        source_software="Custom demo tool",
    )

    # Register the tool
    print("\n1. Registering custom tool...")
    success = registry.register_tool(tool_def, overwrite=True)
    print(f"   Registration {'succeeded' if success else 'failed'}")

    # List all tools
    print("\n2. Listing registered tools...")
    tools = registry.list_tools()
    for t in tools:
        print(f"   - {t.name}: {t.description}")

    # Search for tools
    print("\n3. Searching for 'protein' tools...")
    results = registry.search_tools("protein length calculation", limit=3)
    for tool, score in results:
        print(f"   - {tool.name} (score: {score:.2f})")

    # Validate the tool
    print("\n4. Validating tool...")
    is_valid = registry.validate_tool("demo_protein_length")
    print(f"   Validation {'passed' if is_valid else 'failed'}")

    # Execute the tool
    print("\n5. Executing tool...")
    func = registry.load_tool_function("demo_protein_length")
    if func:
        test_seq = "MKWVTFISLLFLFSSAYSRGVFRR"
        result = func(test_seq)
        print(f"   Input: {test_seq}")
        print(f"   Output: {result} amino acids")

    # Export for context
    print("\n6. Exporting tool documentation...")
    docs = registry.export_for_context(["demo_protein_length"])
    print(docs[:500] + "..." if len(docs) > 500 else docs)

    return registry


def demo_tool_builder_agent():
    """Demonstrate the Tool Builder Agent functionality."""
    from bioagents.agents.tool_builder_agent import (
        extract_tools_from_text,
        list_custom_tools,
        search_custom_tools,
    )

    print("\n" + "=" * 60)
    print("DEMO: Tool Builder Agent")
    print("=" * 60)

    # Sample Methods section text
    methods_text = """
    We performed differential gene expression analysis using DESeq2 (Love et al., 2014).
    Single-cell RNA sequencing data was processed with Scanpy (Wolf et al., 2018) and
    cell type annotation was performed using CellTypist (Dominguez Conde et al., 2022).
    Protein structure predictions were obtained from AlphaFold2 (Jumper et al., 2021).
    Sequence alignments were performed using BLAST (Altschul et al., 1990).
    Gene ontology enrichment analysis was conducted with clusterProfiler (Yu et al., 2012).
    """

    print("\n1. Extracting tools from Methods section...")
    print(f"   Input text: {methods_text[:100]}...")

    # Note: This requires an LLM to be configured
    try:
        result = extract_tools_from_text.invoke({"text": methods_text})
        print("\n   Extracted tools:")
        data = json.loads(result)
        for tool in data.get("tools", []):
            print(f"   - {tool.get('name', 'Unknown')}: {tool.get('task', 'N/A')}")
    except Exception as e:
        print(f"   (Skipping extraction - requires LLM: {e})")

    # List custom tools
    print("\n2. Listing available custom tools...")
    try:
        result = list_custom_tools.invoke({})
        data = json.loads(result)
        print(f"   Found {data.get('total', 0)} tools")
        for tool in data.get("tools", [])[:5]:
            print(f"   - {tool['name']}: {tool['description'][:50]}...")
    except Exception as e:
        print(f"   Error: {e}")

    # Search custom tools
    print("\n3. Searching for specific functionality...")
    try:
        result = search_custom_tools.invoke({"query": "protein sequence analysis"})
        data = json.loads(result)
        print("   Query: 'protein sequence analysis'")
        for r in data.get("results", []):
            print(f"   - {r['name']} (score: {r['score']:.2f})")
    except Exception as e:
        print(f"   Error: {e}")


def demo_smolagents_integration():
    """Demonstrate integration with the Coder Agent via smolagents."""
    from bioagents.tools.tool_builder_wrappers import get_custom_tool_wrappers

    print("\n" + "=" * 60)
    print("DEMO: Smolagents Integration")
    print("=" * 60)

    # Get the custom tool wrappers
    tools = get_custom_tool_wrappers()

    print("\n1. Available smolagents tools for Coder Agent:")
    for tool in tools:
        print(f"   - {tool.name}: {tool.description[:60]}...")

    # Use the search tool
    print("\n2. Using search tool...")
    search_tool = next(t for t in tools if t.name == "search_custom_tools")
    result = search_tool.forward(query="protein", limit=3)
    print(f"   Search results: {result[:200]}...")

    # Use the list tool
    print("\n3. Using list tool...")
    list_tool = next(t for t in tools if t.name == "list_custom_tools")
    result = list_tool.forward()
    print(f"   Tool list: {result[:200]}...")


def demo_full_workflow():
    """Demonstrate a complete workflow with the Tool Builder Agent."""
    print("\n" + "=" * 60)
    print("DEMO: Full Workflow Example")
    print("=" * 60)

    print("""
This demonstrates the complete workflow:

1. User query: "Analyze the amino acid composition of BRCA1"

2. Supervisor routes to Research Agent to fetch the sequence

3. Research Agent finds the sequence but notices no existing
   tool can calculate custom statistics

4. Supervisor routes to Tool Builder Agent with the need

5. Tool Builder Agent:
   a. Searches existing tools: search_custom_tools("amino acid statistics")
   b. Finds no suitable tool
   c. Generates wrapper: generate_tool_wrapper("aa_statistics", ...)
   d. Registers the tool: register_custom_tool(...)
   e. Validates: validate_custom_tool("aa_statistics")

6. Supervisor routes back to Coder Agent

7. Coder Agent:
   a. Finds the new tool: search_custom_tools("amino acid statistics")
   b. Uses it: execute_custom_tool("aa_statistics", {"sequence": "..."})

8. Task complete - tool is now available for future queries!
""")

    print("\nTo see this in action, run:")
    print('  python -m bioagents.main "Analyze amino acid composition of BRCA1"')


if __name__ == "__main__":
    # Run demos
    demo_tool_registry()
    demo_tool_builder_agent()
    demo_smolagents_integration()
    demo_full_workflow()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
