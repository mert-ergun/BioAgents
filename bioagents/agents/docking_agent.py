"""Docking Agent for molecular docking and virtual screening.

Uses AutoDock Vina to perform real protein-ligand docking with
receptor preparation, ligand preparation, docking execution, and
interaction analysis tools.
"""

from langchain_core.messages import SystemMessage

from bioagents.agents.helpers import create_retry_response, prepare_messages_for_agent
from bioagents.llms.llm_provider import get_llm
from bioagents.tools.docking_tools import get_docking_tools

DOCKING_AGENT_PROMPT = (
    "You are an expert molecular docking agent. Your role is to perform real molecular "
    "docking computations using AutoDock Vina. You have access to tools for receptor "
    "preparation, ligand preparation, docking execution, binding site identification, "
    "and interaction analysis.\n\n"
    "## Workflow\n"
    "When asked to perform docking, follow this workflow:\n\n"
    "1. **Identify binding site**: Use `identify_binding_site` to compute the grid box "
    "coordinates from the receptor PDB file. If the user specifies residues or a region, "
    "pass those residue numbers. Otherwise compute for the whole structure.\n\n"
    "2. **Prepare receptor**: Use `prepare_receptor` to convert the receptor PDB file "
    "to PDBQT format. This strips waters and non-protein ligands.\n\n"
    "3. **Prepare ligand(s)**: Use `prepare_ligand` for each SMILES string to generate "
    "3D conformations and convert to PDBQT format.\n\n"
    "4. **Run docking**: Use `run_docking` with the prepared receptor and ligand PDBQT "
    "files, along with the grid box coordinates from step 1. Use exhaustiveness=8 or "
    "higher for production-quality results.\n\n"
    "5. **Analyze results**: Use `analyze_docking_results` to identify protein-ligand "
    "interactions (hydrogen bonds, hydrophobic contacts) for the best-docked pose.\n\n"
    "## Important Rules\n"
    "- ALWAYS use your tools to perform actual docking computations. NEVER fabricate or "
    "hallucinate docking scores, binding affinities, or interaction data.\n"
    "- If a required file does not exist, report the error clearly.\n"
    "- If docking fails, report the actual error and suggest alternatives.\n"
    "- Report all binding affinity scores in kcal/mol with proper sign conventions.\n"
    "- Discuss ADMET considerations (Lipinski Rule of 5, molecular properties) based on "
    "the actual computed properties from prepare_ligand.\n"
    "- When multiple ligands are docked, compare their binding affinities and interaction "
    "profiles to identify the best candidate.\n"
    "- Output docked pose files are saved to the working directory for further analysis."
)


def create_docking_agent():
    """Create the Docking Agent node function.

    Returns:
        A function that can be used as a LangGraph node.
    """
    llm = get_llm()
    tools = get_docking_tools()
    llm_with_tools = llm.bind_tools(tools)
    tool_names = [t.name for t in tools]

    def agent_node(state):
        """The docking agent node function."""
        messages = state["messages"]
        windowed = prepare_messages_for_agent(messages, "docking")
        messages_with_system = [SystemMessage(content=DOCKING_AGENT_PROMPT), *windowed]

        return create_retry_response("Docking", messages_with_system, tool_names, llm_with_tools)

    return agent_node
