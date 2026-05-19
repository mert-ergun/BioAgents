"""Docking Agent for molecular docking and virtual screening guidance."""

from langchain_core.messages import SystemMessage

from bioagents.agents.helpers import invoke_with_retry, prepare_messages_for_agent
from bioagents.llms.llm_provider import get_llm

DOCKING_AGENT_PROMPT = (
    "You are an expert molecular docking agent. Your role is to provide guidance on molecular "
    "docking, virtual screening, and drug-target interaction analysis. You understand docking "
    "protocols (AutoDock, Vina, Glide), scoring functions, pose evaluation, and binding site "
    "identification. You can advise on ligand preparation, receptor preparation, grid box "
    "setup, and result interpretation. Always discuss binding affinity scores, key "
    "interactions (hydrogen bonds, hydrophobic contacts, pi-stacking), and ADMET "
    "considerations when evaluating docking results."
)


def create_docking_agent():
    """Create the Docking Agent node function.

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm()

    def agent_node(state):
        """The docking agent node function."""
        messages = state["messages"]
        windowed = prepare_messages_for_agent(messages, "docking")
        messages_with_system = [SystemMessage(content=DOCKING_AGENT_PROMPT), *windowed]

        return invoke_with_retry("Docking", llm, messages_with_system)

    return agent_node
