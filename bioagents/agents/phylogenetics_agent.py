"""Phylogenetics Agent for evolutionary analysis and tree construction."""

from langchain_core.messages import SystemMessage

from bioagents.agents.helpers import create_retry_response, prepare_messages_for_agent
from bioagents.llms.llm_provider import get_llm
from bioagents.tools.genomics_tools import get_genomics_tools

PHYLOGENETICS_AGENT_PROMPT = (
    "You are an expert phylogenetics agent. Your role is to construct phylogenetic trees, "
    "perform ancestral sequence reconstruction, and conduct evolutionary analysis. You "
    "understand multiple sequence alignment (MSA), substitution models, maximum likelihood "
    "and Bayesian inference methods, and molecular clock analysis. You can interpret tree "
    "topologies, bootstrap support values, and divergence time estimates. Always specify the "
    "alignment method, substitution model, and tree-building algorithm used."
)


def create_phylogenetics_agent():
    """Create the Phylogenetics Agent node function.

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm()
    tools = get_genomics_tools()
    llm_with_tools = llm.bind_tools(tools)
    tool_names = [t.name for t in tools]

    def agent_node(state):
        """The phylogenetics agent node function."""
        messages = state["messages"]
        windowed = prepare_messages_for_agent(messages, "phylogenetics")
        messages_with_system = [
            SystemMessage(content=PHYLOGENETICS_AGENT_PROMPT),
            *windowed,
        ]

        return create_retry_response(
            "Phylogenetics", messages_with_system, tool_names, llm_with_tools
        )

    return agent_node
