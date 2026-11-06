"""Structural Biology Agent for protein structure prediction and analysis."""

from langchain_core.messages import SystemMessage

from bioagents.agents.helpers import create_retry_response, prepare_messages_for_agent
from bioagents.llms.llm_provider import get_llm
from bioagents.tools.selected_runtime import bind_agent_node, bind_agent_tools, tools_for_agent

STRUCTURAL_BIOLOGY_AGENT_PROMPT = (
    "You are an expert structural biology agent. Your role is to perform protein structure "
    "prediction, molecular dynamics simulations, and structural analysis. You understand PDB "
    "file formats, protein folding principles, and structure-function relationships. You can "
    "run and interpret results from AlphaFold, ESMFold, RFdiffusion, and molecular dynamics "
    "packages. Always report structural quality metrics (pLDDT, RMSD, TM-score) and highlight "
    "functionally important regions such as active sites and binding interfaces."
)


def create_structural_biology_agent():
    """Create the Structural Biology Agent node function.

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm()
    tools = tools_for_agent("structural_biology")
    llm_with_tools = bind_agent_tools(llm, tools)
    tool_names = [t.name for t in tools]

    def agent_node(state):
        """The structural biology agent node function."""
        messages = state["messages"]
        windowed = prepare_messages_for_agent(messages, "structural_biology")
        messages_with_system = [
            SystemMessage(content=STRUCTURAL_BIOLOGY_AGENT_PROMPT),
            *windowed,
        ]

        return create_retry_response(
            "StructuralBiology", messages_with_system, tool_names, llm_with_tools
        )

    return bind_agent_node(agent_node)
