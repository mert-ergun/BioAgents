"""Visualization Agent for creating publication-quality plots and charts."""

from langchain_core.messages import SystemMessage

from bioagents.agents.helpers import create_retry_response, prepare_messages_for_agent
from bioagents.llms.llm_provider import get_llm
from bioagents.tools.visualization_tools import get_visualization_tools

VISUALIZATION_AGENT_PROMPT = (
    "You are an expert data visualization agent. Your role is to create publication-quality "
    "plots, charts, heatmaps, and visualizations for biological data. You are proficient "
    "with matplotlib, seaborn, plotly, and specialized bioinformatics visualization libraries. "
    "You can create volcano plots, MA plots, heatmaps, PCA plots, UMAP embeddings, "
    "phylogenetic tree visualizations, genome browser tracks, and protein structure renders. "
    "Always use appropriate color palettes (colorblind-friendly), proper axis labels with "
    "units, legends, and titles. Save figures in high-resolution formats (PNG 300 DPI, SVG, "
    "or PDF) suitable for publication."
)


def create_visualization_agent():
    """Create the Visualization Agent node function.

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm()
    tools = get_visualization_tools()
    llm_with_tools = llm.bind_tools(tools)
    tool_names = [t.name for t in tools]

    def agent_node(state):
        """The visualization agent node function."""
        messages = state["messages"]
        windowed = prepare_messages_for_agent(messages, "visualization")
        messages_with_system = [
            SystemMessage(content=VISUALIZATION_AGENT_PROMPT),
            *windowed,
        ]

        return create_retry_response(
            "Visualization", messages_with_system, tool_names, llm_with_tools
        )

    return agent_node
