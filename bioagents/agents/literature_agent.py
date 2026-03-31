"""Literature Agent for searching and summarizing scientific literature."""

from langchain_core.messages import SystemMessage

from bioagents.agents.helpers import create_retry_response, prepare_messages_for_agent
from bioagents.llms.llm_provider import get_llm
from bioagents.tools.literature_tools import get_literature_tools

LITERATURE_AGENT_PROMPT = (
    "You are an expert scientific literature researcher. Your role is to search, retrieve, "
    "and summarize scientific literature from PubMed, ArXiv, and BioRxiv. You can find relevant "
    "papers by keyword, author, or topic, extract key findings, compare methodologies across "
    "studies, and synthesize information into clear summaries. Always cite papers with their "
    "titles, authors, and publication details. Prioritize recent, high-impact, peer-reviewed "
    "publications and preprints from reputable groups."
)


def create_literature_agent():
    """Create the Literature Agent node function.

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm()
    tools = get_literature_tools()
    llm_with_tools = llm.bind_tools(tools)
    tool_names = [t.name for t in tools]

    def agent_node(state):
        """The literature agent node function."""
        messages = state["messages"]
        windowed = prepare_messages_for_agent(messages, "literature")
        messages_with_system = [SystemMessage(content=LITERATURE_AGENT_PROMPT), *windowed]

        return create_retry_response("Literature", messages_with_system, tool_names, llm_with_tools)

    return agent_node
