"""Paper Replication Agent for orchestrating full paper replication workflows."""

from langchain_core.messages import SystemMessage

from bioagents.agents.helpers import create_retry_response, prepare_messages_for_agent
from bioagents.llms.llm_provider import get_llm
from bioagents.tools.git_tools import get_git_tools
from bioagents.tools.pdf_tools import extract_pdf_text_spacy_layout
from bioagents.tools.web_tools import get_web_tools

PAPER_REPLICATION_AGENT_PROMPT = (
    "You are an expert paper replication agent. Your role is to orchestrate full paper "
    "replication workflows: download the paper, read methods sections, find the associated "
    "GitHub repository, clone it, install dependencies, run experiments, and report results. "
    "You methodically work through each step, verifying success before moving on. When "
    "replication fails, you diagnose the issue and suggest fixes. Always document your "
    "replication process and note any deviations from the original methods.\n\n"
    "When the user provides a local PDF file path, you MUST use the "
    "`extract_pdf_text_spacy_layout` tool to read the PDF content first. "
    "Extract the title, authors, abstract, methods, key findings, and any referenced "
    "GitHub repositories or code URLs from the paper text before proceeding with replication."
)


def create_paper_replication_agent():
    """Create the Paper Replication Agent node function.

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm()
    tools = get_web_tools() + get_git_tools() + [extract_pdf_text_spacy_layout]
    llm_with_tools = llm.bind_tools(tools)
    tool_names = [t.name for t in tools]

    def agent_node(state):
        """The paper replication agent node function."""
        messages = state["messages"]
        windowed = prepare_messages_for_agent(messages, "paper_replication")
        messages_with_system = [
            SystemMessage(content=PAPER_REPLICATION_AGENT_PROMPT),
            *windowed,
        ]

        return create_retry_response(
            "PaperReplication", messages_with_system, tool_names, llm_with_tools
        )

    return agent_node
