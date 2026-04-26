"""Web Browser Agent for navigating the web and extracting data."""

from langchain_core.messages import SystemMessage

from bioagents.agents.helpers import create_retry_response, prepare_messages_for_agent
from bioagents.llms.llm_provider import get_llm
from bioagents.tools.web_tools import get_web_tools

WEB_BROWSER_AGENT_PROMPT = (
    "You are an expert web navigation agent. Your role is to browse the web, fetch "
    "documentation, extract structured data from web pages, and retrieve information from "
    "online resources. You can navigate to URLs, parse HTML content, download files, and "
    "interact with web APIs. Always verify the reliability of sources and present extracted "
    "data in a clean, structured format."
)


def create_web_browser_agent():
    """Create the Web Browser Agent node function.

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm()
    tools = get_web_tools()
    llm_with_tools = llm.bind_tools(tools)
    tool_names = [t.name for t in tools]

    def agent_node(state):
        """The web browser agent node function."""
        messages = state["messages"]
        windowed = prepare_messages_for_agent(messages, "web_browser")
        messages_with_system = [SystemMessage(content=WEB_BROWSER_AGENT_PROMPT), *windowed]

        return create_retry_response("WebBrowser", messages_with_system, tool_names, llm_with_tools)

    return agent_node
