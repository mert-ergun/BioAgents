"""Tool Discovery Agent for searching, discovering, and assigning tools."""

from langchain_core.messages import SystemMessage

from bioagents.agents.helpers import create_retry_response, prepare_messages_for_agent
from bioagents.llms.llm_provider import get_llm
from bioagents.tools.tool_builder_tools import get_tool_builder_tools

TOOL_DISCOVERY_AGENT_PROMPT = (
    "You are an expert tool discovery agent. Your role is to search for available tools, "
    "discover new tools from tool registries, and assign tools to the right agents. You can "
    "install missing packages, register new tool endpoints, and verify tool availability. "
    "When a task requires a tool that is not currently available, search for it, install any "
    "required dependencies, and make it accessible. Always report tool capabilities, "
    "required inputs, expected outputs, and any installation steps performed."
)


def create_tool_discovery_agent():
    """Create the Tool Discovery Agent node function.

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm()
    tools = get_tool_builder_tools()
    llm_with_tools = llm.bind_tools(tools)
    tool_names = [t.name for t in tools]

    def agent_node(state):
        """The tool discovery agent node function."""
        messages = state["messages"]
        windowed = prepare_messages_for_agent(messages, "tool_discovery")
        messages_with_system = [
            SystemMessage(content=TOOL_DISCOVERY_AGENT_PROMPT),
            *windowed,
        ]

        return create_retry_response(
            "ToolDiscovery", messages_with_system, tool_names, llm_with_tools
        )

    return agent_node
