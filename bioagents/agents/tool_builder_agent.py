"""Tool Builder Agent for discovering, creating, and managing custom tools."""

import logging

from langchain_core.messages import SystemMessage

from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt
from bioagents.tools.tool_builder_tools import get_tool_builder_tools

logger = logging.getLogger(__name__)

# Load the tool builder prompt from XML
TOOL_BUILDER_PROMPT = load_prompt("tool_builder")


def create_tool_builder_agent():
    """Create the Tool Builder Agent node function.

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm(prompt_name="tool_builder")
    tools = get_tool_builder_tools()
    llm_with_tools = llm.bind_tools(tools)

    def tool_builder_node(state):
        """The Tool Builder Agent node function."""
        messages = state["messages"]
        messages_with_system = [SystemMessage(content=TOOL_BUILDER_PROMPT), *messages]

        response = llm_with_tools.invoke(messages_with_system)

        return {"messages": [response]}

    return tool_builder_node
