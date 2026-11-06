"""Tool Builder Agent for discovering, creating, and managing custom tools."""

import logging

from langchain_core.messages import SystemMessage

from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt
from bioagents.tools.selected_runtime import bind_agent_node, bind_agent_tools, tools_for_agent

logger = logging.getLogger(__name__)

TOOL_BUILDER_PROMPT = load_prompt("tool_builder")


def create_tool_builder_agent():
    """Create the Tool Builder Agent node function.

    Returns:
        A function that can be used as a LangGraph node
    """
    from bioagents.agents.helpers import create_retry_response, prepare_messages_for_agent

    llm = get_llm(prompt_name="tool_builder")
    tools = tools_for_agent("tool_builder")
    llm_with_tools = bind_agent_tools(llm, tools)
    tool_names = [t.name for t in tools]

    def tool_builder_node(state):
        """The Tool Builder Agent node function."""
        messages = state["messages"]
        windowed = prepare_messages_for_agent(messages, "tool_builder")
        messages_with_system = [SystemMessage(content=TOOL_BUILDER_PROMPT), *windowed]

        return create_retry_response(
            "ToolBuilder", messages_with_system, tool_names, llm_with_tools
        )

    return bind_agent_node(tool_builder_node)
