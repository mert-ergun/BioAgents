"""Analysis Agent for analyzing biological data."""

import logging

from langchain_core.messages import SystemMessage

from bioagents.agents.helpers import create_retry_response
from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

ANALYSIS_AGENT_PROMPT = load_prompt("analysis")


def create_analysis_agent(tools: list):
    """
    Create the Analysis Agent node function.

    Args:
        tools: List of tools available to the agent

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm(prompt_name="analysis")
    llm_with_tools = llm.bind_tools(tools)

    tool_names = [t.name for t in tools]

    def agent_node(state):
        """
        The analysis agent node function.

        Args:
            state: The current AgentState

        Returns:
            A dict with the 'messages' key containing the agent's response
        """
        messages = state["messages"]
        messages_with_system = [SystemMessage(content=ANALYSIS_AGENT_PROMPT), *messages]

        return create_retry_response(
            agent_name="Analysis agent",
            messages_with_system=messages_with_system,
            tool_names=tool_names,
            llm_with_tools=llm_with_tools,
        )

    return agent_node
