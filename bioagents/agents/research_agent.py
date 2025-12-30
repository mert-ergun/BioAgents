"""Research Agent for fetching biological data and conducting literature searches."""

import logging

from langchain_core.messages import SystemMessage

from bioagents.agents.helpers import create_retry_response, extract_task_from_messages
from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

RESEARCH_AGENT_PROMPT = load_prompt("research")


def create_research_agent(tools: list):
    """
    Create the Research Agent node function.

    Args:
        tools: List of tools available to the agent

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm(prompt_name="research")
    llm_with_tools = llm.bind_tools(tools)

    # Get tool names for helpful error messages
    tool_names = [t.name for t in tools]

    def agent_node(state):
        """
        The research agent node function.

        Args:
            state: The current AgentState

        Returns:
            A dict with the 'messages' key containing the agent's response
        """
        messages = state["messages"]
        messages_with_system = [SystemMessage(content=RESEARCH_AGENT_PROMPT), *messages]

        return create_retry_response(
            agent_name="Research agent",
            messages_with_system=messages_with_system,
            tool_names=tool_names,
            llm_with_tools=llm_with_tools,
            task_extractor=extract_task_from_messages,
        )

    return agent_node
