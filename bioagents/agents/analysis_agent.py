"""Analysis Agent for analyzing biological data."""

from langchain_core.messages import SystemMessage

from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt

ANALYSIS_AGENT_PROMPT = load_prompt("analysis")


def create_analysis_agent(tools: list):
    """
    Create the Analysis Agent node function.

    Args:
        tools: List of tools available to the agent

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)

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

        response = llm_with_tools.invoke(messages_with_system)

        return {"messages": [response]}

    return agent_node
