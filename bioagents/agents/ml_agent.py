"""ML Agent for designing and executing machine learning systems."""

import logging

from langchain_core.messages import SystemMessage

from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

ML_AGENT_PROMPT = load_prompt("ml_agent")


def create_ml_agent(tools: list):
    """
    Create the ML Agent node function.

    The ML agent specializes in designing, executing, and optimizing
    machine learning systems for bioinformatics tasks.

    Args:
        tools: List of tools available to the agent (should include ml_tools)

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state):
        """
        The ML agent node function.

        Args:
            state: The current AgentState

        Returns:
            A dict with the 'messages' key containing the agent's response
        """
        messages = state["messages"]

        messages_with_system = [SystemMessage(content=ML_AGENT_PROMPT), *messages]

        response = llm_with_tools.invoke(messages_with_system)

        logger.info("ML Agent: Received LLM response")

        return {"messages": [response]}

    return agent_node
