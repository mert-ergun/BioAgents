"""Critic Agent for scientific validation and domain standard checking."""

from langchain_core.messages import SystemMessage

from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt

CRITIC_AGENT_PROMPT = load_prompt("critic")


def create_critic_agent():
    """
    Create the Critic Agent node function.

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm(prompt_name="critic")

    def agent_node(state):
        """
        The critic agent node function.

        Args:
            state: The current AgentState

        Returns:
            A dict with the 'messages' key containing the agent's response
        """
        messages = state["messages"]

        messages_with_system = [SystemMessage(content=CRITIC_AGENT_PROMPT), *messages]

        response = llm.invoke(messages_with_system)

        return {"messages": [response]}

    return agent_node
