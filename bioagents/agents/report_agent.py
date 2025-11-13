"""Report Agent for synthesizing and presenting findings."""

from langchain_core.messages import SystemMessage

from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt

REPORT_AGENT_PROMPT = load_prompt("report")


def create_report_agent():
    """
    Create the Report Agent node function.

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm()

    def agent_node(state):
        """
        The report agent node function.

        Args:
            state: The current AgentState

        Returns:
            A dict with the 'messages' key containing the agent's response
        """
        messages = state["messages"]

        messages_with_system = [SystemMessage(content=REPORT_AGENT_PROMPT), *messages]

        response = llm.invoke(messages_with_system)

        return {"messages": [response]}

    return agent_node
