"""Supervisor Agent for routing tasks to specialized agents."""

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel

from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt


class RouteResponse(BaseModel):
    """Response from supervisor for routing."""

    next_agent: Literal["research", "analysis", "report", "FINISH"]
    reasoning: str


SUPERVISOR_PROMPT = load_prompt("supervisor")


def create_supervisor_agent(members: list[str]):
    """
    Create the Supervisor Agent for routing.

    Args:
        members: List of available agent names

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm(agent_name="Supervisor")

    options = [*members, "FINISH"]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUPERVISOR_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next? "
                f"Choose from: {', '.join(options)!s}",
            ),
        ]
    )

    supervisor_chain = prompt | llm.with_structured_output(RouteResponse)

    def supervisor_node(state):
        """
        The supervisor node function.

        Args:
            state: The current AgentState

        Returns:
            A dict with next_agent and reasoning for routing
        """
        messages = state["messages"]
        result = supervisor_chain.invoke({"messages": messages})

        return {"next": result.next_agent, "reasoning": result.reasoning, "messages": []}

    return supervisor_node
