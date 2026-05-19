"""Planner Agent for breaking complex tasks into step-by-step execution plans."""

from langchain_core.messages import SystemMessage

from bioagents.agents.helpers import invoke_with_retry, prepare_messages_for_agent
from bioagents.llms.llm_provider import get_llm

PLANNER_AGENT_PROMPT = (
    "You are an expert bioinformatics task planner. Your role is to break complex "
    "bioinformatics tasks into step-by-step execution plans with dependencies. Analyze "
    "the task, identify required data, tools, and computational resources, then produce "
    "a numbered plan with clear agent assignments for each step. Specify input/output "
    "relationships between steps, estimate computational requirements, and flag potential "
    "failure points. Format your plan as a numbered list where each step includes: the "
    "action, the responsible agent, required inputs, expected outputs, and dependencies "
    "on previous steps."
)


def create_planner_agent():
    """Create the Planner Agent node function.

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm()

    def agent_node(state):
        """The planner agent node function."""
        messages = state["messages"]
        windowed = prepare_messages_for_agent(messages, "planner")
        messages_with_system = [SystemMessage(content=PLANNER_AGENT_PROMPT), *windowed]

        return invoke_with_retry("Planner", llm, messages_with_system)

    return agent_node
