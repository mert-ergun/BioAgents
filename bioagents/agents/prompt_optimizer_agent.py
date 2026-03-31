"""Prompt Optimizer Agent for analyzing failures and improving agent prompts."""

from langchain_core.messages import SystemMessage

from bioagents.agents.helpers import invoke_with_retry, prepare_messages_for_agent
from bioagents.llms.llm_provider import get_llm

PROMPT_OPTIMIZER_AGENT_PROMPT = (
    "You are an expert prompt optimization agent. Your role is to analyze agent failures "
    "in the conversation, identify patterns in unsuccessful interactions, and suggest prompt "
    "modifications to improve agent performance. Examine the conversation history for: "
    "(1) empty or off-topic responses, (2) repeated failures on the same task, (3) "
    "misunderstanding of instructions, (4) incorrect tool usage patterns. For each issue "
    "found, propose specific prompt modifications with rationale. Suggest additions, "
    "removals, or rewording of prompt sections that would address the failure mode."
)


def create_prompt_optimizer_agent():
    """Create the Prompt Optimizer Agent node function.

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm()

    def agent_node(state):
        """The prompt optimizer agent node function."""
        messages = state["messages"]
        windowed = prepare_messages_for_agent(messages, "prompt_optimizer")
        messages_with_system = [
            SystemMessage(content=PROMPT_OPTIMIZER_AGENT_PROMPT),
            *windowed,
        ]

        return invoke_with_retry("PromptOptimizer", llm, messages_with_system)

    return agent_node
