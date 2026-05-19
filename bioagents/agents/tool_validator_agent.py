"""Tool Validator Agent for validating tool calls before execution."""

from langchain_core.messages import SystemMessage

from bioagents.agents.helpers import invoke_with_retry, prepare_messages_for_agent
from bioagents.llms.llm_provider import get_llm

TOOL_VALIDATOR_AGENT_PROMPT = (
    "You are an expert tool validation agent. Your role is to validate tool calls before "
    "execution, check argument correctness, detect wrong tool usage, and suggest corrections. "
    "Review the last tool call in the conversation and verify: (1) the tool name matches "
    "the intended operation, (2) all required arguments are present and correctly typed, "
    "(3) argument values are within valid ranges, (4) the tool is appropriate for the "
    "current task. If issues are found, explain the problem and provide the corrected "
    "tool call. If the tool call is valid, confirm it and explain what it will do."
)


def create_tool_validator_agent():
    """Create the Tool Validator Agent node function.

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm()

    def agent_node(state):
        """The tool validator agent node function."""
        messages = state["messages"]
        windowed = prepare_messages_for_agent(messages, "tool_validator")
        messages_with_system = [
            SystemMessage(content=TOOL_VALIDATOR_AGENT_PROMPT),
            *windowed,
        ]

        return invoke_with_retry("ToolValidator", llm, messages_with_system)

    return agent_node
