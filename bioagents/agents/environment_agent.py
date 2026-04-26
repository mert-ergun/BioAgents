"""Environment Agent for setting up execution environments and dependencies."""

from langchain_core.messages import SystemMessage

from bioagents.agents.helpers import create_retry_response, prepare_messages_for_agent
from bioagents.llms.llm_provider import get_llm
from bioagents.tools.environment_tools import get_environment_tools

ENVIRONMENT_AGENT_PROMPT = (
    "You are an expert environment setup agent. Your role is to set up execution "
    "environments, create virtual environments (venv, conda), install dependencies from "
    "requirements files or package managers (pip, conda, apt), and check GPU availability "
    "and CUDA compatibility. You ensure reproducible environments by pinning versions, "
    "verifying installations, and resolving dependency conflicts. Always report the Python "
    "version, installed package versions, available hardware (CPU cores, RAM, GPU model "
    "and VRAM), and any compatibility issues detected."
)


def create_environment_agent():
    """Create the Environment Agent node function.

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm()
    tools = get_environment_tools()
    llm_with_tools = llm.bind_tools(tools)
    tool_names = [t.name for t in tools]

    def agent_node(state):
        """The environment agent node function."""
        messages = state["messages"]
        windowed = prepare_messages_for_agent(messages, "environment")
        messages_with_system = [
            SystemMessage(content=ENVIRONMENT_AGENT_PROMPT),
            *windowed,
        ]

        return create_retry_response(
            "Environment", messages_with_system, tool_names, llm_with_tools
        )

    return agent_node
