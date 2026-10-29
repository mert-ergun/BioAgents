"""Shell Agent for executing shell commands and managing system operations."""

from langchain_core.messages import SystemMessage

from bioagents.agents.helpers import create_retry_response, prepare_messages_for_agent
from bioagents.llms.llm_provider import get_llm
from bioagents.tools.shell_tools import get_shell_tools

SHELL_AGENT_PROMPT = (
    "You are an expert shell execution agent. Your role is to execute shell commands in "
    "the sandbox environment, manage system operations, and install software. You can run "
    "arbitrary commands, chain operations with pipes, manage processes, and handle file "
    "system operations. Always validate commands before execution, use appropriate flags "
    "for non-interactive operation, and capture both stdout and stderr. Report command "
    "exit codes, execution time, and any errors encountered. Prefer safe, idempotent "
    "operations when possible."
)


def build_shell_agent_prompt() -> str:
    """Return the shell prompt plus the canonical workspace root the commands run in."""
    try:
        from bioagents.sandbox.workspace import describe_workspace_root, workspace_root

        root = workspace_root()
    except Exception:  # pragma: no cover - the prompt must survive a sandbox failure
        return SHELL_AGENT_PROMPT
    return (
        f"{SHELL_AGENT_PROMPT}\n\n{describe_workspace_root(root)}\n"
        f"Your commands start in {root}; use paths relative to it (or absolute paths under "
        "it). Do not cd to other prefixes to look for the workspace."
    )


def create_shell_agent():
    """Create the Shell Agent node function.

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm()
    tools = get_shell_tools()
    llm_with_tools = llm.bind_tools(tools)
    tool_names = [t.name for t in tools]

    def agent_node(state):
        """The shell agent node function."""
        messages = state["messages"]
        windowed = prepare_messages_for_agent(messages, "shell")
        messages_with_system = [SystemMessage(content=build_shell_agent_prompt()), *windowed]

        return create_retry_response("Shell", messages_with_system, tool_names, llm_with_tools)

    return agent_node
