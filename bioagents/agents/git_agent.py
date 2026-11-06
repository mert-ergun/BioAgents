"""Git Agent for managing git repositories and version control."""

from langchain_core.messages import SystemMessage

from bioagents.agents.helpers import create_retry_response, prepare_messages_for_agent
from bioagents.llms.llm_provider import get_llm
from bioagents.tools.selected_runtime import bind_agent_node, bind_agent_tools, tools_for_agent

GIT_AGENT_PROMPT = (
    "You are an expert git and version control agent. Your role is to manage git "
    "repositories, clone repos, browse code, check out branches, and navigate repository "
    "structure. You can clone repositories from GitHub, GitLab, and Bitbucket, inspect "
    "commit history, diff changes, switch branches, and locate specific files or code "
    "patterns within repositories. Always verify repository URLs before cloning, report "
    "repository metadata (size, branch count, last commit), and handle authentication "
    "requirements gracefully."
)


def create_git_agent():
    """Create the Git Agent node function.

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm()
    tools = tools_for_agent("git")
    llm_with_tools = bind_agent_tools(llm, tools)
    tool_names = [t.name for t in tools]

    def agent_node(state):
        """The git agent node function."""
        messages = state["messages"]
        windowed = prepare_messages_for_agent(messages, "git")
        messages_with_system = [SystemMessage(content=GIT_AGENT_PROMPT), *windowed]

        return create_retry_response("Git", messages_with_system, tool_names, llm_with_tools)

    return bind_agent_node(agent_node)
