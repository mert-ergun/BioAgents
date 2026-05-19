"""Result Checker Agent for validating outputs and ensuring scientific plausibility."""

from langchain_core.messages import SystemMessage

from bioagents.agents.helpers import invoke_with_retry, prepare_messages_for_agent
from bioagents.llms.llm_provider import get_llm

RESULT_CHECKER_AGENT_PROMPT = (
    "You are an expert result validation agent. Your role is to validate outputs against "
    "expected formats, check scientific plausibility, verify data consistency, and ensure "
    "reproducibility. For each result you review, check: (1) data format correctness "
    "(proper headers, column types, value ranges), (2) scientific plausibility (are p-values "
    "reasonable, are fold changes biologically meaningful, are sequences valid), (3) internal "
    "consistency (do sample counts match, are IDs unique, are cross-references valid), "
    "(4) reproducibility (are random seeds set, are versions documented, are parameters "
    "recorded). Report issues with severity levels: critical, warning, or informational."
)


def create_result_checker_agent():
    """Create the Result Checker Agent node function.

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm()

    def agent_node(state):
        """The result checker agent node function."""
        messages = state["messages"]
        windowed = prepare_messages_for_agent(messages, "result_checker")
        messages_with_system = [
            SystemMessage(content=RESULT_CHECKER_AGENT_PROMPT),
            *windowed,
        ]

        return invoke_with_retry("ResultChecker", llm, messages_with_system)

    return agent_node
