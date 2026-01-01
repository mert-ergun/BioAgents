"""Summary Agent for user-facing workflow output."""

import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

SUMMARY_AGENT_PROMPT = load_prompt("summary")


class ExecutionMode(BaseModel):
    """Model for detecting execution mode."""

    mode: Literal["direct_answer", "full_summary"]
    reasoning: str


def analyze_execution_mode(messages: list) -> tuple[Literal["direct_answer", "full_summary"], str]:
    """
    Analyze the message history to determine if this was a simple single-turn query
    or a complex multi-agent execution.

    Args:
        messages: The list of messages from the state

    Returns:
        Tuple of (mode, reasoning)
    """
    # Count different message types
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]

    # Count tool calls across all AI messages
    total_tool_calls = 0
    for msg in ai_messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            total_tool_calls += len(msg.tool_calls)

    # Logic for determining mode
    is_simple_query = (
        len(human_messages) == 1  # Only the original user query
        and total_tool_calls == 0  # No tools were called
        and len(ai_messages) <= 1  # At most one AI response (direct answer)
    )

    if is_simple_query:
        return (
            "direct_answer",
            "Single user query with no tool calls or agent routing detected.",
        )
    else:
        return (
            "full_summary",
            f"Complex execution: {len(human_messages)} user messages, "
            f"{total_tool_calls} tool calls, {len(ai_messages)} agent messages.",
        )


def create_summary_agent():
    """
    Create the Summary Agent node function.

    The summary agent is the final user-facing output layer.
    It intelligently detects whether to provide a direct answer
    or summarize a complex multi-agent execution.

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm(prompt_name="summary")

    def summary_node(state):
        """
        The summary agent node function.

        Args:
            state: The current AgentState

        Returns:
            A dict with the 'messages' key containing the agent's response
        """
        messages = state["messages"]

        # Analyze the execution to determine output mode
        mode, reasoning = analyze_execution_mode(messages)

        logger.info(f"Summary Agent: Mode='{mode}'. Reasoning: {reasoning}")

        # Create a mode-aware system prompt
        mode_instruction = (
            "EXECUTION MODE: DIRECT ANSWER\n"
            "The user asked a simple question with no agent involvement.\n"
            "Provide a direct, concise answer without mentioning agents or tools.\n"
            "Keep the response focused and natural."
            if mode == "direct_answer"
            else "EXECUTION MODE: FULL SUMMARY\n"
            "Multiple agents and tools were involved in this execution.\n"
            "Summarize the workflow, agents used, tools called, and key results.\n"
            "Use clear section headers and maintain a user-friendly tone."
        )

        # Build messages with system prompt
        messages_with_system = [
            SystemMessage(content=SUMMARY_AGENT_PROMPT),
            SystemMessage(content=mode_instruction),
            *messages,
        ]

        # Invoke the LLM with the message history
        response = llm.invoke(messages_with_system)

        return {"messages": [response]}

    return summary_node