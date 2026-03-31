"""Summary Agent - Final user-facing output layer."""

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from bioagents.agents.helpers import (
    extract_best_content,
    invoke_with_retry,
    is_empty_response,
    prepare_messages_for_agent,
)
from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

SUMMARY_AGENT_PROMPT = load_prompt("summary")


def analyze_execution_mode(messages: list) -> tuple[str, str]:
    """
    Analyze the message history to determine if this was a simple single-turn query
    or a complex multi-agent execution.

    Returns:
        Tuple of (mode, reasoning)
    """
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]

    total_tool_calls = 0
    for msg in ai_messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            total_tool_calls += len(msg.tool_calls)

    is_simple_query = len(human_messages) == 1 and total_tool_calls == 0 and len(ai_messages) <= 1

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
    Create the Summary Agent.

    Returns:
        Agent node function
    """
    load_prompt("summary")
    llm = get_llm(prompt_name="summary")

    def summary_node(state):
        """Summary agent - generates final user output."""
        try:
            memory = state.get("memory", {}) or {}
            messages = state.get("messages", []) or []

            # Find user message
            user_message = None
            for m in messages:
                if isinstance(m, HumanMessage):
                    user_message = m
                    break

            if user_message is None:
                user_message = HumanMessage(content="")

            mode, reasoning = analyze_execution_mode(messages)

            if not has_results:
                # Empty memory: answer directly
                prompt = """
You are a helpful assistant. Directly answer the user's question clearly.
Do not mention agents, tools, or technical systems.
"""
                full_prompt = prompt
            else:
                # Has memory: synthesize findings
                complexity = assess_execution_complexity(memory)
                memory_context = format_memory_for_summary(memory)
                full_prompt = f"""{SUMMARY_AGENT_SYSTEM_PROMPT}

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

        windowed = prepare_messages_for_agent(messages, "summary", summary_mode=True)
        messages_with_system = [
            SystemMessage(content=SUMMARY_AGENT_PROMPT),
            SystemMessage(content=mode_instruction),
            *windowed,
        ]

        result = invoke_with_retry("Summary", llm, messages_with_system, max_retries=1)

        response = result["messages"][0]
        if is_empty_response(response):
            best = extract_best_content(messages)
            if best:
                fallback_content = (
                    "## Workflow Summary\n\n"
                    "The following results were produced during this workflow:\n\n"
                    f"{best}"
                )
            else:
                fallback_content = (
                    "The workflow completed but was unable to generate a summary. "
                    "Please review the agent outputs in the audit log for details."
                )
            return {"messages": [AIMessage(content=fallback_content)]}

        return result

    return summary_node


def format_memory_for_summary(memory: dict) -> str:
    """Format memory for summary generation."""
    lines = []

    for agent_name in sorted(memory.keys()):
        agent_data = memory[agent_name]

        if agent_data.get("status") != "success":
            continue

        lines.append(f"\n{agent_name.upper()}:")

        if agent_data.get("data"):
            try:
                lines.append(json.dumps(agent_data["data"], indent=2))
            except (TypeError, ValueError):
                lines.append(str(agent_data["data"]))

    return "\n".join(lines) if lines else "No completed agent tasks."
