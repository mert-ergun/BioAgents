"""Summary Agent - Final user-facing output layer."""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

SUMMARY_AGENT_PROMPT = load_prompt("summary")

SUMMARY_AGENT_SYSTEM_PROMPT = """
You are the Summary Agent. Generate the final user-facing output.

If memory contains results from multiple agents, synthesize them into a comprehensive summary.
If memory is empty, answer the user's question directly and concisely.

Do NOT mention technical details like agent names or tool calls.
Focus on the scientific/biological findings and results.

Output a well-formatted, human-readable summary.
"""


def create_summary_agent():
    """
    Create the Summary Agent.

    Returns:
        Agent node function
    """
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

            # Check if memory has results
            has_results = any(
                agent_data.get("status") == "success" and agent_data.get("data")
                for agent_data in memory.values()
            )

            if not has_results:
                # Empty memory: answer directly
                prompt = """
You are a helpful assistant. Answer the user's question directly and clearly.
Do not mention agents, tools, or technical systems.
"""
                full_prompt = prompt
            else:
                # Has memory: synthesize findings
                memory_context = format_memory_for_summary(memory)
                full_prompt = f"""{SUMMARY_AGENT_SYSTEM_PROMPT}

SHARED MEMORY STATE:
{memory_context}
"""

            # Invoke LLM
            response = llm.invoke(
                [
                    SystemMessage(content=full_prompt),
                    user_message,
                ]
            )

            raw_output = response.content if hasattr(response, "content") else str(response)

            return {
                "data": {"summary": raw_output},
                "raw_output": raw_output,
                "tool_calls": [],
                "error": None,
            }

        except Exception as e:
            logger.error(f"Summary agent error: {e}", exc_info=True)
            return {
                "data": {"summary": f"Error: {e!s}"},
                "raw_output": str(e),
                "tool_calls": [],
                "error": str(e),
            }

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

    return "\n".join(lines) if lines else "No data available."
