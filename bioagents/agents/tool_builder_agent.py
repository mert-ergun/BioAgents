"""Tool Builder Agent for discovering, creating, and managing custom tools."""

import logging

from langchain_core.messages import SystemMessage

from bioagents.agents.agent_executor import safe_json_output
from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

TOOL_BUILDER_PROMPT = load_prompt("tool_builder")

TOOL_BUILDER_MEMORY_PROMPT = """
You are the Tool Builder Agent in a shared-memory multi-agent system.

CRITICAL: You MUST NOT read other agents' outputs from messages. Read only the shared memory passed to you.

OUTPUT FORMAT (JSON ONLY):
{
  "discovered_tools": [
    {"name": "tool1", "description": "...", "url": "...", "status": "discovered" | "created"}
  ],
  "created_tools": [...],
  "validation_results": [...],
  "completeness": "full" | "partial" | "failed"
}

Return ONLY valid JSON.
"""


def create_tool_builder_agent():
    """
    Create the Tool Builder Agent node function.

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm(prompt_name="tool_builder")

    def tool_builder_node(state):
        """
        The Tool Builder Agent node function.

        Args:
            state: The current AgentState

        Returns:
            A dict with structured data for memory
        """
        try:
            messages = state.get("messages", [])
            messages_with_system = [SystemMessage(content=TOOL_BUILDER_MEMORY_PROMPT), *messages]

            response = llm.invoke(messages_with_system)

            raw_text = response.content if hasattr(response, "content") else str(response)

            # Parse JSON with fallback
            default_json = {
                "discovered_tools": [],
                "created_tools": [],
                "validation_results": [],
                "completeness": "partial",
            }

            structured_data = safe_json_output(raw_text, default_json)

            return {
                "data": structured_data,
                "raw_output": raw_text,
                "tool_calls": [],
                "error": None,
            }

        except Exception as e:
            logger.error(f"Tool builder agent error: {e}", exc_info=True)
            return {
                "data": {
                    "discovered_tools": [],
                    "created_tools": [],
                    "validation_results": [],
                    "completeness": "failed",
                },
                "raw_output": str(e),
                "tool_calls": [],
                "error": str(e),
            }

    return tool_builder_node
