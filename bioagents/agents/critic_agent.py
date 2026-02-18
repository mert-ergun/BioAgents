"""Critic Agent for scientific validation and domain standard checking."""

import logging

from langchain_core.messages import SystemMessage

from bioagents.agents.agent_executor import safe_json_output
from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

CRITIC_AGENT_PROMPT = load_prompt("critic")

CRITIC_AGENT_MEMORY_PROMPT = """
You are the Critic Agent in a shared-memory multi-agent system.

CRITICAL: You MUST NOT read other agents' outputs from messages. Read only the shared memory passed to you.

OUTPUT FORMAT (JSON ONLY):
{
  "validation_results": [
    {"aspect": "scientific_accuracy", "score": 0-10, "issues": ["issue1"], "recommendations": ["rec1"]}
  ],
  "overall_score": 0-10,
  "critical_issues": ["issue1"],
  "completeness": "full" | "partial" | "failed"
}

Return ONLY valid JSON.
"""


def create_critic_agent():
    """
    Create the Critic Agent node function.

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm(prompt_name="critic")

    def agent_node(state):
        """
        The critic agent node function.

        Args:
            state: The current AgentState

        Returns:
            A dict with structured data for memory
        """
        try:
            messages = state.get("messages", [])
            messages_with_system = [SystemMessage(content=CRITIC_AGENT_MEMORY_PROMPT), *messages]

            response = llm.invoke(messages_with_system)

            raw_text = response.content if hasattr(response, "content") else str(response)

            # Parse JSON with fallback
            default_json = {
                "validation_results": [],
                "overall_score": 0,
                "critical_issues": [],
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
            logger.error(f"Critic agent error: {e}", exc_info=True)
            return {
                "data": {
                    "validation_results": [],
                    "overall_score": 0,
                    "critical_issues": [],
                    "completeness": "failed",
                },
                "raw_output": str(e),
                "tool_calls": [],
                "error": str(e),
            }

    return agent_node