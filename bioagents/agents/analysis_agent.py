"""Analysis Agent - Tool-executing implementation."""

import logging

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from bioagents.agents.agent_executor import execute_agent_with_tools, safe_json_output
from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

ANALYSIS_AGENT_PROMPT = load_prompt("analysis")

ANALYSIS_AGENT_SYSTEM_PROMPT = """
You are the Analysis Agent. Your task is to analyze biological sequences and properties.

INSTRUCTIONS:
1. Use your available tools to calculate molecular weight, isoelectric point, etc.
2. Analyze amino acid composition and protein properties
3. Identify key patterns and features
4. Provide a comprehensive analysis summary

FINAL OUTPUT FORMAT:
After analysis, respond with ONLY valid JSON (no markdown, no explanations):

{
    "sequence_properties": {
        "length": 0,
        "molecular_weight": 0.0,
        "isoelectric_point": 0.0,
        "composition": {}
    },
    "key_findings": ["finding1", "finding2"],
    "patterns_detected": ["pattern1"],
    "quality_score": "high" | "medium" | "low",
    "analysis_complete": true,
    "status": "success" | "error",
    "error": null
}

ALWAYS end with valid JSON output.
"""


def create_analysis_agent(tools: list):
    """
    Create the Analysis Agent with tool-executing loop.

    Args:
        tools: List of tool objects

    Returns:
        Agent node function
    """
    llm = get_llm(prompt_name="analysis")
    llm_with_tools = llm.bind_tools(tools)

    # Convert tools list to dict
    tools_dict = {}
    for tool in tools:
        if isinstance(tool, BaseTool):
            tools_dict[tool.name] = tool.func if hasattr(tool, "func") else tool
        elif callable(tool):
            tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", "unknown")
            tools_dict[tool_name] = tool
        else:
            tools_dict[str(tool)] = tool

    logger.info(f"Analysis agent tools: {list(tools_dict.keys())}")

    def analysis_node(state):
        """Analysis agent with tool execution loop."""
        try:
            messages = state.get("messages", [])

            # Find user message
            user_message = None
            for m in messages:
                if isinstance(m, HumanMessage):
                    user_message = m
                    break

            if user_message is None:
                user_message = HumanMessage(content="Analyze the biological data.")

            # Execute agent with tools
            raw_output, tool_calls_used = execute_agent_with_tools(
                llm_with_tools=llm_with_tools,
                system_prompt=ANALYSIS_AGENT_SYSTEM_PROMPT,
                user_message=user_message,
                tools_dict=tools_dict,
                max_iterations=5,
            )

            # Parse JSON with fallback
            default_json = {
                "sequence_properties": {
                    "length": 0,
                    "molecular_weight": 0.0,
                    "isoelectric_point": 0.0,
                    "composition": {},
                },
                "key_findings": [],
                "patterns_detected": [],
                "quality_score": "medium",
                "analysis_complete": False,
                "status": "success" if raw_output else "error",
                "error": None,
            }

            structured_data = safe_json_output(raw_output, default_json)

            return {
                "data": structured_data,
                "raw_output": raw_output,
                "tool_calls": tool_calls_used,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Analysis agent error: {e}", exc_info=True)
            return {
                "data": {
                    "sequence_properties": {
                        "length": 0,
                        "molecular_weight": 0.0,
                        "isoelectric_point": 0.0,
                        "composition": {},
                    },
                    "key_findings": [],
                    "patterns_detected": [],
                    "quality_score": "low",
                    "analysis_complete": False,
                    "status": "error",
                    "error": str(e),
                },
                "raw_output": str(e),
                "tool_calls": [],
                "error": str(e),
            }

    return analysis_node
