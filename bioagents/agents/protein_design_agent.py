"""Protein Design Agent for de novo binder generation and evaluation.

This agent specializes in computational protein design workflows including:
1. Target structure analysis from PDB/AlphaFold
2. Interface identification and hotspot selection
3. De novo binder design using RFdiffusion + ProteinMPNN
4. Structure prediction with AlphaFold-Multimer
5. Binding quality evaluation (iPTM, ipSAE metrics)

NOTE: This agent also handles structural lookup queries (PDB ID searches,
structure retrieval) that the supervisor may route here. For pure data
lookup tasks (fetching sequences, finding IDs), prefer routing to 'research'.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from bioagents.agents.agent_executor import execute_agent_with_tools, safe_json_output
from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

PROTEIN_DESIGN_AGENT_PROMPT = load_prompt("protein_design")

PROTEIN_DESIGN_AGENT_MEMORY_PROMPT = """
You are the Protein Design Agent in a shared-memory multi-agent system.

You handle TWO types of tasks:
1. Full protein design workflows (binder generation, structure prediction, evaluation)
2. Structural lookup queries (finding PDB IDs, fetching structure metadata, simple factual questions about protein structures)

CRITICAL: You MUST NOT read other agents' outputs from messages. Read only the shared memory passed to you.

For LOOKUP tasks (e.g. "find the PDB ID for X", "what is the structure of Y"):
- Use your tools to search/fetch the answer
- Put the answer in "lookup_result" field
- Set completeness to "full"

For DESIGN tasks:
- Run the full design pipeline
- Populate design_candidates, predicted_structures, evaluation fields

OUTPUT FORMAT (JSON ONLY - no markdown, no extra text):
{
  "lookup_result": "direct answer string if this was a lookup/search task, else null",
  "design_candidates": [
    {"id": "candidate1", "sequence": "...", "score": 0.0, "notes": "..."}
  ],
  "predicted_structures": [],
  "evaluation": {},
  "completeness": "full" | "partial" | "failed"
}

Return ONLY valid JSON. Do not wrap in markdown code blocks.
"""


def create_protein_design_agent(tools):
    """
    Create the Protein Design Agent with tool-executing loop.

    Args:
        tools: List of tool objects

    Returns:
        Agent node function
    """
    llm = get_llm(prompt_name="protein_design")
    llm_with_tools = llm.bind_tools(tools)

    # Convert tools list to dict for executor
    tools_dict = {}
    for tool in tools:
        if isinstance(tool, BaseTool):
            tools_dict[tool.name] = tool.func if hasattr(tool, 'func') else tool
        elif hasattr(tool, '__call__'):
            tool_name = getattr(tool, 'name', None) or getattr(tool, '__name__', 'unknown')
            tools_dict[tool_name] = tool
        else:
            tool_name = str(tool)
            tools_dict[tool_name] = tool

    logger.info(f"Protein design agent tools: {list(tools_dict.keys())}")

    def protein_design_node(state):
        """Protein design agent with tool execution loop."""
        try:
            messages = state.get("messages", [])

            # Find user message
            user_message = None
            for m in messages:
                if isinstance(m, HumanMessage):
                    user_message = m
                    break

            if user_message is None:
                user_message = HumanMessage(content="Design protein binders.")

            # Execute agent with tools
            raw_output, tool_calls_used = execute_agent_with_tools(
                llm_with_tools=llm_with_tools,
                system_prompt=PROTEIN_DESIGN_AGENT_MEMORY_PROMPT,
                user_message=user_message,
                tools_dict=tools_dict,
                max_iterations=5,
            )

            logger.info(f"Protein design raw_output: {raw_output}")

            # Default JSON structure — now includes lookup_result
            default_json = {
                "lookup_result": None,
                "design_candidates": [],
                "predicted_structures": [],
                "evaluation": {},
                "completeness": "partial",
            }

            structured_data = safe_json_output(raw_output, default_json)

            # ── FIX 1: If JSON parsing failed (all defaults) but raw_output is a
            #    short plain-text answer (e.g. "1TUP"), treat it as a lookup result.
            #    This rescues cases where the LLM returned a bare string instead of JSON.
            is_all_defaults = (
                not structured_data.get("lookup_result")
                and not structured_data.get("design_candidates")
                and not structured_data.get("predicted_structures")
                and not structured_data.get("evaluation")
            )
            raw_stripped = raw_output.strip() if isinstance(raw_output, str) else ""

            if is_all_defaults and raw_stripped:
                # Check if raw output looks like a short factual answer (≤ 200 chars,
                # no JSON braces) — capture it as lookup_result so summary can use it.
                if len(raw_stripped) <= 200 and "{" not in raw_stripped:
                    logger.info(
                        f"Protein design: raw output looks like a plain-text answer "
                        f"('{raw_stripped}'). Saving as lookup_result."
                    )
                    structured_data["lookup_result"] = raw_stripped
                    structured_data["completeness"] = "full"

            # ── FIX 2: If tools were executed and completeness is still partial,
            #    assume the agent completed its work.
            if tool_calls_used and structured_data.get("completeness") == "partial":
                structured_data["completeness"] = "full"
                logger.info(
                    "Protein design: Setting completeness to 'full' since tools were executed."
                )

            # ── FIX 3: Always preserve the raw_output alongside structured data
            #    so downstream agents (summary) can fall back to it if needed.
            structured_data["_raw_output"] = raw_stripped

            return {
                "data": structured_data,
                "raw_output": raw_output,
                "tool_calls": tool_calls_used,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Protein design agent error: {e}", exc_info=True)
            return {
                "data": {
                    "lookup_result": None,
                    "design_candidates": [],
                    "predicted_structures": [],
                    "evaluation": {},
                    "completeness": "failed",
                    "_raw_output": "",
                },
                "raw_output": str(e),
                "tool_calls": [],
                "error": str(e),
            }

    return protein_design_node