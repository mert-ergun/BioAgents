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

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from bioagents.agents.agent_executor import execute_agent_with_tools, safe_json_output
from bioagents.agents.helpers import resolve_tool_name
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
    from bioagents.agents.helpers import (
        create_retry_response,
        prepare_messages_for_agent,
    )

    llm = get_llm(prompt_name="protein_design")
    llm_with_tools = llm.bind_tools(tools)
    tool_names = [t.name for t in tools]

    # Convert tools list to dict for executor
    tools_dict = {}
    for tool in tools:
        if isinstance(tool, BaseTool):
            tools_dict[tool.name] = tool.func if hasattr(tool, "func") else tool
        else:
            tools_dict[resolve_tool_name(tool)] = tool

    logger.info(f"Protein design agent tools: {list(tools_dict.keys())}")

    def protein_design_node(state):
        """The Protein Design Agent node function."""
        messages = state["messages"]
        windowed = prepare_messages_for_agent(messages, "protein_design")
        messages_with_system = [SystemMessage(content=PROTEIN_DESIGN_AGENT_PROMPT), *windowed]

        return create_retry_response(
            "ProteinDesign", messages_with_system, tool_names, llm_with_tools
        )

    return protein_design_node
