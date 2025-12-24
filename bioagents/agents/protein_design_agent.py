"""Protein Design Agent for de novo binder generation and evaluation.

This agent specializes in computational protein design workflows including:
1. Target structure analysis from PDB/AlphaFold
2. Interface identification and hotspot selection
3. De novo binder design using RFdiffusion + ProteinMPNN
4. Structure prediction with AlphaFold-Multimer
5. Binding quality evaluation (iPTM, ipSAE metrics)
"""

from __future__ import annotations

import logging

from langchain_core.messages import SystemMessage

from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt
from bioagents.tools.protein_design_tools import get_protein_design_tools
from bioagents.tools.structural_tools import get_structural_tools

logger = logging.getLogger(__name__)


PROTEIN_DESIGN_AGENT_PROMPT = load_prompt("protein_design")


def create_protein_design_agent():
    """
    Create the Protein Design Agent.

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm(prompt_name="protein_design")

    # Combine structural and design tools
    tools = get_structural_tools() + get_protein_design_tools()
    llm_with_tools = llm.bind_tools(tools)

    def protein_design_node(state):
        """The Protein Design Agent node function."""
        messages = state["messages"]
        messages_with_system = [SystemMessage(content=PROTEIN_DESIGN_AGENT_PROMPT), *messages]

        response = llm_with_tools.invoke(messages_with_system)

        return {"messages": [response]}

    return protein_design_node


def get_all_protein_design_tools():
    """Return all tools available to the protein design agent."""
    return get_structural_tools() + get_protein_design_tools()
