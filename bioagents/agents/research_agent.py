"""Research Agent for fetching biological data - Tool-executing implementation."""

import logging
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from bioagents.agents.agent_executor import execute_agent_with_tools, safe_json_output
from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

RESEARCH_AGENT_PROMPT = load_prompt("research")

RESEARCH_AGENT_SYSTEM_PROMPT = """
You are the Research Agent. Your task is to fetch biological data, sequences, and analyze scientific literature.

INSTRUCTIONS:
1. Use your available tools to fetch FASTA sequences and literature data
2. Call tools like fetch_uniprot_fasta to get protein sequences
3. When asked to review, summarize, or answer questions based on local PDF papers, MUST use the `search_local_papers_with_paperqa` tool.
4. Integrate the tool's findings into your final summary.
5. Once you have gathered the data, provide a final summary.

FINAL OUTPUT FORMAT:
After gathering data, respond with ONLY valid JSON (no markdown, no explanations):

{
    "fetched_sequences": ["seq1", "seq2"],
    "literature_findings": "Detailed summary of findings from papers or online sources",
    "data_sources": ["source1", "source2"],
    "completeness": "full" | "partial" | "failed",
    "next_steps": "what should be done next",
    "status": "success" | "error",
    "error": null or "error message"
}

If you cannot fetch data, return the JSON with completeness="failed" and status="error".
ALWAYS end with valid JSON output.
"""


def create_research_agent(tools: list):
    """
    Create the Research Agent with tool-executing loop.

    Args:
        tools: List of tool objects

    Returns:
        Agent node function
    """
    llm = get_llm(prompt_name="research")
    llm_with_tools = llm.bind_tools(tools)

    # Convert tools list to dict for executor
    tools_dict = {}
    for tool in tools:
        if isinstance(tool, BaseTool):
            # LangChain BaseTool objects have a `func` attribute
            tools_dict[tool.name] = tool.func if hasattr(tool, 'func') else tool
        elif hasattr(tool, '__call__'):
            # Callable objects or functions
            tool_name = getattr(tool, 'name', None) or getattr(tool, '__name__', 'unknown')
            tools_dict[tool_name] = tool
        else:
            # Fallback: use object's name or str representation
            tool_name = str(tool)
            tools_dict[tool_name] = tool
    
    logger.info(f"Research agent tools: {list(tools_dict.keys())}")

    def research_node(state):
        """Research agent with tool execution loop."""
        try:
            messages = state.get("messages", [])
            
            # Find user message
            user_message = None
            for m in messages:
                if isinstance(m, HumanMessage):
                    user_message = m
                    break
            
            if user_message is None:
                user_message = HumanMessage(content="Fetch biological data.")

            # Execute agent with tools
            raw_output, tool_calls_used = execute_agent_with_tools(
                llm_with_tools=llm_with_tools,
                system_prompt=RESEARCH_AGENT_SYSTEM_PROMPT,
                user_message=user_message,
                tools_dict=tools_dict,
                max_iterations=5,
            )

            # Parse JSON with fallback
            default_json = {
                "fetched_sequences": [],
                "literature_findings": raw_output[:500] if raw_output else "",
                "data_sources": [],
                "completeness": "partial",
                "next_steps": "data retrieval incomplete",
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
            logger.error(f"Research agent error: {e}", exc_info=True)
            return {
                "data": {
                    "fetched_sequences": [],
                    "literature_findings": "",
                    "data_sources": [],
                    "completeness": "failed",
                    "next_steps": "error recovery needed",
                    "status": "error",
                    "error": str(e),
                },
                "raw_output": str(e),
                "tool_calls": [],
                "error": str(e),
            }

    return research_node