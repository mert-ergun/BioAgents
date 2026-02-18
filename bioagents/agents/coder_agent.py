"""Coder Agent for generating and executing code via Jupyter notebooks."""

import logging
import re
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage
from smolagents import CodeAgent, Tool

from bioagents.llms.adapters import LangChainModelAdapter
from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt
from bioagents.sandbox.coder_executor import create_executor
from bioagents.sandbox.coder_helpers import (
    DEFAULT_CODER_IMPORTS,
    build_task_with_output_dir,
    extract_available_data,
    extract_original_query,
    format_coder_result,
)
from bioagents.tools.smol_tool_wrappers import ToolUniverseExecuteTool, ToolUniverseSearchTool

logger = logging.getLogger("BioAgents")

CODER_AGENT_PROMPT = load_prompt("coder")


def create_coder_agent(
    tools: list[Tool] | None = None,
    additional_imports: list[str] | None = None,
    max_steps: int = 20,
) -> CodeAgent:
    """
    Create the Coder Agent instance.

    Args:
        tools: List of tools available to the agent (defaults to ToolUniverse tools)
        additional_imports: Additional Python packages to allow in the sandbox
        max_steps: Maximum number of execution steps

    Returns:
        A CodeAgent instance that can generate and execute Python code via Jupyter notebooks
    """
    if tools is None:
        tools = [ToolUniverseSearchTool(), ToolUniverseExecuteTool()]

    if additional_imports is None:
        additional_imports = DEFAULT_CODER_IMPORTS

    lc_model = get_llm(prompt_name="coder")
    model = LangChainModelAdapter(lc_model)
    executor = create_executor("coder", additional_imports)

    # Escape Jinja2 template syntax in instructions to avoid conflicts
    escaped_instructions = CODER_AGENT_PROMPT.replace("{", "{{").replace("}", "}}")

    from bioagents.sandbox.coder_helpers import PermissiveList

    agent = CodeAgent(
        tools=tools,
        model=model,
        executor=executor,
        additional_authorized_imports=PermissiveList(additional_imports),
        max_steps=max_steps,
        instructions=escaped_instructions,
    )

    return agent


def create_coder_node(agent: CodeAgent) -> Callable:
    """
    Create the Coder Agent node function.

    Args:
        agent: The CodeAgent instance to wrap

    Returns:
        A function that can be used as a LangGraph node
    """

    def create_coder_agent():
        """Create Coder Agent with memory-based output."""
        llm = get_llm(prompt_name="coder")

        CODER_AGENT_MEMORY_PROMPT = """
    You are the Coder Agent. Write code solutions for biological problems.

    OUTPUT FORMAT (JSON):
    {
        "code": "...full working code...",
        "language": "python" | "r" | "javascript",
        "description": "...what this code does...",
        "dependencies": [...list of packages needed...],
        "execution_ready": boolean
    }

    Return ONLY valid JSON.
    """

        def coder_node(state):
            try:
                messages = state["messages"]
                messages_with_system = [
                    SystemMessage(content=CODER_AGENT_MEMORY_PROMPT),
                    *messages
                ]
                
                response = llm.invoke(messages_with_system)
                raw_text = response.content if hasattr(response, "content") else str(response)
                structured_data = parse_json_response(raw_text)

                return {
                    "data": structured_data,
                    "raw_output": raw_text,
                    "tool_calls": [],
                    "error": None,
                }
            except Exception as e:
                logger.error(f"Coder agent error: {e}")
                return {
                    "data": {},
                    "raw_output": "",
                    "tool_calls": [],
                    "error": str(e),
                }

        return coder_node
