"""Coder Agent for generating and executing code via Jupyter notebooks."""

import logging
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
    executor = create_executor(additional_imports)

    # Escape Jinja2 template syntax in instructions to avoid conflicts
    escaped_instructions = CODER_AGENT_PROMPT.replace("{", "{{").replace("}", "}}")

    agent = CodeAgent(
        tools=tools,
        model=model,
        executor=executor,
        additional_authorized_imports=additional_imports,
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

    def coder_node(state: dict[str, Any]) -> dict[str, Any]:
        messages = state["messages"]

        original_query = extract_original_query(messages)
        available_data = extract_available_data(messages)
        output_dir = state.get("output_dir")

        task = build_task_with_output_dir(
            original_query, available_data, output_dir, system_prompt=CODER_AGENT_PROMPT
        )

        try:
            logger.info("Starting coder agent execution")
            result = agent.run(task)
            content = format_coder_result(result)
            return {"messages": [AIMessage(content=content)], "next": "supervisor"}
        except Exception as e:
            import traceback

            error_msg = f"Error executing code: {e}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Coder agent error: {error_msg}")
            return {"messages": [AIMessage(content=error_msg)], "next": "supervisor"}

    return coder_node
