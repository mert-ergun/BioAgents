"""Coder Agent for generating and executing code via Jupyter notebooks."""

import logging
from collections.abc import Callable
from typing import Any

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

    Wraps the smolagents CodeAgent so it:
    - Extracts task context from graph state
    - Executes code in the Jupyter sandbox
    - Returns a shared-memory-compatible dict
      {"data": {...}, "raw_output": str, "tool_calls": [...], "error": str|None}

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

        task = build_task_with_output_dir(original_query, available_data, output_dir)

        try:
            logger.info("Starting coder agent execution")
            result = agent.run(task)
            content = format_coder_result(result)

            # Collect execution steps for the audit trail
            execution_steps: list[dict[str, Any]] = []
            for step in agent.memory.steps:
                if hasattr(step, "task"):
                    # TaskStep — skip, it's just the prompt
                    continue
                step_info: dict[str, Any] = {}
                if hasattr(step, "tool_calls") and step.tool_calls:
                    step_info["tool_calls"] = [
                        {
                            "tool": tc.name,
                            "args": tc.arguments,
                            "result": getattr(tc, "result", None),
                        }
                        for tc in step.tool_calls
                    ]
                if hasattr(step, "observations") and step.observations:
                    step_info["observations"] = str(step.observations)
                if step_info:
                    execution_steps.append(step_info)

            structured_data = {
                "code": content,
                "language": "python",
                "description": f"Code execution result for: {original_query}",
                "execution_steps": execution_steps,
                "execution_ready": True,
            }

            return {
                "data": structured_data,
                "raw_output": content,
                "tool_calls": execution_steps,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Coder agent error: {e}", exc_info=True)
            return {
                "data": {
                    "code": "",
                    "language": "python",
                    "description": "Execution failed",
                    "execution_steps": [],
                    "execution_ready": False,
                },
                "raw_output": str(e),
                "tool_calls": [],
                "error": str(e),
            }

    return coder_node
