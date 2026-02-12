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

            execution_steps: list[dict[str, Any]] = []
            for step in agent.memory.steps:
                if hasattr(step, "task"):
                    continue

                step_data: dict[str, Any] = {
                    "step": len(execution_steps) + 1,
                }

                if hasattr(step, "thought") and step.thought:
                    step_data["thought"] = step.thought
                elif hasattr(step, "model_output") and step.model_output:
                    thought = step.model_output

                    thought = re.sub(r"```python\n[\s\S]*?```", "", thought).strip()
                    if thought:
                        step_data["thought"] = thought

                if hasattr(step, "code") and step.code:
                    step_data["code"] = step.code
                elif hasattr(step, "model_output") and step.model_output:
                    code_match = re.search(r"```python\n([\s\S]*?)```", step.model_output)
                    if code_match:
                        step_data["code"] = code_match.group(1)

                if hasattr(step, "observations") and step.observations:
                    step_data["output"] = str(step.observations)

                if hasattr(step, "logs") and step.logs:
                    step_data["logs"] = step.logs

                if any(k in step_data for k in ["thought", "code", "output", "logs"]):
                    execution_steps.append(step_data)

            logger.info(f"Extracted {len(execution_steps)} execution steps")

            return {
                "messages": [AIMessage(content=content)],
                "next": "supervisor",
                "code_steps": execution_steps,
            }
        except Exception as e:
            import traceback

            error_msg = f"Error executing code: {e}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Coder agent error: {error_msg}")
            return {"messages": [AIMessage(content=error_msg)], "next": "supervisor"}

    return coder_node
