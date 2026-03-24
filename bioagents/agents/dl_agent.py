"""DL Agent for designing and executing deep learning systems."""

import logging
import re
from collections.abc import Callable
from typing import Any

from smolagents import CodeAgent

from bioagents.llms.adapters import LangChainModelAdapter
from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt
from bioagents.sandbox.coder_executor import create_executor
from bioagents.sandbox.coder_helpers import (
    DEFAULT_DL_IMPORTS,
    build_task_with_output_dir,
    extract_available_data,
    extract_original_query,
    format_coder_result,
)
from bioagents.tools.smol_tool_wrappers import ToolUniverseExecuteTool, ToolUniverseSearchTool

logger = logging.getLogger("BioAgents")

DL_AGENT_PROMPT = load_prompt("dl_agent")


def create_dl_agent(
    tools: list | None = None,
    additional_imports: list[str] | None = None,
    max_steps: int = 20,
) -> CodeAgent:
    """
    Create the DL Agent instance.

    Args:
        tools: List of tools available to the agent
        additional_imports: Additional Python packages to allow in the sandbox
        max_steps: Maximum number of execution steps

    Returns:
        A CodeAgent instance that can generate and execute Python code via Jupyter notebooks
    """
    if tools is None:
        tools = [ToolUniverseSearchTool(), ToolUniverseExecuteTool()]

    if additional_imports is None:
        additional_imports = DEFAULT_DL_IMPORTS

    lc_model = get_llm(prompt_name="dl_agent")
    model = LangChainModelAdapter(lc_model)
    executor = create_executor("dl", additional_imports)

    # Escape Jinja2 template syntax in instructions to avoid conflicts
    escaped_instructions = DL_AGENT_PROMPT.replace("{", "{{").replace("}", "}}")

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


def create_dl_node(agent: CodeAgent) -> Callable:
    """
    Create the DL Agent node function.

    Args:
        agent: The CodeAgent instance to wrap

    Returns:
        A function that can be used as a LangGraph node
    """

    def dl_node(state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("messages", [])

        original_query = extract_original_query(messages)
        available_data = extract_available_data(messages)
        output_dir = state.get("output_dir")

        task = build_task_with_output_dir(original_query, available_data, output_dir)

        try:
            logger.info("Starting DL agent execution")
            result = agent.run(task)
            content = format_coder_result(result)

            execution_steps: list[dict[str, Any]] = []

            # Guard against union type: agent.memory may not have .steps
            agent_memory = getattr(agent, "memory", None)
            raw_steps = getattr(agent_memory, "steps", []) if agent_memory is not None else []

            for step in raw_steps:
                if hasattr(step, "task"):
                    continue

                step_data: dict[str, Any] = {"step": len(execution_steps) + 1}

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

            logger.info("Extracted %d execution steps", len(execution_steps))

            structured = {"model_result": {"summary": content, "code_steps": execution_steps}}

            return {"data": structured, "raw_output": content, "tool_calls": [], "error": None}

        except Exception as e:
            import traceback

            error_msg = f"Error executing DL code: {e}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error("DL agent error: %s", error_msg)
            return {"data": {}, "raw_output": "", "tool_calls": [], "error": error_msg}

    return dl_node
