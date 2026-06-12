"""Coder Agent for generating and executing code via Jupyter notebooks."""

import logging
import traceback
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
    append_code_agent_status_footer,
    build_task_with_output_dir,
    collect_code_agent_run_telemetry,
    extract_available_data,
    extract_original_query,
    format_coder_result,
)
from bioagents.tools.smol_tool_wrappers import ToolUniverseExecuteTool, ToolUniverseSearchTool

logger = logging.getLogger("BioAgents")

CODER_AGENT_PROMPT = load_prompt("coder")


def _extract_partial_coder_output(agent: CodeAgent) -> str:
    """Extract the best partial output from a coder agent that timed out mid-run."""
    memory = getattr(agent, "memory", None)
    if not memory or not getattr(memory, "steps", None):
        return ""
    parts: list[str] = []
    for step in memory.steps:
        if hasattr(step, "task"):
            continue
        obs = getattr(step, "observations", None)
        if obs and str(obs).strip():
            parts.append(str(obs).strip())
    if not parts:
        return ""
    combined = "\n\n---\n\n".join(parts[-3:])
    if len(combined) > 4000:
        combined = combined[:4000] + "\n... [truncated]"
    return combined


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
        code_block_tags="markdown",
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
            content = append_code_agent_status_footer(
                content, collect_code_agent_run_telemetry(agent)
            )

            # The LangChainModelAdapter injects a final_answer fallback when the LLM
            # returns empty responses repeatedly. That message has no [CODER_STATUS:]
            # marker, so the supervisor won't know to stop and will re-delegate the
            # same task, causing a kernel-state RecursionError on the next run.
            if (
                "returned empty responses repeatedly" in str(result)
                and "[CODER_STATUS:" not in content
            ):
                content += (
                    "\n\n[CODER_STATUS: max_steps_reached] The coding agent ended in a "
                    "degraded state (LLM returned empty responses). The supervisor must "
                    "choose FINISH or report, not delegate the same execution task to a "
                    "code agent again."
                )

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
            tb = traceback.format_exc()

            is_recursion = isinstance(e, RecursionError)
            is_timeout = isinstance(e, TimeoutError) or "TimeoutError" in str(type(e).__name__)
            if not is_timeout and not is_recursion:
                cause = getattr(e, "__cause__", None) or getattr(e, "__context__", None)
                while cause and not is_timeout:
                    is_timeout = (
                        isinstance(cause, TimeoutError) or "Timeout" in type(cause).__name__
                    )
                    cause = getattr(cause, "__cause__", None) or getattr(cause, "__context__", None)

            if is_recursion:
                content = (
                    "The coding agent encountered an internal recursion error caused by "
                    "reusing a kernel that already completed a previous run.\n\n"
                    "[CODER_STATUS: max_steps_reached] The coding agent ended in a "
                    "degraded state. The supervisor must choose FINISH or report, not "
                    "delegate the same execution task to a code agent again."
                )
                logger.warning("Coder agent RecursionError (kernel state reuse): %s", e)
            elif is_timeout:
                partial_output = _extract_partial_coder_output(agent)
                if partial_output:
                    content = (
                        f"{partial_output}\n\n"
                        f"[CODER_STATUS: timeout] The coding agent's LLM call timed out "
                        f"before completing all steps. The partial results above are the "
                        f"best output available. The supervisor must choose FINISH or report, "
                        f"not delegate the same execution task to a code agent again."
                    )
                else:
                    content = (
                        "The coding agent's LLM call timed out before producing any output.\n\n"
                        "[CODER_STATUS: max_steps_reached] The supervisor must choose FINISH "
                        "or report, not delegate the same execution task to a code agent again."
                    )
                logger.warning("Coder agent timed out (partial output: %s)", bool(partial_output))
            else:
                content = f"Error executing code: {e}\n\nTraceback:\n{tb}"
                logger.error(f"Coder agent error: {content}")

            return {"messages": [AIMessage(content=content)], "next": "supervisor"}

    return coder_node
