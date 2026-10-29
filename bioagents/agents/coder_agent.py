"""Coder Agent for generating and executing code via Jupyter notebooks."""

import logging
import time
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
    SUBAGENT_STEPS_KEY,
    append_code_agent_status_footer,
    append_completion_marker,
    build_partial_result,
    build_task_with_output_dir,
    collect_code_agent_run_telemetry,
    extract_available_data,
    extract_current_instruction,
    extract_original_query,
    format_coder_result,
    infer_output_dir,
    serialize_code_agent_steps,
    summarize_code_agent_steps,
)
from bioagents.sandbox.time_budget import set_recovery_roots
from bioagents.tools.smol_tool_wrappers import ToolUniverseExecuteTool, ToolUniverseSearchTool

logger = logging.getLogger("BioAgents")

CODER_AGENT_PROMPT = load_prompt("coder")


def _sandbox_fatal_error(agent: CodeAgent) -> str | None:
    executor = getattr(agent, "python_executor", None)
    error = getattr(executor, "fatal_error", None)
    return error if isinstance(error, str) and error else None


def _delivered_final_answer(agent: CodeAgent) -> bool:
    steps = getattr(getattr(agent, "memory", None), "steps", None) or []
    return any(getattr(step, "is_final_answer", False) for step in steps)


def _coder_message(content: str, internal_steps: list[dict[str, Any]]) -> AIMessage:
    """The coder's result for the supervisor, carrying its internal steps for run logs."""
    return AIMessage(
        content=content,
        additional_kwargs={
            SUBAGENT_STEPS_KEY: internal_steps,
            "subagent_summary": summarize_code_agent_steps(internal_steps),
        },
    )


def _is_timeout_error(e: BaseException) -> bool:
    if isinstance(e, TimeoutError) or "TimeoutError" in type(e).__name__:
        return True
    cause = getattr(e, "__cause__", None) or getattr(e, "__context__", None)
    while cause:
        if isinstance(cause, TimeoutError) or "Timeout" in type(cause).__name__:
            return True
        cause = getattr(cause, "__cause__", None) or getattr(cause, "__context__", None)
    return False


def _content_for_exception(
    e: Exception, agent: CodeAgent, artifact_dir: str | None, started_at: float
) -> str:
    """Build the supervisor-facing message for a coder run that raised."""
    if isinstance(e, RecursionError):
        logger.warning("Coder agent RecursionError (kernel state reuse): %s", e)
        return (
            "The coding agent encountered an internal recursion error caused by "
            "reusing a kernel that already completed a previous run.\n\n"
            "[CODER_STATUS: max_steps_reached] The coding agent ended in a "
            "degraded state. The supervisor must choose FINISH or report, not "
            "delegate the same execution task to a code agent again."
        )
    if _is_timeout_error(e):
        partial = build_partial_result(agent, "the LLM call timed out", artifact_dir, started_at)
        logger.warning("Coder agent timed out (partial output: %s)", bool(partial))
        if partial:
            return (
                f"{partial}\n\n[CODER_STATUS: timeout] The coding agent's LLM call timed "
                "out before completing all steps. The partial results above are the best "
                "output available. The supervisor must choose FINISH or report, not "
                "delegate the same execution task to a code agent again."
            )
        return (
            "The coding agent's LLM call timed out before producing any output.\n\n"
            "[CODER_STATUS: max_steps_reached] The supervisor must choose FINISH "
            "or report, not delegate the same execution task to a code agent again."
        )
    tb = traceback.format_exc()
    content = f"Error executing code: {e}\n\nTraceback:\n{tb}"
    logger.error("Coder agent error: %s", content)
    partial = build_partial_result(
        agent, f"the coding agent crashed ({e})", artifact_dir, started_at
    )
    return f"{partial}\n\n{content}" if partial else content


def create_coder_agent(
    tools: list[Tool] | None = None,
    additional_imports: list[str] | None = None,
    max_steps: int = 50,
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
        artifact_dir = infer_output_dir(output_dir, original_query)
        # A kernel restart lists files written here so the agent can resume from them.
        set_recovery_roots([artifact_dir, output_dir])

        task = build_task_with_output_dir(
            original_query,
            available_data,
            output_dir,
            current_instruction=extract_current_instruction(messages),
        )
        started_at = time.time()

        try:
            logger.info("Starting coder agent execution")
            result = agent.run(task)
            sandbox_error = _sandbox_fatal_error(agent)
            if sandbox_error:
                partial = build_partial_result(
                    agent, "the code sandbox died", artifact_dir, started_at
                )
                content = f"{partial}\n\n{sandbox_error}" if partial else sandbox_error
                content += (
                    " The supervisor must not delegate the same execution task to a code "
                    "agent again; report the partial results above or FINISH."
                )
            else:
                content = format_coder_result(result)
                content = append_code_agent_status_footer(
                    content, collect_code_agent_run_telemetry(agent)
                )
                if _delivered_final_answer(agent):
                    content = append_completion_marker(content)
                else:
                    partial = build_partial_result(
                        agent,
                        "the coding agent stopped without calling final_answer",
                        artifact_dir,
                        started_at,
                    )
                    if partial:
                        content = f"{content}\n\n{partial}"

            # The adapter's empty-response fallback carries no [CODER_STATUS:] marker,
            # so without this the supervisor re-delegates the same task.
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

            internal_steps = serialize_code_agent_steps(agent)
            structured_data = {
                "code": content,
                "language": "python",
                "description": f"Code execution result for: {original_query}",
                "execution_steps": internal_steps,
                "execution_ready": True,
            }

            # Must travel as a message: state has no data/raw_output channels, so
            # LangGraph drops an update without "messages".
            return {
                "messages": [_coder_message(content, internal_steps)],
                "data": structured_data,
                "raw_output": content,
                "tool_calls": internal_steps,
                "error": None,
            }

        except Exception as e:
            content = _content_for_exception(e, agent, artifact_dir, started_at)
            return {
                "messages": [_coder_message(content, serialize_code_agent_steps(agent))],
                "next": "supervisor",
            }

    return coder_node
