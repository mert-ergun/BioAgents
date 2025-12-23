"""Helper functions for building tasks for the coder agent."""

from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

DEFAULT_CODER_IMPORTS = [
    "pandas",
    "numpy.*",
    "matplotlib.*",
    "scipy.*",
    "bioagents.*",
    "typing",
    "json",
    "os",
    "os.path",
    "sys",
    "pathlib",
]


def extract_original_query(messages: list) -> str | None:
    """
    Extract the original user query from messages.

    Args:
        messages: List of messages in the conversation

    Returns:
        The original user query string, or None if not found
    """
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content
            return str(content) if content is not None else None
    return None


def extract_available_data(messages: list) -> list[str]:
    """
    Extract available data from recent messages.

    Args:
        messages: List of messages in the conversation

    Returns:
        List of available data strings from recent tool outputs and agent messages
    """
    available_data = []

    for msg in messages[-6:]:
        if isinstance(msg, ToolMessage):
            content = str(msg.content)
            tool_name = getattr(msg, "name", "tool")
            if len(content) < 1000:
                available_data.append(f"Data from {tool_name}: {content}")
        elif isinstance(msg, AIMessage):
            content = str(msg.content)
            if len(content) < 500 and any(
                keyword in content.lower()
                for keyword in ["retrieved", "fetched", "sequence", "data", "found"]
            ):
                available_data.append(f"Context: {content}")

    return available_data


def build_task_with_output_dir(
    original_query: str | None,
    available_data: list[str],
    output_dir: str | None,
    system_prompt: str | None = None,
) -> str:
    """
    Build the task string for the coder agent with output directory instructions.

    Args:
        original_query: The original user query
        available_data: List of available data strings from previous messages
        output_dir: Output directory path for saving files
        system_prompt: System prompt to include in task description (for visibility in logs)

    Returns:
        Formatted task string for the coder agent
    """
    task_parts = []

    if original_query:
        task_parts.append(f"TASK: {original_query}\n")

    if available_data:
        task_parts.append("\nAVAILABLE DATA/CONTEXT:\n" + "\n".join(available_data[-3:]))

    # Add system prompt to task description so it's visible in "New run" output
    # (smolagents only shows task in logs, not system prompt)
    if system_prompt:
        task_parts.append(f"\n\nSYSTEM PROMPT / INSTRUCTIONS:\n{system_prompt}")

    task = "\n".join(task_parts) if task_parts else "Please complete the requested task."

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            import os

            project_root = Path.cwd()
            if output_path.is_absolute():
                try:
                    rel_path = os.path.relpath(str(output_path), project_root)
                    output_path_str = str(output_path) if ".." in rel_path else rel_path
                except ValueError:
                    output_path_str = str(output_path)
            else:
                output_path_str = str(output_path)
        except Exception:
            output_path_str = str(output_path)

        task += f"\n\nIMPORTANT: Save all output files (images, data files, CSV, JSON, etc.) to this directory: {output_path_str}"

    return task


def format_coder_result(result: Any) -> str:
    """
    Format the result from the coder agent into a string for the message.

    Args:
        result: The result returned by CodeAgent.run()

    Returns:
        A string representation of the result
    """
    if hasattr(result, "final_answer") and result.final_answer:
        return f"Task completed successfully.\n\nFinal answer: {result.final_answer}"
    elif hasattr(result, "output") and result.output:
        return str(result.output)
    else:
        return str(result)
