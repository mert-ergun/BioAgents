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
    "Bio",
    "Bio.*",
    # PyTorch: bare `torch` is required for `import torch` in smolagents; `torch.*` covers submodules.
    "torch",
    "torch.*",
    "typing",
    "json",
    "os",
    "os.path",
    "posixpath",
    "ntpath",
    "sys",
    "pathlib",
    "io",
    "base64",
    "binascii",
    "subprocess",
    "importlib",
    "pkg_resources",
]

DEFAULT_ML_IMPORTS = [
    *DEFAULT_CODER_IMPORTS,
    "sklearn.*",
    "xgboost",
    "lightgbm",
    "catboost",
    "seaborn",
    "joblib",
    "statsmodels.*",
]

DEFAULT_DL_IMPORTS = [
    *DEFAULT_ML_IMPORTS,
    "torchvision.*",
    "torchaudio.*",
    "tensorflow.*",
    "keras.*",
    "tensorboard",
    "transformers",
    "datasets",
    "tqdm",
]


class PermissiveList(list):
    """A list that contains everything, used to allow all imports in sandboxed environments."""

    def __contains__(self, item: Any) -> bool:
        return True


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
) -> str:
    """
    Build the task string for the coder agent with output directory instructions.

    Args:
        original_query: The original user query
        available_data: List of available data strings from previous messages
        output_dir: Output directory path for saving files

    Returns:
        Formatted task string for the coder agent
    """
    task_parts = []

    if original_query:
        task_parts.append(f"TASK: {original_query}\n")

    if available_data:
        task_parts.append("\nAVAILABLE DATA/CONTEXT:\n" + "\n".join(available_data[-3:]))

    # System prompt is already passed to CodeAgent via instructions parameter,
    # so we don't need to include it in the task description to avoid cluttering logs

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
    final_answer = getattr(result, "final_answer", None)
    if not final_answer:
        output = getattr(result, "output", None)
        return str(output) if output else str(result)

    if isinstance(final_answer, dict):
        lines = ["Task completed successfully.\n"]
        for key, value in final_answer.items():
            formatted_key = key.replace("_", " ").title()
            if isinstance(value, str) and (
                value.endswith(".png") or value.endswith(".jpg") or value.endswith(".pdf")
            ):
                lines.append(f"- **{formatted_key}**: `{value}`")
            else:
                lines.append(f"- **{formatted_key}**:\n{value}")
        return "\n".join(lines)

    return f"Task completed successfully.\n\nFinal answer: {final_answer}"


def collect_code_agent_run_telemetry(agent: Any) -> dict[str, bool]:
    """
    Scan smolagents CodeAgent step observations for failure modes that should not
    trigger another identical delegation (max steps, parse loops, sandbox import denials).
    """
    memory = getattr(agent, "memory", None)
    if not memory or not getattr(memory, "steps", None):
        return {
            "max_steps_reached": False,
            "repeated_parse_errors": False,
            "import_denied": False,
        }
    combined = ""
    for step in memory.steps:
        if hasattr(step, "task"):
            continue
        obs = getattr(step, "observations", None)
        if obs is not None:
            combined += str(obs)
    import_failed = (
        "ImportError:" in combined
        and ("cannot open shared object" in combined or "No module named" in combined)
        and combined.count("ImportError:") >= 2
    )
    return {
        "max_steps_reached": "Reached max steps" in combined,
        "repeated_parse_errors": combined.count("Error in code parsing") >= 3,
        "import_denied": "is not allowed" in combined and "Import" in combined,
        "import_failed": import_failed,
    }


def append_code_agent_status_footer(content: str, telemetry: dict[str, bool]) -> str:
    """Append machine-readable markers for the supervisor to stop re-delegation loops."""
    markers: list[str] = []
    if telemetry.get("max_steps_reached"):
        markers.append("[CODER_STATUS: max_steps_reached]")
    if telemetry.get("repeated_parse_errors"):
        markers.append("[CODER_STATUS: repeated_parse_errors]")
    if telemetry.get("import_denied"):
        markers.append("[CODER_STATUS: import_denied]")
    if telemetry.get("import_failed"):
        markers.append("[CODER_STATUS: import_failed]")
    if not markers:
        return content
    footer = (
        "\n\n"
        + " ".join(markers)
        + " The coding agent ended in a degraded state. The supervisor must choose FINISH "
        "or report, not delegate the same execution task to a code agent again."
    )
    return content + footer
