"""Helper functions for building tasks for the coder agent."""

import importlib.util
import os
import re
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

_tooluniverse_installed = importlib.util.find_spec("tooluniverse") is not None
if _tooluniverse_installed:
    DEFAULT_CODER_IMPORTS.append("tooluniverse")


DEFAULT_ML_IMPORTS = [
    *DEFAULT_CODER_IMPORTS,
    "sklearn.*",
    "joblib",
]

if os.getenv("BIOAGENTS_ENABLE_OPTIONAL_ML_IMPORTS", "").lower() in ("1", "true", "yes"):
    DEFAULT_ML_IMPORTS.extend(
        [
            "xgboost",
            "lightgbm",
            "catboost",
            "seaborn",
            "statsmodels.*",
        ]
    )

DEFAULT_DL_IMPORTS = [
    *DEFAULT_ML_IMPORTS,
    "transformers",
    "tqdm",
]

if os.getenv("BIOAGENTS_ENABLE_OPTIONAL_DL_IMPORTS", "").lower() in ("1", "true", "yes"):
    DEFAULT_DL_IMPORTS.extend(
        [
            "torchvision.*",
            "torchaudio.*",
            "tensorflow.*",
            "keras.*",
            "tensorboard",
            "datasets",
        ]
    )


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


SUPERVISOR_TASK_PREFIX = "[SUPERVISOR TASK]"


def extract_current_instruction(messages: list) -> str | None:
    """The supervisor's most recent instruction, if it has given one.

    Without this a code agent always re-read the first HumanMessage, so every
    re-delegation restarted the whole pipeline from the original request instead of
    doing the step the supervisor actually asked for.
    """
    for msg in reversed(messages):
        if not isinstance(msg, HumanMessage):
            continue
        content = str(msg.content or "")
        if SUPERVISOR_TASK_PREFIX in content:
            return content.split(SUPERVISOR_TASK_PREFIX, 1)[1].strip() or None
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
    current_instruction: str | None = None,
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

    if current_instruction and original_query and current_instruction != original_query:
        task_parts.append(f"TASK: {current_instruction}\n")
        task_parts.append(f"\nORIGINAL REQUEST (context only): {original_query}\n")
    elif current_instruction:
        task_parts.append(f"TASK: {current_instruction}\n")
    elif original_query:
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


COMPLETION_MARKER = "[CODER_STATUS: completed]"


def append_completion_marker(content: str) -> str:
    """Mark a delivered final_answer so the supervisor need not infer completion.

    Completion used to be guessed from the prose, with patterns borrowed from model
    training ("Accuracy:", "Epoch 1/10"). A bioinformatics run that ended with
    "TASK COMPLETE" and a file list matched none of them, so finished work was
    re-delegated until the workflow timed out.
    """
    if "[CODER_STATUS:" in content:
        return content
    return (
        f"{content}\n\n{COMPLETION_MARKER} The coding agent delivered its final answer. "
        "The task is done: use this result, do not delegate it again."
    )


#: Key under which a code agent's internal smolagents steps travel on its AIMessage
#: (``additional_kwargs``) so the experiment runner can persist them in run.json.
SUBAGENT_STEPS_KEY = "subagent_steps"

_STEP_FIELD_LIMIT = 3000


def _clip(value: Any, limit: int = _STEP_FIELD_LIMIT) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def serialize_code_agent_steps(agent: Any, limit: int = _STEP_FIELD_LIMIT) -> list[dict[str, Any]]:
    """Return a JSON-safe record of every internal step a smolagents agent took.

    Captures, per action step: the code executed, the observations, the error (if
    any), timing and whether it delivered the final answer. Planning steps record the
    plan. The task step (the prompt) is skipped.
    """
    memory = getattr(agent, "memory", None)
    steps = getattr(memory, "steps", None) or []
    records: list[dict[str, Any]] = []
    for step in steps:
        if hasattr(step, "task") and not hasattr(step, "step_number"):
            continue  # TaskStep: just the prompt
        if hasattr(step, "plan") and not hasattr(step, "step_number"):
            records.append({"type": "planning", "plan": _clip(step.plan, limit)})
            continue
        record: dict[str, Any] = {"type": "action", "step": getattr(step, "step_number", None)}
        code = getattr(step, "code_action", None)
        if code:
            record["code"] = _clip(code, limit)
        observations = getattr(step, "observations", None)
        if observations:
            record["observations"] = _clip(observations, limit)
        error = getattr(step, "error", None)
        if error is not None:
            record["error"] = _clip(getattr(error, "message", None) or error, limit)
        timing = getattr(step, "timing", None)
        duration = getattr(timing, "duration", None) if timing is not None else None
        if duration is not None:
            record["duration_s"] = round(float(duration), 2)
        if getattr(step, "is_final_answer", False):
            record["is_final_answer"] = True
            record["final_answer"] = _clip(getattr(step, "action_output", ""), limit)
        records.append(record)
    return records


def summarize_code_agent_steps(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts for a run record: steps taken, how many errored, whether it finished."""
    actions = [s for s in steps if s.get("type") == "action"]
    return {
        "step_count": len(actions),
        "error_count": sum(1 for s in actions if s.get("error")),
        "delivered_final_answer": any(s.get("is_final_answer") for s in actions),
    }


_OUTPUT_DIR_PATTERN = re.compile(
    r"Save all generated files for this task under\s+(\S+?)\s+relative to the shared project root"
)


def infer_output_dir(output_dir: str | None, query: str | None) -> str | None:
    """Prefer the state's output_dir; otherwise read the task directory from the prompt."""
    if output_dir:
        return output_dir
    if not query:
        return None
    match = _OUTPUT_DIR_PATTERN.search(query)
    return match.group(1) if match else None


def list_recent_artifacts(output_dir: str | None, since: float, limit: int = 40) -> list[str]:
    """Files under ``output_dir`` written at or after ``since`` (epoch seconds)."""
    if not output_dir:
        return []
    root = Path(output_dir)
    if not root.is_dir():
        return []
    recent: list[tuple[float, str]] = []
    try:
        for path in root.rglob("*"):
            try:
                if path.is_file() and path.stat().st_mtime >= since:
                    recent.append((path.stat().st_mtime, str(path)))
            except OSError:
                continue
    except OSError:
        return []
    recent.sort(reverse=True)
    return [p for _, p in recent[:limit]]


def build_partial_result(
    agent: Any,
    reason: str,
    output_dir: str | None = None,
    since: float | None = None,
    max_chars: int = 4000,
) -> str:
    """Salvage what a code agent computed when it did not deliver a final answer.

    Combines the last successful observations with the files it wrote, so a result
    that was computed but never passed to ``final_answer`` still reaches the supervisor.
    """
    memory = getattr(agent, "memory", None)
    observations: list[str] = []
    for step in getattr(memory, "steps", None) or []:
        if hasattr(step, "task") and not hasattr(step, "step_number"):
            continue
        obs = getattr(step, "observations", None)
        if obs and str(obs).strip() and getattr(step, "error", None) is None:
            observations.append(str(obs).strip())
    parts: list[str] = [f"PARTIAL RESULT — {reason}"]
    if observations:
        combined = "\n\n---\n\n".join(observations[-3:])
        parts.append("Last successful execution output:\n" + _clip(combined, max_chars))
    artifacts = list_recent_artifacts(output_dir, since) if since is not None else []
    if artifacts:
        parts.append("Files written during this run:\n" + "\n".join(f"- {a}" for a in artifacts))
    if len(parts) == 1:
        return ""
    parts.append(
        "These results were computed by the code agent but it did not deliver a final "
        "answer; verify them from the files above rather than re-running the whole task."
    )
    return "\n\n".join(parts)
