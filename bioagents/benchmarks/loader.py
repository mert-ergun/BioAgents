"""Loader for STELLA benchmark tasks."""

import logging
import re
from collections.abc import Iterator
from pathlib import Path

import yaml

from bioagents.benchmarks.models import STELLATask

logger = logging.getLogger(__name__)


def _fix_yaml_indentation(content: str) -> str:
    """
    Fix common YAML indentation issues that cause parsing errors.

    Uses a context-aware approach to fix indentation line by line.
    """
    lines = content.split("\n")
    fixed_lines = []
    indent_stack = [0]  # Track indentation levels

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip empty lines and comments
        if not stripped or (stripped.startswith("#") and ":" not in stripped):
            fixed_lines.append(line)
            i += 1
            continue

        current_indent = len(line) - len(line.lstrip())

        # Check if this is a key-value pair
        if ":" in stripped and not stripped.startswith("-"):
            # This is a key
            # Find the correct parent indent from stack
            # Pop stack until we find a parent with less indent
            while indent_stack and indent_stack[-1] >= current_indent:
                indent_stack.pop()

            # If current indent is wrong (should be 2 more than parent)
            if indent_stack:
                expected_indent = indent_stack[-1] + 2
                if current_indent != expected_indent and current_indent < expected_indent:
                    # Fix the indent
                    line = " " * expected_indent + stripped
                    current_indent = expected_indent

            # Push this key's indent to stack
            indent_stack.append(current_indent)
            fixed_lines.append(line)
            i += 1
            continue

        # Check if this is a list item
        if stripped.startswith("- "):
            # Find parent indent
            while indent_stack and indent_stack[-1] >= current_indent:
                indent_stack.pop()

            # List items should be at parent + 2
            if indent_stack:
                expected_indent = indent_stack[-1] + 2
                if current_indent != expected_indent and current_indent < expected_indent:
                    line = " " * expected_indent + stripped
                    current_indent = expected_indent

            # Push list item indent (for nested fields)
            indent_stack.append(current_indent)
            fixed_lines.append(line)
            i += 1
            continue

        # For other lines (values, etc.), try to maintain context
        # If it looks like it should be indented more, fix it
        if i > 0:
            prev_line = fixed_lines[-1] if fixed_lines else ""
            prev_stripped = prev_line.strip()

            if prev_stripped.endswith(":") and not prev_stripped.startswith("#"):
                prev_indent = len(prev_line) - len(prev_line.lstrip())
                if current_indent <= prev_indent and prev_indent > 0:
                    # This should be a child
                    line = " " * (prev_indent + 2) + stripped
                    current_indent = prev_indent + 2

        fixed_lines.append(line)
        i += 1

    return "\n".join(fixed_lines)


def load_stella_tasks(tasks_file: str | Path) -> list[STELLATask]:
    """
    Load STELLA benchmark tasks from YAML file.

    The YAML file contains multiple task sections separated by comment lines.
    Each section starts with "# Task N:" followed by YAML content.

    Args:
        tasks_file: Path to the tasks YAML file

    Returns:
        List of STELLATask objects
    """
    tasks_file = Path(tasks_file)
    if not tasks_file.exists():
        raise FileNotFoundError(f"Tasks file not found: {tasks_file}")

    with tasks_file.open("r", encoding="utf-8") as f:
        content = f.read()

    # Find all task separators
    task_markers = list(re.finditer(r"^# Task \d+:", content, flags=re.MULTILINE))

    if not task_markers:
        logger.warning("No task markers found in YAML file")
        return []

    tasks = []

    # Extract each task section
    for i, marker in enumerate(task_markers):
        # Get the start position (after the marker)
        start_pos = marker.end()

        # Get the end position (start of next marker, or end of file)
        end_pos = task_markers[i + 1].start() if i + 1 < len(task_markers) else len(content)

        # Extract the section
        section = content[start_pos:end_pos]

        # Remove separator lines (# =============================================================================)
        section = re.sub(r"^# =+$", "", section, flags=re.MULTILINE)
        section = section.strip()
        if not section:
            continue

        # Find the start of YAML content (first field like "task_name:")
        lines = section.split("\n")
        yaml_start = -1
        for j, line in enumerate(lines):
            stripped = line.strip()
            # Look for YAML key (task_name:)
            if stripped.startswith("task_name:"):
                yaml_start = j
                break

        if yaml_start < 0:
            # No task_name found in this section
            logger.debug(f"No task_name found in section {i + 1}")
            continue

        # Extract YAML content from task_name to end of section
        yaml_content = "\n".join(lines[yaml_start:])

        # Clean up: remove comment-only lines but keep inline comments and valid YAML
        yaml_lines = yaml_content.split("\n")
        cleaned_lines = []
        for line in yaml_lines:
            stripped = line.strip()
            # Skip comment-only lines (that start with # and don't have : before the #)
            if stripped.startswith("#") and ":" not in stripped:
                continue
            cleaned_lines.append(line)

        yaml_content = "\n".join(cleaned_lines)

        # Fix common YAML indentation issues before parsing
        yaml_content = _fix_yaml_indentation(yaml_content)

        # Try to parse as YAML
        try:
            task_data = yaml.safe_load(yaml_content)
            if task_data and isinstance(task_data, dict) and "task_name" in task_data:
                task = STELLATask.from_dict(task_data)
                tasks.append(task)
            else:
                logger.debug(f"Section {i + 1} did not parse to dict with task_name")
        except yaml.YAMLError as e:
            # Log YAML parsing errors only in debug mode to reduce noise
            logger.debug(f"Skipping section {i + 1} due to YAML error: {e}")
            continue
        except (TypeError, AttributeError) as e:
            logger.warning(f"Skipping section {i + 1} due to error: {e}")
            continue
        except Exception as e:
            logger.warning(f"Skipping section {i + 1} due to unexpected error: {e}")
            continue

    logger.info(f"Loaded {len(tasks)} tasks from {tasks_file}")
    return tasks


def get_tasks_by_type(tasks: list[STELLATask], task_type: str) -> list[STELLATask]:
    """Filter tasks by type."""
    return [task for task in tasks if task.task_type == task_type]


def iter_tasks(tasks: list[STELLATask]) -> Iterator[tuple[int, STELLATask]]:
    """Iterator over tasks with indices."""
    yield from enumerate(tasks, start=1)
