"""File tools for reading, writing, and listing files in the sandbox workspace.

Agents only redo work they cannot tell is already done. Two things make that failure
mode possible, and both are handled here:

- *path resolution*: every tool below resolves a path through
  :mod:`bioagents.tools.workspace_paths`, so a container ``/app/...`` path, a
  project-root-relative path and a workspace-relative path all find the same file.
  A listing tool that quietly answers "empty" to a path it simply could not resolve
  tells the agent the work is missing, and the agent repeats it.
- *write feedback*: ``write_local_file`` reports whether it created, reused or
  replaced a file, and by default writes a new distinctly-named version rather than
  overwriting. Replacing an existing artifact is an explicit decision, never an
  accident.
"""

import datetime
import json
from pathlib import Path

from langchain_core.tools import tool

from bioagents.sandbox.sandbox_manager import get_sandbox
from bioagents.sandbox.workspace import to_portable_path
from bioagents.tools.workspace_paths import prepare_output_file, resolve_existing_path

# Reading a file this far past the context budget helps nobody; truncate with a notice.
MAX_READ_CHARS = 100_000
_TEXT_SUFFIXES = {".csv", ".tsv", ".txt", ".fasta", ".fa", ".fastq", ".fq"}


def _existing(file_path: str) -> Path | None:
    """Resolve ``file_path`` to an existing path, trying every agent path form."""
    try:
        return resolve_existing_path(file_path)
    except ValueError:  # traversal or outside the workspace — treat as not found
        return None


@tool
def read_local_file(file_path: str) -> str:
    """Read the contents of a file in the sandbox workspace.

    Args:
        file_path: Path to the file. Workspace-relative, project-relative, absolute
            or a container '/app/...' path all work.

    Returns:
        The file contents as a string, or an error message.
    """
    try:
        try:
            content = get_sandbox().read_file(file_path)
        except FileNotFoundError:
            # The agent may have reported the path from a different filesystem view
            # (container /app path, project-relative path, or a different case).
            resolved = _existing(file_path)
            if resolved is None or not resolved.is_file():
                return f"Error: File '{file_path}' not found."
            content = resolved.read_text(errors="replace")
        if len(content) > MAX_READ_CHARS:
            return content[:MAX_READ_CHARS] + "\n\n... [truncated — file exceeds 100KB] ..."
        return content
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found."
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def write_local_file(file_path: str, content: str, overwrite: bool = False) -> str:
    """Write content to a file in the sandbox workspace.

    Existing files are never clobbered by accident: if the path is already taken the
    content goes to a new, distinctly-named file (``results.csv`` -> ``results_v2.csv``)
    and the reply says so. Pass overwrite=True only when you deliberately want the
    previous version gone.

    Args:
        file_path: Path for the file, relative to the shared workspace or absolute.
        content: The text content to write.
        overwrite: Replace an existing file instead of writing a new version.

    Returns:
        Confirmation naming the path written and whether it was created, reused,
        replaced, or written as a new version alongside the existing file.
    """
    sandbox = get_sandbox()
    if not isinstance(getattr(sandbox, "workdir", None), Path):
        # A sandbox that is not backed by a real filesystem (a test double, or a
        # remote executor): delegate the write and report it plainly.
        try:
            return (
                f"Successfully wrote {len(content)} characters to: "
                f"{to_portable_path(sandbox.write_file(file_path, content))}"
            )
        except Exception as e:
            return f"Error writing file: {e}"

    try:
        target, action = prepare_output_file(file_path, overwrite=overwrite)
    except Exception as e:
        return f"Error writing file: {e}"

    try:
        if action != "create":
            original = target if action == "overwrite" else target.with_name(_origin_name(target))
            verdict = _guard_existing(original, content)
            if verdict is not None:
                return verdict

        target.write_text(content)
        portable = to_portable_path(target)
        return (
            f"Successfully wrote {len(content)} characters to: {portable}\n{_note(action, target)}"
        )
    except Exception as e:
        return f"Error writing file: {e}"


def _origin_name(versioned: Path) -> str:
    """Recover the original filename that ``unique_sibling`` versioned away from."""
    stem = versioned.stem.rsplit("_v", 1)[0]
    return f"{stem}{versioned.suffix}"


def _guard_existing(original: Path, content: str) -> str | None:
    """Refuse writes that would destroy or needlessly duplicate a good artifact."""
    if not original.is_file():
        return None
    try:
        previous = original.read_text(errors="replace")
    except Exception:  # unreadable (e.g. written by the container as another user)
        return None

    portable = to_portable_path(original)
    if previous == content:
        return (
            f"Already up to date: {portable} ({len(previous)} characters).\n"
            f"Content is identical, so nothing was rewritten. Reuse this file."
        )
    if not content.strip() and previous.strip():
        return (
            f"Refused: {portable} already holds {len(previous)} characters of content and "
            f"the new content is empty. A successful result must not be replaced by an "
            f"empty one. If the previous result is genuinely wrong, write the correction "
            f"to a different filename."
        )
    return None


def _note(action: str, target: Path) -> str:
    """Explain what happened to anything that was already at the path."""
    portable = to_portable_path(target)
    shared = (
        "Other agents (including the Coder agent, which runs in a container) can "
        "open this file at exactly this path."
    )
    if action == "overwrite":
        return f"Replaced the previous contents of {portable} (overwrite=True). {shared}"
    if action == "new_version":
        original = to_portable_path(target.with_name(_origin_name(target)))
        return (
            f"NOTE: {original} already existed, so this was saved alongside it as a new "
            f"version instead of replacing it. Check whether the existing file already "
            f"answers your task before producing another version; pass overwrite=True "
            f"only if you intend to discard the earlier result. {shared}"
        )
    return f"Created a new file. {shared}"


@tool
def check_file_exists(path: str) -> str:
    """Check whether a file or directory already exists in the shared workspace.

    Use this before re-downloading, re-computing or re-writing anything: another agent
    may have produced it already. The answer is a plain yes/no, not an error message.

    Args:
        path: Path to check. Workspace-relative, project-relative, absolute or a
            container '/app/...' path all work.

    Returns:
        JSON with 'exists' plus, when it does exist, the portable path, type and size.
    """
    try:
        resolved = _existing(path)
        if resolved is None:
            return json.dumps(
                {
                    "query": path,
                    "exists": False,
                    "note": "Nothing at this path. Safe to create it.",
                },
                indent=2,
            )
        info: dict = {
            "query": path,
            "exists": True,
            "path": to_portable_path(resolved),
            "type": "directory" if resolved.is_dir() else "file",
        }
        if resolved.is_dir():
            info["n_entries"] = sum(1 for _ in resolved.iterdir())
        else:
            stat = resolved.stat()
            info["size_bytes"] = stat.st_size
            info["modified"] = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
        info["note"] = (
            "This already exists. Reuse it rather than regenerating it, unless you have "
            "a concrete reason to believe it is wrong."
        )
        return json.dumps(info, indent=2)
    except Exception as e:
        return f"Error checking path: {e}"


@tool
def list_local_directory(path: str = ".") -> str:
    """List the contents of a directory in the sandbox workspace.

    Args:
        path: Directory path relative to the shared workspace (default '.').

    Returns:
        JSON list of directory entries with name, type, and size. A directory that
        does not exist is reported as such — distinctly from one that is empty.
    """
    try:
        resolved = _existing(path)
        if resolved is None:
            return json.dumps(
                {
                    "directory": path,
                    "exists": False,
                    "entries": [],
                    "note": (
                        "This directory does not exist (it is NOT an empty directory). "
                        "Check the path before concluding that earlier work is missing."
                    ),
                },
                indent=2,
            )
        if not resolved.is_dir():
            return json.dumps(
                {
                    "directory": to_portable_path(resolved),
                    "exists": True,
                    "entries": [],
                    "note": "This path is a file, not a directory.",
                },
                indent=2,
            )

        entries = [
            {
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else 0,
            }
            for item in sorted(resolved.iterdir())
        ]
        payload = {
            "directory": to_portable_path(resolved),
            "exists": True,
            "entries": entries,
        }
        if not entries:
            payload["note"] = "The directory exists but is empty."
        return json.dumps(payload, indent=2)
    except Exception as e:
        return f"Error listing directory: {e}"


@tool
def get_file_info(file_path: str) -> str:
    """Get detailed information about a file in the sandbox.

    Args:
        file_path: Path to the file, relative to the shared workspace or absolute.

    Returns:
        JSON string with file size, type, modification time, and permissions.
    """
    try:
        fp = _existing(file_path)
        if fp is None:
            return f"Error: '{file_path}' does not exist."

        stat = fp.stat()
        info = {
            "path": to_portable_path(fp),
            "name": fp.name,
            "type": "directory" if fp.is_dir() else "file",
            "size_bytes": stat.st_size,
            "size_human": _human_size(stat.st_size),
            "modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "is_symlink": fp.is_symlink(),
            "suffix": fp.suffix,
        }

        if fp.is_file() and fp.suffix in _TEXT_SUFFIXES:
            try:
                with fp.open(errors="replace") as handle:
                    info["line_count"] = sum(1 for _ in handle)
            except Exception:  # nosec B110
                pass

        return json.dumps(info, indent=2)
    except Exception as e:
        return f"Error getting file info: {e}"


def _human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable size string."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def get_file_tools() -> list:
    """Return all file tools."""
    return [
        read_local_file,
        write_local_file,
        check_file_exists,
        list_local_directory,
        get_file_info,
    ]


__all__ = [
    "check_file_exists",
    "get_file_info",
    "get_file_tools",
    "get_sandbox",
    "list_local_directory",
    "read_local_file",
    "write_local_file",
]
