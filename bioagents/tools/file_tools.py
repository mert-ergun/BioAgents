"""File tools for reading, writing, and listing files in the sandbox workspace."""

import json
from pathlib import Path

from langchain_core.tools import tool

from bioagents.sandbox.sandbox_manager import get_sandbox


@tool
def read_local_file(file_path: str) -> str:
    """Read the contents of a file in the sandbox workspace.

    Args:
        file_path: Path to the file, relative to the sandbox root or absolute.

    Returns:
        The file contents as a string, or an error message.
    """
    try:
        sandbox = get_sandbox()
        content = sandbox.read_file(file_path)
        if len(content) > 100000:
            return content[:100000] + "\n\n... [truncated — file exceeds 100KB] ..."
        return content
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found."
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def write_local_file(file_path: str, content: str) -> str:
    """Write content to a file in the sandbox workspace.

    Creates parent directories if they don't exist.

    Args:
        file_path: Path for the file, relative to sandbox root or absolute.
        content: The text content to write.

    Returns:
        Confirmation with the full path where the file was written.
    """
    try:
        sandbox = get_sandbox()
        full_path = sandbox.write_file(file_path, content)
        return f"Successfully wrote {len(content)} characters to: {full_path}"
    except Exception as e:
        return f"Error writing file: {e}"


@tool
def list_local_directory(path: str = ".") -> str:
    """List the contents of a directory in the sandbox workspace.

    Args:
        path: Directory path relative to the sandbox root (default '.').

    Returns:
        JSON list of directory entries with name, type, and size.
    """
    try:
        sandbox = get_sandbox()
        entries = sandbox.list_directory(path)
        if not entries:
            return f"Directory '{path}' is empty or does not exist."
        return json.dumps(entries, indent=2)
    except Exception as e:
        return f"Error listing directory: {e}"


@tool
def get_file_info(file_path: str) -> str:
    """Get detailed information about a file in the sandbox.

    Args:
        file_path: Path to the file, relative to sandbox root or absolute.

    Returns:
        JSON string with file size, type, modification time, and permissions.
    """
    try:
        sandbox = get_sandbox()
        fp = Path(file_path)
        if not fp.is_absolute():
            fp = sandbox.workdir / fp

        if not fp.exists():
            return f"Error: '{file_path}' does not exist."

        stat = fp.stat()
        import datetime

        info = {
            "path": str(fp),
            "name": fp.name,
            "type": "directory" if fp.is_dir() else "file",
            "size_bytes": stat.st_size,
            "size_human": _human_size(stat.st_size),
            "modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "is_symlink": fp.is_symlink(),
            "suffix": fp.suffix,
        }

        if fp.is_file() and fp.suffix in {".csv", ".tsv", ".txt", ".fasta", ".fa", ".fastq", ".fq"}:
            try:
                line_count = sum(1 for _ in fp.open())
                info["line_count"] = line_count
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
    return [read_local_file, write_local_file, list_local_directory, get_file_info]
