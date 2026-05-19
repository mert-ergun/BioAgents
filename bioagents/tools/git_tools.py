"""Git tools for repository operations in the sandbox environment."""

import json

from langchain_core.tools import tool

from bioagents.sandbox.sandbox_manager import get_sandbox


@tool
def git_clone_repo(repo_url: str, target_dir: str = "") -> str:
    """Clone a git repository into the sandbox workspace.

    Args:
        repo_url: URL of the git repository to clone.
        target_dir: Optional target directory name within the sandbox. If empty,
                    uses the default name from the repo URL.

    Returns:
        Result message indicating success or failure.
    """
    try:
        sandbox = get_sandbox()
        result = sandbox.git_clone(repo_url, target_dir or None)
        if result["success"]:
            dirname = target_dir or repo_url.rstrip("/").split("/")[-1].replace(".git", "")
            return f"Successfully cloned '{repo_url}' into '{dirname}'.\n{result['stdout']}"
        return f"Failed to clone repository: {result['stderr']}"
    except Exception as e:
        return f"Error cloning repository: {e}"


@tool
def list_repo_files(repo_path: str, pattern: str = "*") -> str:
    """List files in a cloned repository matching a glob pattern.

    Args:
        repo_path: Path to the repository directory (relative to sandbox root).
        pattern: Glob pattern for filtering files (default '*' for all).

    Returns:
        JSON list of matching file paths, or an error message.
    """
    try:
        sandbox = get_sandbox()
        full_path = sandbox.workdir / repo_path
        if not full_path.is_dir():
            return f"Error: Directory '{repo_path}' not found in sandbox."

        matches = sorted(
            str(p.relative_to(full_path)) for p in full_path.rglob(pattern) if p.is_file()
        )
        if not matches:
            return f"No files matching pattern '{pattern}' in '{repo_path}'."
        return json.dumps(matches[:500])
    except Exception as e:
        return f"Error listing files: {e}"


@tool
def read_repo_file(file_path: str) -> str:
    """Read the contents of a file from a cloned repository in the sandbox.

    Args:
        file_path: Path to the file relative to the sandbox root.

    Returns:
        The file contents as a string, or an error message.
    """
    try:
        sandbox = get_sandbox()
        content = sandbox.read_file(file_path)
        if len(content) > 50000:
            return content[:50000] + "\n\n... [truncated, file too large] ..."
        return content
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found in sandbox."
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def git_checkout_branch(repo_path: str, branch: str) -> str:
    """Checkout a specific branch in a cloned repository.

    Args:
        repo_path: Path to the repository relative to sandbox root.
        branch: Branch name to checkout.

    Returns:
        Result message indicating success or failure.
    """
    try:
        sandbox = get_sandbox()
        result = sandbox.run_command(f"git checkout {branch}", cwd=repo_path)
        if result["success"]:
            return f"Checked out branch '{branch}' in '{repo_path}'.\n{result['stdout']}"
        result = sandbox.run_command(
            f"git fetch origin {branch} && git checkout {branch}", cwd=repo_path
        )
        if result["success"]:
            return (
                f"Fetched and checked out branch '{branch}' in '{repo_path}'.\n{result['stdout']}"
            )
        return f"Failed to checkout branch '{branch}': {result['stderr']}"
    except Exception as e:
        return f"Error checking out branch: {e}"


def get_git_tools() -> list:
    """Return all git tools."""
    return [git_clone_repo, list_repo_files, read_repo_file, git_checkout_branch]
