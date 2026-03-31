"""Shell execution tools for running commands in a sandboxed environment."""

import json

from langchain_core.tools import tool

from bioagents.sandbox.sandbox_manager import get_sandbox


@tool
def run_shell_command(command: str, timeout: int = 120) -> str:
    """Execute a shell command in the sandboxed environment.

    Args:
        command: The shell command to execute.
        timeout: Maximum seconds to wait for the command to complete (default 120).

    Returns:
        JSON string with stdout, stderr, return code, and success status.
    """
    try:
        sandbox = get_sandbox()
        result = sandbox.run_command(command, timeout=timeout)
        return json.dumps(
            {
                "success": result["success"],
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "returncode": result["returncode"],
            }
        )
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@tool
def install_python_package(package_name: str) -> str:
    """Install a Python package using pip in the sandbox.

    Args:
        package_name: Name of the package to install (e.g. 'numpy', 'biopython>=1.80').

    Returns:
        Installation result with success status and output.
    """
    try:
        sandbox = get_sandbox()
        result = sandbox.install_package(package_name)
        if result["success"]:
            return f"Successfully installed '{package_name}'.\n{result['stdout']}"
        return f"Failed to install '{package_name}': {result['stderr']}"
    except Exception as e:
        return f"Error installing package: {e}"


@tool
def check_installed_packages() -> str:
    """List all Python packages installed in the sandbox environment.

    Returns:
        A list of installed packages and their versions.
    """
    try:
        sandbox = get_sandbox()
        result = sandbox.run_command("pip list --format=columns", timeout=30)
        if result["success"]:
            return result["stdout"]
        return f"Failed to list packages: {result['stderr']}"
    except Exception as e:
        return f"Error checking packages: {e}"


def get_shell_tools() -> list:
    """Return all shell tools."""
    return [run_shell_command, install_python_package, check_installed_packages]
