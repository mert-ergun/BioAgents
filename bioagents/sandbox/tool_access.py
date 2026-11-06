"""Check generated cells before they can import registered service implementations.

This is an execution-interface guard, not an OS security boundary for hostile
Python. Ordinary installed Python libraries and file operations remain available.
"""

from __future__ import annotations

import ast

from bioagents.tools.tool_selection import get_selection


def check_generated_tool_access(code: str, selection=None) -> None:
    if (selection or get_selection()).mode == "all":
        return
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return  # Let the Python executor report the original syntax error.
    for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
            if node.module == "bioagents" and any(alias.name == "tools" for alias in node.names):
                modules.append("bioagents.tools")
        elif isinstance(node, ast.Call) and node.args:
            function = node.func
            dynamic = (isinstance(function, ast.Name) and function.id == "__import__") or (
                isinstance(function, ast.Attribute) and function.attr == "import_module"
            )
            if (
                dynamic
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                modules = [node.args[0].value]
        if any(
            name == "tooluniverse"
            or name.startswith("tooluniverse.")
            or name == "bioagents.tools"
            or name.startswith("bioagents.tools.")
            for name in modules
        ):
            raise ValueError(
                "Direct registered-tool imports are unavailable in selected mode. Check the current tool introduction list and call an exact bound tool name."
            )
