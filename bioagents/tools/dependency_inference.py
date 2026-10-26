"""Infer executable dependencies from actual Python calls, never error prose."""

import ast
import re
import shlex


def extract_system_dependencies_from_code(code: str) -> list[str]:
    """Find literal executables used by shutil.which, subprocess or os.system.

    Dynamic expressions are deliberately left to explicit dependency metadata.
    Comments, docstrings and exception messages are not executable evidence.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    dependencies = set()
    process_calls = {
        f"subprocess.{name}" for name in ("run", "call", "check_call", "check_output", "Popen")
    }
    builtins = {
        "sh",
        "bash",
        "echo",
        "cat",
        "grep",
        "ls",
        "cd",
        "mkdir",
        "rm",
        "cp",
        "mv",
        "tar",
        "zip",
        "unzip",
        "uv",
        "conda",
        "mamba",
        "env",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        prefix, separator, suffix = name.partition(".")
        name = aliases.get(prefix, prefix) + (separator + suffix if separator else "")
        if name not in process_calls | {"shutil.which", "os.system", "os.popen"}:
            continue
        argument = (
            node.args[0]
            if node.args
            else next(
                (kw.value for kw in node.keywords if kw.arg in {"args", "cmd", "command"}), None
            )
        )
        if isinstance(argument, (ast.List, ast.Tuple)):
            argument = argument.elts[0] if argument.elts else None
        try:
            executable = ast.literal_eval(argument) if argument is not None else None
        except (ValueError, TypeError):
            continue
        if not isinstance(executable, str):
            continue
        if name != "shutil.which":
            try:
                parts = shlex.split(executable)
            except ValueError:
                continue
            executable = parts[0] if parts else ""
        if (
            re.fullmatch(r"[a-zA-Z0-9_][a-zA-Z0-9_.+-]*", executable)
            and executable not in builtins
            and not re.fullmatch(r"(?:python|pip)\d*(?:\.\d+)*", executable)
        ):
            dependencies.add(executable)
    return sorted(dependencies)
