"""Static checks that generated tool code matches the installed environment.

LLM-written wrappers often import packages that are not installed or call
third-party APIs with keyword arguments that the installed version does not
accept (e.g. RDKit ``BRICSBuild(minFragmentSize=...)``). Both are cheap to detect
before the first execution: imports via ``importlib.util.find_spec`` and keyword
arguments via ``inspect.signature`` on the imported symbols.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

_IMPORT_ERRORS = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}
_UNRESOLVED = object()


def _guarded_by_import_error(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """True when an import sits in a ``try`` whose handlers catch import failures."""
    child, parent = node, parents.get(node)
    while parent is not None:
        if isinstance(parent, ast.Try) and child in parent.body:
            for handler in parent.handlers:
                if handler.type is None:
                    return True
                names = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
                if any(ast.unparse(n).split(".")[-1] in _IMPORT_ERRORS for n in names):
                    return True
        child, parent = parent, parents.get(parent)
    return False


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name.split(".")[0]) is not None
    except (ImportError, ValueError):
        return False


def _is_third_party(module: str) -> bool:
    top = module.split(".")[0]
    return top not in sys.stdlib_module_names and top not in {"bioagents", "__future__"}


def _resolve(dotted: str) -> Any:
    """Import the longest importable prefix of ``dotted`` and getattr the rest."""
    parts = dotted.split(".")
    for split in range(len(parts), 0, -1):
        try:
            obj: Any = importlib.import_module(".".join(parts[:split]))
        except Exception:  # nosec B112 - any import failure means unresolvable
            continue
        try:
            for attribute in parts[split:]:
                obj = getattr(obj, attribute)
        except AttributeError:
            return _UNRESOLVED
        return obj
    return _UNRESOLVED


def _dotted_call_target(func: ast.expr) -> list[str] | None:
    parts: list[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if not isinstance(func, ast.Name):
        return None
    parts.append(func.id)
    return parts[::-1]


def _keyword_problems(call: ast.Call, target: str, obj: Any) -> list[str]:
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return []
    parameters = signature.parameters.values()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
        return []
    accepted = {
        p.name
        for p in parameters
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return [
        f"{target}() got an unexpected keyword argument '{kw.arg}' "
        f"(installed signature: {target}{signature})"
        for kw in call.keywords
        if kw.arg is not None and kw.arg not in accepted
    ]


def check_generated_code(code: str) -> list[str]:
    """Return human-readable problems; an empty list means no drift was detected."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"SyntaxError: {exc}"]
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    problems: list[str] = []
    aliases: dict[str, str] = {}
    missing_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                aliases[bound] = alias.name if alias.asname else bound
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            modules = [node.module]
            for alias in node.names:
                if alias.name != "*":
                    aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        else:
            continue
        if _guarded_by_import_error(node, parents):
            continue
        for module in modules:
            if module not in missing_modules and not _module_available(module):
                missing_modules.add(module)
                problems.append(f"Import '{module}' does not resolve in the current environment")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.keywords:
            continue
        parts = _dotted_call_target(node.func)
        if not parts or parts[0] not in aliases:
            continue
        dotted = ".".join([aliases[parts[0]], *parts[1:]])
        if not _is_third_party(dotted) or dotted.split(".")[0] in {
            m.split(".")[0] for m in missing_modules
        }:
            continue
        obj = _resolve(dotted)
        if obj is _UNRESOLVED or not callable(obj):
            continue
        problems.extend(_keyword_problems(node, ".".join(parts), obj))
    return problems
