"""Discover registered metadata without importing unselected implementations."""

from __future__ import annotations

import ast
import importlib
import importlib.metadata
import json
from functools import lru_cache
from pathlib import Path

TOOLS_DIR = Path(__file__).parent
# Ownership follows tool families. New tools in a family need no agent edits.
FAMILY_ROLES = {
    "docking_tools": ("docking",),
    "genomics_tools": ("genomics", "phylogenetics"),
    "proteomics_tools": ("research", "data_acquisition"),
    "literature_tools": ("literature", "research"),
    "visualization_tools": ("visualization",),
    "structural_tools": ("structural_biology", "research"),
    "cofold_tools": ("structural_biology",),
    "esm_tools": ("analysis", "structural_biology"),
    "analysis_tools": ("analysis",),
    "rdkit_tools": ("analysis", "docking"),
    "web_tools": ("web_browser", "research", "data_acquisition", "paper_replication"),
    "git_tools": ("git", "paper_replication"),
    "environment_tools": ("environment",),
    "shell_tools": ("shell",),
    "tool_builder_tools": ("tool_builder", "tool_discovery"),
    "transcriptomics_tools": ("transcriptomics",),
}


@lru_cache(maxsize=4)
def local_index(directory: Path | None = None) -> dict[str, tuple[str, str]]:
    """Index top-level @tool definitions by AST; do not import every tool family."""
    directory = directory or TOOLS_DIR
    result = {}
    for path in sorted(directory.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                called = decorator.func if isinstance(decorator, ast.Call) else decorator
                if not (isinstance(called, ast.Name) and called.id == "tool"):
                    continue
                name = node.name
                if (
                    isinstance(decorator, ast.Call)
                    and decorator.args
                    and isinstance(decorator.args[0], ast.Constant)
                ):
                    value = decorator.args[0].value
                    if not isinstance(value, str):
                        raise ValueError(f"Local tool name must be a string: {path.name}")
                    name = value
                if name in result:
                    raise ValueError(f"Duplicate registered local tool: {name}")
                result[name] = (f"bioagents.tools.{path.stem}", node.name)
    return result


def custom_entries() -> dict[str, dict]:
    path = TOOLS_DIR / "custom_tools/registry.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("tools", data) if isinstance(data, dict) else data
    if isinstance(entries, dict):
        return {name: entry for name, entry in entries.items() if isinstance(entry, dict)}
    return {
        entry["name"]: entry for entry in entries if isinstance(entry, dict) and "name" in entry
    }


@lru_cache(maxsize=1)
def sdk_entries() -> dict[str, dict]:
    """Read installed ToolUniverse JSON metadata without constructing an SDK client."""
    try:
        root = Path(
            str(importlib.metadata.distribution("tooluniverse").locate_file("tooluniverse/data"))
        )
    except importlib.metadata.PackageNotFoundError:
        return {}
    result: dict[str, dict] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, list):
            for entry in data:
                if (
                    isinstance(entry, dict)
                    and isinstance(entry.get("name"), str)
                    and isinstance(entry.get("parameter"), dict)
                ):
                    result.setdefault(entry["name"], entry)
    return result


def custom_schema(entry: dict) -> dict:
    properties, required = {}, []
    for parameter in entry.get("parameters", []):
        name = parameter["name"]
        kind = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "dict": "object",
            "list": "array",
        }.get(parameter["type"], parameter["type"])
        item = {"type": kind, "description": parameter.get("description", "")}
        if kind == "array":
            item["items"] = {}
        if parameter.get("enum"):
            item["enum"] = parameter["enum"]
        if parameter.get("default") is not None:
            item["default"] = parameter["default"]
        properties[name] = item
        if parameter.get("required", True):
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


def describe_tool(name: str, selection) -> dict:
    """Resolve the original description and parameter schema for one selected name."""
    from bioagents.tools.agent_manifest import UNAVAILABLE_STUB_TOOLS

    if name in UNAVAILABLE_STUB_TOOLS:
        raise ValueError(f"Selected tool has no implementation: {name}")
    backend = dict(selection.aliases).get(name)
    if name in selection.tool_universe_tools or backend:
        entry = sdk_entries().get(backend or name)
        if entry is None:
            raise ValueError(f"ToolUniverse metadata is unavailable for selected tool {name}")
        return {
            "name": name,
            "description": entry.get("description", name),
            "schema": entry["parameter"],
            "source": "sdk",
            "backend": backend or name,
            "roles": ("research", "literature", "genomics", "transcriptomics"),
        }
    index = local_index()
    if name in index:
        module, attribute = index[name]
        obj = getattr(importlib.import_module(module), attribute)
        return {
            "name": name,
            "description": obj.description,
            "schema": obj.args_schema
            if isinstance(obj.args_schema, dict)
            else (
                obj.args_schema.model_json_schema()
                if obj.args_schema
                else {"type": "object", "properties": {}}
            ),
            "source": "local",
            "module": module,
            "attribute": attribute,
            "roles": FAMILY_ROLES.get(module.rsplit(".", 1)[-1], ("research",)),
        }
    entry = custom_entries().get(name)
    if entry is not None:
        return {
            "name": name,
            "description": entry.get("description", name),
            "schema": custom_schema(entry),
            "source": "custom",
            "roles": ("research",),
        }
    raise ValueError(f"Unknown selected local tool: {name}. Check its registered name.")
