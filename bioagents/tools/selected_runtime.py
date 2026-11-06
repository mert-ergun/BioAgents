"""Build named adapters and enforce the same frozen selection at execution."""

from __future__ import annotations

import copy
import importlib
import json
import tempfile
from functools import lru_cache
from typing import TYPE_CHECKING

from langchain_core.tools import StructuredTool

from bioagents.tools.tool_catalog import describe_tool
from bioagents.tools.tool_selection import (
    ToolSelection,
    active_selection,
    get_selection,
    selection_scope,
    unavailable_tool,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_CODE_SOURCES: list[tempfile.TemporaryDirectory] = []

ROLES = (
    "research",
    "analysis",
    "tool_builder",
    "protein_design",
    "literature",
    "web_browser",
    "paper_replication",
    "data_acquisition",
    "genomics",
    "transcriptomics",
    "structural_biology",
    "phylogenetics",
    "tool_discovery",
    "shell",
    "git",
    "environment",
    "visualization",
    "docking",
)


def execute_selected(name: str, arguments: dict, selection: ToolSelection | None = None):
    """Reject unknown targets before importing implementations or creating clients."""
    selection = selection or get_selection()
    current = active_selection()
    if not selection.allows(name) or (current is not None and not current.allows(name)):
        return unavailable_tool(name)
    if selection.mode == "all":
        from bioagents.tools.tool_universe import DEFAULT_WRAPPER

        return DEFAULT_WRAPPER.execute_tool(name, arguments)
    metadata = describe_tool(name, selection)
    with selection_scope(selection):
        if metadata["source"] == "sdk":
            from bioagents.tools.tool_universe import DEFAULT_WRAPPER

            return DEFAULT_WRAPPER._execute_selected_sdk(metadata["backend"], arguments, selection)
        if metadata["source"] == "local":
            implementation = getattr(
                importlib.import_module(metadata["module"]), metadata["attribute"]
            )
            return implementation.invoke(arguments)
        from bioagents.tools.tool_registry import get_registry

        function = get_registry().load_tool_function(name)
        if function is None:
            raise RuntimeError(f"Selected custom tool {name} has no callable implementation")
        return function(**arguments)


@lru_cache(maxsize=16)
def selected_metadata(selection: ToolSelection) -> tuple[dict, ...]:
    return tuple(describe_tool(name, selection) for name in selection.names)


def named_tool(metadata: dict, selection: ToolSelection) -> StructuredTool:
    from bioagents.prompts.tool_context import redact_tool_references

    name = metadata["name"]

    def invoke(**arguments):
        result = execute_selected(name, arguments, selection)
        return result if isinstance(result, str) else json.dumps(result, default=str)

    return StructuredTool.from_function(
        func=invoke,
        name=name,
        description=redact_tool_references(metadata["description"], selection),
        args_schema=copy.deepcopy(metadata["schema"]),
        infer_schema=False,
    )


@lru_cache(maxsize=16)
def selected_tool_map(selection: ToolSelection) -> dict[str, list]:
    mapping: dict[str, list] = {role: [] for role in ROLES}
    for metadata in selected_metadata(selection):
        adapter = named_tool(metadata, selection)
        for role in metadata["roles"]:
            mapping[role].append(adapter)
    return mapping


def tools_for_agent(
    role: str, tools: list | None = None, *, extra_tools: list | None = None
) -> list:
    selection = get_selection()
    if selection.mode == "selected":
        # Resolve roles centrally even for standalone factories and caller lists.
        return list(selected_tool_map(selection)[role])
    if tools is not None:
        return list(tools)
    from bioagents.tools.agent_manifest import build_agent_tool_map, filter_bindable_tools

    return filter_bindable_tools(list(build_agent_tool_map()[role]) + list(extra_tools or []))


def bind_agent_tools(llm, tools: list):
    return llm.bind_tools(tools) if tools else llm


def _expanded_schema(value, definitions, seen=()):
    if isinstance(value, list):
        return [_expanded_schema(item, definitions, seen) for item in value]
    if not isinstance(value, dict):
        return value
    if "$ref" in value:
        reference = value["$ref"].rsplit("/", 1)[-1]
        if reference in seen or reference not in definitions:
            raise ValueError(
                "Selected code-agent schema contains an unresolved recursive reference"
            )
        return _expanded_schema(
            {**definitions[reference], **{k: v for k, v in value.items() if k != "$ref"}},
            definitions,
            (*seen, reference),
        )
    return {key: _expanded_schema(item, definitions, seen) for key, item in value.items()}


@lru_cache(maxsize=16)
def selected_code_tools(selection: ToolSelection) -> tuple:
    """Create serializable smolagents classes without adding project pip requirements."""
    import hashlib
    import sys
    from pathlib import Path

    from bioagents.prompts.tool_context import redact_tool_references

    identity = hashlib.sha256(selection.to_json().encode()).hexdigest()[:20]
    source_directory = tempfile.TemporaryDirectory(prefix="bioagents-selected-")
    _CODE_SOURCES.append(source_directory)
    path = Path(source_directory.name) / f"tools_{identity}.py"
    classes, names = ["from smolagents import Tool\n"], []
    for index, metadata in enumerate(selected_metadata(selection)):
        schema = metadata["schema"]
        inputs = _expanded_schema(
            copy.deepcopy(schema.get("properties", {})), schema.get("$defs", {})
        )
        required = set(schema.get("required", []))
        for name, parameter in inputs.items():
            parameter.setdefault("type", "any")
            parameter.setdefault("description", name)
            parameter["nullable"] = name not in required
        class_name = f"SelectedTool{index}"
        names.append(class_name)
        classes.append(
            f"class {class_name}(Tool):\n"
            f"    name = {metadata['name']!r}\n"
            f"    description = {redact_tool_references(metadata['description'], selection)!r}\n"
            "    inputs = {}\n"
            "    output_type = 'string'\n"
            "    skip_forward_signature_validation = True\n"
            "    def __init__(self):\n"
            "        import json\n"
            f"        self.inputs = json.loads({json.dumps(inputs)!r})\n"
            "        super().__init__()\n"
            "    def forward(self, **arguments):\n"
            "        import importlib\n"
            "        import json\n"
            "        runtime = importlib.import_module('bioagents.tools.selected_runtime')\n"
            "        access = importlib.import_module('bioagents.tools.tool_selection')\n"
            f"        selection = access.parse_selection(json.loads({selection.to_json()!r}))\n"
            f"        result = runtime.execute_selected({metadata['name']!r}, arguments, selection)\n"
            "        return result if isinstance(result, str) else json.dumps(result, default=str)\n"
        )
    source = "\n".join(classes)
    if not path.exists() or path.read_text() != source:
        path.write_text(source, encoding="utf-8")
    module_name = f"_bioagents_selected_{identity}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot construct selected code-agent wrappers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return tuple(getattr(module, name)() for name in names)


def code_agent_tools(tools: list | None = None) -> list:
    selection = get_selection()
    if selection.mode == "selected":
        return list(selected_code_tools(selection))
    if tools is not None:
        return list(tools)
    from bioagents.tools.smol_tool_wrappers import ToolUniverseExecuteTool, ToolUniverseSearchTool

    return [ToolUniverseSearchTool(), ToolUniverseExecuteTool()]


def bind_agent_node[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Keep an existing graph isolated from configuration changes in later graphs."""
    from functools import wraps

    selection = get_selection()

    @wraps(function)
    def scoped(*args: P.args, **kwargs: P.kwargs) -> R:
        with selection_scope(selection):
            return function(*args, **kwargs)

    return scoped


def with_new_tool_selection(function):
    """Read configuration once and scope the complete graph construction."""
    from functools import wraps

    from bioagents.tools import tool_selection

    @wraps(function)
    def construct(*args, **kwargs):
        selection = tool_selection.read_selection()
        tool_selection._PROCESS_SELECTION = selection
        with selection_scope(selection):
            return function(*args, **kwargs)

    return construct
