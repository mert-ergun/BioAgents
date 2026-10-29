"""Tool outcomes must not turn non-execution or load failures into success."""

import ast
import json
from pathlib import Path

import pytest

from bioagents.tools.result_status import tool_result_status
from bioagents.tools.tool_builder_wrappers import get_all_custom_tools_for_coder
from bioagents.tools.tool_registry import ToolDefinition, ToolParameter, ToolRegistry


@pytest.mark.parametrize(
    "result",
    [
        None,
        False,
        0,
        3.2,
        "plain result",
        '"plain result"',
        [],
        ["error"],
        [{"status": "inactive"}],
        {"value": 42},
        {"status": "success"},
        {"status": " OK "},
        {"status": "completed"},
    ],
)
def test_legacy_data_and_success_results_remain_valid(result):
    assert tool_result_status(result) == "success"


@pytest.mark.parametrize(
    "status",
    [
        "error",
        "partial",
        "capability_unavailable",
        "timeout",
        "failed",
        "pending",
        "skipped",
        "dry_run",
        "check_only",
        "manual_execution_required",
        "use_google_colab",
        "unexpected_status",
    ],
)
@pytest.mark.parametrize("encode", [False, True])
def test_non_success_status_is_preserved_in_dictionaries_and_json_strings(status, encode):
    result = {"status": status}
    assert tool_result_status(json.dumps(result) if encode else result) == status


@pytest.mark.parametrize("status", [None, False, 0, {}, [], "", "   "])
def test_invalid_explicit_status_is_an_error(status):
    assert tool_result_status({"status": status}) == "error"


def test_cyclic_or_excessively_nested_envelopes_fail_closed(monkeypatch):
    result = {"status": "success"}
    result["result"] = result
    assert tool_result_status(result) == "error"

    def recursion_failure(_):
        raise RecursionError("JSON nesting limit exceeded")

    monkeypatch.setattr(json, "loads", recursion_failure)
    assert tool_result_status("deeply nested JSON") == "error"


@pytest.mark.parametrize(
    "code",
    [
        "from pathlib import Path\n",
        "from pathlib import Path\nbroken = 42\n",
        "def another_function():\n    return 42\n",
    ],
)
def test_registry_never_loads_an_unrelated_callable_as_the_tool(tmp_path, code):
    registry = ToolRegistry(tools_dir=tmp_path)
    definition = ToolDefinition(
        name="broken",
        description="Broken entry",
        category="testing",
        parameters=[],
        return_type="object",
        return_description="Result",
        code=code,
    )
    assert registry.register_tool(definition)
    assert registry.load_tool_function("broken") is None
    assert registry.load_tool_function("broken") is None  # Cached module follows the same rule.
    assert not registry.validate_tool("broken")
    assert not registry.get_tool("broken").validated
    assert registry.get_tool("broken").validation_error


def test_published_registry_embedded_code_matches_all_tool_sources():
    tools_dir = Path(__file__).resolve().parents[1] / "bioagents/tools/custom_tools"
    registry = json.loads((tools_dir / "registry.json").read_text())["tools"]
    for name, entry in registry.items():
        assert entry["code"] == (tools_dir / f"{name}.py").read_text(), name


@pytest.mark.parametrize("name", ["analyze_pdb_chain", "gwas_get_associations_by_disease"])
def test_legacy_tool_optional_parameters_match_the_published_function_signature(name):
    tools_dir = Path(__file__).resolve().parents[1] / "bioagents/tools/custom_tools"
    entry = json.loads((tools_dir / "registry.json").read_text())["tools"][name]
    function = next(
        node
        for node in ast.parse(entry["code"]).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    defaults = {
        arg.arg: ast.literal_eval(default)
        for arg, default in zip(
            function.args.args[-len(function.args.defaults) :], function.args.defaults
        )
    }
    for parameter in entry["parameters"]:
        assert parameter["required"] is (parameter["name"] not in defaults)
        if parameter["name"] in defaults:
            assert parameter["default"] == defaults[parameter["name"]]


@pytest.mark.parametrize(
    ("python_type", "schema_type"),
    [
        ("str", "string"),
        ("int", "integer"),
        ("float", "number"),
        ("bool", "boolean"),
        ("dict", "object"),
        ("list", "array"),
        ("string", "string"),
        ("Any", "any"),
    ],
)
def test_python_type_aliases_are_valid_in_tool_schemas(python_type, schema_type):
    assert ToolParameter("value", python_type, "Input").to_schema()["type"] == schema_type


def test_all_published_tools_can_be_exposed_to_coder_without_running_them(monkeypatch):
    from bioagents.tools import tool_registry

    tools_dir = Path(__file__).resolve().parents[1] / "bioagents/tools/custom_tools"
    registry = ToolRegistry(tools_dir=tools_dir)
    monkeypatch.setattr(tool_registry, "get_registry", lambda: registry)
    tools = get_all_custom_tools_for_coder(validated_only=False)
    assert {tool.name for tool in tools} == {entry.name for entry in registry.list_tools()}
    # Constructing wrappers must not import/execute the underlying scientific tools.
    assert not registry._loaded_modules
