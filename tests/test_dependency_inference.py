"""Dependency inference must use executable calls rather than missing-file prose."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from bioagents.tools import tool_builder_tools as tools
from bioagents.tools.dependency_inference import extract_system_dependencies_from_code
from bioagents.tools.tool_registry import ToolRegistry


@pytest.mark.parametrize(
    "code",
    [
        'raise FileNotFoundError(f"Directory not found: {path}")',
        'raise RuntimeError("samtools not found")',
        '# shutil.which("samtools")\nprint("Directory not found")',
        '"""subprocess.run(["samtools", "view"])"""',
        'example = "shutil.which(\\"bwa\\")"',
        'import subprocess\nsubprocess.run([sys.executable, "-m", "pip", "install", "x"])',
        "invalid python syntax [",
    ],
)
def test_error_prose_comments_and_examples_never_become_system_packages(code):
    assert extract_system_dependencies_from_code(code) == []


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ('import shutil\nshutil.which("samtools")', ["samtools"]),
        ('import shutil as s\ns.which("obabel")', ["obabel"]),
        ('from shutil import which as find\nfind("bwa")', ["bwa"]),
        ('import subprocess as sp\nsp.run(["blastp", "-version"])', ["blastp"]),
        (
            'from subprocess import check_call as run\nrun(args=["bedtools", "--help"])',
            ["bedtools"],
        ),
        ('import os\nos.system("samtools --version")', ["samtools"]),
        ('import subprocess\nsubprocess.run(["python3.12", "script.py"])', []),
        ('import subprocess\nsubprocess.run(["/opt/bin/custom", "--help"])', []),
    ],
)
def test_literal_executable_calls_and_aliases_are_recognized(code, expected):
    assert extract_system_dependencies_from_code(code) == expected


def test_directory_error_never_triggers_installer_after_registration(tmp_path, monkeypatch):
    registry = ToolRegistry(tools_dir=tmp_path)
    monkeypatch.setattr(tools, "get_registry", lambda: registry)
    response = json.loads(
        tools.register_custom_tool.invoke(
            {
                "name": "directory_listing",
                "description": "List files",
                "category": "utility",
                "code": 'def directory_listing():\n    raise FileNotFoundError("Directory not found")\n',
                "parameters_json": "[]",
                "return_type": "dict",
                "return_description": "Files",
            }
        )
    )
    assert response["status"] == "success"
    assert registry.get_tool("directory_listing").system_dependencies == []
    install = Mock()
    monkeypatch.setattr(tools, "run_install_command", install)
    result = json.loads(tools.execute_custom_tool.invoke({"tool_name": "directory_listing"}))
    assert result["status"] == "error"
    assert "Directory not found" in result["message"]
    install.assert_not_called()


def test_published_workspace_utility_never_attempts_system_installs(tmp_path, monkeypatch):
    from bioagents.tools.tool_registry import ToolDefinition

    path = Path(__file__).resolve().parents[1] / "bioagents/tools/custom_tools/registry.json"
    entry = json.loads(path.read_text())["tools"]["list_workspace_files"]
    registry = ToolRegistry(tools_dir=tmp_path / "tools")
    registry.register_tool(ToolDefinition.from_dict(entry))
    monkeypatch.setattr(tools, "get_registry", lambda: registry)
    monkeypatch.setattr(
        tools.shutil,
        "which",
        lambda name: "/fake/" + name if name in {"mamba", "conda", "apt-get"} else None,
    )
    install = Mock()
    monkeypatch.setattr(tools, "run_install_command", install)
    result = json.loads(
        tools.execute_custom_tool.invoke(
            {
                "tool_name": "list_workspace_files",
                "arguments_json": json.dumps({"path": str(tmp_path)}),
            }
        )
    )
    assert result["status"] == "success"
    install.assert_not_called()


def test_scikit_learn_distribution_uses_its_actual_import_name(monkeypatch):
    import builtins

    imports = []

    def load(name):
        imports.append(name)
        if name != "sklearn":
            raise ImportError(name)
        return Mock()

    monkeypatch.setattr(builtins, "__import__", load)
    assert tools._missing_python_dependencies(["scikit-learn>=1.0"]) == []
    assert imports == ["sklearn"]


@pytest.mark.parametrize("first", [TimeoutError("temporary timeout"), ""])
def test_wrapper_generation_retries_transient_failure_or_empty_content(monkeypatch, first):
    llm = Mock()
    llm.invoke.side_effect = [
        first if isinstance(first, Exception) else Mock(content=first),
        Mock(content='{"code":"def example(): pass"}'),
    ]
    monkeypatch.setattr(tools, "get_llm", lambda: llm)
    monkeypatch.setattr(tools.time, "sleep", lambda _: None)
    result = tools.generate_tool_wrapper.invoke(
        {
            "tool_name": "example",
            "tool_description": "Test",
            "tool_category": "test",
            "parameters_json": "[]",
            "return_type": "dict",
            "return_description": "Result",
            "implementation_notes": "Test only",
        }
    )
    assert json.loads(result)["code"] == "def example(): pass"
    assert llm.invoke.call_count == 2


def test_wrapper_retry_never_overruns_an_expired_workflow_budget(monkeypatch):
    llm = Mock()
    monkeypatch.setattr(tools, "remaining_workflow_seconds", lambda: 0)
    with pytest.raises(TimeoutError, match="deadline"):
        tools._invoke_wrapper(llm, "test")
    llm.invoke.assert_not_called()


def test_wrapper_permanent_errors_are_not_retried(monkeypatch):
    llm = Mock()
    llm.invoke.side_effect = ValueError("invalid provider configuration")
    with pytest.raises(ValueError):
        tools._invoke_wrapper(llm, "test")
    assert llm.invoke.call_count == 1


def test_load_validation_failure_keeps_the_import_reason(tmp_path):
    from bioagents.tools.tool_registry import ToolDefinition

    registry = ToolRegistry(tools_dir=tmp_path)
    registry.register_tool(
        ToolDefinition(
            name="missing_import",
            description="Test",
            category="test",
            parameters=[],
            return_type="dict",
            return_description="Result",
            code="import definitely_missing_fixture_package\ndef missing_import(): return {}\n",
        )
    )
    assert not registry.validate_tool("missing_import")
    reloaded = ToolRegistry(tools_dir=tmp_path)
    assert (
        "definitely_missing_fixture_package" in reloaded.get_tool("missing_import").validation_error
    )
