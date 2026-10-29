"""Implausible system dependencies must never trigger repeated install attempts (#62)."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

from bioagents.tools import tool_builder_tools as tools
from bioagents.tools.dependency_inference import extract_system_dependencies_from_code
from bioagents.tools.tool_registry import ToolDefinition, ToolRegistry


def _definition(**overrides):
    values = {
        "name": "fixture_tool",
        "description": "Fixture",
        "category": "test",
        "parameters": [],
        "return_type": "dict",
        "return_description": "Result",
        "code": 'def fixture_tool():\n    return {"status": "success"}\n',
    }
    return ToolDefinition(**{**values, **overrides})


def _register_args(**overrides):
    return {
        "name": "fixture_tool",
        "description": "Fixture",
        "category": "test",
        "code": 'def fixture_tool():\n    return {"status": "success"}\n',
        "parameters_json": "[]",
        "return_type": "dict",
        "return_description": "Result",
        **overrides,
    }


def test_directory_not_found_message_yields_no_dependencies():
    code = 'def f(path):\n    raise FileNotFoundError(f"Directory not found: {path}")\n'
    assert extract_system_dependencies_from_code(code) == []


def test_real_shutil_which_call_is_still_detected():
    code = 'import shutil\n\ndef f():\n    return shutil.which("samtools")\n'
    assert extract_system_dependencies_from_code(code) == ["samtools"]


def test_screening_rejects_prose_but_keeps_installable_and_resolvable_names(monkeypatch):
    monkeypatch.setattr(tools.shutil, "which", lambda n: "/bin/Rscript" if n == "Rscript" else None)
    kept, status = tools._screen_system_dependencies(
        ["Directory", "samtools", "Rscript", "two words", "", 3, "samtools"]
    )
    assert kept == ["samtools", "Rscript"]
    assert status["samtools"] == "missing"
    assert status["Rscript"] == "available"
    assert status["Directory"].startswith("rejected")
    assert status["two words"].startswith("rejected")


def test_explicit_prose_dependency_is_rejected_and_recorded_at_registration(tmp_path, monkeypatch):
    registry = ToolRegistry(tools_dir=tmp_path)
    monkeypatch.setattr(tools, "get_registry", lambda: registry)
    monkeypatch.setattr(tools.shutil, "which", lambda _: None)
    response = json.loads(
        tools.register_custom_tool.invoke(
            _register_args(system_dependencies_json='["Directory", "samtools"]')
        )
    )
    assert response["status"] == "success"
    reloaded = ToolRegistry(tools_dir=tmp_path).get_tool("fixture_tool")
    assert reloaded.system_dependencies == ["samtools"]
    assert reloaded.system_dependency_status["samtools"] == "missing"
    assert reloaded.system_dependency_status["Directory"].startswith("rejected")


def test_persisted_bad_entry_is_sanitized_at_execution_without_installs(tmp_path, monkeypatch):
    registry = ToolRegistry(tools_dir=tmp_path)
    registry.register_tool(_definition(system_dependencies=["Directory"]))
    monkeypatch.setattr(tools, "get_registry", lambda: registry)
    monkeypatch.setattr(
        tools.shutil,
        "which",
        lambda n: "/fake/" + n if n in {"mamba", "conda", "apt-get"} else None,
    )
    install = Mock()
    monkeypatch.setattr(tools, "run_install_command", install)
    result = json.loads(tools.execute_custom_tool.invoke({"tool_name": "fixture_tool"}))
    assert result["status"] == "success"
    install.assert_not_called()
    reloaded = ToolRegistry(tools_dir=tmp_path).get_tool("fixture_tool")
    assert reloaded.system_dependencies == []
    assert reloaded.system_dependency_status["Directory"].startswith("rejected")


def test_failed_system_install_is_persisted_and_not_retried(tmp_path, monkeypatch):
    registry = ToolRegistry(tools_dir=tmp_path)
    registry.register_tool(_definition(system_dependencies=["samtools"]))
    monkeypatch.setattr(tools, "get_registry", lambda: registry)
    monkeypatch.setattr(
        tools.shutil, "which", lambda n: "/fake/" + n if n in {"mamba", "conda"} else None
    )
    install = Mock(return_value=SimpleNamespace(returncode=1, stdout="", stderr="no package"))
    monkeypatch.setattr(tools, "run_install_command", install)

    tools.execute_custom_tool.invoke({"tool_name": "fixture_tool"})
    assert install.call_count == 2  # mamba, then conda
    reloaded = ToolRegistry(tools_dir=tmp_path).get_tool("fixture_tool")
    assert reloaded.system_dependency_status["samtools"] == "install_failed"

    tools.execute_custom_tool.invoke({"tool_name": "fixture_tool"})
    assert install.call_count == 2
