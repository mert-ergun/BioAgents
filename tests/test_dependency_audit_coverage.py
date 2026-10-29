"""Every install attempt, including failed and blocked ones, reaches the audit (#70)."""

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bioagents.benchmarks.use_case_models import (
    ExperimentConfig,
    ExperimentRun,
    RunResult,
    UseCase,
)
from bioagents.sandbox import dependency_events as events
from bioagents.tools import tool_builder_tools as tools
from bioagents.tools.tool_registry import ToolDefinition, ToolRegistry


@pytest.fixture
def audit_run(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOAGENTS_RUN_ID", "b" * 32)
    monkeypatch.setenv("BIOAGENTS_DEPENDENCY_EVENTS_DIR", str(tmp_path))
    monkeypatch.delenv("BIOAGENTS_USE_CASE_ID", raising=False)
    monkeypatch.delenv(events.DISABLE_INSTALLS_ENV, raising=False)
    return events.RunAudit("b" * 32, tmp_path)


def _events(tmp_path):
    return [json.loads(line) for line in (tmp_path / events.EVENTS_FILE).read_text().splitlines()]


def _tool_registry(tmp_path, **overrides):
    registry = ToolRegistry(tools_dir=tmp_path / "tools")
    values = {
        "name": "tsne_fingerprint_scatter",
        "description": "Fixture",
        "category": "test",
        "parameters": [],
        "return_type": "dict",
        "return_description": "Result",
        "code": 'def tsne_fingerprint_scatter():\n    return {"status": "success"}\n',
    }
    registry.register_tool(ToolDefinition(**{**values, **overrides}))
    return registry


def test_failed_scikit_learn_install_is_counted_with_reason(audit_run, tmp_path, monkeypatch):
    registry = _tool_registry(tmp_path, dependencies=["scikit-learn"])
    monkeypatch.setattr(tools, "get_registry", lambda: registry)
    monkeypatch.setattr(tools, "_missing_python_dependencies", lambda deps: list(deps))
    monkeypatch.setattr(tools, "filter_install_requests", lambda deps: (list(deps), []))
    monkeypatch.setattr(
        events.subprocess,
        "run",
        Mock(return_value=SimpleNamespace(returncode=1, stdout="", stderr="no network")),
    )
    tools.execute_custom_tool.invoke({"tool_name": "tsne_fingerprint_scatter"})

    summary = audit_run.summary()
    assert summary["install_attempts"] >= 1
    assert summary["outcomes"] == {"failed": 1}
    assert summary["by_phase"] == {"execution": 1}
    assert summary["by_tool"] == {"tsne_fingerprint_scatter": 1}
    started = [e for e in _events(tmp_path) if e["kind"] == "install_started"]
    assert started[0]["packages"] == ["scikit-learn"]
    assert started[0]["component"] == "tool_builder"
    assert started[0]["reason"] == "missing_python_dependency"
    assert started[0]["command"][1:4] == ["-m", "pip", "install"]


def test_installed_but_unimportable_package_is_recorded(audit_run, tmp_path, monkeypatch):
    registry = _tool_registry(tmp_path, dependencies=["scikit-learn"])
    monkeypatch.setattr(tools, "get_registry", lambda: registry)
    monkeypatch.setattr(tools, "_missing_python_dependencies", lambda deps: list(deps))
    monkeypatch.setattr(tools, "filter_install_requests", lambda deps: (list(deps), []))
    monkeypatch.setattr(
        events.subprocess,
        "run",
        Mock(return_value=SimpleNamespace(returncode=0, stdout="", stderr="")),
    )
    tools.execute_custom_tool.invoke({"tool_name": "tsne_fingerprint_scatter"})
    summary = audit_run.summary()
    assert summary["outcomes"] == {"success": 1}
    assert summary["event_counts"]["install_unverified"] == 1


def test_disabled_installs_are_blocked_but_still_counted(audit_run, tmp_path, monkeypatch):
    monkeypatch.setenv(events.DISABLE_INSTALLS_ENV, "1")
    run = Mock()
    monkeypatch.setattr(events.subprocess, "run", run)
    result = events.run_install_command(
        ["pip", "install", "scikit-learn"], reason="test", capture_output=True, text=True
    )
    run.assert_not_called()
    assert result.returncode == 1
    assert "disabled" in result.stderr
    summary = audit_run.summary()
    assert summary["install_attempts"] == 1
    assert summary["outcomes"] == {"blocked": 1}
    finished = [e for e in _events(tmp_path) if e["kind"] == "install_finished"]
    assert finished[0]["error_type"] == "dynamic_installs_disabled"


def test_sudo_only_apt_and_pip_guard_skips_are_blocked_attempts(audit_run, tmp_path, monkeypatch):
    registry = _tool_registry(
        tmp_path, system_dependencies=["samtools"], dependencies=["smolagents"]
    )
    monkeypatch.setattr(tools, "get_registry", lambda: registry)
    monkeypatch.setattr(
        tools.shutil, "which", lambda n: "/fake/apt-get" if n == "apt-get" else None
    )
    monkeypatch.setattr(tools, "_missing_python_dependencies", lambda deps: list(deps))
    run = Mock()
    monkeypatch.setattr(events.subprocess, "run", run)
    tools.execute_custom_tool.invoke({"tool_name": "tsne_fingerprint_scatter"})
    run.assert_not_called()

    summary = audit_run.summary()
    assert summary["install_attempts"] == 2
    assert summary["outcomes"] == {"blocked": 2}
    finished = {e["error_type"] for e in _events(tmp_path) if e["kind"] == "install_finished"}
    assert finished == {"requires_sudo", "pip_guard"}
    managers = {e["manager"] for e in _events(tmp_path) if e["kind"] == "install_started"}
    assert managers == {"apt-get", "pip"}


def test_run_json_aggregates_dependency_audit_counts():
    def result(audit):
        return RunResult(
            use_case_id="u",
            use_case_name="U",
            prompt="p",
            execution_time=1.0,
            total_steps=1,
            workflow_completed=True,
            agent_flow=[],
            tool_calls=[],
            final_messages=[],
            dependency_audit=audit,
        )

    run = ExperimentRun(
        run_id="r",
        experiment_config=ExperimentConfig(name="cfg"),
        use_cases=[UseCase(id="u", name="U", prompt="p")],
        results=[
            result(
                {
                    "install_attempts": 2,
                    "outcomes": {"failed": 1, "blocked": 1},
                    "by_component": {"tool_builder": 2},
                }
            ),
            result({"install_attempts": 1, "outcomes": {"failed": 1}, "by_component": {"x": 1}}),
            result({}),
        ],
        started_at=datetime.now(),
    )
    aggregate = run.to_dict()["aggregates"]["dependency_audit"]
    assert aggregate == {
        "install_attempts": 3,
        "outcomes": {"failed": 2, "blocked": 1},
        "by_component": {"tool_builder": 2, "x": 1},
    }
