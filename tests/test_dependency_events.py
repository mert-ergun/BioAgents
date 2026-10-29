"""Install evidence covers failed ToolBuilder calls even with zero coder actions."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from bioagents.benchmarks.runner import run_use_case
from bioagents.benchmarks.use_case_models import UseCase
from bioagents.sandbox import dependency_events as events
from bioagents.tools import tool_builder_tools as tools
from bioagents.tools.tool_registry import ToolDefinition, ToolRegistry


@pytest.fixture
def audit_run(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOAGENTS_RUN_ID", "a" * 32)
    monkeypatch.setenv("BIOAGENTS_DEPENDENCY_EVENTS_DIR", str(tmp_path))
    monkeypatch.delenv("BIOAGENTS_USE_CASE_ID", raising=False)
    events.enable_dependency_tracking()
    return events.RunAudit("a" * 32, tmp_path)


def installer(tmp_path, name="pip", exit_code=0):
    path = tmp_path / name
    path.write_text(f"#!/bin/sh\nexit {exit_code}\n")
    path.chmod(0o700)
    return str(path)


@pytest.mark.parametrize("manager", ["pip", "pip3", "uv", "conda", "mamba", "apt", "apt-get"])
@pytest.mark.parametrize("exit_code", [0, 7])
def test_executed_installers_have_one_attempt_and_a_real_outcome(
    audit_run, tmp_path, manager, exit_code
):
    command = [installer(tmp_path, manager, exit_code)]
    if manager == "uv":
        command.append("pip")
    command += ["install", "example-package"]
    subprocess.run(command, capture_output=True)
    summary = audit_run.summary()
    assert summary["install_attempts"] == 1
    assert summary["outcomes"] == {"success" if exit_code == 0 else "failed": 1}


def test_repeated_wait_poll_and_communicate_do_not_double_count(audit_run, tmp_path):
    process = subprocess.Popen([installer(tmp_path), "install", "example"], stdout=subprocess.PIPE)
    process.communicate()
    process.wait()
    process.poll()
    assert audit_run.summary()["install_attempts"] == 1
    assert audit_run.summary()["event_counts"]["install_finished"] == 1


def test_positional_popen_environment_keeps_standard_api_compatibility(audit_run, tmp_path):
    process = subprocess.Popen(
        [installer(tmp_path), "install", "example"],
        -1,
        None,
        None,
        subprocess.PIPE,
        subprocess.PIPE,
        None,
        True,
        False,
        None,
        dict(os.environ),
    )
    process.communicate()
    assert process.returncode == 0
    assert audit_run.summary()["install_attempts"] == 1


def test_preexisting_popen_references_are_observed_without_guessing_the_outcome(
    audit_run, tmp_path
):
    original_popen = subprocess.Popen.__mro__[1]
    process = original_popen([installer(tmp_path), "install", "example"])
    assert process.wait(timeout=10) == 0
    assert audit_run.summary()["install_attempts"] == 1
    assert audit_run.summary()["outcomes"] == {"unknown": 1}


def test_spawn_failure_is_an_attempt_with_a_failed_outcome(audit_run, tmp_path):
    with pytest.raises(FileNotFoundError):
        subprocess.run([str(tmp_path / "pip"), "install", "example"])
    assert audit_run.summary()["outcomes"] == {"failed": 1}


def test_installer_deadline_is_counted_as_timeout(audit_run, tmp_path):
    path = Path(installer(tmp_path))
    path.write_text("#!/bin/sh\nexec sleep 2\n")
    with pytest.raises(subprocess.TimeoutExpired):
        events.run_install_command(
            [str(path), "install", "example"], reason="missing_python_dependency", timeout=0.02
        )
    assert audit_run.summary()["install_attempts"] == 1
    assert audit_run.summary()["outcomes"] == {"timeout": 1}


def test_os_system_install_is_observed_but_printed_advice_is_not(audit_run, tmp_path):
    os.system("echo 'pip install example' > /dev/null")
    assert audit_run.summary()["install_attempts"] == 0
    os.system(f"{installer(tmp_path)} install example")
    assert audit_run.summary()["outcomes"] == {"success": 1}


def test_mockable_install_boundary_preserves_failure_source_and_redacts_credentials(
    audit_run, monkeypatch, tmp_path
):
    monkeypatch.setattr(events.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=9)))
    with events.dependency_scope(component="tool_builder", tool="example", phase="execution"):
        events.run_install_command(
            ["pip", "install", "sample", "--index-url", "https://user:secret@example.com/simple"],
            reason="missing_python_dependency",
        )
    content = (tmp_path / events.EVENTS_FILE).read_text()
    assert "secret" not in content and "user:" not in content
    records = [json.loads(line) for line in content.splitlines()]
    start = records[0]
    assert start["packages"] == ["sample"]
    assert start["component"] == "tool_builder" and start["tool"] == "example"
    assert start["phase"] == "execution" and start["reason"] == "missing_python_dependency"
    assert audit_run.summary()["outcomes"] == {"failed": 1}


def test_event_file_symlink_cannot_redirect_writes(audit_run, tmp_path):
    victim = tmp_path / "preserve.txt"
    victim.write_text("preserve")
    (tmp_path / events.EVENTS_FILE).symlink_to(victim)
    events.record_dependency_event("guard_skipped", packages=["torch"], audit=audit_run)
    assert victim.read_text() == "preserve"
    assert audit_run.errors


def test_host_prepares_journal_without_truncating_existing_events(audit_run, tmp_path):
    audit_run.prepare()
    path = tmp_path / events.EVENTS_FILE
    owner = path.stat().st_uid
    events.record_dependency_event("guard_skipped", audit=audit_run, packages=["torch"])
    original = path.read_bytes()
    audit_run.prepare()
    assert path.read_bytes() == original
    assert path.stat().st_uid == owner
    assert path.stat().st_mode & 0o777 == 0o600


def test_unexecuted_shell_branch_is_unknown_instead_of_a_fake_install(audit_run):
    os.system("false && pip install never-executed-fixture")
    summary = audit_run.summary()
    assert summary["install_attempts"] == 0
    assert summary["unobserved_install_branches"] == 1


def test_quoted_install_advice_in_shell_compound_is_not_a_gap(audit_run):
    os.system("echo 'pip install example' > /dev/null && true")
    assert audit_run.summary()["install_attempts"] == 0
    assert audit_run.summary()["unobserved_install_branches"] == 0


def test_failed_toolbuilder_install_appears_in_run_json_without_coder_actions(
    audit_run, tmp_path, monkeypatch
):
    registry = ToolRegistry(tools_dir=tmp_path / "tools")
    registry.register_tool(
        ToolDefinition(
            name="needs_package",
            description="Fixture",
            category="test",
            parameters=[],
            return_type="dict",
            return_description="Result",
            dependencies=["missing-fixture-package"],
            code='def needs_package():\n    return {"status":"capability_unavailable"}\n',
        )
    )
    monkeypatch.setattr(tools, "get_registry", lambda: registry)
    monkeypatch.setattr(
        tools, "_missing_python_dependencies", lambda _: ["missing-fixture-package"]
    )
    monkeypatch.setattr(
        events.subprocess,
        "run",
        Mock(
            return_value=SimpleNamespace(
                returncode=1, stdout="install failed", stderr="install failed"
            )
        ),
    )

    def work(_):
        tools.execute_custom_tool.invoke({"tool_name": "needs_package"})
        return {"messages": [AIMessage(content="Install failed; no computation performed")]}

    graph = StateGraph(dict)
    graph.add_node("tool_builder", work)
    graph.add_edge(START, "tool_builder")
    graph.add_edge("tool_builder", END)
    result = run_use_case(UseCase(id="audit", name="Audit", prompt="Test"), graph=graph.compile())
    assert result.subagent_internal_step_counts() == {}
    summary = result.to_dict()["dependency_audit"]
    assert summary["install_attempts"] == 1
    assert summary["outcomes"] == {"failed": 1}
    assert summary["by_component"] == {"tool_builder": 1}
    report = json.loads((tmp_path / "dependency_audit.json").read_text())
    assert report["install_attempts"] == 1
    assert report["outcomes"] == {"failed": 1}


def test_separate_python_processes_preserve_scope_and_append_valid_events(audit_run, tmp_path):
    executable = installer(tmp_path)
    code = (
        "from bioagents.sandbox.dependency_events import enable_dependency_tracking; "
        "enable_dependency_tracking(); import subprocess; "
        f"subprocess.run([{executable!r}, 'install', 'example'])"
    )
    with events.dependency_scope(component="tool_builder", phase="validation", tool="example"):
        environment = {**os.environ, **events.dependency_environment()}
    processes = [subprocess.Popen([sys.executable, "-c", code], env=environment) for _ in range(2)]
    for process in processes:
        assert process.wait(timeout=10) == 0
    summary = audit_run.summary()
    assert summary["install_attempts"] == 2
    assert summary["outcomes"] == {"success": 2}
    assert summary["by_component"] == {"tool_builder": 2}
    assert summary["tracking_errors"] == []
    records = [
        json.loads(line) for line in (tmp_path / events.EVENTS_FILE).read_text().splitlines()
    ]
    assert all(
        record["phase"] == "validation" and record["tool"] == "example" for record in records
    )


def test_pip_show_of_package_named_install_is_not_an_install_attempt(audit_run, tmp_path):
    subprocess.run([installer(tmp_path), "show", "install"])
    assert audit_run.summary()["install_attempts"] == 0


def test_framework_kernel_install_has_a_real_failed_outcome(audit_run, monkeypatch):
    from bioagents.sandbox.resilient_executor import ResilientDockerExecutor

    monkeypatch.setattr(
        "bioagents.sandbox.resilient_executor.filter_install_requests",
        lambda _: (["example"], ["torch"]),
    )
    monkeypatch.setattr(
        events.subprocess,
        "run",
        Mock(return_value=SimpleNamespace(returncode=1, stdout="", stderr="failed")),
    )
    executor = object.__new__(ResilientDockerExecutor)
    executor.logger = Mock()
    executor.run_code_raise_errors = lambda code: exec(compile(code, "<fake-kernel>", "exec"), {})
    with pytest.raises(RuntimeError, match="installation failed"):
        executor.install_packages(["example", "torch"])
    summary = audit_run.summary()
    assert summary["install_attempts"] == 1
    assert summary["outcomes"] == {"failed": 1}
    assert summary["event_counts"]["guard_skipped"] == 1
    assert summary["by_component"] == {"framework": 1}
