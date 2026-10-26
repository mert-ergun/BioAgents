"""Exercise venv dispatch with real isolated interpreters, without model downloads."""

import json
import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest

from bioagents.sandbox import python_environments as environments


@pytest.fixture
def isolated_environments(tmp_path, monkeypatch):
    root = tmp_path / "venvs"
    for name in environments.ENVIRONMENTS:
        directory = root / name
        venv.EnvBuilder(with_pip=False, symlinks=True).create(directory)
        site = (
            directory
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        (site / "environment_marker.py").write_text(f"NAME = {name!r}\n")
    monkeypatch.setenv("BIOAGENTS_VENV_ROOT", str(root))
    monkeypatch.setenv("PIP_CONSTRAINT", str(tmp_path / "coder-only-constraints.txt"))
    monkeypatch.setenv("UV_CONSTRAINT", str(tmp_path / "coder-only-constraints.txt"))
    monkeypatch.chdir(tmp_path)
    return root


@pytest.mark.parametrize("name", environments.ENVIRONMENTS)
def test_each_venv_uses_its_own_packages_and_shares_workspace(isolated_environments, name):
    result = environments.run_in_environment(
        name,
        "import environment_marker,json,os,sys; from pathlib import Path; "
        "Path('shared.txt').write_text(environment_marker.NAME); "
        "print(json.dumps({'name': environment_marker.NAME,'prefix':sys.prefix,"
        "'venv':os.environ['VIRTUAL_ENV'],'guard':os.getenv('PIP_CONSTRAINT'),"
        "'uv_guard':os.getenv('UV_CONSTRAINT')}))",
    )
    data = json.loads(result.stdout)
    assert data["name"] == name
    assert data["prefix"] == data["venv"] == str(isolated_environments / name)
    assert data["guard"] is data["uv_guard"] is None
    assert Path("shared.txt").read_text() == name
    assert "PIP_CONSTRAINT" in os.environ


def test_cli_executes_the_named_python(isolated_environments):
    result = subprocess.run(
        [
            sys.executable,
            environments.__file__,
            "pytdc",
            "--",
            "-c",
            "import environment_marker; print(environment_marker.NAME)",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert result.stdout.strip() == "pytdc"


@pytest.mark.parametrize("name", environments.ENVIRONMENTS)
@pytest.mark.parametrize("cli", [False, True])
def test_isolated_python_and_cli_keep_runtime_audit_without_local_scanner(
    isolated_environments, tmp_path, monkeypatch, name, cli
):
    from bioagents.sandbox.dependency_events import RunAudit

    run_id = "d" * 32
    journal = tmp_path / "audit"
    journal.mkdir()
    monkeypatch.setenv("BIOAGENTS_RUN_ID", run_id)
    monkeypatch.setenv("BIOAGENTS_DEPENDENCY_EVENTS_DIR", str(journal))
    fake_pip = tmp_path / "pip"
    fake_pip.write_text("#!/bin/sh\nexit 7\n")
    fake_pip.chmod(0o700)
    code = (
        "from __future__ import annotations\n"
        "import subprocess, environment_marker\n"
        f"subprocess.run([{str(fake_pip)!r}, 'install', 'fixture-package'])\n"
        "print(environment_marker.NAME)\n"
    )
    if cli:
        result = subprocess.run(
            [sys.executable, environments.__file__, name, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    else:
        result = environments.run_in_environment(name, code)
    assert result.stdout.strip() == name
    summary = RunAudit(run_id, journal).summary()
    assert summary["install_attempts"] == 1
    assert summary["outcomes"] == {"failed": 1}
    assert not summary["tracking_errors"]


def test_child_error_is_reported_with_stderr(isolated_environments):
    with pytest.raises(subprocess.CalledProcessError) as error:
        environments.run_in_environment("boltz", "raise RuntimeError('calculation failed')")
    assert "calculation failed" in error.value.stderr


def test_child_timeout_raises(isolated_environments):
    with pytest.raises(subprocess.TimeoutExpired):
        environments.run_in_environment("pytdc", "import time; time.sleep(10)", timeout=0.1)


def test_cofold_prefers_the_preinstalled_boltz_environment(isolated_environments, monkeypatch):
    from bioagents.tools import cofold_tools

    monkeypatch.delenv("BIOAGENTS_BOLTZ_CMD", raising=False)
    cli = isolated_environments / "boltz" / "bin" / "boltz"
    cli.write_text("#!/bin/sh\nexit 0\n")
    monkeypatch.setattr(cofold_tools.shutil, "which", lambda name: "/other/boltz")
    backend = cofold_tools.detect_cofold_backend()
    assert backend.name == "boltz-venv"
    assert backend.command == (str(cli),)


def test_explicit_boltz_backend_still_wins(isolated_environments, monkeypatch):
    from bioagents.tools import cofold_tools

    monkeypatch.setenv("BIOAGENTS_BOLTZ_CMD", "/custom/boltz")
    backend = cofold_tools.detect_cofold_backend()
    assert backend.name == "boltz-explicit"
    assert backend.command == ("/custom/boltz",)


def test_missing_environment_does_not_fall_back(monkeypatch, tmp_path):
    monkeypatch.setenv("BIOAGENTS_VENV_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="Build"):
        environments.run_in_environment("boltz", "print('should not execute')")


@pytest.mark.parametrize("name", ["../coder", "unknown", "coder/bin/python"])
def test_unknown_environment_is_rejected(name):
    with pytest.raises(ValueError, match="Unknown Python environment"):
        environments.environment_python(name)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_is_rejected(timeout):
    with pytest.raises(ValueError, match="timeout"):
        environments.run_in_environment("coder", "pass", timeout=timeout)
