"""Standalone imports and native conversion must fail clearly and safely."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from bioagents.tools.custom_tools import vina_dock_ligands as docking


def test_standalone_custom_tool_imports_without_langchain():
    script = """
import importlib.abc
import sys
class NoLangChain(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'langchain_core' or fullname.startswith('langchain_core.'):
            raise ModuleNotFoundError('deliberately missing langchain_core', name='langchain_core')
sys.meta_path.insert(0, NoLangChain())
import bioagents.tools
from bioagents.tools.custom_tools import vina_dock_ligands
assert callable(vina_dock_ligands.vina_dock_ligands)
try:
    from bioagents.tools import calculate_molecular_weight
except ModuleNotFoundError as exc:
    assert 'requires langchain-core' in str(exc)
else:
    raise AssertionError('Missing dependency was hidden')
try:
    import bioagents.tools.tool_universe
except ModuleNotFoundError as exc:
    assert 'wrappers require langchain-core' in str(exc)
else:
    raise AssertionError('ToolUniverse dependency was hidden')
"""
    subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True, timeout=10
    )


def test_lazy_import_does_not_swallow_unrelated_import_errors(monkeypatch):
    import bioagents.tools as tools

    def broken_import(name):
        raise ImportError("broken internal import")

    monkeypatch.setattr(tools, "import_module", broken_import)
    with pytest.raises(ImportError, match="broken internal import"):
        tools.__getattr__("calculate_molecular_weight")


def test_public_exports_resolve_with_dependencies_installed():
    import bioagents.tools as tools

    assert all(getattr(tools, name) is not None for name in tools.__all__)


@pytest.fixture
def cli(monkeypatch):
    monkeypatch.setattr(docking.shutil, "which", lambda name: "/fake/obabel")


def test_cli_conversion_atomically_publishes_fresh_output(tmp_path, monkeypatch, cli):
    target = tmp_path / "out.pdbqt"
    target.write_text("old output")

    def convert(command, **kwargs):
        assert command[0] == "/fake/obabel"
        assert kwargs["timeout"] == 60
        temporary = Path(command[command.index("-O") + 1])
        assert temporary != target
        temporary.write_text("new output")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(docking.subprocess, "run", convert)
    docking._convert_with_obabel("input.pdb", target)
    assert target.read_text() == "new output"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("returncode", [-6, 1, 0])
def test_failed_or_empty_cli_output_does_not_reuse_stale_file(
    tmp_path, monkeypatch, cli, returncode
):
    target = tmp_path / "out.pdbqt"
    target.write_text("stale output")
    monkeypatch.setattr(
        docking.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=returncode, stderr="conversion failed"),
    )
    with pytest.raises(RuntimeError, match="conversion failed"):
        docking._receptor_to_pdbqt("input.pdb", target)
    assert target.read_text() == "stale output"
    assert list(tmp_path.iterdir()) == [target]


def test_cli_timeout_cleans_up_temporary_file(tmp_path, monkeypatch, cli):
    def timed_out(*args, **kwargs):
        raise subprocess.TimeoutExpired("obabel", 60)

    monkeypatch.setattr(docking.subprocess, "run", timed_out)
    with pytest.raises(subprocess.TimeoutExpired):
        docking._convert_with_obabel("input.pdb", tmp_path / "out.pdbqt")
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX child-signal isolation")
def test_native_child_abort_does_not_kill_calling_process(tmp_path, monkeypatch):
    executable = tmp_path / "obabel"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import os, resource, signal\n"
        "resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
        "os.kill(os.getpid(), signal.SIGABRT)\n"
    )
    executable.chmod(0o700)
    monkeypatch.setattr(docking.shutil, "which", lambda name: str(executable))
    with pytest.raises(RuntimeError, match="exit -6"):
        docking._convert_with_obabel("input.pdb", tmp_path / "out.pdbqt")
    assert list(tmp_path.iterdir()) == [executable]


def test_missing_cli_never_falls_back_to_in_process_openbabel(monkeypatch, tmp_path):
    monkeypatch.setattr(docking.shutil, "which", lambda name: None)

    def unexpected_import(*args, **kwargs):
        raise AssertionError("Native fallback must not be imported")

    monkeypatch.setattr(docking, "_ensure", unexpected_import)
    with pytest.raises(RuntimeError, match="CLI is required"):
        docking._receptor_to_pdbqt("input.pdb", tmp_path / "out.pdbqt")


def test_missing_dependency_is_not_installed_at_runtime(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError("missing", name=name)

    monkeypatch.setattr(docking.importlib, "import_module", missing)
    with pytest.raises(ModuleNotFoundError, match="sandbox image before execution"):
        docking._ensure("rdkit")


def test_registered_source_matches_updated_tool_file():
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "bioagents/tools/custom_tools/registry.json").read_text())
    assert registry["tools"]["vina_dock_ligands"]["code"] == Path(docking.__file__).read_text()
