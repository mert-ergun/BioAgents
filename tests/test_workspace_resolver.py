"""Canonical workspace root and the shared path resolver (issue #68)."""

import json
from pathlib import Path

import pytest

from bioagents.sandbox import workspace
from bioagents.sandbox.workspace import (
    WORKSPACE_ROOT_ENV,
    WorkspacePathError,
    describe_shared_workspace,
    resolve_path_arguments,
    resolve_workspace_path,
    workspace_root,
)
from bioagents.tools import tool_builder_tools
from bioagents.tools.tool_registry import ToolDefinition, ToolParameter, ToolRegistry

EXPERIMENT = "3e871bcce0d04949af2f27a2d7d1b0b2"


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """Project root with an experiment workspace, as in the benchmark runs."""
    project = tmp_path / "project"
    root = project / "generated_artifacts" / "experiments" / EXPERIMENT / "default"
    task = root / "001_task1_rmsd_calculation"
    task.mkdir(parents=True)
    (task / "4GV1.pdb").write_text("ATOM\n")
    (task / "blast").mkdir()
    (tmp_path / "outside.txt").write_text("secret")
    monkeypatch.setenv("BIOAGENTS_PROJECT_ROOT", str(project))
    monkeypatch.setenv("BIOAGENTS_SANDBOX_DIR", str(root.parent))
    monkeypatch.setenv(WORKSPACE_ROOT_ENV, str(root))
    monkeypatch.chdir(project)
    return project, root.resolve(), task.resolve()


def test_workspace_root_honours_env(layout):
    _, root, _ = layout
    assert workspace_root() == root
    assert workspace_root().is_absolute()


def test_workspace_root_defaults_to_sandbox(tmp_path, monkeypatch):
    from bioagents.sandbox import sandbox_manager

    monkeypatch.delenv(WORKSPACE_ROOT_ENV, raising=False)
    sandbox = sandbox_manager.SandboxManager(workspace_id="run")
    monkeypatch.setattr(sandbox_manager, "get_sandbox", lambda *a, **k: sandbox)
    assert workspace_root() == sandbox.workdir.resolve()


def test_sandbox_uses_canonical_root_as_shell_cwd(layout):
    from bioagents.sandbox.sandbox_manager import SandboxManager

    _, root, _ = layout
    sandbox = SandboxManager()
    assert sandbox.workdir == root
    result = sandbox.run_command(f'pwd && echo "${WORKSPACE_ROOT_ENV}"')
    assert result["stdout"].split() == [str(root), str(root)]


@pytest.mark.parametrize(
    "given",
    [
        "001_task1_rmsd_calculation/4GV1.pdb",
        f"generated_artifacts/experiments/{EXPERIMENT}/default/001_task1_rmsd_calculation/4GV1.pdb",
        f"/app/generated_artifacts/experiments/{EXPERIMENT}/default/001_task1_rmsd_calculation/4GV1.pdb",
        "sandbox_workdir/default/001_task1_rmsd_calculation/4GV1.pdb",
        (
            f"generated_artifacts/experiments/{EXPERIMENT}/default/generated_artifacts/"
            f"experiments/{EXPERIMENT}/default/001_task1_rmsd_calculation/4GV1.pdb"
        ),
        "001_task1_rmsd_calculation/4gv1.pdb",
        "001_TASK1_RMSD_CALCULATION/4gv1.PDB",
        "./001_task1_rmsd_calculation/../001_task1_rmsd_calculation/4GV1.pdb",
    ],
)
def test_every_convention_resolves_to_one_absolute_path(layout, given):
    _, _, task = layout
    assert resolve_workspace_path(given, must_exist=True) == task / "4GV1.pdb"


def test_absolute_host_path_with_doubled_prefix_is_collapsed(layout):
    _, root, task = layout
    doubled = f"{root}/generated_artifacts/experiments/{EXPERIMENT}/default/{task.name}/blast"
    assert resolve_workspace_path(doubled, must_exist=True) == task / "blast"


def test_missing_output_resolves_under_root_without_creating(layout):
    _, root, _ = layout
    target = resolve_workspace_path("new/out")
    assert target == root / "new" / "out"
    assert not target.exists()


@pytest.mark.parametrize("given", ["../" * 40 + "etc/passwd", "/etc/passwd"])
def test_escaping_the_workspace_is_refused(layout, given):
    with pytest.raises(WorkspacePathError, match="escapes the workspace"):
        resolve_workspace_path(given)


def test_strict_roots_refuse_a_sibling_of_the_workspace(layout):
    _, root, _ = layout
    with pytest.raises(WorkspacePathError, match="escapes the workspace"):
        resolve_workspace_path("../../../../../outside.txt", allowed_roots=[root])


def test_missing_input_raises_with_root(layout):
    _, root, _ = layout
    with pytest.raises(FileNotFoundError, match=str(root)):
        resolve_workspace_path("nope.pdb", must_exist=True)


def test_empty_path_is_rejected(layout):
    with pytest.raises(WorkspacePathError):
        resolve_workspace_path("  ")


def test_path_arguments_are_canonicalised(layout):
    _, root, task = layout
    args = {
        "input_file": "sandbox_workdir/default/001_task1_rmsd_calculation/4gv1.pdb",
        "output_dir": "001_task1_rmsd_calculation/out",
        "pdb_id": "4GV1",
        "source_url": "https://example.org/a.pdb",
        "n": 3,
    }
    new_args, resolved = resolve_path_arguments(args)
    assert resolved == {
        "input_file": str(task / "4GV1.pdb"),
        "output_dir": str(task / "out"),
    }
    assert new_args == {**args, **resolved}
    assert args["output_dir"] == "001_task1_rmsd_calculation/out"  # input not mutated
    assert root.is_dir()


def test_missing_input_argument_is_refused(layout):
    with pytest.raises(WorkspacePathError, match="molecules_file"):
        resolve_path_arguments({"molecules_file": "missing.smi"})


def test_prompt_states_the_canonical_root(layout):
    _, root, _ = layout
    note = describe_shared_workspace(root)
    assert f"WORKSPACE ROOT: {root}" in note
    assert WORKSPACE_ROOT_ENV in note
    assert f"/app/generated_artifacts/experiments/{EXPERIMENT}/default" in note


def test_shell_prompt_includes_root(layout):
    from bioagents.agents.shell_agent import build_shell_agent_prompt

    _, root, _ = layout
    assert str(root) in build_shell_agent_prompt()


def test_list_workspace_files_accepts_doubled_prefix(layout):
    from bioagents.tools.custom_tools.list_workspace_files import list_workspace_files

    _, _, task = layout
    doubled = (
        f"generated_artifacts/experiments/{EXPERIMENT}/default/generated_artifacts/"
        f"experiments/{EXPERIMENT}/default/{task.name}"
    )
    result = list_workspace_files(doubled, "*.pdb")
    assert result["directory"] == str(task)
    assert result["n_files"] == 1


@pytest.fixture
def path_tool(layout, tmp_path, monkeypatch):
    registry = ToolRegistry(tools_dir=tmp_path / "tools")
    definition = ToolDefinition(
        name="echo_paths",
        description="Echo the paths it was given",
        category="utility",
        parameters=[
            ToolParameter("input_file", "string", "Input", required=True),
            ToolParameter("output_dir", "string", "Output", required=False, default=""),
        ],
        return_type="dict",
        return_description="The paths",
        code=(
            "def echo_paths(input_file: str, output_dir: str = ''):\n"
            "    return {'status': 'success', 'input_file': input_file, 'output_dir': output_dir}\n"
        ),
    )
    assert registry.register_tool(definition)
    monkeypatch.setattr(tool_builder_tools, "get_registry", lambda: registry)
    return layout


def _execute(args: dict) -> dict:
    return json.loads(
        tool_builder_tools.execute_custom_tool.invoke(
            {"tool_name": "echo_paths", "arguments_json": json.dumps(args)}
        )
    )


def test_execute_custom_tool_passes_and_reports_absolute_paths(path_tool):
    _, root, task = path_tool
    result = _execute(
        {
            "input_file": f"/app/generated_artifacts/experiments/{EXPERIMENT}/default/"
            f"{task.name}/4gv1.pdb",
            "output_dir": f"{task.name}/blast",
        }
    )
    assert result["status"] == "success"
    expected = {"input_file": str(task / "4GV1.pdb"), "output_dir": str(task / "blast")}
    assert result["resolved_paths"] == expected
    assert result["result"]["input_file"] == expected["input_file"]
    assert result["result"]["output_dir"] == expected["output_dir"]
    assert Path(result["result"]["output_dir"]).is_relative_to(root)


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ({"input_file": "absent.pdb"}, "does not exist"),
        ({"input_file": "001_task1_rmsd_calculation/4GV1.pdb", "output_dir": "/etc"}, "escapes"),
    ],
)
def test_execute_custom_tool_rejects_bad_paths_before_running(path_tool, args, match):
    result = _execute(args)
    assert result["status"] == "error"
    assert match in result["message"]
    assert "schema" in result


def test_module_exports_env_name():
    assert workspace.WORKSPACE_ROOT_ENV == "BIOAGENTS_WORKSPACE_ROOT"
