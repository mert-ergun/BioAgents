"""Host tools must resolve artifacts from the same workspace as code agents."""

import pytest

from bioagents.tools.custom_tools.list_workspace_files import list_workspace_files


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    project = tmp_path / "project"
    workdir = project / "artifacts" / "run"
    workdir.mkdir(parents=True)
    monkeypatch.setenv("BIOAGENTS_PROJECT_ROOT", str(project))
    monkeypatch.setattr("bioagents.tools.workspace_paths.workspace_dir", lambda: workdir)
    monkeypatch.chdir(project)
    (workdir / "result.json").write_text("{}")
    (workdir / "report.txt").write_text("done")
    (workdir / "subdirectory").mkdir()
    # A symlink must not leak file metadata from outside the workspace.
    (workdir / "linked.json").symlink_to(tmp_path / "outside.json")
    (tmp_path / "outside.json").write_text("outside")
    return project, workdir


@pytest.mark.parametrize("path", [".", "/app/artifacts/run", "artifacts/run", "result.json"])
def test_listing_resolves_workspace_container_and_file_paths(workspace, path):
    _, workdir = workspace
    result = list_workspace_files(path, "*.json")
    assert result["status"] == "success"
    assert result["directory"] == str(workdir)
    assert result["n_files"] == 1
    assert result["files"][0]["name"] == "result.json"
    assert result["files"][0]["size_bytes"] == 2


def test_nonexistent_directory_is_not_created(workspace):
    _, workdir = workspace
    with pytest.raises(FileNotFoundError, match="Directory not found"):
        list_workspace_files("missing")
    assert not (workdir / "missing").exists()


def test_traversal_is_rejected(workspace):
    with pytest.raises(ValueError, match="traversal"):
        list_workspace_files("../outside")
