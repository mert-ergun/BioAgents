"""Verifies a file written by one agent is findable and readable by another.

Regression context: file tools reported absolute host paths like
``/home/user/BioAgents/sandbox_workdir/default/4gv1.pdb``. The coder/ml/dl agents run in
a container that mounts the project root at ``/app``, so that path does not exist for
them — a file one agent produced was effectively invisible to the next.
"""

from __future__ import annotations

import json

import pytest

from bioagents.sandbox.workspace import (
    CONTAINER_PROJECT_ROOT,
    project_root,
    to_host_path,
    to_portable_path,
)
from bioagents.tools.file_tools import (
    get_file_info,
    list_local_directory,
    read_local_file,
    write_local_file,
)


@pytest.fixture
def sandboxed(tmp_path, monkeypatch):
    """Point both the sandbox and the project root at an isolated directory."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BIOAGENTS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("BIOAGENTS_SANDBOX_DIR", str(tmp_path / "sandbox_workdir"))

    import bioagents.sandbox.sandbox_manager as sm

    monkeypatch.setattr(sm, "SANDBOX_BASE_DIR", tmp_path / "sandbox_workdir")
    monkeypatch.setattr(sm, "_sandbox_instances", {}, raising=False)
    monkeypatch.setattr(sm, "_default_sandbox", None, raising=False)
    return tmp_path


def _reported_path(write_result: str) -> str:
    return write_result.split("to: ", 1)[1].splitlines()[0].strip()


def test_written_path_is_not_an_absolute_host_path(sandboxed) -> None:
    """An absolute host path is unusable inside the code-execution container."""
    result = write_local_file.invoke({"file_path": "4gv1.pdb", "content": "ATOM pretend"})
    path = _reported_path(result)

    assert not path.startswith("/"), f"file tool reported a non-portable absolute path: {path}"


def test_second_agent_can_read_what_first_agent_wrote(sandboxed) -> None:
    written = write_local_file.invoke(
        {"file_path": "structures/4gv1.pdb", "content": "ATOM      1  N   MET A   1"}
    )
    path = _reported_path(written)

    assert read_local_file.invoke({"file_path": path}) == "ATOM      1  N   MET A   1"


def test_container_style_path_resolves_back_to_the_same_file(sandboxed) -> None:
    """The coder agent reports /app/... paths; other agents must still open them."""
    written = write_local_file.invoke({"file_path": "out/result.csv", "content": "a,b\n1,2"})
    path = _reported_path(written)

    container_path = f"{CONTAINER_PROJECT_ROOT}/{path}"

    assert read_local_file.invoke({"file_path": container_path}) == "a,b\n1,2"
    assert json.loads(get_file_info.invoke({"file_path": container_path}))["path"] == path


def test_directory_listing_reports_a_portable_location(sandboxed) -> None:
    write_local_file.invoke({"file_path": "shared/a.txt", "content": "x"})

    listing = json.loads(list_local_directory.invoke({"path": "shared"}))

    assert not listing["directory"].startswith("/")
    assert [e["name"] for e in listing["entries"]] == ["a.txt"]


def test_portable_path_round_trips(sandboxed) -> None:
    target = project_root() / "sandbox_workdir" / "default" / "x.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("data")

    portable = to_portable_path(target)

    assert not portable.startswith("/")
    assert to_host_path(portable).read_text() == "data"
    assert to_host_path(f"{CONTAINER_PROJECT_ROOT}/{portable}").read_text() == "data"


def test_paths_outside_the_project_root_are_left_absolute(sandboxed) -> None:
    """Rewriting an unshareable path would hide that it is unshareable."""
    outside = "/etc/hostname"

    assert to_portable_path(outside) == outside
