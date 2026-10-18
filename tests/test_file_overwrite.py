"""Guards against the task2 failure: a good artifact silently replaced by a worse one.

In that run the workflow ran twice. The second pass could not tell that the first pass
had already produced ``chembl_akt1_ligands.csv``, re-derived it with different
parameters, and left the deliverable describing ten molecules that were never docked
while the ranking described ten different ones.

Two things have to hold for that to be impossible:
- an agent can ask "does this exist?" and get a true answer for *every* path form it
  might be holding (workspace-relative, project-relative, container ``/app/...``);
- a write that lands on an existing artifact is an explicit decision, and a successful
  result is never replaced by an empty one.
"""

from __future__ import annotations

import json

import pytest

from bioagents.sandbox.workspace import CONTAINER_PROJECT_ROOT, to_portable_path
from bioagents.tools.file_tools import (
    check_file_exists,
    list_local_directory,
    read_local_file,
    write_local_file,
)
from bioagents.tools.workspace_paths import (
    prepare_output_file,
    resolve_existing_path,
    unique_sibling,
)

GOOD_CSV = "chembl_id,smiles\nCHEMBL2177824,Nc1ncccc1\n"
OTHER_CSV = "chembl_id,smiles\nCHEMBL3985936,CS(=O)(=O)N1\n"


@pytest.fixture
def sandboxed(tmp_path, monkeypatch):
    """Point the sandbox and the project root at an isolated directory."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BIOAGENTS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("BIOAGENTS_SANDBOX_DIR", str(tmp_path / "sandbox_workdir"))

    import bioagents.sandbox.sandbox_manager as sm

    monkeypatch.setattr(sm, "SANDBOX_BASE_DIR", tmp_path / "sandbox_workdir")
    monkeypatch.setattr(sm, "_default_sandbox", None, raising=False)
    return tmp_path


def _reported_path(write_result: str) -> str:
    return write_result.split("to: ", 1)[1].splitlines()[0].strip()


# --- existence must be answerable, in every path form ------------------------------


def test_check_file_exists_is_false_before_and_true_after_a_write(sandboxed) -> None:
    before = json.loads(check_file_exists.invoke({"path": "ligands.csv"}))
    assert before["exists"] is False

    write_local_file.invoke({"file_path": "ligands.csv", "content": GOOD_CSV})

    after = json.loads(check_file_exists.invoke({"path": "ligands.csv"}))
    assert after["exists"] is True
    assert after["type"] == "file"
    assert after["size_bytes"] == len(GOOD_CSV)


def test_existence_is_true_for_the_container_path_an_agent_reports(sandboxed) -> None:
    """The coder agent reports /app/... paths; a later existence check must agree."""
    written = write_local_file.invoke({"file_path": "task/ligands.csv", "content": GOOD_CSV})
    portable = _reported_path(written)

    container = f"{CONTAINER_PROJECT_ROOT}/{portable}"

    assert json.loads(check_file_exists.invoke({"path": container}))["exists"] is True
    assert read_local_file.invoke({"file_path": container}) == GOOD_CSV


def test_missing_directory_is_not_reported_as_empty(sandboxed) -> None:
    """'Empty' told the agent earlier work was gone; it has to say 'does not exist'."""
    listing = json.loads(list_local_directory.invoke({"path": "never_created"}))

    assert listing["exists"] is False
    assert "does not exist" in listing["note"]


def test_empty_directory_is_distinguishable_from_a_missing_one(sandboxed) -> None:
    (sandboxed / "sandbox_workdir" / "default" / "empty_dir").mkdir(parents=True)

    listing = json.loads(list_local_directory.invoke({"path": "empty_dir"}))

    assert listing["exists"] is True
    assert listing["entries"] == []


def test_directory_listing_finds_a_container_style_path(sandboxed) -> None:
    write_local_file.invoke({"file_path": "task/ligands.csv", "content": GOOD_CSV})
    portable = to_portable_path(sandboxed / "sandbox_workdir" / "default" / "task")

    listing = json.loads(
        list_local_directory.invoke({"path": f"{CONTAINER_PROJECT_ROOT}/{portable}"})
    )

    assert listing["exists"] is True
    assert [e["name"] for e in listing["entries"]] == ["ligands.csv"]


# --- an existing artifact is never clobbered by accident ---------------------------


def test_a_second_write_does_not_replace_the_first_result(sandboxed) -> None:
    """The task2 regression: a later, different result must not take the same name."""
    first = _reported_path(
        write_local_file.invoke({"file_path": "ligands.csv", "content": GOOD_CSV})
    )
    second = write_local_file.invoke({"file_path": "ligands.csv", "content": OTHER_CSV})

    assert "already existed" in second
    assert _reported_path(second) != first
    assert read_local_file.invoke({"file_path": first}) == GOOD_CSV
    assert read_local_file.invoke({"file_path": _reported_path(second)}) == OTHER_CSV


def test_the_new_version_is_distinctly_named(sandboxed) -> None:
    write_local_file.invoke({"file_path": "ligands.csv", "content": GOOD_CSV})

    second = _reported_path(
        write_local_file.invoke({"file_path": "ligands.csv", "content": OTHER_CSV})
    )

    assert second.endswith("ligands_v2.csv")


def test_explicit_overwrite_replaces_the_file_and_says_so(sandboxed) -> None:
    first = _reported_path(
        write_local_file.invoke({"file_path": "ligands.csv", "content": GOOD_CSV})
    )

    result = write_local_file.invoke(
        {"file_path": "ligands.csv", "content": OTHER_CSV, "overwrite": True}
    )

    assert _reported_path(result) == first
    assert "Replaced" in result
    assert read_local_file.invoke({"file_path": first}) == OTHER_CSV


def test_a_successful_result_is_never_replaced_by_an_empty_one(sandboxed) -> None:
    path = _reported_path(
        write_local_file.invoke({"file_path": "ligands.csv", "content": GOOD_CSV})
    )

    refused = write_local_file.invoke(
        {"file_path": "ligands.csv", "content": "", "overwrite": True}
    )

    assert refused.startswith("Refused")
    assert read_local_file.invoke({"file_path": path}) == GOOD_CSV


def test_rewriting_identical_content_reuses_the_existing_file(sandboxed) -> None:
    """Re-deriving the same answer should not spawn a v2 of an identical file."""
    write_local_file.invoke({"file_path": "ligands.csv", "content": GOOD_CSV})

    again = write_local_file.invoke({"file_path": "ligands.csv", "content": GOOD_CSV})

    assert "Already up to date" in again
    assert json.loads(check_file_exists.invoke({"path": "ligands_v2.csv"}))["exists"] is False


def test_a_fresh_write_reports_that_it_created_the_file(sandboxed) -> None:
    result = write_local_file.invoke({"file_path": "ligands.csv", "content": GOOD_CSV})

    assert "Created a new file" in result


# --- the shared path helpers other tools build on ----------------------------------


def test_resolve_existing_path_returns_none_instead_of_raising(sandboxed) -> None:
    assert resolve_existing_path("nothing/here.csv") is None


def test_resolve_existing_path_finds_directories_too(sandboxed) -> None:
    target = sandboxed / "sandbox_workdir" / "default" / "docking"
    target.mkdir(parents=True)

    assert resolve_existing_path("docking") == target


def test_resolve_existing_path_rejects_traversal(sandboxed) -> None:
    with pytest.raises(ValueError):
        resolve_existing_path("../../etc/passwd")


def test_unique_sibling_keeps_counting_up(sandboxed) -> None:
    base = sandboxed / "sandbox_workdir" / "default"
    base.mkdir(parents=True, exist_ok=True)
    (base / "a.csv").write_text("1")
    (base / "a_v2.csv").write_text("2")

    assert unique_sibling(base / "a.csv").name == "a_v3.csv"


def test_prepare_output_file_reports_the_decision_it_made(sandboxed) -> None:
    target, action = prepare_output_file("out/results.json")
    assert action == "create"
    target.write_text("{}")

    _, again = prepare_output_file("out/results.json")
    assert again == "new_version"

    _, forced = prepare_output_file("out/results.json", overwrite=True)
    assert forced == "overwrite"


def test_prepare_output_file_anchors_relative_paths_at_the_workspace(sandboxed) -> None:
    """A relative path must not land in the host process cwd, where no agent looks."""
    target, _ = prepare_output_file("results.csv")

    assert target.parent == (sandboxed / "sandbox_workdir" / "default")
