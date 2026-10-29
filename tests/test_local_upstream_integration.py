"""Regression checks for the selected local improvements on top of PR #60."""

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bioagents.sandbox import coder_executor
from bioagents.tools.custom_tools import analyze_protein_ligand_interactions as interactions
from bioagents.tools.custom_tools import pdb_fetch_structure as pdb_fetch
from bioagents.tools.custom_tools import vina_dock_ligands as docking


@pytest.mark.parametrize("identifier", ["../../outside", "/tmp/outside", "../bad", "a/b", "a\\b"])
def test_ligand_identifiers_cannot_escape_output_directory(tmp_path, identifier):
    path = tmp_path / "ligands.json"
    path.write_text(json.dumps({"molecules": [{"chembl_id": identifier, "smiles": "CC"}]}))
    with pytest.raises(ValueError, match="Unsafe ligand identifier"):
        docking._load_ligands(path)


def test_duplicate_ligand_identifiers_are_rejected(tmp_path):
    path = tmp_path / "ligands.json"
    path.write_text(json.dumps([{"chembl_id": "same", "smiles": "CC"}] * 2))
    with pytest.raises(ValueError, match="Duplicate"):
        docking._load_ligands(path)


@pytest.fixture
def docking_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BIOAGENTS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr("bioagents.tools.workspace_paths.workspace_dir", lambda: tmp_path)
    receptor = tmp_path / "4GV1.pdbqt"
    receptor.write_text("provided receptor")
    ligands = tmp_path / "ligands.json"
    ligands.write_text(
        json.dumps(
            {
                "molecules": [
                    {"chembl_id": "A", "smiles": "CC"},
                    {"chembl_id": "B", "smiles": "CCC"},
                ]
            }
        )
    )
    monkeypatch.setattr(docking, "_choose_center", lambda *a: ([0.0, 0.0, 0.0], "test"))
    monkeypatch.setitem(sys.modules, "vina", SimpleNamespace(Vina=Mock()))

    def dock_one(_vina, ligand, output, *_):
        pose = output / f"{ligand['chembl_id']}_pose.pdbqt"
        pose.write_text("docked pose")
        return {**ligand, "status": "success", "affinity_kcal_mol": -5.0, "pose_pdbqt": str(pose)}

    execute = Mock(side_effect=dock_one)
    monkeypatch.setattr(docking, "_dock_one", execute)
    return receptor, ligands, tmp_path / "output", execute, dock_one


def test_partial_docking_is_not_reported_as_complete_success(docking_run):
    receptor, ligands, output, execute, dock_one = docking_run

    def partial(vina, ligand, out, *args):
        if ligand["chembl_id"] == "B":
            raise RuntimeError("ligand failed")
        return dock_one(vina, ligand, out, *args)

    execute.side_effect = partial
    result = docking.vina_dock_ligands(str(receptor), str(ligands), str(output))
    assert result["status"] == "partial"
    assert result["n_docked"] == result["n_failed"] == 1


def test_resume_uses_latest_version_and_matches_inputs(docking_run):
    receptor, ligands, output, execute, _ = docking_run
    first = docking.vina_dock_ligands(str(receptor), str(ligands), str(output))
    original = Path(first["results_json"]).read_bytes()
    second = docking.vina_dock_ligands(str(receptor), str(ligands), str(output), resume=True)
    assert execute.call_count == 2
    assert first["results_json"] != second["results_json"]
    assert Path(first["results_json"]).read_bytes() == original
    # If only the versioned result remains, resume must still find it.
    Path(first["results_json"]).unlink()
    docking.vina_dock_ligands(str(receptor), str(ligands), str(output), resume=True)
    assert execute.call_count == 2
    # A different box changes the calculation and must not reuse old affinities.
    docking.vina_dock_ligands(
        str(receptor), str(ligands), str(output), resume=True, box_size=[25, 25, 25]
    )
    assert execute.call_count == 4


def test_failed_conversion_cannot_reuse_old_receptor(tmp_path, monkeypatch):
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("ATOM  test\nEND\n")
    old = tmp_path / "receptor.pdbqt"
    old.write_text("previous valid receptor")
    monkeypatch.setattr(docking.shutil, "which", lambda _: "/fake/obabel")
    monkeypatch.setattr(
        docking.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=0, stderr=""))
    )
    with pytest.raises(RuntimeError, match="conversion failed"):
        docking._prepare_receptor(receptor, tmp_path)
    assert old.read_text() == "previous valid receptor"


@pytest.mark.parametrize("box", [[0, 1, 1], [math.inf, 1, 1], [1, 2], []])
def test_invalid_docking_box_fails_before_io(box):
    with pytest.raises(ValueError, match="box_size"):
        docking.vina_dock_ligands("missing", "missing", box_size=box)


def test_interaction_angles_are_computed_without_name_error():
    assert interactions._angle([1, 0, 0], [0, 0, 0], [0, 1, 0]) == pytest.approx(90)
    assert interactions._angle([1, 0, 0], [0, 0, 0], [-1, 0, 0]) == pytest.approx(180)


def test_missing_interaction_dependency_is_not_installed(monkeypatch):
    monkeypatch.setattr(
        interactions.importlib,
        "import_module",
        Mock(side_effect=ModuleNotFoundError("missing", name="rdkit")),
    )
    with pytest.raises(ModuleNotFoundError, match="sandbox image"):
        interactions._ensure("rdkit")


def test_pdb_download_requires_explicit_target():
    with pytest.raises(TypeError):
        pdb_fetch.pdb_fetch_structure()


def test_registry_embedded_sources_match_selected_tools():
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "bioagents/tools/custom_tools/registry.json").read_text())
    for name in [
        "vina_dock_ligands",
        "chembl_get_active_inhibitors",
        "pdb_fetch_structure",
        "analyze_protein_ligand_interactions",
    ]:
        assert (
            registry["tools"][name]["code"]
            == (root / f"bioagents/tools/custom_tools/{name}.py").read_text()
        )
    pdb_id = registry["tools"]["pdb_fetch_structure"]["parameters"][0]
    assert pdb_id["required"] and pdb_id["default"] is None


def test_missing_image_never_falls_back_to_host_execution(monkeypatch):
    monkeypatch.setenv("USE_LOCAL_EXECUTOR", "false")
    monkeypatch.delenv("BIOAGENTS_SANDBOX_ALLOW_BUILD", raising=False)
    monkeypatch.setattr(coder_executor, "_existing_image_id", lambda _: None)
    local = Mock()
    monkeypatch.setattr(coder_executor, "_local_executor", local)
    lazy = coder_executor.create_executor("coder", [])
    assert not lazy.started
    with pytest.raises(RuntimeError, match="No existing sandbox image"):
        lazy("print('never on host')")
    local.assert_not_called()


def test_image_build_requires_explicit_opt_in(monkeypatch):
    from bioagents.sandbox import resilient_executor

    monkeypatch.setenv("USE_LOCAL_EXECUTOR", "false")
    monkeypatch.setenv("BIOAGENTS_SANDBOX_ALLOW_BUILD", "true")
    monkeypatch.setattr(coder_executor, "_existing_image_id", lambda _: None)
    monkeypatch.setattr(coder_executor, "find_available_port", lambda: 8892)
    remote = Mock()
    monkeypatch.setattr(resilient_executor, "ResilientDockerExecutor", remote)
    coder_executor.create_executor("coder", [])("pass")
    assert remote.call_args.kwargs["build_new_image"] is True


def test_interaction_reader_uses_only_first_vina_pose(tmp_path):
    pose = tmp_path / "multi_pose.pdbqt"
    atom = "ATOM      1  C   LIG A   1       1.000   2.000   3.000  1.00  0.00     0.000 C\n"
    pose.write_text("MODEL 1\n" + atom + "ENDMDL\nMODEL 2\n" + atom + "ENDMDL\n")
    atoms, _, _, _ = interactions._load_ligand(None, str(pose))
    assert len(atoms) == 1


def test_download_rejects_error_page_then_uses_fallback(tmp_path, monkeypatch):
    good = "ATOM      1  C   ALA A   1       1.000   2.000   3.000  1.00  0.00           C\nEND\n"
    responses = [
        SimpleNamespace(text="<html>ATOM END server error</html>", raise_for_status=lambda: None),
        SimpleNamespace(text=good, raise_for_status=lambda: None),
    ]
    get = Mock(side_effect=responses)
    monkeypatch.setattr(pdb_fetch.requests, "get", get)
    result = pdb_fetch.pdb_fetch_structure("4GV1", str(tmp_path))
    assert get.call_count == 2
    assert "ebi.ac.uk" in result["source_url"]
    assert Path(result["pdb_file"]).read_text() == good


def test_shared_image_includes_chemistry_and_executor_dependencies():
    assert {"rdkit", "vina", "meeko", "gemmi", "openbabel-wheel", "langchain-core"} <= set(
        coder_executor.sandbox_packages()
    )
    assert "smolagents" in coder_executor.build_dockerfile()


def test_adapter_quotes_multiline_summary_as_data():
    from langchain_core.messages import AIMessage

    from bioagents.llms.adapters import LangChainModelAdapter

    summary = 'A "quoted" answer with \\slashes\nand another line'
    adapter = LangChainModelAdapter(Mock(invoke=Mock(return_value=AIMessage(content=summary))))
    output = adapter.generate([]).content
    code = output.split("```python\n", 1)[1].split("```", 1)[0]
    answers = []
    exec(compile(code, "<summary>", "exec"), {"final_answer": answers.append})
    assert answers == [summary]


@pytest.mark.parametrize("marker", ["empty_response", "repeated_code"])
def test_adapter_failures_are_not_reported_as_completed_tasks(marker):
    from langchain_core.messages import AIMessage

    from bioagents.agents.supervisor_helpers import (
        check_code_agent_task_completed,
        check_coder_should_force_finish,
    )
    from bioagents.sandbox.coder_helpers import append_completion_marker

    content = f"[CODER_STATUS: {marker}] No result was computed."
    assert "[CODER_STATUS: completed]" not in append_completion_marker(content)
    message = AIMessage(content=content, name="Coder")
    assert not check_code_agent_task_completed([message])[0]
    done, reason = check_coder_should_force_finish([message])
    assert done
    assert "completed successfully" not in reason
