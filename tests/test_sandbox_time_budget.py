"""Per-tool cell budgets stay below the kernel deadline and survive restarts (issue #67)."""

from __future__ import annotations

import json
import os
import time
from unittest.mock import Mock

import pytest
from smolagents.local_python_executor import CodeOutput
from smolagents.utils import AgentError

from bioagents import limits
from bioagents.sandbox import checkpoint, time_budget
from bioagents.sandbox.resilient_executor import (
    KERNEL_RESTARTED_MESSAGE,
    KernelRestartedError,
    ResilientDockerExecutor,
)


def _sleep_then_return(seconds: float) -> str:
    time.sleep(seconds)
    return "finished"


def _raise_value_error() -> str:
    raise ValueError("bad input")


@pytest.fixture(autouse=True)
def _reset_recovery_roots():
    yield
    time_budget.set_recovery_roots([])


@pytest.fixture
def clean_env(monkeypatch):
    for name in (
        "BIOAGENTS_CODE_EXECUTION_TIMEOUT_SEC",
        "BIOAGENTS_CODER_EXEC_TIMEOUT",
        time_budget.CELL_TIMEOUT_ENV,
        *time_budget.IN_CELL_ENVS,
        *time_budget.FORWARDED_BUDGET_ENVS,
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# -- deadlines and budgets --------------------------------------------------------------
def test_default_tiers_come_from_limits(clean_env):
    assert time_budget.cell_execution_timeout_sec() == limits.DEFAULT_CELL_EXECUTION_TIMEOUT_SEC
    assert limits.CELL_SETUP_TIMEOUT_SEC == 30.0
    assert set(limits.TOOL_CELL_BUDGETS_SEC) >= {"blast", "esmfold", "vina", "model_staging"}


def test_container_cell_timeout_wins_over_host_settings(clean_env):
    clean_env.setenv("BIOAGENTS_CODE_EXECUTION_TIMEOUT_SEC", "900")
    clean_env.setenv(time_budget.CELL_TIMEOUT_ENV, "300")
    assert time_budget.active_cell_timeout_sec() == 300
    clean_env.setenv(time_budget.CELL_TIMEOUT_ENV, "nonsense")
    assert time_budget.active_cell_timeout_sec() == 900


@pytest.mark.parametrize("cell_timeout", [2.0, 30.0, 120.0, 300.0, 1800.0, 7200.0])
@pytest.mark.parametrize("tool", ["blast", "esmfold", "vina", "model_staging", "unknown_tool"])
def test_tool_budget_is_always_strictly_below_the_cell_deadline(tool, cell_timeout):
    budget = time_budget.tool_cell_budget(tool, cell_timeout=cell_timeout)
    assert 0 < budget < cell_timeout


def test_vina_at_1800s_cell_no_longer_equals_the_kill_threshold():
    """The issue's Vina call ran right up to the 1800 s kill."""
    budget = time_budget.tool_cell_budget("vina", cell_timeout=1800.0)
    assert budget <= 1800.0 - limits.TOOL_CELL_BUDGET_MARGIN_SEC


def test_requested_budget_only_shrinks_the_configured_one():
    assert time_budget.tool_cell_budget("vina", requested=100, cell_timeout=1800) == 100
    configured = limits.TOOL_CELL_BUDGETS_SEC["vina"]
    assert time_budget.tool_cell_budget(
        "vina", requested=10_000, cell_timeout=100_000
    ) == pytest.approx(configured)


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("run_docking", "vina"),
        ("AutoDock_Vina_dock", "vina"),
        ("run_esmfold", "esmfold"),
        ("BLAST_protein_search", "blast"),
        ("model_staging", "model_staging"),
        ("from_pretrained", "model_staging"),
        ("something_else", None),
        (None, None),
    ],
)
def test_budget_key_resolves_tool_aliases(name, key):
    assert time_budget.budget_key(name) == key


def test_blast_budget_respects_a_short_cell_deadline(clean_env):
    clean_env.setenv("BIOAGENTS_BLAST_IN_CELL", "1")
    clean_env.setenv(time_budget.CELL_TIMEOUT_ENV, "120")
    from bioagents.tools.genomics_tools import _blast_budget

    assert 0 < _blast_budget(0) < 120


def test_sandbox_env_forwards_deadline_and_overrides(clean_env):
    clean_env.setenv("BIOAGENTS_CODE_EXECUTION_TIMEOUT_SEC", "600")
    clean_env.setenv("BIOAGENTS_VINA_CELL_BUDGET_SEC", "400")
    env = time_budget.sandbox_time_budget_env()
    assert env[time_budget.CELL_TIMEOUT_ENV] == "600"
    assert env["BIOAGENTS_VINA_CELL_BUDGET_SEC"] == "400"
    assert env["BIOAGENTS_IN_SANDBOX_CELL"] == "1"
    assert "BIOAGENTS_ESMFOLD_CELL_BUDGET_SEC" not in env


def test_in_cell_detection_honours_overrides(clean_env):
    clean_env.setenv("BIOAGENTS_IN_SANDBOX_CELL", "0")
    clean_env.setenv("BIOAGENTS_BLAST_IN_CELL", "1")
    assert time_budget.in_sandbox_cell() is False
    clean_env.delenv("BIOAGENTS_IN_SANDBOX_CELL")
    assert time_budget.in_sandbox_cell() is True


# -- prompt context ---------------------------------------------------------------------
def test_guidance_states_deadline_budgets_and_checkpointing():
    text = time_budget.time_budget_guidance(cell_timeout=300.0, workflow_remaining=1234.0)
    assert "300s" in text
    assert "vina" in text and "esmfold" in text and "blast" in text
    assert "save_checkpoint" in text
    assert "1234s remain" in text


def test_task_includes_time_budget_only_for_the_docker_sandbox(clean_env):
    from bioagents.sandbox.coder_helpers import build_task_with_output_dir

    clean_env.setenv("USE_LOCAL_EXECUTOR", "false")
    assert "[TIME BUDGET]" in build_task_with_output_dir("Dock it", [], None)
    clean_env.setenv("USE_LOCAL_EXECUTOR", "true")
    assert "[TIME BUDGET]" not in build_task_with_output_dir("Dock it", [], None)


# -- recovery notice --------------------------------------------------------------------
def test_recent_files_are_bounded_and_skip_hidden_dirs(tmp_path):
    old = tmp_path / "old.csv"
    old.write_text("x")
    os.utime(old, (1, 1))
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "index").write_text("x")
    since = time.time() - 5
    for i in range(5):
        (tmp_path / f"new_{i}.pdb").write_text("x")
    files = time_budget.recent_workspace_files([tmp_path], since, limit=3)
    assert len(files) == 3
    assert all("new_" in f for f in files)
    assert len(time_budget.recent_workspace_files([tmp_path], since, max_entries=2)) <= 2


def test_recovery_notice_lists_recent_files(tmp_path):
    (tmp_path / "docked_poses.pdbqt").write_text("x")
    notice = time_budget.recovery_notice(time.time() - 60, roots=[tmp_path])
    assert notice.startswith(KERNEL_RESTARTED_MESSAGE)
    assert "docked_poses.pdbqt" in notice
    assert time_budget.recovery_notice(None) == KERNEL_RESTARTED_MESSAGE


@pytest.fixture
def executor():
    value = object.__new__(ResilientDockerExecutor)
    value._init_resilience()
    value.ws_url = "ws://localhost/api/kernels/test/channels"
    value.base_url = "http://localhost"
    value.kernel_id = "test"
    value.logger = Mock()
    return value


def test_kernel_restart_error_lists_files_written_this_run(executor, monkeypatch, tmp_path):
    (tmp_path / "esmfold_prediction.pdb").write_text("ATOM")
    time_budget.set_recovery_roots([tmp_path])
    executor._run_started_at = time.time() - 60
    monkeypatch.setattr(executor, "_ensure_session", lambda: None)
    execute = Mock(side_effect=[KernelRestartedError("Kernel entered restarting state"), None])
    monkeypatch.setattr(executor, "_raw_execute", execute)
    with pytest.raises(AgentError) as info:
        executor.resilient_run("fold()")
    assert "ALL variables" in str(info.value)
    assert "esmfold_prediction.pdb" in str(info.value)


def test_timeout_restart_suggests_splitting_and_checkpointing(executor, monkeypatch, tmp_path):
    import requests

    time_budget.set_recovery_roots([tmp_path])
    executor._run_started_at = time.time()
    monkeypatch.setattr(executor, "_ensure_session", lambda: None)
    ok = CodeOutput(output=None, logs="", is_final_answer=False)
    monkeypatch.setattr(
        executor, "_raw_execute", Mock(side_effect=[TimeoutError("exceeded 300s"), ok])
    )
    monkeypatch.setattr(requests, "post", Mock())
    with pytest.raises(AgentError, match="save_checkpoint"):
        executor.resilient_run("dock_everything()")


def test_setup_cells_use_the_configured_setup_tier(executor, monkeypatch):
    from bioagents.sandbox import resilient_executor

    monkeypatch.setattr(resilient_executor, "CELL_SETUP_TIMEOUT_SEC", 45.0)
    execute = Mock()
    monkeypatch.setattr(executor, "_raw_execute", execute)
    executor._apply_setup()
    assert execute.call_args.kwargs["timeout"] == 45.0


def test_send_tools_marks_the_run_start(executor, monkeypatch):
    monkeypatch.setattr(executor, "_apply_setup", lambda: None)
    executor.installed_packages = []
    before = time.time()
    executor.send_tools({})
    assert executor._run_started_at >= before


# -- bounded native calls ---------------------------------------------------------------
def test_budgeted_process_returns_result_within_budget(clean_env):
    clean_env.setenv(time_budget.CELL_TIMEOUT_ENV, "600")
    result = time_budget.run_in_budgeted_process(
        _sleep_then_return, {"seconds": 0}, tool="vina", budget=60
    )
    assert result == "finished"


def test_budgeted_process_stops_before_the_cell_deadline(clean_env):
    clean_env.setenv(time_budget.CELL_TIMEOUT_ENV, "600")
    started = time.monotonic()
    result = json.loads(
        time_budget.run_in_budgeted_process(
            _sleep_then_return, {"seconds": 60}, tool="vina", budget=1.0, hint="dock fewer"
        )
    )
    assert time.monotonic() - started < 30
    assert result["status"] == "timeout"
    assert result["cell_bounded"] is True
    assert result["budget_sec"] == 1.0
    assert result["cell_timeout_sec"] == 600
    assert "dock fewer" in result["message"]


def test_budgeted_process_reports_errors(clean_env):
    result = json.loads(
        time_budget.run_in_budgeted_process(_raise_value_error, {}, tool="esmfold", budget=60)
    )
    assert result == {"status": "error", "tool": "esmfold", "message": "ValueError: bad input"}


def test_docking_runs_inline_outside_the_sandbox(clean_env, monkeypatch):
    from bioagents.tools import docking_tools

    clean_env.setenv("BIOAGENTS_IN_SANDBOX_CELL", "0")
    impl = Mock(return_value='{"status": "success"}')
    bounded = Mock()
    monkeypatch.setattr(docking_tools, "_run_docking_impl", impl)
    monkeypatch.setattr(docking_tools, "run_in_budgeted_process", bounded)
    args = {"receptor_pdbqt_path": "r.pdbqt", "ligand_pdbqt_path": "l.pdbqt"}
    docking_tools.run_docking.invoke({**args, "center_x": 0, "center_y": 0, "center_z": 0})
    impl.assert_called_once()
    bounded.assert_not_called()


def test_docking_is_bounded_inside_the_sandbox(clean_env, monkeypatch):
    from bioagents.tools import docking_tools

    clean_env.setenv("BIOAGENTS_IN_SANDBOX_CELL", "1")
    bounded = Mock(return_value='{"status": "timeout"}')
    monkeypatch.setattr(docking_tools, "run_in_budgeted_process", bounded)
    args = {"receptor_pdbqt_path": "r.pdbqt", "ligand_pdbqt_path": "l.pdbqt"}
    docking_tools.run_docking.invoke({**args, "center_x": 0, "center_y": 0, "center_z": 0})
    assert bounded.call_args.args[0] is docking_tools._run_docking_impl
    assert bounded.call_args.kwargs["tool"] == "vina"


def test_esmfold_is_bounded_inside_the_sandbox(clean_env, monkeypatch):
    from bioagents.sandbox import time_budget as tb
    from bioagents.tools import structural_tools

    clean_env.setenv("BIOAGENTS_IN_SANDBOX_CELL", "1")
    bounded = Mock(return_value='{"status": "timeout"}')
    monkeypatch.setattr(tb, "run_in_budgeted_process", bounded)
    structural_tools.run_esmfold.invoke({"sequence": "MKV"})
    assert bounded.call_args.args[0] is structural_tools._run_esmfold_impl
    assert bounded.call_args.kwargs["tool"] == "esmfold"


# -- checkpoint helper ------------------------------------------------------------------
def test_checkpoint_round_trips_json_and_pickle(tmp_path):
    path = checkpoint.save_checkpoint("scores", {"a": [1, 2]}, directory=tmp_path)
    assert path.endswith("scores.json")
    assert checkpoint.load_checkpoint("scores", directory=tmp_path) == {"a": [1, 2]}

    checkpoint.save_checkpoint("scores", {1, 2, 3}, directory=tmp_path)
    assert checkpoint.load_checkpoint("scores", directory=tmp_path) == {1, 2, 3}
    assert not (tmp_path / "scores.json").exists()
    assert checkpoint.list_checkpoints(tmp_path) == ["scores"]


def test_checkpoint_defaults_and_env_directory(tmp_path, monkeypatch):
    monkeypatch.setenv(checkpoint.CHECKPOINT_DIR_ENV, str(tmp_path / "ckpt"))
    assert checkpoint.load_checkpoint("missing", default="none") == "none"
    checkpoint.save_checkpoint("step1", [1])
    assert (tmp_path / "ckpt" / "step1.json").is_file()


@pytest.mark.parametrize("name", ["../escape", "a/b", "", ".hidden", "x" * 200])
def test_checkpoint_rejects_unsafe_names(tmp_path, name):
    with pytest.raises(ValueError):
        checkpoint.save_checkpoint(name, 1, directory=tmp_path)
