"""An agent's finished work must reach the supervisor, and be recognised as finished.

Both halves failed in the recorded runs: results returned on non-channel keys were
dropped by LangGraph without an error, and a coder that said "TASK COMPLETE" was not
believed because completion was detected with model-training vocabulary.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from bioagents.agents.supervisor_helpers import (
    check_code_agent_task_completed,
    check_coder_should_force_finish,
)
from bioagents.graph import STATE_CHANNELS, ensure_result_is_visible
from bioagents.sandbox.coder_helpers import COMPLETION_MARKER, append_completion_marker


def test_result_on_non_channel_keys_is_rebuilt_into_a_message():
    result = {"data": {"x": 1}, "raw_output": "RMSD = 1.42 A", "tool_calls": [], "error": None}

    fixed = ensure_result_is_visible(result, "Coder")

    assert fixed["messages"]
    assert "RMSD = 1.42 A" in fixed["messages"][0].content


def test_an_error_on_non_channel_keys_still_reaches_the_supervisor():
    fixed = ensure_result_is_visible({"error": "ChEMBL 500", "raw_output": None}, "Analysis")

    assert "ChEMBL 500" in fixed["messages"][0].content
    assert "AGENT_ERROR" in fixed["messages"][0].content


def test_existing_messages_are_left_alone():
    original = AIMessage(content="done")
    result = {"messages": [original], "data": {}}

    assert ensure_result_is_visible(result, "Coder")["messages"] == [original]


def test_an_empty_result_becomes_an_explicit_no_output_message():
    fixed = ensure_result_is_visible({}, "Research")

    assert "NO_OUTPUT" in fixed["messages"][0].content
    assert "Research" in fixed["messages"][0].content


def test_a_pure_routing_update_is_not_given_a_fake_message():
    result = {"next": "supervisor"}

    assert ensure_result_is_visible(result, "Coder") == result


def test_state_channel_list_matches_the_graph_state():
    from bioagents.graph import AgentState

    declared = set(AgentState.__annotations__)

    assert declared <= STATE_CHANNELS


@pytest.mark.parametrize(
    "agent_module,factory",
    [
        ("bioagents.agents.analysis_agent", "create_analysis_agent"),
        ("bioagents.agents.research_agent", "create_research_agent"),
    ],
)
def test_agent_nodes_never_return_only_non_channel_keys(agent_module, factory):
    """Static guard: a return without `messages` is invisible to the supervisor."""
    import ast
    import importlib
    import pathlib

    module = importlib.import_module(agent_module)
    source = ast.parse(pathlib.Path(module.__file__).read_text())

    offenders = []
    for node in ast.walk(source):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Dict):
            continue
        keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
        if not keys or keys & STATE_CHANNELS:
            continue
        offenders.append(node.lineno)

    assert offenders == [], f"{agent_module}: returns without a state channel at {offenders}"


def test_delivered_final_answer_is_marked_for_the_supervisor():
    content = append_completion_marker("TASK COMPLETE: RMSD computed.")

    assert COMPLETION_MARKER in content


def test_completion_marker_does_not_override_a_degraded_status():
    degraded = "partial output [CODER_STATUS: max_steps_reached] ..."

    assert append_completion_marker(degraded) == degraded


def test_the_marker_alone_proves_completion():
    message = AIMessage(content=f"short answer {COMPLETION_MARKER}", name="Coder")

    done, reason = check_code_agent_task_completed([message])

    assert done is True
    assert "final answer" in reason


@pytest.mark.parametrize(
    "opening",
    [
        "TASK COMPLETE: 4GV1 Co-folding and Ligand RMSD Calculation",
        "TASK COMPLETED AUTONOMOUSLY - CDK2 Mutation-Fitness Study",
        "TASK COMPLETE — 100 molecules generated from capivasertib",
        "ANALYSIS COMPLETE. All requested metrics were produced",
    ],
)
def test_plain_language_completion_is_recognised(opening):
    """These are the real opening lines from runs the supervisor restarted anyway."""
    body = opening + "\n\nOUTPUT FILES:\n" + "- results.csv with the computed values\n" * 20
    message = AIMessage(content=body, name="Coder")

    assert check_code_agent_task_completed([message])[0] is True


def test_a_denial_of_completion_is_not_read_as_completion():
    body = "I was not able to complete the task. " * 20
    message = AIMessage(content=body, name="Coder")

    assert check_code_agent_task_completed([message])[0] is False


def test_completion_is_only_accepted_from_a_code_agent():
    body = "TASK COMPLETE\n" + "details " * 100
    message = AIMessage(content=body, name="Research")

    assert check_code_agent_task_completed([message])[0] is False


def test_a_lost_sandbox_still_stops_redelegation():
    message = AIMessage(content="[CODER_STATUS: sandbox_unrecoverable]", name="Coder")

    assert check_coder_should_force_finish([message])[0] is True


def test_supervisor_finishes_instead_of_rerunning_a_completed_coder():
    """End to end over the helper the supervisor node calls first."""
    messages = [
        HumanMessage(content="compute the RMSD"),
        AIMessage(
            content=f"TASK COMPLETE: ligand RMSD = 1.42 A\n{'results written. ' * 40}"
            f"\n\n{COMPLETION_MARKER}",
            name="Coder",
        ),
    ]

    done, _ = check_code_agent_task_completed(messages)

    assert done is True


def test_code_agent_runs_the_supervisors_latest_instruction():
    """Re-delegation restarted the whole pipeline because only the first message was read."""
    from bioagents.sandbox.coder_helpers import (
        build_task_with_output_dir,
        extract_current_instruction,
    )

    messages = [
        HumanMessage(content="Create 100 molecules and plot them"),
        AIMessage(content="first attempt", name="Coder"),
        HumanMessage(
            content="[SUPERVISOR TASK] Only redraw the t-SNE plot from molecules_100.json",
            name="Supervisor",
        ),
    ]

    instruction = extract_current_instruction(messages)
    task = build_task_with_output_dir(
        "Create 100 molecules and plot them", [], None, current_instruction=instruction
    )

    assert task.startswith("TASK: Only redraw the t-SNE plot")
    assert "ORIGINAL REQUEST (context only)" in task


def test_the_original_request_is_used_when_the_supervisor_gave_none():
    from bioagents.sandbox.coder_helpers import (
        build_task_with_output_dir,
        extract_current_instruction,
    )

    messages = [HumanMessage(content="Compute the RMSD")]

    assert extract_current_instruction(messages) is None
    assert build_task_with_output_dir("Compute the RMSD", [], None).startswith(
        "TASK: Compute the RMSD"
    )


def test_a_timeout_reports_the_files_the_tool_already_wrote(tmp_path):
    """Claiming "partial results unavailable" sent the agent off to redo finished work."""
    from bioagents.tools.tool_universe import _timeout_report

    (tmp_path / "chembl_akt1_ligands.csv").write_text("chembl_id,smiles\nCHEMBL1,CCO\n")

    report = _timeout_report("chembl_get_active_inhibitors", 30.0, {"output_dir": str(tmp_path)})

    assert "timed out after 30s" in report
    assert "chembl_akt1_ligands.csv" in report
    assert "inspect them before running anything again" in report


def test_a_timeout_with_no_output_says_so_plainly(tmp_path):
    from bioagents.tools.tool_universe import _timeout_report

    report = _timeout_report("some_tool", 30.0, {"output_dir": str(tmp_path)})

    assert "No output files were found" in report


def test_chembl_tool_refuses_to_default_to_akt1():
    from bioagents.tools.custom_tools.chembl_get_active_inhibitors import (
        chembl_get_active_inhibitors,
    )

    out = chembl_get_active_inhibitors()

    assert out["status"] == "error"
    assert "No target given" in out["message"]


def test_target_prediction_reports_unavailable_instead_of_a_fixed_gene_panel():
    """The node always fell through to EGFR/CDK2/BRAF with invented probabilities."""
    from bioagents.workflows.drug_discovery import fetch_nodes

    assert not hasattr(fetch_nodes, "_stub_target_panel")

    node = fetch_nodes.SwissTargetPredictionNode()
    out = node.run({"smiles": ""})

    assert out["predicted_targets"] == []
    assert out["available"] is False


def test_the_sandbox_container_runs_as_the_host_user(monkeypatch):
    """Root-owned artifacts made host agents treat finished work as stale relics."""
    import os

    from bioagents.sandbox.coder_executor import _sandbox_user

    monkeypatch.delenv("BIOAGENTS_SANDBOX_UID", raising=False)
    assert _sandbox_user() == f"{os.getuid()}:{os.getgid()}"


def test_image_tag_changes_when_the_recipe_changes():
    from bioagents.sandbox.coder_executor import sandbox_image_tag

    assert sandbox_image_tag("FROM a") != sandbox_image_tag("FROM b")
    assert sandbox_image_tag("FROM a") == sandbox_image_tag("FROM a")


def test_blast_budget_stays_under_the_sandbox_cell_limit(monkeypatch):
    """A 1200 s budget in a 300 s cell killed the kernel and lost the RID."""
    monkeypatch.setenv("BIOAGENTS_BLAST_IN_CELL", "1")
    from bioagents.tools.tool_universe import _tool_time_budget

    assert _tool_time_budget("BLAST_protein_search") < 300
