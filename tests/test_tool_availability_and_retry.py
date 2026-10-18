"""Unavailable tools are never offered, transient failures retry, and specialists keep
work the code agents would otherwise absorb."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from bioagents.agents.supervisor_helpers import (
    check_coder_routing_guard,
    count_code_agent_routes,
    describe_workspace_artifacts,
    suggest_specialist_for_task,
)
from bioagents.tools.agent_manifest import (
    UNAVAILABLE_STUB_TOOLS,
    agent_tool_names,
    filter_bindable_tools,
)
from bioagents.tools.tool_retry import (
    annotate_exhausted,
    invoke_with_retry,
    is_transient_error,
    should_retry,
)


def test_stub_tools_are_absent_from_every_agent_inventory():
    inventory = agent_tool_names()
    offered = {name for names in inventory.values() for name in names}
    assert not (offered & UNAVAILABLE_STUB_TOOLS)


def test_filter_drops_tools_marked_unavailable_by_attribute():
    @tool
    def pretend_tool(x: str) -> str:
        """A tool that cannot actually run."""
        return x

    @tool
    def real_tool(x: str) -> str:
        """A tool that works."""
        return x

    pretend_tool.func.__bioagents_unavailable__ = True

    kept = filter_bindable_tools([pretend_tool, real_tool])

    assert [t.name for t in kept] == ["real_tool"]


def test_filter_removes_duplicate_tool_names():
    @tool
    def only_once(x: str) -> str:
        """Registered twice, bound once."""
        return x

    assert len(filter_bindable_tools([only_once, only_once])) == 1


def test_rdkit_tools_are_bound_so_molecular_properties_need_no_coder():
    inventory = agent_tool_names()
    assert "compute_molecular_descriptors" in inventory["analysis"]
    assert "compute_molecular_fingerprint" in inventory["analysis"]
    assert "search_similar_molecules_tool" in inventory["analysis"]
    assert "compute_molecular_descriptors" in inventory["docking"]


@pytest.mark.parametrize(
    "text",
    [
        "HTTPSConnectionPool: Read timed out",
        "Server returned HTTP 503",
        "429 Too Many Requests",
        "Connection reset by peer",
        "ToolUniverse SDK error: Bad Gateway",
    ],
)
def test_transient_errors_are_recognised(text):
    assert is_transient_error(text)


@pytest.mark.parametrize(
    "text",
    [
        "ValidationError: field required",
        "No such file or directory: receptor.pdbqt",
        "unexpected keyword argument 'foo'",
        '{"status": "error", "error_type": "capability_unavailable"}',
        "aggrescan is not installed in your environment",
    ],
)
def test_permanent_errors_are_not_retried(text):
    assert not is_transient_error(text)


def _error_payload(text: str) -> dict:
    return {"messages": [ToolMessage(content=text, tool_call_id="1", name="t", status="error")]}


def _ok_payload(text: str = "done") -> dict:
    return {"messages": [ToolMessage(content=text, tool_call_id="1", name="t")]}


def test_transient_failure_is_retried_until_it_succeeds():
    calls = []
    results = [_error_payload("HTTP 503 service unavailable"), _ok_payload("real data")]

    def fake_invoke(*_args, **_kwargs):
        calls.append(1)
        return results[len(calls) - 1]

    out = invoke_with_retry(fake_invoke, {}, max_attempts=3, sleep=lambda _s: None)

    assert len(calls) == 2
    assert out["messages"][0].content == "real data"


def test_permanent_failure_is_returned_without_retrying():
    calls = []

    def fake_invoke(*_args, **_kwargs):
        calls.append(1)
        return _error_payload("ValidationError: bad SMILES")

    invoke_with_retry(fake_invoke, {}, max_attempts=3, sleep=lambda _s: None)

    assert len(calls) == 1


def test_exhausted_retries_tell_the_agent_to_stop_calling_the_tool():
    def fake_invoke(*_args, **_kwargs):
        return _error_payload("Connection reset by peer")

    out = invoke_with_retry(fake_invoke, {}, max_attempts=2, sleep=lambda _s: None)

    content = out["messages"][0].content
    assert "failed 2 times" in content
    assert "do not call this tool again" in content
    assert out["messages"][0].status == "error"


def test_a_batch_with_one_success_is_not_retried():
    payload = {
        "messages": [
            ToolMessage(content="HTTP 503", tool_call_id="1", name="a", status="error"),
            ToolMessage(content="ok", tool_call_id="2", name="b"),
        ]
    }

    retry, _ = should_retry(payload)

    assert retry is False


def test_annotate_exhausted_preserves_tool_call_id():
    out = annotate_exhausted(_error_payload("timeout"), 3)
    assert out["messages"][0].tool_call_id == "1"


def test_workspace_listing_names_files_and_previews_small_json(tmp_path):
    (tmp_path / "molecules_100.json").write_text(json.dumps({"molecules": ["CCO", "c1ccccc1"]}))
    (tmp_path / "big.bin").write_bytes(b"\x00" * 2048)

    described = describe_workspace_artifacts(tmp_path)

    assert "molecules_100.json" in described
    assert "CCO" in described  # preview, so the supervisor sees it is real data
    assert "big.bin" in described
    assert "do not ask an agent to re-download" in described.lower()


def test_workspace_listing_is_empty_when_nothing_exists(tmp_path):
    assert describe_workspace_artifacts(tmp_path) == ""


def test_workspace_listing_survives_a_missing_directory(tmp_path):
    assert describe_workspace_artifacts(tmp_path / "nope") == ""


def test_workspace_listing_skips_dot_directories(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main")

    assert describe_workspace_artifacts(tmp_path) == ""


MEMBERS = ["research", "analysis", "coder", "docking", "genomics", "report", "data_acquisition"]


def test_specialist_is_preferred_when_it_holds_a_matching_tool():
    replacement, reason = check_coder_routing_guard(
        [], MEMBERS, "compute molecular descriptors for the 100 molecules"
    )

    assert replacement in {"analysis", "docking"}
    assert "already holds tools" in reason


def test_genuine_code_task_still_reaches_the_coder():
    replacement, _ = check_coder_routing_guard(
        [], MEMBERS, "write a script that implements an algorithm for custom clustering"
    )

    assert replacement is None


def test_code_agents_are_capped_per_run():
    messages = [
        HumanMessage(content="task"),
        AIMessage(content="step one", name="Coder"),
        AIMessage(content="step two", name="Coder"),
    ]

    replacement, reason = check_coder_routing_guard(messages, MEMBERS, "keep going")

    assert replacement == "report"
    assert "already run 2 times" in reason


def test_count_code_agent_routes_counts_only_code_agents():
    messages = [
        AIMessage(content="a", name="Coder"),
        AIMessage(content="b", name="Research"),
        AIMessage(content="c", name="ml"),
    ]

    assert count_code_agent_routes(messages) == 2


def test_no_specialist_suggested_for_an_unrelated_task():
    assert suggest_specialist_for_task("ponder the meaning of the data", MEMBERS) is None


def test_async_retry_recovers_from_a_transient_failure():
    import asyncio

    from bioagents.tools.tool_retry import ainvoke_with_retry

    calls = []
    results = [_error_payload("Read timed out"), _ok_payload("recovered")]

    async def fake_ainvoke(*_args, **_kwargs):
        calls.append(1)
        return results[len(calls) - 1]

    async def no_sleep(_delay):
        return None

    out = asyncio.run(ainvoke_with_retry(fake_ainvoke, {}, max_attempts=3, sleep=no_sleep))

    assert len(calls) == 2
    assert out["messages"][0].content == "recovered"


def test_tool_node_wrapper_retries_transient_failures():
    """The retry layer is wired into the node every agent's tools run through."""
    from bioagents.truncating_tool_node import make_truncating_tool_node

    node = make_truncating_tool_node([])
    calls = []
    results = [_error_payload("HTTP 502 bad gateway"), _ok_payload("data for AKT1")]

    def flaky_inner(*_args, **_kwargs):
        calls.append(1)
        return results[len(calls) - 1]

    node._inner.invoke = flaky_inner

    import bioagents.tools.tool_retry as retry_module

    original = retry_module.time.sleep
    retry_module.time.sleep = lambda _s: None
    try:
        out = node.invoke({"messages": []})
    finally:
        retry_module.time.sleep = original

    assert len(calls) == 2
    assert "data for AKT1" in out["messages"][0].content


def test_approval_node_also_retries():
    from bioagents.truncating_tool_node import make_approval_tool_node

    node = make_approval_tool_node([])
    calls = []
    results = [_error_payload("Connection reset by peer"), _ok_payload("second try")]

    def flaky_inner(*_args, **_kwargs):
        calls.append(1)
        return results[len(calls) - 1]

    node._inner.invoke = flaky_inner

    import bioagents.tools.tool_retry as retry_module

    original = retry_module.time.sleep
    retry_module.time.sleep = lambda _s: None
    try:
        out = node.invoke({"messages": []})
    finally:
        retry_module.time.sleep = original

    assert len(calls) == 2
    assert "second try" in out["messages"][0].content


def test_sandbox_unrecoverable_stops_further_code_delegation():
    from bioagents.agents.supervisor_helpers import check_coder_should_force_finish

    messages = [
        AIMessage(
            content="[CODER_STATUS: sandbox_unrecoverable] kernel gone",
            name="Coder",
        )
    ]

    finish, reason = check_coder_should_force_finish(messages)

    assert finish is True
    assert "sandbox" in reason.lower()


def test_cofold_tool_is_available_to_the_docking_agent():
    assert "run_boltz" in agent_tool_names()["docking"]


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("compute the ecfp4 fingerprints for these molecules", "analysis"),
        ("calculate QED and SA score for 100 molecules", "analysis"),
        ("run a tanimoto similarity search over the library", "analysis"),
        ("run a BLAST search for the candidate sequences", "genomics"),
        ("prepare the receptor and run docking for the ligands", "docking"),
        ("create a bar chart of the binding affinities", "visualization"),
    ],
)
def test_capability_phrasings_reach_the_agent_that_holds_the_tool(task, expected):
    """Tasks are phrased by capability, not by tool name; the guard must still match."""
    members = [
        "research",
        "analysis",
        "coder",
        "docking",
        "genomics",
        "visualization",
        "report",
    ]

    assert check_coder_routing_guard([], members, task)[0] == expected


@pytest.mark.parametrize(
    "task",
    [
        "make a t-SNE plot of the molecules coloured by LogP",
        "cluster the molecules by fingerprint similarity",
        "compute the RMSD between the predicted and crystal ligand",
        "write a script that implements our custom scoring algorithm",
    ],
)
def test_work_with_no_tool_still_reaches_the_coder(task):
    """The guard must not strand a task on a specialist that cannot perform it."""
    members = ["research", "analysis", "coder", "docking", "visualization", "report"]

    assert check_coder_routing_guard([], members, task)[0] is None


@pytest.mark.parametrize(
    ("tool_name", "minimum"),
    [
        ("BLAST_protein_search", 1200.0),
        ("vina_dock_ligands", 600.0),
        ("chembl_get_active_inhibitors", 120.0),
    ],
)
def test_long_running_tools_get_a_budget_they_can_finish_in(tool_name, minimum):
    """A 30 s cap reported healthy services as broken; the budget must fit the work."""
    from bioagents.tools.tool_universe import _tool_time_budget

    assert _tool_time_budget(tool_name) >= minimum


def test_ordinary_tools_keep_a_short_budget():
    from bioagents.tools.tool_universe import _tool_time_budget

    assert _tool_time_budget("UniProt_get_entry_by_accession") == 30.0


def test_tooluniverse_blast_is_served_by_the_local_implementation():
    """ToolUniverse's BLAST polls qblast forever; ours submits once and resumes by RID."""
    from bioagents.tools.tool_universe import _OVERRIDDEN_TOOLS

    module, attr = _OVERRIDDEN_TOOLS["BLAST_protein_search"]
    assert module == "bioagents.tools.genomics_tools"

    import importlib

    assert callable(getattr(importlib.import_module(module), attr))


def test_smol_tools_do_not_make_the_sandbox_pip_install_bioagents():
    """smolagents pip-installs each tool's parsed imports, so none may name `bioagents`."""
    from bioagents.tools.smol_tool_wrappers import (
        ToolUniverseExecuteTool,
        ToolUniverseSearchTool,
        get_esm_smol_tools,
    )

    for tool_obj in [ToolUniverseSearchTool(), ToolUniverseExecuteTool(), *get_esm_smol_tools()]:
        assert tool_obj.to_dict()["requirements"] == ["smolagents"]
