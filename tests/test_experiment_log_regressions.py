"""Regressions for the Task 1/2 console failures captured on 25 October 2026."""

import json
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from bioagents.agents import supervisor_agent
from bioagents.agents.supervisor_helpers import check_tool_builder_execution_success
from bioagents.benchmarks.models import STELLATask
from bioagents.benchmarks.runner import run_benchmark_task, run_use_case
from bioagents.benchmarks.use_case_models import ExperimentConfig, FailureMode, UseCase
from bioagents.tools import tool_builder_tools, tool_registry
from bioagents.tools.tool_builder_wrappers import (
    CustomToolExecuteTool,
    create_smolagent_tool_from_registry,
)
from bioagents.tools.tool_registry import ToolDefinition, ToolParameter, ToolRegistry


def interrupt_graph(payload):
    graph = StateGraph(dict)
    graph.add_node("download", lambda state: {"messages": [AIMessage(content="Structure saved")]})

    def ask(state):
        interrupt(payload)

    graph.add_node("ask", ask)
    graph.add_edge(START, "download")
    graph.add_edge("download", "ask")
    graph.add_edge("ask", END)
    return graph.compile()


@pytest.mark.parametrize("kind", ["clarification", "stuck", "api_key_request"])
def test_real_langgraph_interrupt_is_incomplete_and_preserves_output(kind):
    result = run_use_case(
        UseCase(id="task", name="interrupt regression", prompt="Compute a ligand ranking"),
        ExperimentConfig(name="test"),
        graph=interrupt_graph({"type": kind, "question": "Which ligand should be used?"}),
    )
    assert result.failure_mode == FailureMode.INCOMPLETE
    assert not result.workflow_completed
    assert "awaiting user input" in result.error_message
    assert "Which ligand" in result.error_message
    assert "tuple" not in result.error_message
    assert result.total_steps == 1
    assert result.agent_flow == ["download"]
    assert result.raw_output == "Structure saved"
    assert result.final_messages == ["Structure saved"]
    assert result.agent_steps[0].messages[0]["content"] == "Structure saved"


def test_legacy_benchmark_also_handles_interrupt_without_marking_success():
    result = run_benchmark_task(
        STELLATask(task_name="test", task_type="ranking", task_description="Rank", input={}),
        graph=interrupt_graph({"question": "Missing input"}),
    )
    assert not result.workflow_completed
    assert result.total_steps == 1
    assert result.raw_output == "Structure saved"
    assert "Missing input" in result.error_message


def test_natural_graph_completion_and_task_output_directory():
    output = "generated_artifacts/experiments/test/default/001_task"

    def work(state):
        assert state["output_dir"] == output
        return {"messages": [AIMessage(content="Ranking saved")]}

    graph = StateGraph(dict)
    graph.add_node("work", work)
    graph.add_edge(START, "work")
    graph.add_edge("work", END)
    result = run_use_case(
        UseCase(id="task", name="test", prompt="Rank ligands"),
        ExperimentConfig(name="test", extra={"output_dirs": {"task": output}}),
        graph=graph.compile(),
    )
    assert result.workflow_completed
    assert result.failure_mode == FailureMode.COMPLETED
    assert result.raw_output == "Ranking saved"


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("bad_state", [("unexpected",), ["unexpected"], "unexpected", 0])
def test_unexpected_node_state_is_incomplete_and_preserves_previous_output(
    legacy, bad_state, caplog
):
    graph = Mock()
    graph.stream.return_value = iter(
        [{"download": {"messages": [AIMessage(content="Saved structure")]}}, {"custom": bad_state}]
    )
    if legacy:
        result = run_benchmark_task(
            STELLATask(task_name="test", task_type="ranking", task_description="Rank", input={}),
            graph=graph,
        )
    else:
        result = run_use_case(UseCase(id="task", name="test", prompt="Rank ligands"), graph=graph)
        assert result.failure_mode == FailureMode.INCOMPLETE
        assert result.agent_steps[-1].agent == "custom"
    assert not result.workflow_completed
    assert result.raw_output == "Saved structure"
    assert "custom" in result.error_message
    assert type(bad_state).__name__ in result.error_message
    assert "expected dict or None" in caplog.text


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("bad_update", [None, ("node", {}), {}])
def test_invalid_outer_stream_update_preserves_previous_results(legacy, bad_update):
    graph = Mock()
    graph.stream.return_value = iter(
        [{"download": {"messages": [AIMessage(content="Saved structure")]}}, bad_update]
    )
    if legacy:
        result = run_benchmark_task(
            STELLATask(task_name="test", task_type="ranking", task_description="Rank", input={}),
            graph=graph,
        )
    else:
        result = run_use_case(UseCase(id="task", name="test", prompt="Rank ligands"), graph=graph)
        assert result.failure_mode == FailureMode.INCOMPLETE
    assert not result.workflow_completed
    assert result.raw_output == "Saved structure"
    assert "Unexpected graph stream update" in result.error_message


@pytest.mark.parametrize("legacy", [False, True])
def test_none_node_update_is_an_empty_step_and_does_not_skip_completion_checks(legacy):
    graph = Mock()
    graph.stream.return_value = iter(
        [{"empty": None}, {"answer": {"messages": [AIMessage(content="Ranking saved")]}}]
    )
    if legacy:
        result = run_benchmark_task(
            STELLATask(task_name="test", task_type="ranking", task_description="Rank", input={}),
            graph=graph,
        )
    else:
        result = run_use_case(UseCase(id="task", name="test", prompt="Rank ligands"), graph=graph)
    assert result.workflow_completed
    assert result.total_steps == 2
    assert result.raw_output == "Ranking saved"


@pytest.fixture
def custom_registry(tmp_path, monkeypatch):
    registry = ToolRegistry(tools_dir=tmp_path)
    definition = ToolDefinition(
        name="ligand_query",
        description="Query a ligand target",
        category="database",
        parameters=[
            ToolParameter("target_name", "string", "Target", required=True),
            ToolParameter("n_molecules", "integer", "Count", required=False, default=10),
        ],
        return_type="dict",
        return_description="Query result",
        code=(
            "def ligand_query(target_name: str, n_molecules: int = 10):\n"
            '    if target_name == "raise": raise RuntimeError("query failed")\n'
            '    return {"status": target_name, "n_molecules": n_molecules}\n'
        ),
    )
    assert registry.register_tool(definition)
    monkeypatch.setattr(tool_builder_tools, "get_registry", lambda: registry)
    return registry


@pytest.mark.parametrize("operation", ["list", "search", "code"])
def test_tool_discovery_includes_required_parameters_defaults_and_invocation(
    custom_registry, operation
):
    if operation == "list":
        entry = json.loads(tool_builder_tools.list_custom_tools.invoke({}))["tools"][0]
    elif operation == "search":
        entry = json.loads(tool_builder_tools.search_custom_tools.invoke({"query": "ligand"}))[
            "results"
        ][0]
    else:
        entry = json.loads(tool_builder_tools.get_tool_code.invoke({"tool_name": "ligand_query"}))
    assert entry["execution_tool"] == "execute_custom_tool"
    assert entry["schema"]["required"] == ["target_name"]
    parameters = {p["name"]: p for p in entry["parameters"]}
    assert parameters["target_name"]["required"] is True
    assert parameters["n_molecules"]["default"] == 10


@pytest.mark.parametrize("args", [{"target": "AKT1"}, {"target_name": "AKT1", "n": 10}, {}, []])
def test_wrong_parameters_return_schema_before_loading_or_dependency_install(custom_registry, args):
    with (
        patch.object(custom_registry, "load_tool_function") as load,
        patch.object(tool_builder_tools.subprocess, "run") as process,
    ):
        result = json.loads(
            tool_builder_tools.execute_custom_tool.invoke(
                {"tool_name": "ligand_query", "arguments_json": json.dumps(args)}
            )
        )
    assert result["status"] == "error"
    assert result["schema"]["required"] == ["target_name"]
    assert "n_molecules" in result["schema"]["properties"]
    load.assert_not_called()
    process.assert_not_called()


@pytest.mark.parametrize(
    "status",
    [
        "error",
        "partial",
        "success",
        "capability_unavailable",
        "timeout",
        "failed",
        "pending",
        "skipped",
        "dry_run",
        "check_only",
        "manual_execution_required",
        "use_google_colab",
        "unexpected_status",
    ],
)
def test_custom_tool_result_status_reaches_supervisor(custom_registry, status):
    result = json.loads(
        tool_builder_tools.execute_custom_tool.invoke(
            {"tool_name": "ligand_query", "arguments_json": json.dumps({"target_name": status})}
        )
    )
    assert result["status"] == status
    assert result["result"]["status"] == status
    assert check_tool_builder_execution_success(
        [ToolMessage(content=json.dumps(result), tool_call_id="query")]
    )[0] is (status == "success")


@pytest.mark.parametrize(
    "status", ["error", "partial", "capability_unavailable", "dry_run", "check_only", "pending"]
)
def test_old_success_envelope_does_not_hide_a_tool_failure(status):
    message = ToolMessage(
        content=json.dumps({"status": "success", "tool": "query", "result": {"status": status}}),
        tool_call_id="query",
    )
    assert not check_tool_builder_execution_success([message])[0]


def test_validation_without_arguments_never_executes_required_argument_tool(custom_registry):
    with patch.object(custom_registry, "load_tool_function", return_value=Mock()) as load:
        result = json.loads(
            tool_builder_tools.validate_custom_tool.invoke({"tool_name": "ligand_query"})
        )
    assert result["status"] == "success"
    load.return_value.assert_not_called()


@pytest.mark.parametrize(
    "args",
    [{}, []]
    + [
        {"target_name": status}
        for status in ("raise", "error", "partial", "capability_unavailable", "dry_run", "pending")
    ],
)
def test_explicit_invalid_or_failed_validation_never_passes(custom_registry, args):
    result = json.loads(
        tool_builder_tools.validate_custom_tool.invoke(
            {"tool_name": "ligand_query", "test_args_json": json.dumps(args)}
        )
    )
    assert result["status"] == "error"
    assert not custom_registry.get_tool("ligand_query").validated
    assert custom_registry.get_tool("ligand_query").validation_error


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"status": "capability_unavailable"}, "capability_unavailable"),
        ('{"status": "error", "error_type": "capability_unavailable"}', "error"),
        ('{"status": "dry_run"}', "dry_run"),
        ({"status": "success", "result": '{"status": "partial"}'}, "partial"),
        ({"status": "success", "success": False}, "error"),
        ({"status": "success", "error": "dependency missing"}, "error"),
        ({"status": "success", "available": False}, "capability_unavailable"),
        ({"status": "success", "performed_any_computation": False}, "not_performed"),
        ({"status": "success", "value": 42}, "success"),
        ("plain legacy result", "success"),
    ],
)
def test_execution_wrappers_validation_and_supervisor_agree_on_outcome(
    custom_registry, monkeypatch, payload, expected
):
    monkeypatch.setattr(tool_registry, "get_registry", lambda: custom_registry)
    monkeypatch.setattr(
        custom_registry, "load_tool_function", lambda *args, **kwargs: lambda **_: payload
    )
    langchain_result = json.loads(
        tool_builder_tools.execute_custom_tool.invoke(
            {"tool_name": "ligand_query", "arguments_json": '{"target_name": "unused"}'}
        )
    )
    coder_result = json.loads(CustomToolExecuteTool().forward("ligand_query"))
    for result in (langchain_result, coder_result):
        assert result["status"] == expected
        assert result["result"] == payload
        # Even an old success envelope must not conceal a structured failure.
        result["status"] = "success"
        assert check_tool_builder_execution_success(
            [ToolMessage(content=json.dumps(result), tool_call_id="query")]
        )[0] is (expected == "success")
    assert custom_registry.validate_tool("ligand_query", {}) is (expected == "success")
    definition = custom_registry.get_tool("ligand_query")
    assert definition.validated is (expected == "success")
    if expected != "success":
        assert expected in definition.validation_error


@pytest.mark.parametrize("kind", ["missing_tool", "exception", "invalid_args", "pending"])
@pytest.mark.parametrize("wrapper", ["langchain", "coder"])
def test_latest_execution_failure_prevents_reusing_an_old_success(
    custom_registry, monkeypatch, kind, wrapper
):
    monkeypatch.setattr(tool_registry, "get_registry", lambda: custom_registry)
    name = "missing" if kind == "missing_tool" else "ligand_query"
    args = {"target_name": "raise" if kind == "exception" else "pending"}
    if kind == "invalid_args":
        args = {}
    if wrapper == "langchain":
        response = tool_builder_tools.execute_custom_tool.invoke(
            {"tool_name": name, "arguments_json": json.dumps(args)}
        )
    else:
        response = CustomToolExecuteTool().forward(name, args)
    assert json.loads(response)["status"] != "success"
    older = ToolMessage(
        content='{"status":"success","tool":"old_query","result":{"value":1}}',
        tool_call_id="old",
    )
    latest = ToolMessage(content=response, tool_call_id="latest")
    assert check_tool_builder_execution_success([older, latest]) == (False, "")
    assert check_tool_builder_execution_success([latest, older]) == (True, "old_query")


def test_direct_coder_tool_uses_function_defaults_and_rejects_invalid_arguments(
    custom_registry, monkeypatch
):
    monkeypatch.setattr(tool_registry, "get_registry", lambda: custom_registry)
    tool = create_smolagent_tool_from_registry("ligand_query")
    assert tool.output_type == "object"
    assert tool(target_name="success") == {"status": "success", "n_molecules": 10}
    assert tool(target_name="success", n_molecules=3)["n_molecules"] == 3
    with pytest.raises(TypeError):
        tool.forward(unknown_argument="value")


def successful_download_messages():
    return [
        HumanMessage(
            content="Retrieve inhibitors, dock them against the receptor, and rank interactions"
        ),
        ToolMessage(
            content=json.dumps(
                {
                    "status": "success",
                    "tool": "chembl_get_active_inhibitors",
                    "result": {"status": "success"},
                }
            ),
            tool_call_id="query",
        ),
        AIMessage(
            name="ToolBuilder",
            content="[MAX_TOOL_ROUNDS] Data retrieved; docking and interaction analysis remain.",
        ),
    ]


def test_successful_query_does_not_force_supervisor_to_skip_docking(monkeypatch):
    chain = Mock()
    chain.invoke.return_value = supervisor_agent.RouteResponse(
        next_agent="docking", reasoning="Retrieved ligands still need docking"
    )
    prompt = Mock()
    prompt.__or__ = Mock(return_value=chain)
    monkeypatch.setattr(supervisor_agent, "get_llm", Mock())
    monkeypatch.setattr(supervisor_agent, "_agent_tool_inventory", lambda: "")
    monkeypatch.setattr(supervisor_agent, "describe_workspace_artifacts", lambda: "")
    with patch.object(supervisor_agent.ChatPromptTemplate, "from_messages", return_value=prompt):
        node = supervisor_agent.create_supervisor_agent(["docking", "tool_builder", "report"])
    result = node({"messages": successful_download_messages()})
    assert result["next"] == "docking"
    chain.invoke.assert_called_once()
    assert not any("[EXECUTION_SUCCESS]" in str(m.content) for m in result["messages"])


def test_routing_loop_does_not_treat_a_query_as_completed_analysis(monkeypatch):
    monkeypatch.setattr(supervisor_agent, "get_llm", Mock())
    monkeypatch.setattr(
        supervisor_agent, "check_for_repeated_routing", lambda _: (True, "ToolBuilder")
    )
    monkeypatch.setattr(supervisor_agent, "check_tool_builder_success", lambda _: (False, ""))
    node = supervisor_agent.create_supervisor_agent(["tool_builder", "analysis", "report"])
    result = node({"messages": successful_download_messages()})
    assert result["next"] == "analysis"
    assert not any("task is complete" in str(m.content) for m in result["messages"])


def test_success_claims_in_prose_or_old_markers_are_not_execution_evidence():
    assert not check_tool_builder_execution_success(
        [AIMessage(name="ToolBuilder", content="I have successfully executed the tool")]
    )[0]
    assert not check_tool_builder_execution_success(
        [
            SystemMessage(
                content="[EXECUTION_SUCCESS] ToolBuilder successfully executed tool 'query'"
            )
        ]
    )[0]
