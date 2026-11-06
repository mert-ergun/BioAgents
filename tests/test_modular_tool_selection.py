"""Offline selection contracts: metadata, named adapters and denied side effects."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph

from bioagents.prompts.tool_context import apply_tool_context, selected_system_messages
from bioagents.tools import tool_selection as selection_module
from bioagents.tools.selected_runtime import (
    ROLES,
    execute_selected,
    selected_code_tools,
    selected_tool_map,
    tools_for_agent,
)
from bioagents.tools.tool_catalog import describe_tool, sdk_entries
from bioagents.tools.tool_selection import (
    ToolSelection,
    get_selection,
    parse_selection,
    read_selection,
    selection_scope,
)


@pytest.fixture
def selection():
    data = json.loads(Path(selection_module.DEFAULT_FILE).read_text())
    with selection_scope(parse_selection(data)) as _:
        yield get_selection()


def test_exact_operator_profile(selection):
    assert len(selection.local_tools) == 11
    assert len(selection.tool_universe_tools) == 9
    assert selection.names[-1] == "ZINC_search_by_smiles"
    assert dict(selection.aliases) == {"genmol": "NvidiaNIM_genmol"}
    names = {tool.name for tools in selected_tool_map(selection).values() for tool in tools}
    assert names == set(selection.names)
    assert "tool_universe_call_tool" not in names
    assert "run_boltz" not in names


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        {},
        {"local_tools": ["x", "x"], "tool_universe_tools": []},
        {"local_tools": ["not a name"], "tool_universe_tools": []},
        {"local_tools": [], "tool_universe_tools": [], "unexpected": True},
        {
            "local_tools": [],
            "tool_universe_tools": [],
            "tool_universe_aliases": {"extra": "backend"},
        },
    ],
)
def test_invalid_profiles_fail_closed(data):
    with pytest.raises(ValueError):
        parse_selection(data)


def test_operator_mode_and_bad_json_never_fall_back_to_all(tmp_path, monkeypatch):
    path = tmp_path / "tools.json"
    path.write_text("invalid JSON")
    monkeypatch.setenv("BIOAGENTS_TOOL_SELECTION_FILE", str(path))
    monkeypatch.setenv("BIOAGENTS_TOOL_MODE", "selected")
    monkeypatch.delenv("BIOAGENTS_TOOL_SELECTION_SNAPSHOT", raising=False)
    with pytest.raises(ValueError):
        read_selection()
    monkeypatch.setenv("BIOAGENTS_TOOL_MODE", "typo")
    with pytest.raises(ValueError):
        read_selection()
    monkeypatch.setenv("BIOAGENTS_TOOL_MODE", "all")
    assert read_selection().mode == "all"


def test_unknown_configured_name_is_reported_before_binding():
    selection = ToolSelection("selected", ("unregistered_fixture",))
    with pytest.raises(ValueError, match="Unknown selected local tool"):
        selected_tool_map(selection)


def test_name_only_addition_preserves_schema_and_executes_local_io(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"local_tools": ["write_local_file"], "tool_universe_tools": []}))
    selection = parse_selection(json.loads(path.read_text()))
    with selection_scope(selection):
        tools = tools_for_agent("research")
        assert [tool.name for tool in tools] == ["write_local_file"]
        metadata = describe_tool("write_local_file", selection)
        assert tools[0].args_schema == metadata["schema"]
        output = tmp_path / "actual.txt"
        result = tools[0].invoke({"file_path": str(output), "content": "fixture output"})
        assert output.read_text() == "fixture output", result
    # Another run cannot use a callable cached from the wider profile.
    with selection_scope(ToolSelection("selected")):
        blocked = json.loads(tools[0].invoke({"file_path": str(output), "content": "unexpected"}))
        assert blocked["error_code"] == "tool_not_available"
        assert output.read_text() == "fixture output"


def test_sdk_adapter_retains_original_metadata_without_running_a_service(selection, monkeypatch):
    from bioagents.tools import tool_universe

    client = Mock()
    client.run.return_value = {"fixture": True}
    factory = Mock(return_value=client)
    wrapper = tool_universe.ToolUniverseWrapper(tool_factory=factory)
    monkeypatch.setattr(tool_universe, "DEFAULT_WRAPPER", wrapper)
    name = "ChEMBL_search_molecules"
    metadata = describe_tool(name, selection)
    assert metadata["schema"] == sdk_entries()[name]["parameter"]
    result = execute_selected(name, {"query": "fixture"}, selection)
    assert json.loads(result)["result"] == {"fixture": True}
    client.run.assert_called_once_with({"name": name, "arguments": {"query": "fixture"}})
    loaded = client.load_tools.call_args.kwargs["include_tools"]
    assert set(loaded) == set(selection.tool_universe_tools) | {"NvidiaNIM_genmol"}


def test_alias_routes_to_only_its_declared_sdk_backend(selection, monkeypatch):
    from bioagents.tools import tool_universe

    client = Mock()
    client.run.return_value = {"fixture": True}
    wrapper = tool_universe.ToolUniverseWrapper(tool_factory=Mock(return_value=client))
    monkeypatch.setattr(tool_universe, "DEFAULT_WRAPPER", wrapper)
    execute_selected("genmol", {"smiles": "fixture"}, selection)
    assert client.run.call_args.args[0]["name"] == "NvidiaNIM_genmol"
    blocked = json.loads(wrapper.execute_tool("NvidiaNIM_genmol", {}))
    assert blocked["error_code"] == "tool_not_available"
    assert client.run.call_count == 1


@pytest.mark.parametrize(
    "name", ["write_local_file", "analyze_pdb_chain", "PubMed_search_articles", "invented_tool"]
)
def test_gateway_blocks_disabled_routes_before_any_implementation(selection, monkeypatch, name):
    from bioagents.tools import tool_universe

    factory = Mock(side_effect=AssertionError("Client must not be created"))
    wrapper = tool_universe.ToolUniverseWrapper(tool_factory=factory)
    local = Mock(side_effect=AssertionError("Implementation must not be resolved"))
    monkeypatch.setattr(tool_universe, "get_local_tool", local)
    result = json.loads(wrapper.execute_tool(name, "invalid arguments"))
    assert result["error_code"] == "tool_not_available"
    assert result["execution_state"] == "not_started"
    assert result["retryable"] is False
    assert "introduction list" in result["message"]
    factory.assert_not_called()
    local.assert_not_called()


def test_discovery_is_offline_and_only_returns_selected_names(selection):
    from bioagents.tools.tool_universe import ToolUniverseWrapper

    factory = Mock(side_effect=AssertionError("Discovery must not create a client"))
    wrapper = ToolUniverseWrapper(tool_factory=factory)
    result = json.loads(wrapper.find_tools("protein structure molecules", limit=100))
    assert {item["name"] for item in result["results"]} <= set(selection.names)
    factory.assert_not_called()


@pytest.mark.parametrize("role", ROLES)
def test_standalone_factory_binds_the_canonical_selected_schema(selection, monkeypatch, role):
    module = importlib.import_module(f"bioagents.agents.{role}_agent")
    llm = Mock()
    monkeypatch.setattr(module, "get_llm", Mock(return_value=llm))
    node = getattr(module, f"create_{role}_agent")()
    expected = [tool.name for tool in selected_tool_map(selection)[role]]
    if expected:
        assert [t.name for t in llm.bind_tools.call_args.args[0]] == expected
    else:
        llm.bind_tools.assert_not_called()
    assert callable(node)


@pytest.mark.parametrize("role", ["coder", "ml", "dl"])
def test_code_agents_expose_only_named_selected_tools(selection, monkeypatch, role):
    module = importlib.import_module(f"bioagents.agents.{role}_agent")
    constructor = Mock()
    monkeypatch.setattr(module, "CodeAgent", constructor)
    monkeypatch.setattr(module, "get_llm", Mock())
    monkeypatch.setattr(module, "LangChainModelAdapter", Mock())
    monkeypatch.setattr(module, "create_executor", Mock())
    getattr(module, f"create_{role}_agent")()
    assert {t.name for t in constructor.call_args.kwargs["tools"]} == set(selection.names)
    assert "score_mutations_esm" not in constructor.call_args.kwargs["instructions"]


def test_named_code_tools_serialize_without_project_pip_install(selection):
    from smolagents.remote_executors import get_tools_definition_code

    tools = selected_code_tools(selection)
    assert {t.name for t in tools} == set(selection.names)
    for selected in tools:
        assert "bioagents" not in selected.to_dict()["requirements"]
    source = get_tools_definition_code({t.name: t for t in tools})
    namespace = {}
    exec(compile(source, "<selected-tools-fixture>", "exec"), namespace)
    assert all(callable(namespace[name]) for name in selection.names)


def test_prompt_refresh_is_idempotent_and_preserves_non_tool_rules(selection):
    text = "Keep timeout evidence. Use run_boltz. Never fabricate a result."
    first = apply_tool_context(text)
    second = apply_tool_context(first)
    assert first == second
    assert "run_boltz" not in first
    assert "Keep timeout evidence" in first and "Never fabricate" in first
    assert first.count("[ACTIVE TOOL CATALOGUE]") == 1
    for name in selection.names:
        assert name in first
    user = HumanMessage(content="The previous tool was run_boltz")
    refreshed = selected_system_messages([SystemMessage(content=text), user])
    assert refreshed[-1] is user


def test_toolnode_returns_actionable_rejection_for_hallucinated_names(selection):
    from bioagents.truncating_tool_node import make_approval_tool_node

    calls = []

    @tool
    def fixture_side_effect(value: str) -> str:
        """Record a harmless fixture call."""
        calls.append(value)
        return value

    builder = StateGraph(MessagesState)
    builder.add_node("tools", make_approval_tool_node([fixture_side_effect]))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    request = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "fixture_side_effect",
                        "args": {"value": "must not run"},
                        "id": "fixture-call",
                    }
                ],
            )
        ]
    }
    message = builder.compile().invoke(request)["messages"][-1]
    result = json.loads(message.content)
    assert result["error_code"] == "tool_not_available"
    assert message.tool_call_id == "fixture-call"
    assert calls == []


@pytest.mark.parametrize(
    "code",
    [
        "import tooluniverse",
        "from bioagents.tools import file_tools",
        "import bioagents.tools as tools",
        "import importlib; importlib.import_module('tooluniverse')",
    ],
)
def test_generated_cells_cannot_import_tool_implementations(selection, code):
    from bioagents.sandbox.tool_access import check_generated_tool_access

    with pytest.raises(ValueError, match="introduction list"):
        check_generated_tool_access(code)


def test_ordinary_python_imports_are_not_registered_service_calls(selection):
    from bioagents.sandbox.tool_access import check_generated_tool_access

    check_generated_tool_access("import math; from pathlib import Path; print(math.sqrt(4))")


def test_registry_hides_disabled_metadata_and_rechecks_cached_function(tmp_path, monkeypatch):
    from bioagents.tools import tool_catalog, tool_registry
    from bioagents.tools.tool_registry import ToolDefinition, ToolParameter, ToolRegistry

    registry = ToolRegistry(tmp_path / "custom_tools")
    definition = ToolDefinition(
        name="fixture_echo",
        description="Echo a fixture value.",
        category="fixture",
        parameters=[ToolParameter("value", "string", "Fixture input")],
        return_type="string",
        return_description="Fixture result",
        code="calls = 0\ndef fixture_echo(value):\n    global calls\n    calls += 1\n    return value\n",
    )
    with selection_scope(ToolSelection("all")):
        assert registry.register_tool(definition)
    monkeypatch.setattr(tool_registry, "get_registry", lambda: registry)
    monkeypatch.setattr(tool_catalog, "TOOLS_DIR", tmp_path)
    tool_catalog.local_index.cache_clear()
    try:
        selected = ToolSelection("selected", ("fixture_echo",))
        with selection_scope(selected):
            metadata = describe_tool("fixture_echo", selected)
            assert metadata["description"] == definition.description
            adapter = tools_for_agent("research")[0]
            assert adapter.invoke({"value": "fixture"}) == "fixture"
            cached = registry.load_tool_function("fixture_echo")
        with selection_scope(ToolSelection("selected")):
            assert registry.get_tool("fixture_echo") is None
            assert registry.list_tools() == []
            assert registry.search_tools("fixture") == []
            assert registry.load_tool_function("fixture_echo") is None
            assert cached("must not run")["error_code"] == "tool_not_available"
            assert registry._loaded_modules["fixture_echo"].calls == 1
    finally:
        tool_catalog.local_index.cache_clear()


def test_sdk_client_is_recreated_for_a_different_selection():
    from bioagents.tools.tool_universe import ToolUniverseWrapper

    clients = [Mock(), Mock()]
    factory = Mock(side_effect=clients)
    wrapper = ToolUniverseWrapper(tool_factory=factory)
    with selection_scope(ToolSelection("selected", (), ("ChEMBL_search_molecules",))):
        assert wrapper._ensure_client() is clients[0]
        assert wrapper._ensure_client() is clients[0]
    with selection_scope(ToolSelection("selected", (), ("ClinVar_search_variants",))):
        assert wrapper._ensure_client() is clients[1]
    assert factory.call_count == 2
    clients[1].load_tools.assert_called_once_with(include_tools=["ClinVar_search_variants"])


def test_a_bound_node_keeps_its_snapshot_and_restores_the_caller():
    from bioagents.tools.selected_runtime import bind_agent_node

    original = ToolSelection("selected", ("fixture_first",))
    with selection_scope(original):
        node = bind_agent_node(lambda: get_selection().names)
    with selection_scope(ToolSelection("selected")):
        assert node() == ("fixture_first",)
        assert get_selection().names == ()


def test_mixed_batch_is_not_partially_executed():
    from bioagents.truncating_tool_node import make_approval_tool_node

    calls = []

    @tool
    def fixture_allowed(value: str) -> str:
        """Record a fixture value."""
        calls.append(value)
        return value

    selected = ToolSelection("selected", ("fixture_allowed",))
    with selection_scope(selected):
        node = make_approval_tool_node([fixture_allowed])
    request = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "fixture_allowed", "args": {"value": "must not run"}, "id": "allowed"},
                    {"name": "fixture_unknown", "args": {}, "id": "unknown"},
                ],
            )
        ]
    }
    messages = node.invoke(request)["messages"]
    assert {m.tool_call_id for m in messages} == {"allowed", "unknown"}
    assert json.loads(messages[-1].content)["error_code"] == "tool_not_available"
    assert calls == []


def test_serialized_code_adapter_executes_name_only_file_tool(tmp_path):
    from smolagents.remote_executors import get_tools_definition_code

    selected = parse_selection({"local_tools": ["write_local_file"], "tool_universe_tools": []})
    with selection_scope(selected):
        tools = selected_code_tools(selected)
        source = get_tools_definition_code({t.name: t for t in tools})
        namespace = {}
        exec(compile(source, "<selected-file-fixture>", "exec"), namespace)
        output = tmp_path / "serialized.txt"
        namespace["write_local_file"](file_path=str(output), content="serialized fixture")
        assert output.read_text() == "serialized fixture"
    with selection_scope(ToolSelection("selected")):
        rejected = json.loads(
            namespace["write_local_file"](file_path=str(output), content="unexpected")
        )
        assert rejected["error_code"] == "tool_not_available"
        assert output.read_text() == "serialized fixture"


def test_conflicting_or_duplicate_snapshots_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOAGENTS_TOOL_MODE", "selected")
    monkeypatch.setenv("BIOAGENTS_TOOL_SELECTION_SNAPSHOT", '{"mode":"all"}')
    with pytest.raises(ValueError):
        read_selection()
    monkeypatch.delenv("BIOAGENTS_TOOL_SELECTION_SNAPSHOT")
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"local_tools":[],"local_tools":["write_local_file"],"tool_universe_tools":[]}'
    )
    monkeypatch.setenv("BIOAGENTS_TOOL_SELECTION_FILE", str(path))
    with pytest.raises(ValueError, match="Duplicate"):
        read_selection()


@pytest.mark.parametrize("factory_name", ["make_truncating_tool_node", "make_approval_tool_node"])
@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("input_shape", ["state", "messages", "calls"])
def test_all_toolnode_entrypoints_reject_disabled_calls(factory_name, async_call, input_shape):
    import asyncio

    from bioagents import truncating_tool_node

    calls = []

    @tool
    def fixture_disabled(value: str) -> str:
        """Record a fixture side effect."""
        calls.append(value)
        return value

    with selection_scope(ToolSelection("selected")):
        node = getattr(truncating_tool_node, factory_name)([fixture_disabled])
    request_call = {
        "name": "fixture_disabled",
        "args": {"value": "must not run"},
        "id": "disabled",
        "type": "tool_call",
    }
    message = AIMessage(content="", tool_calls=[request_call])
    request = (
        {"messages": [message]}
        if input_shape == "state"
        else ([message] if input_shape == "messages" else [request_call])
    )
    output = asyncio.run(node.ainvoke(request)) if async_call else node.invoke(request)
    result = json.loads(output["messages"][0].content)
    assert result["error_code"] == "tool_not_available"
    assert output["messages"][0].tool_call_id == "disabled"
    assert calls == []


def test_supervisor_inventory_refreshes_when_the_session_changes():
    from bioagents.agents.supervisor_agent import _agent_tool_inventory

    with selection_scope(ToolSelection("selected", ("write_local_file",))):
        first = _agent_tool_inventory()
        assert "write_local_file" in first
        assert "- coder: write_local_file" in first
    with selection_scope(ToolSelection("selected")):
        second = _agent_tool_inventory()
        assert "write_local_file" not in second
        assert "- coder: " in second


def test_all_mode_deduplicates_extra_tools():
    from bioagents.tools.tool_universe import tool_universe_call_tool

    with selection_scope(ToolSelection("all")):
        tools = tools_for_agent("literature", extra_tools=[tool_universe_call_tool])
        names = [t.name for t in tools]
        assert names.count("tool_universe_call_tool") == 1
        assert len(names) == len(set(names))


def test_lazy_container_keeps_selection_and_forwards_only_required_credentials(monkeypatch):
    from bioagents.sandbox import coder_executor, dependency_events, resilient_executor
    from bioagents.tools import tool_catalog

    captured = {}

    def construct(**kwargs):
        captured.update(kwargs)
        captured["selection"] = get_selection()
        return Mock()

    monkeypatch.setenv("USE_LOCAL_EXECUTOR", "false")
    monkeypatch.delenv("BIOAGENTS_RUN_ID", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "fixture-key")
    monkeypatch.setenv("UNSELECTED_API_KEY", "fixture-unused")
    monkeypatch.setattr(coder_executor, "_existing_image_id", lambda _: "fixture-image")
    monkeypatch.setattr(coder_executor, "build_dockerfile", lambda: "FROM fixture")
    monkeypatch.setattr(coder_executor, "find_available_port", lambda: 8888)
    monkeypatch.setattr(resilient_executor, "ResilientDockerExecutor", construct)
    monkeypatch.setattr(dependency_events, "current_audit", lambda: None)
    monkeypatch.setattr(
        tool_catalog,
        "sdk_entries",
        lambda: {
            "NvidiaNIM_genmol": {"required_api_keys": ["NVIDIA_API_KEY"]},
            "fixture_unused": {"required_api_keys": ["UNSELECTED_API_KEY"]},
        },
    )
    selected = ToolSelection("selected", ("genmol",), (), (("genmol", "NvidiaNIM_genmol"),))
    with selection_scope(selected):
        executor = coder_executor.create_executor("coder", [])
        assert not executor.started
    with selection_scope(ToolSelection("selected")):
        executor._ensure()
        assert get_selection().names == ()
    assert captured["selection"] == selected
    environment = captured["container_run_kwargs"]["environment"]
    assert environment["BIOAGENTS_TOOL_MODE"] == "selected"
    assert parse_selection(json.loads(environment["BIOAGENTS_TOOL_SELECTION_SNAPSHOT"])) == selected
    assert environment["NVIDIA_API_KEY"] == "fixture-key"
    assert "UNSELECTED_API_KEY" not in environment


def test_supervisor_uses_frozen_prompt_catalogue_without_a_live_model(monkeypatch):
    from langchain_core.runnables import RunnableLambda

    from bioagents.agents import supervisor_agent

    received = []

    def route(messages):
        received.extend(messages)
        return supervisor_agent.RouteResponse(next_agent="FINISH", reasoning="Fixture complete")

    llm = Mock()
    llm.with_structured_output.return_value = RunnableLambda(route)
    monkeypatch.setattr(supervisor_agent, "get_llm", lambda **_: llm)
    monkeypatch.setattr(supervisor_agent, "describe_workspace_artifacts", lambda: "")
    monkeypatch.setattr(supervisor_agent, "check_query_ambiguity", lambda _: None)
    selected = ToolSelection("selected", ("write_local_file",))
    with selection_scope(selected):
        node = supervisor_agent.create_supervisor_agent(["research", "report"])
    with selection_scope(ToolSelection("selected")):
        node({"messages": [HumanMessage(content="Summarize the available file operations.")]})
        assert get_selection().names == ()
    system = "\n".join(str(m.content) for m in received if isinstance(m, SystemMessage))
    assert "write_local_file" in system
    assert "run_boltz" not in system
    assert system.count("[ACTIVE TOOL CATALOGUE]") == 1


def test_plain_llm_retry_path_receives_only_active_introductions():
    from bioagents.agents.helpers import invoke_with_retry

    llm = Mock()
    llm.invoke.return_value = AIMessage(content="Fixture response")
    user = HumanMessage(content="Preserve this previous run_boltz reference.")
    with selection_scope(ToolSelection("selected", ("write_local_file",))):
        invoke_with_retry(
            "Fixture", llm, [SystemMessage(content="Use run_boltz for this role."), user]
        )
    sent = llm.invoke.call_args.args[0]
    assert "run_boltz" not in sent[0].content
    assert "write_local_file" in sent[0].content
    assert sent[-1] is user


@pytest.mark.parametrize(
    "role",
    [
        "critic",
        "planner",
        "prompt_optimizer",
        "result_checker",
        "tool_validator",
        "summary",
        "report",
    ],
)
def test_plain_llm_agents_keep_their_selection_snapshot(monkeypatch, role):
    module = importlib.import_module(f"bioagents.agents.{role}_agent")
    observed = []

    def respond(*args, **kwargs):
        observed.append(get_selection())
        return {"messages": [AIMessage(content="Fixture response")]}

    monkeypatch.setattr(module, "get_llm", Mock())
    monkeypatch.setattr(module, "invoke_with_retry", respond)
    selected = ToolSelection("selected", ("write_local_file",))
    with selection_scope(selected):
        node = getattr(module, f"create_{role}_agent")()
    with selection_scope(ToolSelection("selected")):
        node({"messages": [HumanMessage(content="Summarize file operations.")], "memory": {}})
        assert get_selection().names == ()
    assert observed == [selected]


def test_new_graph_construction_reads_once_and_restores_the_caller(monkeypatch):
    from bioagents.tools.selected_runtime import bind_agent_node, with_new_tool_selection

    first = ToolSelection("selected", ("write_local_file",))
    second = ToolSelection("selected")
    reader = Mock(side_effect=[first, second])
    monkeypatch.setattr(selection_module, "read_selection", reader)
    monkeypatch.setattr(selection_module, "_PROCESS_SELECTION", None)

    @with_new_tool_selection
    def build():
        return bind_agent_node(lambda: get_selection())

    caller = ToolSelection("selected", ("fixture_caller",))
    with selection_scope(caller):
        first_node = build()
        second_node = build()
        assert get_selection() == caller
        assert first_node() == first
        assert second_node() == second
    assert reader.call_count == 2


def _isolated_ui_registry():
    import ast

    source = Path("frontend/server.py").read_text()
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_agent_registry_with_live_tools"
    )
    namespace = {
        "AGENT_REGISTRY": {
            role: {"tools": ["run_boltz"], "description": "Use run_boltz."}
            for role in ("research", "analysis", "coder", "ml", "dl", "fixture_unknown_role")
        },
        "logger": Mock(),
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "<ui-catalogue-fixture>", "exec"),
        namespace,
    )
    return namespace["_agent_registry_with_live_tools"]


def test_ui_catalogue_only_advertises_selected_tools(selection):
    registry = _isolated_ui_registry()()
    for role, entry in registry.items():
        assert set(entry["tools"]) <= set(selection.names)
        assert "run_boltz" not in entry["description"]
        if role in {"coder", "ml", "dl"}:
            assert set(entry["tools"]) == set(selection.names)
    assert registry["fixture_unknown_role"]["tools"] == []


def test_ui_does_not_fall_back_to_unrestricted_lists_on_invalid_selection(selection, monkeypatch):
    from bioagents.tools import agent_manifest

    monkeypatch.setattr(
        agent_manifest, "agent_tool_names", Mock(side_effect=ValueError("Invalid fixture profile"))
    )
    with pytest.raises(ValueError, match="Invalid fixture profile"):
        _isolated_ui_registry()()


def test_custom_worker_keeps_the_selection_context(monkeypatch):
    from bioagents.tools import tool_registry
    from bioagents.tools.tool_universe import ToolUniverseWrapper

    registry = Mock()
    registry.load_tool_function.return_value = get_selection
    monkeypatch.setattr(tool_registry, "get_registry", lambda: registry)
    selected = ToolSelection("selected", ("fixture_custom",))
    with selection_scope(selected):
        found, result = ToolUniverseWrapper._execute_custom_tool("fixture_custom", {})
    assert found
    assert result == selected
