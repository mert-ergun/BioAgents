"""Keeps the agent→tool mapping honest and single-sourced.

Regression context: the mapping lived in two places that had drifted — the real wiring
inside create_graph, and a hardcoded catalogue in frontend/server.py that advertised
tool names (fetch_target_structure, analyze_interface, predict_structure,
evaluate_binders) which did not exist anywhere in the codebase. Nothing could answer
"does this agent actually have a tool for this task?", so the supervisor routed blind.
"""

from __future__ import annotations

import pytest

from bioagents.tools.agent_manifest import (
    NON_TOOL_AGENTS,
    agent_tool_names,
    build_agent_tool_map,
    describe_agent_tools,
    find_agents_with_tool,
)

MANIFEST = agent_tool_names()


def test_every_agent_in_the_manifest_owns_at_least_one_tool() -> None:
    empty = [agent for agent, tools in MANIFEST.items() if not tools]
    assert not empty, f"agents declared as tool-using but given no tools: {empty}"


def test_manifest_and_non_tool_agents_do_not_overlap() -> None:
    overlap = sorted(set(MANIFEST) & NON_TOOL_AGENTS)
    assert not overlap, f"agents listed both as tool-using and non-tool: {overlap}"


@pytest.mark.parametrize("agent", sorted(MANIFEST))
def test_tools_are_real_invocable_objects(agent: str) -> None:
    for tool in build_agent_tool_map()[agent]:
        assert getattr(tool, "name", None), f"{agent} was given a tool with no name"
        assert hasattr(tool, "invoke"), f"{agent}'s tool {tool} is not agent-invocable"


def test_frontend_registry_advertises_only_tools_that_exist() -> None:
    """The API must not promise capabilities the wiring does not provide."""
    from frontend.server import _agent_registry_with_live_tools

    real_tool_names = {name for names in MANIFEST.values() for name in names}
    offenders: dict[str, list[str]] = {}

    for agent, entry in _agent_registry_with_live_tools().items():
        if agent in NON_TOOL_AGENTS:
            continue
        unknown = [t for t in entry.get("tools", []) if t not in real_tool_names]
        if unknown:
            offenders[agent] = unknown

    assert not offenders, f"API advertises non-existent tools: {offenders}"


def test_find_agents_with_tool_locates_owners() -> None:
    assert "docking" in find_agents_with_tool("run_docking")
    assert find_agents_with_tool("a_tool_that_does_not_exist") == []


def test_esm_scoring_is_reachable_by_some_agent() -> None:
    """task4 asked for ESM mutation fitness; no agent held an ESM tool at the time."""
    assert find_agents_with_tool("score_mutations_esm"), (
        "no agent can score mutations with ESM — the capability is unreachable"
    )


def test_supervisor_inventory_names_agents_and_their_tools() -> None:
    rendered = describe_agent_tools()
    for agent in MANIFEST:
        assert agent in rendered, f"supervisor inventory omits agent {agent}"
    assert "run_docking" in rendered
    assert "coder" in rendered, "inventory must explain the sandbox agents too"


def test_graph_wires_agents_from_the_manifest() -> None:
    """create_graph must consume the manifest, not a second hand-maintained copy."""
    import inspect

    import bioagents.graph as graph

    source = inspect.getsource(graph.create_graph)
    assert "build_agent_tool_map(" in source, (
        "create_graph no longer builds its tool lists from the manifest; the mapping has "
        "forked again."
    )
