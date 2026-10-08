"""Tool-callability smoke tests.

Every tool an agent can reach is exercised through ``tool.invoke(<dict>)`` --
the exact interface a LangGraph agent uses -- so that passing here means an
agent really can call the tool, not merely that the underlying Python function
exists.

Two layers:

* structural (default, offline, fast): naming, descriptions, arg schemas,
  duplicate names, and a validation probe proving the tool is wired up.
* live (``--runlive``): real invocations with realistic arguments, asserting the
  result is a real result rather than a cheerful lie.

Run the live layer with::

    uv run pytest tests/test_tool_callability.py --runlive

A human-readable inventory is produced by ``uv run python -m tests.tool_audit``.
"""

from __future__ import annotations

import warnings

import pytest
from pydantic import ValidationError

from tests.tool_inventory import (
    AGENT_TOOL_MAP,
    ALL_TOOLS,
    MIN_DESCRIPTION_LENGTH,
    analyse_structure,
    build_invalid_payload,
    find_duplicate_names_in_lists,
    find_phantom_tools,
    find_shadowed_names,
    unaudited_graph_factories,
)
from tests.tool_live_cases import (
    KNOWN_LIVE_FAILURES,
    LIVE_CASES,
    LiveCase,
    LiveContext,
)

# ---------------------------------------------------------------------------
# Known-defect ledgers.
#
# These are explicit, visible expected-failures. A tool listed here is a known
# bug, not an accepted standard: fixing the tool flips the entry green and the
# test then asks for the entry to be removed.
# ---------------------------------------------------------------------------

#: Tools that return a hardcoded success string without using their own inputs.
#: Replacing a stub with a real implementation removes it from this set
#: automatically -- the test only fails on *new* phantoms.
#: Empty, and it must stay empty. All eleven original offenders (run_esm2, run_esm3,
#: run_saprot, run_alphafold2, run_boltz, run_esmfold, run_abodybuilder3, run_unimol,
#: run_rfdiffusion, run_proteinmpnn, run_aggrescan3d) now either do real work or report
#: an honest failure. Any name appearing here again is a regression, not a baseline.
KNOWN_PHANTOM_TOOLS: frozenset[str] = frozenset()

#: Tools whose description never tells the agent what it gets back.
KNOWN_RETURN_DOC_GAPS: frozenset[str] = frozenset()

#: Tools with arguments the agent gets no documentation for at all -- neither a
#: schema field description nor a mention in the tool description.
KNOWN_UNDOCUMENTED_ARGS: frozenset[str] = frozenset()

#: Tool names bound twice inside one tool list handed to ``llm.bind_tools()``.
KNOWN_DUPLICATE_BINDINGS: frozenset[tuple[str, str]] = frozenset(
    {
        ("repair_invalid_smiles", "agent:rdkit_validator"),
        ("repair_invalid_smiles", "factory:get_all_rdkit_tools"),
    }
)

TOOL_NAMES = sorted(ALL_TOOLS)


def _warn_if_fixed(fixed: set[str], ledger_name: str) -> None:
    if fixed:
        warnings.warn(
            f"These tools are no longer defective and can be removed from "
            f"{ledger_name}: {sorted(fixed)}",
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# 1. Structural checks over every tool (offline, always run)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("tool_name", TOOL_NAMES)
def test_tool_has_name_and_description(tool_name: str) -> None:
    """Every tool exposes a usable name and a substantive description."""
    tool = ALL_TOOLS[tool_name]

    assert tool.name and tool.name.strip(), "tool has an empty name"
    assert tool.name.isidentifier(), f"tool name {tool.name!r} is not a valid identifier"

    description = (tool.description or "").strip()
    assert description, f"{tool_name} has no description"
    assert len(description) >= MIN_DESCRIPTION_LENGTH, (
        f"{tool_name} description is {len(description)} chars, "
        f"minimum is {MIN_DESCRIPTION_LENGTH}: {description!r}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("tool_name", TOOL_NAMES)
def test_tool_description_documents_return_value(tool_name: str) -> None:
    """An agent must be told what the tool gives back before it calls it."""
    report = analyse_structure(ALL_TOOLS[tool_name])
    if tool_name in KNOWN_RETURN_DOC_GAPS and not report.documents_return:
        pytest.xfail(f"{tool_name} is a known return-documentation gap")
    assert report.documents_return, (
        f"{tool_name} description never says what it returns: {ALL_TOOLS[tool_name].description!r}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("tool_name", TOOL_NAMES)
def test_tool_has_args_schema_with_documented_fields(tool_name: str) -> None:
    """``args_schema`` exists and every field is documented for the agent."""
    tool = ALL_TOOLS[tool_name]
    report = analyse_structure(tool)

    assert getattr(tool, "args_schema", None) is not None, (
        f"{tool_name} has no args_schema; an agent cannot build a tool call"
    )
    if tool_name in KNOWN_UNDOCUMENTED_ARGS and report.undocumented_args:
        pytest.xfail(f"{tool_name} has known undocumented arguments")
    assert not report.undocumented_args, (
        f"{tool_name} arguments are invisible to the agent "
        f"(no schema description and not named in the description): "
        f"{report.undocumented_args}"
    )


@pytest.mark.unit
def test_description_ledgers_are_not_stale() -> None:
    """Known-gap ledgers must describe reality, so a regression cannot hide."""
    reports = {name: analyse_structure(tool) for name, tool in ALL_TOOLS.items()}
    _warn_if_fixed(
        {n for n in KNOWN_RETURN_DOC_GAPS if reports[n].documents_return},
        "KNOWN_RETURN_DOC_GAPS",
    )
    _warn_if_fixed(
        {n for n in KNOWN_UNDOCUMENTED_ARGS if not reports[n].undocumented_args},
        "KNOWN_UNDOCUMENTED_ARGS",
    )
    unknown = (KNOWN_RETURN_DOC_GAPS | KNOWN_UNDOCUMENTED_ARGS) - set(ALL_TOOLS)
    assert not unknown, f"ledgers reference tools that no longer exist: {sorted(unknown)}"


@pytest.mark.unit
def test_inventory_covers_every_factory_graph_uses() -> None:
    """The harness must not drift behind ``bioagents/graph.py``.

    If someone wires a new ``get_*_tools()`` factory into an agent, this fails
    until the factory is added to the audited surface.
    """
    missing = sorted(unaudited_graph_factories())
    assert not missing, "graph.py binds tool factories this harness does not audit: " + ", ".join(
        missing
    )


@pytest.mark.unit
def test_no_shadowed_tool_names() -> None:
    """No tool name may resolve to two different tool objects."""
    shadowed = find_shadowed_names()
    assert not shadowed, (
        "tool names bound to more than one distinct tool object:\n  "
        + "\n  ".join(str(f) for f in shadowed)
    )


@pytest.mark.unit
def test_no_duplicate_tool_names_in_a_bound_list() -> None:
    """A tool list handed to ``bind_tools()`` must not repeat a name.

    Providers reject or silently collapse duplicate function names, which makes
    one of the two tools unreachable for the agent.
    """
    found = {(f.tool_name, f.location) for f in find_duplicate_names_in_lists()}
    new = found - KNOWN_DUPLICATE_BINDINGS
    assert not new, "duplicate tool names inside a single bound tool list:\n  " + "\n  ".join(
        f"{name} in {location}" for name, location in sorted(new)
    )
    _warn_if_fixed(
        {f"{n} in {loc}" for n, loc in KNOWN_DUPLICATE_BINDINGS - found},
        "KNOWN_DUPLICATE_BINDINGS",
    )


@pytest.mark.unit
@pytest.mark.parametrize("tool_name", TOOL_NAMES)
def test_tool_is_invocable_through_agent_interface(tool_name: str) -> None:
    """``tool.invoke(...)`` reaches schema validation instead of blowing up.

    This is the real agent entry point. A bad import, a broken decorator, or a
    schema the agent cannot satisfy shows up here as ImportError/AttributeError
    rather than a pydantic ValidationError.

    The probe deliberately sends an *invalid* payload so the tool body never
    runs -- safe even for destructive tools. Tools that take no arguments at all
    are called for real, since there is nothing to invalidate and they are
    read-only by construction.
    """
    tool = ALL_TOOLS[tool_name]
    payload = build_invalid_payload(tool)

    if payload is None:
        result = tool.invoke({})
        assert result is not None, f"{tool_name} returned None for a no-arg call"
        return

    with pytest.raises(ValidationError) as excinfo:
        tool.invoke(payload)

    assert excinfo.value.error_count() > 0, (
        f"{tool_name} raised an empty ValidationError for payload {payload!r}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("agent_name", sorted(AGENT_TOOL_MAP))
def test_agent_has_at_least_one_tool(agent_name: str) -> None:
    """Every tool-using agent in the graph really receives tools."""
    tools = AGENT_TOOL_MAP[agent_name]
    assert tools, f"agent {agent_name} is bound to an empty tool list"


# ---------------------------------------------------------------------------
# 2. Phantom-tool detection
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_no_phantom_tools() -> None:
    """No tool may fake success while ignoring its own inputs.

    A phantom tool ignores at least one declared argument and returns a
    hardcoded success-looking string. Agents trust such a result completely,
    so this is the most damaging possible tool defect.

    Known offenders are listed in ``KNOWN_PHANTOM_TOOLS``. Replacing a stub with
    a real implementation makes this test pass for that tool with no edit here.
    """
    detected = set(find_phantom_tools())
    new_phantoms = detected - KNOWN_PHANTOM_TOOLS
    assert not new_phantoms, (
        "new phantom tools detected -- they return a hardcoded success string "
        f"without using their own arguments: {sorted(new_phantoms)}"
    )
    _warn_if_fixed(KNOWN_PHANTOM_TOOLS - detected, "KNOWN_PHANTOM_TOOLS")


# ---------------------------------------------------------------------------
# 3. Live smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_every_tool_has_a_live_case_or_an_explicit_skip() -> None:
    """Nothing is silently untested: every tool is in the live table."""
    missing = sorted(set(ALL_TOOLS) - set(LIVE_CASES))
    assert not missing, "tools with no live case and no documented skip reason: " + ", ".join(
        missing
    )
    stale = sorted(set(LIVE_CASES) - set(ALL_TOOLS))
    assert not stale, "live cases for tools that no longer exist: " + ", ".join(stale)
    unknown = sorted(set(KNOWN_LIVE_FAILURES) - set(ALL_TOOLS))
    assert not unknown, "KNOWN_LIVE_FAILURES references tools that no longer exist: " + ", ".join(
        unknown
    )


@pytest.mark.unit
@pytest.mark.parametrize("tool_name", TOOL_NAMES)
def test_live_case_is_well_formed(tool_name: str) -> None:
    """A live case either runs something or explains why it cannot."""
    case = LIVE_CASES[tool_name]
    if case.skip_reason:
        assert len(case.skip_reason) > 20, (
            f"{tool_name} skip reason is too vague: {case.skip_reason!r}"
        )
        assert case.args is None, f"{tool_name} has both args and a skip reason"
    else:
        assert case.check is not None, f"{tool_name} live case has no assertion"


@pytest.mark.live
@pytest.mark.parametrize("tool_name", TOOL_NAMES)
def test_tool_live_invocation(tool_name: str, live_context: LiveContext) -> None:
    """Call the tool for real through ``tool.invoke`` and check the result."""
    case: LiveCase = LIVE_CASES[tool_name]
    if case.skip_reason:
        pytest.skip(case.skip_reason)

    tool = ALL_TOOLS[tool_name]
    args = case.resolve_args(live_context)
    result = tool.invoke(args)

    assert result is not None, f"{tool_name} returned None"
    assert case.check is not None
    try:
        case.check(result)
    except AssertionError as exc:
        diagnosis = KNOWN_LIVE_FAILURES.get(tool_name)
        if diagnosis:
            # Deliberately still a failure: this is a tracked defect in the tool,
            # not an accepted behaviour of the harness.
            raise AssertionError(
                f"{tool_name} is a known-broken tool -- {diagnosis}\n\n{exc}"
            ) from exc
        raise
