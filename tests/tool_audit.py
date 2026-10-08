"""Standalone tool-callability audit report.

Walks every agent-reachable tool, runs the structural checks, optionally runs
the live invocations, and prints one row per tool.

Usage::

    uv run python -m tests.tool_audit              # structural only
    uv run python -m tests.tool_audit --live       # also invoke tools for real
    uv run python -m tests.tool_audit --live --only fetch_pdb_structure,search_pubmed

Exit code is non-zero if any structural check fails (live failures are reported
but, by default, do not fail the run -- use ``--fail-on-live`` to change that).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from tests.test_tool_callability import (
    KNOWN_DUPLICATE_BINDINGS,
    KNOWN_PHANTOM_TOOLS,
    KNOWN_RETURN_DOC_GAPS,
    KNOWN_UNDOCUMENTED_ARGS,
)
from tests.tool_inventory import (
    ALL_TOOLS,
    analyse_structure,
    build_invalid_payload,
    find_duplicate_names_in_lists,
    find_phantom_tools,
    find_shadowed_names,
    is_phantom_tool,
    owners_of,
)
from tests.tool_live_cases import KNOWN_LIVE_FAILURES, LIVE_CASES, LiveContext

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIPPED"
NOT_RUN = "not-run"


@dataclass
class Row:
    """One audited tool."""

    name: str
    owners: str
    has_schema: bool
    description_ok: bool
    invocable: str
    phantom: bool
    live_status: str
    live_detail: str
    problems: list[str]

    @property
    def structural_ok(self) -> bool:
        return self.has_schema and self.invocable == PASS


def _probe_invocable(tool) -> tuple[str, str]:
    """Check that ``tool.invoke`` reaches schema validation."""
    payload = build_invalid_payload(tool)
    try:
        if payload is None:
            result = tool.invoke({})
            return (
                (PASS, "no-arg call succeeded") if result is not None else (FAIL, "returned None")
            )
        tool.invoke(payload)
    except ValidationError:
        return PASS, "rejected invalid args at the schema"
    except Exception as exc:
        return FAIL, f"{type(exc).__name__}: {exc}"
    return FAIL, "accepted an invalid payload without validating"


def _run_live(name: str, ctx: LiveContext) -> tuple[str, str]:
    case = LIVE_CASES.get(name)
    if case is None:
        return FAIL, "no live case defined"
    if case.skip_reason:
        return SKIP, case.skip_reason
    try:
        args = case.resolve_args(ctx)
        result = ALL_TOOLS[name].invoke(args)
        assert case.check is not None
        case.check(result)
    except AssertionError as exc:
        return FAIL, f"assertion: {exc}".replace("\n", " ")[:200]
    except Exception as exc:
        return FAIL, f"{type(exc).__name__}: {exc}".replace("\n", " ")[:200]
    return PASS, "real result verified"


def build_rows(live: bool, only: set[str] | None, ctx: LiveContext | None) -> list[Row]:
    rows: list[Row] = []
    for name in sorted(ALL_TOOLS):
        if only and name not in only:
            continue
        tool = ALL_TOOLS[name]
        report = analyse_structure(tool)
        invocable, invocable_detail = _probe_invocable(tool)

        problems = list(report.problems)
        if invocable != PASS:
            problems.append(f"not invocable: {invocable_detail}")

        description_ok = report.description_ok
        if name in KNOWN_RETURN_DOC_GAPS or name in KNOWN_UNDOCUMENTED_ARGS:
            problems = [f"KNOWN: {p}" for p in problems]

        live_status, live_detail = (NOT_RUN, "")
        if live and ctx is not None:
            live_status, live_detail = _run_live(name, ctx)
        elif not live:
            case = LIVE_CASES.get(name)
            if case is not None and case.skip_reason:
                live_status, live_detail = SKIP, case.skip_reason

        rows.append(
            Row(
                name=name,
                owners=", ".join(owners_of(name)),
                has_schema=report.has_schema,
                description_ok=description_ok,
                invocable=invocable,
                phantom=is_phantom_tool(tool),
                live_status=live_status,
                live_detail=live_detail,
                problems=problems,
            )
        )
    return rows


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def print_report(rows: list[Row], live: bool) -> None:
    widths = (34, 26, 6, 8, 9, 8, 9)
    header = (
        f"{'TOOL':<{widths[0]}} {'AGENT(S)':<{widths[1]}} {'SCHEMA':<{widths[2]}} "
        f"{'DESC':<{widths[3]}} {'INVOCABLE':<{widths[4]}} {'PHANTOM':<{widths[5]}} "
        f"{'LIVE':<{widths[6]}}"
    )
    print("=" * len(header))
    print("BioAgents tool-callability audit")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{_truncate(row.name, widths[0]):<{widths[0]}} "
            f"{_truncate(row.owners, widths[1]):<{widths[1]}} "
            f"{('yes' if row.has_schema else 'NO'):<{widths[2]}} "
            f"{('ok' if row.description_ok else 'BAD'):<{widths[3]}} "
            f"{row.invocable:<{widths[4]}} "
            f"{('YES' if row.phantom else '-'):<{widths[5]}} "
            f"{row.live_status:<{widths[6]}}"
        )

    print()
    print("-" * len(header))
    total = len(rows)
    print(f"Tools audited: {total}")
    print(f"  schema present      : {sum(r.has_schema for r in rows)}/{total}")
    print(f"  description ok      : {sum(r.description_ok for r in rows)}/{total}")
    print(f"  invocable via agent : {sum(r.invocable == PASS for r in rows)}/{total}")
    print(f"  phantom tools       : {sum(r.phantom for r in rows)}")
    if live:
        print(f"  live PASS           : {sum(r.live_status == PASS for r in rows)}")
        print(f"  live FAIL           : {sum(r.live_status == FAIL for r in rows)}")
        print(f"  live SKIPPED        : {sum(r.live_status == SKIP for r in rows)}")

    _print_section(
        "NOT AGENT-CALLABLE",
        [f"{r.name}: {'; '.join(r.problems)}" for r in rows if r.invocable != PASS],
    )
    _print_section(
        "BAD / MISSING DESCRIPTIONS",
        [f"{r.name}: {'; '.join(r.problems)}" for r in rows if not r.description_ok],
    )
    _print_section(
        "PHANTOM TOOLS (hardcoded success, inputs ignored)",
        [
            f"{name}{'' if name in KNOWN_PHANTOM_TOOLS else '   <-- NEW, not in ledger'}"
            for name in find_phantom_tools()
        ],
    )
    _print_section(
        "SHADOWED TOOL NAMES (same name, different objects)",
        [str(f) for f in find_shadowed_names()],
    )
    _print_section(
        "DUPLICATE NAMES INSIDE ONE BOUND TOOL LIST",
        [
            f"{f}{'' if (f.tool_name, f.location) in KNOWN_DUPLICATE_BINDINGS else '   <-- NEW'}"
            for f in find_duplicate_names_in_lists()
        ],
    )
    if live:
        _print_section(
            "LIVE FAILURES",
            [
                f"{r.name}: {r.live_detail}"
                + (
                    f"\n      diagnosis: {KNOWN_LIVE_FAILURES[r.name]}"
                    if r.name in KNOWN_LIVE_FAILURES
                    else "   <-- NEW, undiagnosed"
                )
                for r in rows
                if r.live_status == FAIL
            ],
        )
    _print_section(
        "DIAGNOSED LIVE DEFECTS (tracked findings)",
        [f"{name}: {detail}" for name, detail in sorted(KNOWN_LIVE_FAILURES.items())],
    )
    _print_section(
        "SKIPPED LIVE TESTS (untested on purpose)",
        [f"{r.name}: {r.live_detail}" for r in rows if r.live_status == SKIP],
    )


def _print_section(title: str, lines: list[str]) -> None:
    print()
    print(f"## {title} ({len(lines)})")
    if not lines:
        print("  none")
        return
    for line in lines:
        print(f"  - {line}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="also invoke tools for real")
    parser.add_argument("--only", default="", help="comma-separated tool names to audit")
    parser.add_argument(
        "--fail-on-live",
        action="store_true",
        help="exit non-zero when a live invocation fails",
    )
    args = parser.parse_args(argv)
    only = {n.strip() for n in args.only.split(",") if n.strip()} or None

    ctx = None
    tmpdir = None
    sandbox_manager = None
    original_base = original_default = None
    if args.live:
        from bioagents.sandbox import sandbox_manager as sandbox_manager

        tmpdir = tempfile.TemporaryDirectory(prefix="bioagents_tool_audit_")
        root = Path(tmpdir.name)
        workspace = root / "workspace"
        sandbox_dir = root / "sandbox"
        workspace.mkdir()
        sandbox_dir.mkdir()
        original_base = sandbox_manager.SANDBOX_BASE_DIR
        original_default = sandbox_manager._default_sandbox
        sandbox_manager.SANDBOX_BASE_DIR = sandbox_dir
        sandbox_manager._default_sandbox = None
        ctx = LiveContext(workspace=workspace, sandbox_dir=sandbox_dir)

    try:
        rows = build_rows(live=args.live, only=only, ctx=ctx)
    except Exception:
        traceback.print_exc()
        return 2
    finally:
        if sandbox_manager is not None:
            sandbox_manager.SANDBOX_BASE_DIR = original_base
            sandbox_manager._default_sandbox = original_default
        if tmpdir is not None:
            tmpdir.cleanup()

    print_report(rows, live=args.live)

    structural_failures = [r for r in rows if not r.structural_ok]
    missing_cases = sorted(set(ALL_TOOLS) - set(LIVE_CASES))
    new_phantoms = sorted(set(find_phantom_tools()) - KNOWN_PHANTOM_TOOLS)
    live_failures = [r for r in rows if r.live_status == FAIL]

    print()
    if missing_cases:
        print(f"ERROR: tools with no live case or skip reason: {missing_cases}")
    if new_phantoms:
        print(f"ERROR: new phantom tools not in the ledger: {new_phantoms}")

    failed = bool(structural_failures or missing_cases or new_phantoms)
    if args.fail_on_live and live_failures:
        failed = True

    print("RESULT:", "FAIL" if failed else "OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
