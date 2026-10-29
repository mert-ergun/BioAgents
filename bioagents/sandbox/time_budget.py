"""Per-tool time budgets that keep long tool calls below the sandbox cell deadline.

The coder sandbox kills a Jupyter cell when it exceeds its deadline and restarts the
kernel, wiping every variable (issue #67: a Vina docking hit the 1800 s limit after
its results were already on disk and cost ~30 minutes of recovery). BLAST already
avoided that by capping its in-cell budget (``BLAST_CELL_BUDGET_SEC``); this module
generalizes the pattern:

* :func:`tool_cell_budget` gives every long-running tool (blast, esmfold, vina, model
  staging, ...) a budget that is always strictly below the active cell deadline;
* :func:`sandbox_time_budget_env` forwards the cell deadline and the budget overrides
  into the container, where the tools actually run;
* :func:`run_in_budgeted_process` runs a blocking native call (Vina, ESMFold) in a
  child process and returns a resumable ``timeout`` status before the kernel is killed;
* :func:`time_budget_guidance` and :func:`recovery_notice` tell the agent about the
  deadline up front and about the files that survived a kernel restart.
"""

from __future__ import annotations

import contextvars
import json
import logging
import math
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bioagents import limits

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

logger = logging.getLogger(__name__)

#: Set by the host on the container: the user-cell deadline the executor enforces.
CELL_TIMEOUT_ENV = "BIOAGENTS_CELL_TIMEOUT_SEC"
#: Forces in-cell detection either way; the BLAST-specific name is kept as an alias.
IN_CELL_ENVS = ("BIOAGENTS_IN_SANDBOX_CELL", "BIOAGENTS_BLAST_IN_CELL")

#: Host settings the in-container tools read at import time, so they must be forwarded.
FORWARDED_BUDGET_ENVS = (
    "BIOAGENTS_BLAST_TIMEOUT_SEC",
    "BIOAGENTS_BLAST_CELL_BUDGET_SEC",
    "BIOAGENTS_ESMFOLD_CELL_BUDGET_SEC",
    "BIOAGENTS_VINA_CELL_BUDGET_SEC",
    "BIOAGENTS_MODEL_STAGING_CELL_BUDGET_SEC",
    "BIOAGENTS_DEFAULT_TOOL_CELL_BUDGET_SEC",
    "BIOAGENTS_TOOL_CELL_BUDGET_MARGIN_SEC",
)

#: Tool-name spellings mapped onto a budget key in ``limits.TOOL_CELL_BUDGETS_SEC``.
_TOOL_ALIASES = {
    "docking": "vina",
    "autodock": "vina",
    "run_docking": "vina",
    "run_esmfold": "esmfold",
    "staging": "model_staging",
    "model_download": "model_staging",
    "from_pretrained": "model_staging",
}

#: Directories never worth scanning for agent output after a restart.
_SKIP_DIRS = frozenset(
    {"node_modules", "__pycache__", ".git", ".venv", "venv", ".mypy_cache", ".pytest_cache"}
)

KERNEL_RESTARTED_MESSAGE = (
    "The Jupyter kernel died while executing this code (typically out-of-memory, a "
    "native crash, e.g. a pip install replacing an already-imported extension, or a cell "
    "that ran past its deadline). A fresh kernel was started and final_answer plus the "
    "tools were restored, but ALL variables and imports from previous steps are gone: "
    "recompute what you need (results already written to files are still on disk)."
)


# -- deadlines ------------------------------------------------------------------------
def _positive_finite(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return value


def cell_execution_timeout_sec() -> float:
    """The user-cell deadline the host executor enforces.

    ``BIOAGENTS_CODE_EXECUTION_TIMEOUT_SEC`` overrides the legacy
    ``BIOAGENTS_CODER_EXEC_TIMEOUT``; without either, ``DEFAULT_CELL_EXECUTION_TIMEOUT_SEC``.
    """
    raw = os.getenv("BIOAGENTS_CODE_EXECUTION_TIMEOUT_SEC")
    if raw is None:
        raw = os.getenv("BIOAGENTS_CODER_EXEC_TIMEOUT")
    value = limits.DEFAULT_CELL_EXECUTION_TIMEOUT_SEC if raw is None else float(raw)
    return _positive_finite(value, "Code execution timeout")


def active_cell_timeout_sec() -> float:
    """The deadline of the cell this code runs in (inside the container) or will run in."""
    raw = os.getenv(CELL_TIMEOUT_ENV, "").strip()
    if raw:
        try:
            return _positive_finite(float(raw), CELL_TIMEOUT_ENV)
        except ValueError:
            logger.warning("Ignoring invalid %s=%r", CELL_TIMEOUT_ENV, raw)
    return cell_execution_timeout_sec()


def cell_margin_sec(cell_timeout: float) -> float:
    """Seconds a tool must leave before the cell deadline; always > 0 and < the deadline."""
    return min(limits.TOOL_CELL_BUDGET_MARGIN_SEC, cell_timeout * 0.25)


def clamp_below_cell(budget: float, cell_timeout: float | None = None) -> float:
    """Clamp ``budget`` so it ends strictly before the cell deadline."""
    timeout = active_cell_timeout_sec() if cell_timeout is None else cell_timeout
    ceiling = timeout - cell_margin_sec(timeout)
    return float(max(0.0, min(budget, ceiling)))


def budget_key(tool: str | None) -> str | None:
    """Map a tool name onto its budget key, or None for an unknown tool."""
    if not tool:
        return None
    lowered = tool.strip().lower()
    budgets = limits.TOOL_CELL_BUDGETS_SEC
    if lowered in budgets:
        return lowered
    if lowered in _TOOL_ALIASES:
        return _TOOL_ALIASES[lowered]
    matches = [key for key in (*budgets, *_TOOL_ALIASES) if key in lowered]
    if not matches:
        return None
    best = max(matches, key=len)
    return _TOOL_ALIASES.get(best, best)


def tool_cell_budget(
    tool: str | None, requested: float | None = None, cell_timeout: float | None = None
) -> float:
    """Seconds ``tool`` may run inside one sandbox cell.

    The configured per-tool budget (or ``requested``, if smaller and positive) clamped
    strictly below the cell deadline, so the call returns before the kernel is killed.
    """
    key = budget_key(tool)
    budget = limits.TOOL_CELL_BUDGETS_SEC.get(key or "", limits.DEFAULT_TOOL_CELL_BUDGET_SEC)
    if requested is not None and requested > 0:
        budget = min(budget, requested)
    return clamp_below_cell(budget, cell_timeout)


def in_sandbox_cell() -> bool:
    """True when this code runs inside the sandbox's Jupyter kernel (which has a deadline)."""
    for name in IN_CELL_ENVS:
        override = os.getenv(name, "").strip().lower()
        if override in ("1", "true", "yes"):
            return True
        if override in ("0", "false", "no"):
            return False
    # ipykernel is only imported where a kernel actually runs, never in the host process.
    if "ipykernel" in sys.modules:
        return True
    ipython = sys.modules.get("IPython")
    if ipython is None:
        return False
    try:
        shell = ipython.get_ipython()
    except Exception:
        return False
    # A plain terminal IPython has no cell timeout; only the kernel (ZMQ) shell does.
    return shell is not None and type(shell).__name__ == "ZMQInteractiveShell"


def workflow_remaining_sec() -> float | None:
    """Seconds left on the whole workflow's wall clock, when one is set."""
    try:
        from bioagents.llms import timeout_llm

        deadline = timeout_llm._workflow_deadline
    except Exception:
        return None
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def sandbox_time_budget_env() -> dict[str, str]:
    """Environment the container needs so in-cell tools see the host's deadlines."""
    env = {CELL_TIMEOUT_ENV: f"{cell_execution_timeout_sec():g}", IN_CELL_ENVS[0]: "1"}
    for name in FORWARDED_BUDGET_ENVS:
        value = os.getenv(name)
        if value is not None and value.strip():
            env[name] = value.strip()
    return env


# -- prompt context -------------------------------------------------------------------
def time_budget_guidance(
    cell_timeout: float | None = None, workflow_remaining: float | None = None
) -> str:
    """Tell the agent how long a cell may run and how to survive a kernel restart."""
    timeout = cell_execution_timeout_sec() if cell_timeout is None else cell_timeout
    budgets = ", ".join(
        f"{name} {tool_cell_budget(name, cell_timeout=timeout):.0f}s"
        for name in limits.TOOL_CELL_BUDGETS_SEC
    )
    lines = [
        "[TIME BUDGET]",
        f"- Each code cell is killed after {timeout:.0f}s; the kernel then restarts and "
        "ALL variables are lost (files on disk survive).",
        f"- Long tool calls inside a cell are capped to return first: {budgets}. A "
        "result with status 'timeout' and cell_bounded: true means the call was stopped "
        "on purpose — retry with smaller inputs or settings, never in a loop.",
        "- Split long work into several smaller cells and write each intermediate result "
        "to disk as soon as it exists: from bioagents.sandbox.checkpoint import "
        "save_checkpoint, load_checkpoint. Check load_checkpoint(name) before recomputing.",
    ]
    remaining = workflow_remaining_sec() if workflow_remaining is None else workflow_remaining
    if remaining is not None:
        lines.append(f"- About {remaining:.0f}s remain for the whole run; plan accordingly.")
    return "\n".join(lines)


# -- recovery after a restart ---------------------------------------------------------
_recovery_roots: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "bioagents_recovery_roots", default=()
)


def set_recovery_roots(roots: Iterable[str | Path | None]) -> None:
    """Directories to scan for recent output when the kernel restarts in this context."""
    _recovery_roots.set(tuple(str(root) for root in roots if root))


def recovery_roots() -> tuple[str, ...]:
    """The registered scan roots, or the default shared sandbox workspace."""
    roots = _recovery_roots.get()
    if roots:
        return roots
    from bioagents.sandbox.workspace import project_root

    return (str(project_root() / os.getenv("BIOAGENTS_SANDBOX_DIR", "sandbox_workdir")),)


def recent_workspace_files(
    roots: Sequence[str | Path],
    since: float,
    limit: int | None = None,
    max_entries: int | None = None,
) -> list[str]:
    """Files under ``roots`` modified at or after ``since``, newest first.

    Bounded: at most ``max_entries`` paths are examined and hidden or tool-cache
    directories are skipped, so this stays cheap on a large workspace.
    """
    limit = limits.RECOVERY_RECENT_FILES_LIMIT if limit is None else limit
    max_entries = limits.RECOVERY_SCAN_MAX_ENTRIES if max_entries is None else max_entries
    if limit <= 0:
        return []
    from bioagents.sandbox.workspace import to_portable_path

    found: list[tuple[float, str]] = []
    examined = 0
    for root in roots:
        if not Path(root).is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in _SKIP_DIRS]
            for filename in filenames:
                examined += 1
                if examined > max_entries:
                    break
                path = Path(dirpath) / filename
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    continue
                if mtime >= since:
                    found.append((mtime, to_portable_path(path)))
            if examined > max_entries:
                break
    found.sort(reverse=True)
    unique = list(dict.fromkeys(path for _, path in found))
    return unique[:limit]


def recovery_notice(since: float | None, roots: Sequence[str | Path] | None = None) -> str:
    """The kernel-restart message plus files written since ``since``."""
    if since is None:
        return KERNEL_RESTARTED_MESSAGE
    try:
        files = recent_workspace_files(recovery_roots() if roots is None else roots, since)
    except Exception as exc:  # the notice must never break recovery itself
        logger.debug("Recent-file scan failed: %s", exc)
        return KERNEL_RESTARTED_MESSAGE
    if not files:
        return KERNEL_RESTARTED_MESSAGE
    listing = "\n".join(f"- {path}" for path in files)
    return (
        f"{KERNEL_RESTARTED_MESSAGE}\nFiles written to the workspace during this run "
        f"(newest first) — load these instead of recomputing:\n{listing}"
    )


# -- bounded native calls -------------------------------------------------------------
def _child_entry(conn: Any, target: Callable[..., Any], kwargs: dict[str, Any]) -> None:
    try:
        conn.send(("ok", target(**kwargs)))
    except BaseException as exc:  # report everything; the parent decides
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def _stop(process: Any) -> None:
    if process.is_alive():
        process.terminate()
        process.join(5)
    if process.is_alive():
        process.kill()
        process.join(5)


def timeout_result(tool: str, budget: float, hint: str = "", **extra: Any) -> str:
    """JSON a budgeted tool returns when it was stopped before the cell deadline."""
    message = (
        f"{tool} did not finish within its {budget:.0f}s in-cell budget and was stopped "
        f"before the sandbox's {active_cell_timeout_sec():.0f}s cell deadline, so the "
        "kernel and its variables survived."
    )
    body = {
        "status": "timeout",
        "tool": tool,
        "cell_bounded": True,
        "budget_sec": round(budget, 1),
        "cell_timeout_sec": round(active_cell_timeout_sec(), 1),
        "message": f"{message} {hint}".strip(),
        **extra,
    }
    return json.dumps(body, indent=2)


def run_in_budgeted_process(
    target: Callable[..., str],
    kwargs: dict[str, Any],
    *,
    tool: str,
    budget: float | None = None,
    hint: str = "",
) -> str:
    """Run ``target(**kwargs)`` in a child process bounded by the tool's cell budget.

    Native calls such as ``Vina.dock`` or an ESMFold forward pass cannot be interrupted
    in-process, so running them inline lets them outlive the cell and take the kernel
    down. A child process can be stopped; files it already wrote stay on disk.
    ``target`` must be an importable module-level function returning a string.
    """
    seconds = tool_cell_budget(tool, requested=budget)
    if seconds <= 0:
        return timeout_result(tool, 0.0, hint)
    ctx = multiprocessing.get_context("spawn")  # fork is unsafe in a threaded kernel
    receiver, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_child_entry, args=(sender, target, kwargs))
    started = time.monotonic()
    process.start()
    sender.close()
    try:
        if not receiver.poll(seconds):
            return timeout_result(
                tool, seconds, hint, elapsed_sec=round(time.monotonic() - started, 1)
            )
        try:
            status, value = receiver.recv()
        except EOFError:
            process.join(5)
            return json.dumps(
                {
                    "status": "error",
                    "tool": tool,
                    "message": f"{tool} worker exited with code {process.exitcode} "
                    "without a result (likely out of memory).",
                }
            )
        if status == "ok":
            return str(value)
        return json.dumps({"status": "error", "tool": tool, "message": value})
    finally:
        receiver.close()
        _stop(process)
