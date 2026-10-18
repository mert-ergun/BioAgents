"""Resolve file paths that agents pass to host-side tools.

Agents report paths in several forms, and a tool that only tries ``Path(p).exists()``
against the process working directory misses most of them:

- relative to the run workspace (``001_task/4GV1.pdb``) — the form agents use most,
  because shell/file agents run with the workspace as their cwd, while host-side tools
  run with the project root as cwd;
- relative to the project root (``generated_artifacts/experiments/<id>/default/...``);
- container paths under ``/app`` reported by code-execution agents;
- absolute host paths;
- with the wrong case (``4gv1.pdb`` for a file saved as ``4GV1.pdb``) — PDB IDs are
  case-insensitive, so agents freely lowercase them.

``resolve_input_file`` tries all of these and, when nothing matches, raises a
``FileNotFoundError`` that lists the files that *do* exist so the caller can correct
the path instead of giving up.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from bioagents.sandbox.workspace import CONTAINER_PROJECT_ROOT, project_root

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

# Bound the candidate scan so a huge workspace cannot stall a tool call.
_MAX_SCAN_DEPTH = 4
_MAX_CANDIDATES = 25
# Cap on distinctly-named retries before a tool should stop re-deriving the same artifact.
_MAX_VERSIONS = 20


def workspace_dir() -> Path | None:
    """Return the active run workspace (the shell/file agents' cwd), if one exists."""
    try:
        from bioagents.sandbox.sandbox_manager import get_sandbox

        return get_sandbox().workdir.resolve()
    except Exception as exc:  # pragma: no cover - defensive; sandbox is optional
        logger.debug("No sandbox workspace available: %s", exc)
        return None


def search_bases() -> list[Path]:
    """Directories a relative path may be relative to, most specific first."""
    bases: list[Path] = []
    for base in (workspace_dir(), project_root(), Path.cwd().resolve()):
        if base is not None and base not in bases:
            bases.append(base)
    return bases


def allowed_roots() -> list[Path]:
    """Roots that resolved tool paths must stay within."""
    roots = list(search_bases())
    sandbox_env = os.environ.get("BIOAGENTS_SANDBOX_DIR")
    if sandbox_env:
        roots.append(Path(sandbox_env).expanduser().resolve())
    roots.append(Path(tempfile.gettempdir()).resolve())
    return roots


def _reject_traversal(path_str: str) -> None:
    if ".." in Path(path_str).parts:
        raise ValueError(f"Path traversal rejected: {path_str}")


def _check_allowed(path: Path, original: str) -> Path:
    resolved = path.resolve()
    if not any(resolved.is_relative_to(root) for root in allowed_roots()):
        roots = ", ".join(str(r) for r in search_bases())
        raise ValueError(f"Path outside allowed workspace: {original}. Allowed roots: {roots}")
    return resolved


def _from_container(path_str: str) -> Path | None:
    if path_str == CONTAINER_PROJECT_ROOT:
        return project_root()
    if path_str.startswith(CONTAINER_PROJECT_ROOT + "/"):
        return project_root() / path_str[len(CONTAINER_PROJECT_ROOT) + 1 :]
    return None


def _case_insensitive_match(path: Path) -> Path | None:
    """Find ``path`` ignoring the case of its final component."""
    parent = path.parent
    if not parent.is_dir():
        return None
    wanted = path.name.lower()
    matches = [p for p in parent.iterdir() if p.name.lower() == wanted]
    return matches[0] if len(matches) == 1 else None


def _candidate_paths(path_str: str) -> list[Path]:
    container = _from_container(path_str)
    if container is not None:
        return [container]
    raw = Path(path_str).expanduser()
    if raw.is_absolute():
        return [raw]
    return [base / raw for base in search_bases()]


def _scan(bases: Iterable[Path], suffixes: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    seen: set[Path] = set()
    for base in bases:
        if not base.is_dir():
            continue
        for root, dirs, files in os.walk(base):
            depth = len(Path(root).relative_to(base).parts)
            dirs[:] = [d for d in dirs if not d.startswith(".")] if depth < _MAX_SCAN_DEPTH else []
            for name in files:
                if suffixes and not name.lower().endswith(suffixes):
                    continue
                full = (Path(root) / name).resolve()
                if full in seen:
                    continue
                seen.add(full)
                try:
                    found.append(str(full.relative_to(base)))
                except ValueError:
                    found.append(str(full))
                if len(found) >= _MAX_CANDIDATES:
                    return found
    return found


def resolve_input_file(
    path_str: str, label: str = "Input file", suffixes: tuple[str, ...] = ()
) -> Path:
    """Resolve an existing input file from any path form an agent might report.

    Args:
        path_str: Path as given by the agent.
        label: Human-readable name used in error messages (e.g. ``"Receptor PDB"``).
        suffixes: Lower-case extensions used to list candidates when nothing matches.

    Returns:
        The resolved, existing file path.

    Raises:
        ValueError: On path traversal or a path outside the allowed roots.
        FileNotFoundError: When no candidate exists; the message lists files that do.
    """
    if not path_str or not str(path_str).strip():
        raise FileNotFoundError(f"{label} path is empty.")
    path_str = str(path_str).strip()
    _reject_traversal(path_str)

    tried = _candidate_paths(path_str)
    for candidate in tried:
        hit = candidate if candidate.is_file() else _case_insensitive_match(candidate)
        if hit is not None and hit.is_file():
            return _check_allowed(hit, path_str)

    bases = [b for b in search_bases() if b.is_dir()]
    candidates = _scan(bases[:1], suffixes)
    lines = [f"{label} not found: {path_str}", "Tried:"]
    lines += [f"  - {p}" for p in tried]
    if candidates:
        lines.append(f"Existing {'/'.join(suffixes) or ''} files in the workspace ({bases[0]}):")
        lines += [f"  - {c}" for c in candidates]
    else:
        lines.append("No matching files exist in the workspace yet — create or download it first.")
    raise FileNotFoundError("\n".join(lines))


def resolve_output_dir(path_str: str | None, default_subdir: str = "") -> Path:
    """Resolve (and create) an output directory.

    Relative paths are anchored at the run workspace — where every other agent will
    look for them — rather than at the host process's cwd.
    """
    base = workspace_dir() or Path.cwd().resolve()
    if not path_str or not str(path_str).strip():
        target = base / default_subdir if default_subdir else base
    else:
        path_str = str(path_str).strip()
        _reject_traversal(path_str)
        container = _from_container(path_str)
        raw = Path(path_str).expanduser()
        if container is not None:
            target = container
        elif raw.is_absolute():
            target = raw
        else:
            target = _relative_output_target(raw, base)
    target = _check_allowed(target, str(path_str or target))
    target.mkdir(parents=True, exist_ok=True)
    return target


def _relative_output_target(raw: Path, workspace: Path) -> Path:
    """Anchor a relative output dir, accepting project-root-relative workspace paths."""
    root = project_root()
    as_project = (root / raw).resolve()
    # ``generated_artifacts/experiments/<id>/default/out`` is project-root relative and
    # already points inside the workspace; anchoring it at the workspace would double it.
    if as_project.is_relative_to(workspace):
        return as_project
    return workspace / raw


def resolve_existing_path(path_str: str) -> Path | None:
    """Resolve an existing file *or directory* from any path form an agent reports.

    Unlike :func:`resolve_input_file` this answers the plain question "is this there?"
    and never raises on a miss, so an agent can get a truthful yes/no instead of an
    error string it has to interpret.

    Returns:
        The resolved path if it exists, otherwise ``None``.

    Raises:
        ValueError: On path traversal or a path outside the allowed roots.
    """
    if not path_str or not str(path_str).strip():
        return None
    path_str = str(path_str).strip()
    _reject_traversal(path_str)

    for candidate in _candidate_paths(path_str):
        hit = candidate if candidate.exists() else _case_insensitive_match(candidate)
        if hit is not None and hit.exists():
            return _check_allowed(hit, path_str)
    return None


def unique_sibling(path: Path) -> Path:
    """Return a not-yet-existing path next to ``path`` (``a.csv`` -> ``a_v2.csv``).

    Used instead of clobbering an artifact: a second attempt lands beside the first,
    so neither result is lost and the difference stays visible.
    """
    if not path.exists():
        return path
    for version in range(2, _MAX_VERSIONS + 1):
        candidate = path.with_name(f"{path.stem}_v{version}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(
        f"{path.name} already has {_MAX_VERSIONS} versions in {path.parent}; "
        "clean up before writing another."
    )


def prepare_output_file(
    path: str | Path, *, overwrite: bool = False, default_dir: str | None = None
) -> tuple[Path, str]:
    """Decide where a tool should write, given something may already be there.

    This is the one place that turns "a file exists" into an explicit decision rather
    than an accidental clobber. A tool that reuses it cannot silently replace a good
    artifact with a later, worse one.

    Args:
        path: Target file path, in any form an agent might report.
        overwrite: ``True`` replaces an existing file; ``False`` writes a new,
            distinctly-named sibling instead.
        default_dir: Directory for a bare filename (defaults to the run workspace).

    Returns:
        ``(target, action)`` where action is ``"create"``, ``"overwrite"`` or
        ``"new_version"``. ``target`` is the path the caller should write to.
    """
    raw = Path(str(path).strip()).expanduser()
    container = _from_container(str(path).strip())
    if container is not None:
        target = container
    elif raw.is_absolute():
        target = raw
    elif raw.parent != Path():
        target = resolve_output_dir(str(raw.parent)) / raw.name
    else:
        target = resolve_output_dir(default_dir) / raw.name
    target = _check_allowed(target, str(path))
    target.parent.mkdir(parents=True, exist_ok=True)

    if not target.exists():
        return target, "create"
    if overwrite:
        return target, "overwrite"
    return unique_sibling(target), "new_version"
