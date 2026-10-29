"""Shared workspace paths that are valid for every agent.

Agents do not all see the same filesystem. File-tool agents (data_acquisition, research,
...) run in the host process and resolve paths against ``sandbox_workdir/<workspace>``.
The code-writing agents (coder, ml, dl) run inside a Docker container that mounts the
project root at ``/app``.

An absolute host path such as ``/home/user/BioAgents/sandbox_workdir/default/4gv1.pdb``
therefore does not exist inside the container, so a file one agent wrote was reported
with a path the next agent could not open. Reporting paths relative to the project root
fixes this: the same string resolves correctly on the host and at ``/app`` in the
container.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

# Mount point of the project root inside the code-execution container.
CONTAINER_PROJECT_ROOT = "/app"
# Overrides the canonical workspace root; also exported to shell commands.
WORKSPACE_ROOT_ENV = "BIOAGENTS_WORKSPACE_ROOT"
# Legacy sandbox base directory name agents still prefix paths with.
_LEGACY_SANDBOX_DIR = "sandbox_workdir"
# Argument names treated as filesystem paths at the custom-tool boundary.
_PATH_ARG_NAMES = frozenset({"path", "directory", "input_file", "output_dir", "outdir", "workdir"})
_PATH_ARG_SUFFIXES = ("_path", "_dir", "_file", "_directory", "_folder")
# Markers of a path a tool writes to (need not exist yet) rather than reads from.
_OUTPUT_ARG_MARKERS = ("output", "save", "dest", "target", "work", "cache", "tmp", "temp", "log")


def project_root() -> Path:
    """Return the project root that the execution container mounts."""
    configured = os.getenv("BIOAGENTS_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.cwd().resolve()


def to_portable_path(path: str | Path) -> str:
    """Render a path so every agent can open it, whichever filesystem view it has.

    Paths inside the project root are returned relative to it, which resolves correctly
    both on the host and under ``/app`` in the container. Paths outside the project root
    are returned unchanged — they are genuinely not shareable, and silently rewriting
    them would hide that.
    """
    resolved = Path(path).expanduser()
    try:
        resolved = resolved.resolve()
    except OSError:
        return str(path)

    try:
        return str(resolved.relative_to(project_root()))
    except ValueError:
        return str(resolved)


def to_host_path(path: str | Path) -> Path:
    """Resolve a path reported by any agent back to a real host path.

    Accepts project-relative paths, absolute host paths, and container paths beginning
    with ``/app`` (which agents running in the container will naturally report).
    """
    raw = str(path)
    root = project_root()

    if raw.startswith(CONTAINER_PROJECT_ROOT + "/"):
        return root / raw[len(CONTAINER_PROJECT_ROOT) + 1 :]
    if raw == CONTAINER_PROJECT_ROOT:
        return root

    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate
    return root / candidate


def describe_workspace_root(workspace_dir: str | Path) -> str:
    """State the canonical workspace root and how relative paths resolve against it."""
    absolute = Path(workspace_dir).expanduser().resolve()
    portable = to_portable_path(absolute)
    if Path(portable).is_absolute():
        container = "it is not mounted inside the code-execution container"
    else:
        container = (
            f"inside the code-execution container (cwd {CONTAINER_PROJECT_ROOT}) the same "
            f"directory is {CONTAINER_PROJECT_ROOT}/{portable}"
        )
    return (
        f"WORKSPACE ROOT: {absolute} (also in ${WORKSPACE_ROOT_ENV}).\n"
        "Every relative path is resolved against this root: shell commands run with it "
        "as their cwd, and file tools and registered custom tools resolve relative, "
        f"{CONTAINER_PROJECT_ROOT}/... and project-relative paths onto it (custom tools "
        f"echo the absolute paths they used in `resolved_paths`). {container[0].upper()}"
        f"{container[1:]}, so Python code run there must use that path."
    )


def describe_shared_workspace(workspace_dir: str | Path) -> str:
    """Build the workspace note injected into agent context.

    Agents can only reuse each other's output if they know where it lands and in which
    form paths are exchanged, so this is stated explicitly rather than assumed.
    """
    return (
        "[SHARED WORKSPACE]\n"
        f"{describe_workspace_root(workspace_dir)}\n"
        "- Files written by any agent are readable by every other agent at that path.\n"
        "- Report file paths relative to the workspace root (e.g. 001_task/result.csv). "
        "Do not guess other prefixes such as sandbox_workdir/... or /app/generated_"
        "artifacts/...; if a path fails, list the workspace root instead of retrying "
        "variants.\n"
        "- Before reporting that a file is missing, list the directory to check: another "
        "agent may have saved it under a different name.\n"
        "- Before downloading, computing or writing an artifact, check whether it is "
        "already there (check_file_exists, or list the directory). An earlier step in "
        "this same run may have produced it; redoing it wastes the run and can replace "
        "a good result with a worse one.\n"
        "- Never overwrite an existing artifact by default. Writing the same filename "
        "again saves a new version alongside it instead; only force a replacement when "
        "you have established that the existing file is wrong."
    )


class WorkspacePathError(ValueError):
    """A path cannot be resolved to a usable location inside the workspace."""


def workspace_root() -> Path:
    """Return the canonical, absolute workspace root shared by every agent and tool.

    ``BIOAGENTS_WORKSPACE_ROOT`` wins when set; otherwise it is the active sandbox
    workspace (``$BIOAGENTS_SANDBOX_DIR/<workspace_id>``), which is also the shell cwd.
    """
    configured = os.getenv(WORKSPACE_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    from bioagents.sandbox.sandbox_manager import get_sandbox

    return get_sandbox().workdir.resolve()


def default_allowed_roots(root: Path) -> list[Path]:
    """Roots a resolved path may live under.

    The workspace, the project root, the sandbox base and the system temp directory —
    the same set ``bioagents.tools.workspace_paths`` has always allowed.
    """
    roots = [root.resolve(), project_root()]
    sandbox_env = os.getenv("BIOAGENTS_SANDBOX_DIR")
    if sandbox_env:
        roots.append(Path(sandbox_env).expanduser().resolve())
    roots.append(Path(tempfile.gettempdir()).resolve())
    return roots


def _anchor_sequences(root: Path) -> list[tuple[str, ...]]:
    """Path-part sequences that denote the workspace itself when they appear in a path."""
    anchors: list[tuple[str, ...]] = [(_LEGACY_SANDBOX_DIR, root.name)]
    sandbox_env = os.getenv("BIOAGENTS_SANDBOX_DIR")
    if sandbox_env:
        anchors.append((Path(sandbox_env).name, root.name))
    try:
        relative = root.relative_to(project_root()).parts
    except ValueError:
        relative = ()
    # A single-part anchor (e.g. "default") is too generic to match safely.
    if len(relative) >= 2:
        anchors.append(relative)
    return anchors


def strip_workspace_prefix(path_str: str, root: Path) -> Path | None:
    """Map a path that carries a workspace prefix onto ``root``.

    Handles ``sandbox_workdir/<ws>/x``, project-relative
    ``generated_artifacts/experiments/<id>/default/x`` and doubled prefixes such as
    ``.../default/generated_artifacts/experiments/<id>/default/x``: everything up to the
    *last* occurrence of a workspace prefix is dropped. Returns ``None`` when the path
    carries no workspace prefix.
    """
    parts = Path(path_str).parts
    end = -1
    for anchor in _anchor_sequences(root):
        size = len(anchor)
        for start in range(len(parts) - size + 1):
            if parts[start : start + size] == anchor:
                end = max(end, start + size)
    if end < 0:
        return None
    return root.joinpath(*parts[end:])


def case_insensitive_lookup(path: Path) -> Path | None:
    """Return the existing path matching ``path`` with the case of any component ignored.

    Returns ``None`` when nothing matches or a component is ambiguous.
    """
    if path.exists():
        return path
    current = Path(path.anchor) if path.is_absolute() else Path()
    for part in path.parts[1:] if path.is_absolute() else path.parts:
        candidate = current / part
        if candidate.exists():
            current = candidate
            continue
        if not current.is_dir():
            return None
        wanted = part.lower()
        matches = [p for p in current.iterdir() if p.name.lower() == wanted]
        if len(matches) != 1:
            return None
        current = matches[0]
    return current


def _ensure_within(path: Path, original: str, roots: list[Path]) -> Path:
    resolved = path.resolve()
    if not any(resolved.is_relative_to(r.resolve()) for r in roots):
        allowed = ", ".join(str(r) for r in roots)
        raise WorkspacePathError(
            f"Path escapes the workspace (traversal rejected): {original!r} resolves to "
            f"{resolved}, outside {allowed}"
        )
    return resolved


def _raw_candidates(path_str: str, root: Path) -> list[Path]:
    """Absolute candidates for ``path_str``, most likely first."""
    if path_str == CONTAINER_PROJECT_ROOT or path_str.startswith(CONTAINER_PROJECT_ROOT + "/"):
        path_str = str(project_root() / path_str[len(CONTAINER_PROJECT_ROOT) :].lstrip("/"))
    stripped = strip_workspace_prefix(path_str, root)
    if stripped is not None:
        return [stripped]
    raw = Path(path_str).expanduser()
    if raw.is_absolute():
        return [raw]
    candidates = [root / raw]
    if project_root() != root:
        candidates.append(project_root() / raw)
    return candidates


def resolve_workspace_path(
    path: str | Path,
    *,
    root: Path | None = None,
    must_exist: bool = False,
    allowed_roots: list[Path] | None = None,
) -> Path:
    """Resolve any path form an agent reports to one canonical absolute path.

    Accepted forms: workspace-relative (``001_task/x.pdb``), project-relative
    (``generated_artifacts/experiments/<id>/default/...``), ``/app/...`` container paths,
    legacy ``sandbox_workdir/<ws>/...``, doubled workspace prefixes and absolute host
    paths. Relative paths resolve against the workspace root first. An existing path is
    found even when the case of any component differs.

    Args:
        path: Path as given by the agent.
        root: Workspace root (defaults to :func:`workspace_root`).
        must_exist: Raise :class:`FileNotFoundError` when nothing exists there.
        allowed_roots: Roots the result must stay within (defaults to
            :func:`default_allowed_roots`).

    Raises:
        WorkspacePathError: Empty path, or a path that escapes the allowed roots.
        FileNotFoundError: ``must_exist`` is set and no candidate exists.
    """
    path_str = str(path).strip()
    if not path_str:
        raise WorkspacePathError("Path is empty.")
    root = (root or workspace_root()).resolve()
    roots = allowed_roots if allowed_roots is not None else default_allowed_roots(root)

    candidates = [Path(os.path.normpath(c)) for c in _raw_candidates(path_str, root)]
    for candidate in candidates:
        hit = case_insensitive_lookup(candidate)
        if hit is not None:
            return _ensure_within(hit, path_str, roots)
    if must_exist:
        tried = ", ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            f"Path not found: {path_str!r} (tried {tried}; workspace root is {root})"
        )
    return _ensure_within(candidates[0], path_str, roots)


def is_path_argument(name: str) -> bool:
    """Whether a tool argument name denotes a filesystem path."""
    lowered = name.lower()
    return lowered in _PATH_ARG_NAMES or lowered.endswith(_PATH_ARG_SUFFIXES)


def is_output_argument(name: str) -> bool:
    """Whether a path argument is written to (and so need not exist yet)."""
    lowered = name.lower()
    return lowered.startswith("out") or any(m in lowered for m in _OUTPUT_ARG_MARKERS)


def resolve_path_arguments(
    args: dict[str, Any], *, root: Path | None = None
) -> tuple[dict[str, Any], dict[str, str]]:
    """Canonicalise the path-like arguments of a tool call at the tool boundary.

    Every string argument whose name looks like a path (``*_path``, ``*_dir``,
    ``*_file``, ``output_dir``, ``input_file``, ...) is resolved with
    :func:`resolve_workspace_path`. Input paths must exist; output paths only have to
    stay inside the workspace. URLs and non-string values pass through unchanged.

    Returns:
        ``(new_args, resolved_paths)``: a new argument dict carrying absolute paths, and
        a ``{argument: absolute path}`` map to report back to the agent.

    Raises:
        WorkspacePathError: A path escapes the workspace or a required input is missing.
    """
    path_args = {
        name: value
        for name, value in args.items()
        if is_path_argument(name) and isinstance(value, str) and value.strip()
        if "://" not in value
    }
    if not path_args:
        return dict(args), {}
    root = (root or workspace_root()).resolve()
    resolved: dict[str, str] = {}
    for name, value in path_args.items():
        try:
            target = resolve_workspace_path(
                value, root=root, must_exist=not is_output_argument(name)
            )
        except FileNotFoundError as exc:
            raise WorkspacePathError(f"Input path for '{name}' does not exist: {exc}") from exc
        except WorkspacePathError as exc:
            raise WorkspacePathError(f"Invalid path for '{name}': {exc}") from exc
        resolved[name] = str(target)
    return {**args, **resolved}, resolved
