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
from pathlib import Path

# Mount point of the project root inside the code-execution container.
CONTAINER_PROJECT_ROOT = "/app"


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


def describe_shared_workspace(workspace_dir: str | Path) -> str:
    """Build the workspace note injected into agent context.

    Agents can only reuse each other's output if they know where it lands and in which
    form paths are exchanged, so this is stated explicitly rather than assumed.
    """
    portable = to_portable_path(workspace_dir)
    return (
        "[SHARED WORKSPACE]\n"
        f"All agents share one working directory: {portable} (relative to the project "
        f"root). Inside the code-execution container the project root is mounted at "
        f"{CONTAINER_PROJECT_ROOT}, so the same relative path works there too.\n"
        "- Files written by any agent are readable by every other agent at that path.\n"
        "- Always report file paths relative to the project root, never as absolute host "
        "paths — an absolute host path does not exist inside the code sandbox.\n"
        "- Before reporting that a file is missing, list the directory to check: another "
        "agent may have saved it under a different name."
    )
