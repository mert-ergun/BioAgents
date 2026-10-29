"""Tiny checkpoint helpers for sandbox code: save intermediate results to disk.

A sandbox cell that runs past its deadline restarts the kernel and loses every
variable; files survive. Agent code saves each intermediate result as soon as it
exists and reloads it on the next step instead of recomputing::

    from bioagents.sandbox.checkpoint import save_checkpoint, load_checkpoint

    poses = load_checkpoint("docking_poses")
    if poses is None:
        poses = expensive_docking()
        save_checkpoint("docking_poses", poses)

JSON-serializable values are stored as ``<name>.json`` (readable by any agent); other
objects fall back to ``<name>.pkl``. Writes are atomic, so a kill mid-write never
leaves a truncated checkpoint behind.
"""

from __future__ import annotations

import json
import os

# Checkpoints are written and read back by the same sandbox agent only.
import pickle  # nosec B403
import re
import tempfile
from pathlib import Path
from typing import Any

CHECKPOINT_DIR_ENV = "BIOAGENTS_CHECKPOINT_DIR"
DEFAULT_CHECKPOINT_DIR = "sandbox_workdir/checkpoints"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def checkpoint_dir(directory: str | Path | None = None) -> Path:
    """The directory checkpoints live in (created on demand)."""
    path = Path(directory or os.getenv(CHECKPOINT_DIR_ENV) or DEFAULT_CHECKPOINT_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validated(name: str) -> str:
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name) or ".." in name:
        raise ValueError(
            "Checkpoint names must be 1-128 characters of letters, digits, '_', '-' or "
            "'.', and must not contain path separators"
        )
    return name


def _atomic_write(target: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        Path(tmp).replace(target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def save_checkpoint(name: str, value: Any, directory: str | Path | None = None) -> str:
    """Save ``value`` under ``name`` and return the written path."""
    base = checkpoint_dir(directory)
    stem = _validated(name)
    try:
        data = json.dumps(value).encode()
        target, stale = base / f"{stem}.json", base / f"{stem}.pkl"
    except (TypeError, ValueError):
        data = pickle.dumps(value)
        target, stale = base / f"{stem}.pkl", base / f"{stem}.json"
    _atomic_write(target, data)
    stale.unlink(missing_ok=True)  # never leave an older value that load would prefer
    return str(target)


def load_checkpoint(name: str, default: Any = None, directory: str | Path | None = None) -> Any:
    """Load the value saved under ``name``, or ``default`` when there is none."""
    base = checkpoint_dir(directory)
    stem = _validated(name)
    json_path, pickle_path = base / f"{stem}.json", base / f"{stem}.pkl"
    if json_path.is_file():
        return json.loads(json_path.read_text())
    if pickle_path.is_file():
        return pickle.loads(pickle_path.read_bytes())  # nosec B301
    return default


def list_checkpoints(directory: str | Path | None = None) -> list[str]:
    """Names of the saved checkpoints."""
    base = checkpoint_dir(directory)
    return sorted({p.stem for p in base.iterdir() if p.is_file() and p.suffix in (".json", ".pkl")})
