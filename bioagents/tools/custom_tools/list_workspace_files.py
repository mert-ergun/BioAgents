"""
list_workspace_files.py

Utility for listing files in a workspace directory, optionally filtered by a
glob pattern. Designed so agents can resolve filenames (with sizes and
modification times) before reading or reusing shared workspace artifacts.

Uses the shared workspace resolver for host and container paths.
"""

import fnmatch
import logging
from datetime import datetime

from bioagents.tools.workspace_paths import resolve_existing_path

logger = logging.getLogger(__name__)


def list_workspace_files(path: str, pattern: str = "*") -> dict:
    """
    List files in a directory (optionally filtered by glob pattern) with
    names, sizes and modification times.

    Parameters
    ----------
    path : str
        Directory to list. Relative paths are resolved against the run
        workspace; '/app' paths and absolute paths are accepted. A file path
        is accepted and treated as its parent directory.
    pattern : str, optional
        Glob pattern filter, e.g. '*.json'. Default '*' (all files).
        Directories are always excluded from results.

    Returns
    -------
    dict
        {
            'status': 'success',
            'directory': str,   # directory as given/resolved
            'pattern': str,
            'n_files': int,
            'files': [
                {'name': str, 'size_bytes': int, 'modified': ISO-8601 str},
                ...
            ]
        }
        Files are sorted by name.

    Raises
    ------
    ValueError
        If `pattern` is not a non-empty string.
    FileNotFoundError
        If the resolved directory does not exist.

    Examples
    --------
    >>> result = list_workspace_files('/app/data', pattern='*.json')
    >>> result['status']
    'success'
    """
    # --- Validate pattern -------------------------------------------------
    if not isinstance(pattern, str) or not pattern:
        raise ValueError(f"pattern must be a non-empty string (got {pattern!r})")

    # --- Resolve path -----------------------------------------------------
    if not isinstance(path, str) or not path:
        raise ValueError(f"path must be a non-empty string (got {path!r})")

    p = resolve_existing_path(path)
    if p is None:
        raise FileNotFoundError(f"Directory not found: {path!r}")
    if p.is_file():
        p = p.parent
    if not p.is_dir():
        raise FileNotFoundError(f"Directory not found: {path!r}")

    # --- Collect entries --------------------------------------------------
    logger.info("Listing directory %s (pattern=%s)", p, pattern)
    files = []
    for entry in sorted(p.iterdir()):
        if entry.is_symlink() or not entry.is_file():
            continue
        if not fnmatch.fnmatch(entry.name, pattern):
            continue
        stat = entry.stat()
        files.append(
            {
                "name": entry.name,
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            }
        )

    logger.info("Listed %d file(s) matching %r in %s", len(files), pattern, p)
    return {
        "status": "success",
        "directory": str(p),
        "pattern": pattern,
        "n_files": len(files),
        "files": files,
    }
