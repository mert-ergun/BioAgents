"""Load and freeze the operator-owned tool allowlist for a run."""

from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILE = Path(__file__).with_name("selected_tools.json")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


@dataclass(frozen=True)
class ToolSelection:
    mode: str
    local_tools: tuple[str, ...] = ()
    tool_universe_tools: tuple[str, ...] = ()
    aliases: tuple[tuple[str, str], ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return self.local_tools + self.tool_universe_tools

    def allows(self, name: str, *, backend: bool = False) -> bool:
        if self.mode == "all":
            return True
        return name in self.names or (backend and name in dict(self.aliases).values())

    def to_json(self) -> str:
        return json.dumps(
            {
                "mode": self.mode,
                "local_tools": self.local_tools,
                "tool_universe_tools": self.tool_universe_tools,
                "tool_universe_aliases": dict(self.aliases),
            },
            sort_keys=True,
        )


def _names(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(n, str) or not _NAME.fullmatch(n) for n in value
    ):
        raise ValueError(f"{field} must contain exact Python tool identifiers")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} contains duplicate names")
    return tuple(value)


def parse_selection(data: object, *, mode: str = "selected") -> ToolSelection:
    if mode not in {"all", "selected"}:
        raise ValueError("BIOAGENTS_TOOL_MODE must be 'all' or 'selected'")
    if mode == "all":
        return ToolSelection("all")
    if not isinstance(data, dict):
        raise ValueError("Tool selection must be a JSON object")
    unknown = set(data) - {"mode", "local_tools", "tool_universe_tools", "tool_universe_aliases"}
    if unknown:
        raise ValueError(f"Unknown tool selection fields: {sorted(unknown)}")
    if "mode" in data and data["mode"] != mode:
        raise ValueError("Selection snapshot mode does not match the configured mode")
    local = _names(data.get("local_tools"), "local_tools")
    sdk = _names(data.get("tool_universe_tools"), "tool_universe_tools")
    if set(local) & set(sdk):
        raise ValueError("Local and SDK public names must not overlap")
    aliases = data.get("tool_universe_aliases", {})
    if not isinstance(aliases, dict) or any(n not in local for n in aliases):
        raise ValueError("Aliases must refer to selected local public names")
    targets = _names(list(aliases.values()), "alias backends")
    if set(local) & set(targets):
        raise ValueError("An alias backend cannot shadow a local public name")
    return ToolSelection("selected", local, sdk, tuple(aliases.items()))


def _decode_selection(text: str) -> object:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate tool selection field: {key}")
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=unique_object)


def read_selection() -> ToolSelection:
    """Read operator configuration; malformed configuration never enables all mode."""
    mode = os.getenv("BIOAGENTS_TOOL_MODE", "selected").strip().lower()
    if mode == "all":
        return ToolSelection("all")
    if mode != "selected":
        raise ValueError("BIOAGENTS_TOOL_MODE must be 'all' or 'selected'")
    snapshot = os.getenv("BIOAGENTS_TOOL_SELECTION_SNAPSHOT")
    if snapshot:
        data = _decode_selection(snapshot)
        return parse_selection(data, mode=mode)
    path = Path(os.getenv("BIOAGENTS_TOOL_SELECTION_FILE", str(DEFAULT_FILE))).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return parse_selection(_decode_selection(path.read_text(encoding="utf-8")))


_SCOPE: ContextVar[ToolSelection | None] = ContextVar("bioagents_tool_selection", default=None)
_PROCESS_SELECTION: ToolSelection | None = None


def get_selection() -> ToolSelection:
    """Use the active run snapshot, or the process snapshot for standalone calls."""
    global _PROCESS_SELECTION
    active = _SCOPE.get()
    if active is not None:
        return active
    if _PROCESS_SELECTION is None:
        _PROCESS_SELECTION = read_selection()
    return _PROCESS_SELECTION


@contextmanager
def selection_scope(selection: ToolSelection) -> Iterator[None]:
    token = _SCOPE.set(selection)
    try:
        yield
    finally:
        _SCOPE.reset(token)


def unavailable_tool(name: str) -> dict:
    return {
        "tool": name,
        "status": "error",
        "error_code": "tool_not_available",
        "message": "No tool with this name is available in this session. Check the current tool introduction list and use an exact listed name.",
        "execution_state": "not_started",
        "retryable": False,
    }


def is_tool_enabled(name: str, *, backend: bool = False) -> bool:
    return get_selection().allows(name, backend=backend)


def active_selection() -> ToolSelection | None:
    """Return the explicitly scoped run selection, without reading configuration."""
    return _SCOPE.get()
