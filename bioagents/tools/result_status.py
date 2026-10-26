"""Interpret structured tool outcomes without turning non-execution into success."""

from __future__ import annotations

import json
from typing import Any

_SUCCESS_STATUSES = frozenset({"success", "ok", "completed", "complete", "done", "passed"})


def tool_result_status(result: Any, *, _depth: int = 0) -> str:
    """Preserve explicit outcomes, including JSON strings and execution envelopes.

    Plain text, numbers, lists and dictionaries without outcome fields remain valid
    legacy data. Arbitrary data rows are not scanned for status-like properties.
    Unknown explicit statuses are non-success; malformed/deep envelopes are errors.
    """
    if _depth >= 16:
        return "error"
    if isinstance(result, str):
        try:
            decoded = json.loads(result)
        except RecursionError:
            return "error"
        except ValueError:
            return "success"
        return tool_result_status(decoded, _depth=_depth + 1)
    if not isinstance(result, dict):
        return "success"

    if "status" in result:
        status = result["status"]
        if not isinstance(status, str) or not status.strip():
            return "error"
        status = status.strip().lower()
        if status not in _SUCCESS_STATUSES:
            return status
    if result.get("success") is False or result.get("error"):
        return "error"
    if result.get("available") is False:
        return "capability_unavailable"
    if result.get("performed_any_computation") is False:
        return "not_performed"
    if "result" in result and any(key in result for key in ("status", "tool", "tool_name")):
        return tool_result_status(result["result"], _depth=_depth + 1)
    return "success"
