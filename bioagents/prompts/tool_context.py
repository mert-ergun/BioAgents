"""Render introductions from the active catalogue and remove disabled references."""

from __future__ import annotations

import re

from langchain_core.messages import SystemMessage

from bioagents.tools.tool_catalog import custom_entries, local_index, sdk_entries
from bioagents.tools.tool_selection import ToolSelection, get_selection

_START = "[ACTIVE TOOL CATALOGUE]"
_END = "[/ACTIVE TOOL CATALOGUE]"
_WORD = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def redact_tool_references(text: str, selection: ToolSelection | None = None) -> str:
    selection = selection or get_selection()
    if selection.mode == "all":
        return text
    known = set(local_index()) | set(custom_entries()) | set(sdk_entries())
    aliases = {backend: alias for alias, backend in selection.aliases}

    def replace(match):
        name = match.group()
        if name in aliases:
            return aliases[name]
        return "[unavailable tool]" if name in known and name not in selection.names else name

    return _WORD.sub(replace, text)


def render_tool_catalogue(selection: ToolSelection | None = None) -> str:
    from bioagents.tools.selected_runtime import selected_metadata

    selection = selection or get_selection()
    if selection.mode == "all":
        return ""
    lines = [
        _START,
        "Only the following registered tools are enabled for this run.",
        "Use only your bound schemas. A role title does not grant additional tools.",
        "Do not import the raw tool SDK or recreate a disabled service with direct HTTP.",
        "If a tool is unavailable, check this introduction list; do not invent a replacement or result.",
    ]
    if not selection.names:
        lines.append("No registered service tools are enabled.")
    for metadata in selected_metadata(selection):
        lines.append(
            f"- {metadata['name']}: {redact_tool_references(metadata['description'], selection)}"
        )
    lines.append(_END)
    return "\n".join(lines)


def apply_tool_context(text: str, *, include_catalogue: bool = True) -> str:
    if get_selection().mode == "all":
        return text
    base = re.sub(re.escape(_START) + ".*?" + re.escape(_END), "", text, flags=re.DOTALL)

    def selected_example(match):
        example = match.group()
        return (
            "[Tool example omitted: its registered capability is unavailable.]"
            if redact_tool_references(example) != example
            else example
        )

    base = re.sub(r"```[^\n]*\n.*?```", selected_example, base, flags=re.DOTALL)
    base = redact_tool_references(base).strip()
    return base + ("\n\n" + render_tool_catalogue() if include_catalogue else "")


def selected_system_messages(messages: list) -> list:
    if get_selection().mode == "all":
        return messages
    result, included = [], False
    for message in messages:
        if isinstance(message, SystemMessage) and isinstance(message.content, str):
            message = message.model_copy(
                update={
                    "content": apply_tool_context(message.content, include_catalogue=not included)
                }
            )
            included = True
        result.append(message)
    if not included:
        result.insert(0, SystemMessage(content=render_tool_catalogue()))
    return result
