"""Wrap LangGraph ToolNode so tool results are capped before entering message state."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.prebuilt import ToolNode

from bioagents.limits import MAX_TOOL_OUTPUT_CHARS


def _text_len(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        n = 0
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                n += len(str(block.get("text", "")))
            elif isinstance(block, str):
                n += len(block)
        return n
    return len(str(content))


def _truncate_content(content: Any, max_chars: int) -> Any:
    if max_chars <= 0:
        return content
    if _text_len(content) <= max_chars:
        return content
    if isinstance(content, str):
        return content[:max_chars] + "\n... [truncated by BioAgents: output size cap]"
    if isinstance(content, list):
        out = []
        used = 0
        for block in content:
            if used >= max_chars:
                break
            if isinstance(block, dict) and block.get("type") == "text":
                text = str(block.get("text", ""))
                take = min(len(text), max_chars - used)
                nb = dict(block)
                nb["text"] = text[:take] + ("\n... [truncated]" if take < len(text) else "")
                out.append(nb)
                used += take
            else:
                s = str(block)
                take = min(len(s), max_chars - used)
                out.append(s[:take])
                used += take
        return out
    return str(content)[:max_chars] + "\n... [truncated by BioAgents: output size cap]"


def truncate_tool_messages_payload(payload: Any, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> Any:
    """Truncate ToolMessage contents inside a ToolNode return value (dict, list, or ToolMessage)."""
    if isinstance(payload, ToolMessage):
        return ToolMessage(
            content=_truncate_content(payload.content, max_chars),
            tool_call_id=payload.tool_call_id,
            name=getattr(payload, "name", "") or "",
            status=getattr(payload, "status", "success"),
        )

    if isinstance(payload, dict) and "messages" in payload:
        msgs = payload["messages"]
        if isinstance(msgs, list):
            new_msgs = [truncate_tool_messages_payload(m, max_chars) for m in msgs]
            return {**payload, "messages": new_msgs}
        return payload

    if isinstance(payload, list):
        return [truncate_tool_messages_payload(m, max_chars) for m in payload]

    return payload


class TruncatingToolRunnable(Runnable):
    """ToolNode wrapper that caps tool outputs before they are merged into state."""

    def __init__(self, tools: list):
        super().__init__()
        self._inner = ToolNode(tools)
        self._max_chars = MAX_TOOL_OUTPUT_CHARS

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        out = self._inner.invoke(input, config, **kwargs)
        return truncate_tool_messages_payload(out, self._max_chars)

    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        out = await self._inner.ainvoke(input, config, **kwargs)
        return truncate_tool_messages_payload(out, self._max_chars)


def make_truncating_tool_node(tools: list) -> TruncatingToolRunnable:
    """Build a ToolNode that caps every tool result before it is merged into graph state."""
    return TruncatingToolRunnable(tools)
