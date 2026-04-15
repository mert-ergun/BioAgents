"""Wrap LangGraph ToolNode so tool results are capped before entering message state."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.prebuilt import ToolNode

from bioagents.limits import MAX_TOOL_OUTPUT_CHARS
from bioagents.tools.tool_policy import ToolPolicy, get_default_policy

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Approval gate
# ---------------------------------------------------------------------------

class ApprovalToolRunnable(Runnable):
    """ToolNode wrapper that adds policy evaluation + truncation.

    Policy check results are embedded in ToolMessage content so the frontend
    can display approval requests.  Tools requiring approval are blocked on
    first call (returning a denial message).  If the user approves the tool
    via the frontend, it is added to the session's approved list and the
    agent's next retry will succeed.
    """

    def __init__(
        self,
        tools: list,
        policy: ToolPolicy | None = None,
    ):
        super().__init__()
        self._inner = ToolNode(tools)
        self._max_chars = MAX_TOOL_OUTPUT_CHARS
        self._policy = policy or get_default_policy()

    # ------------------------------------------------------------------
    # Core invoke
    # ------------------------------------------------------------------

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        """Execute tools with policy check and truncation."""
        messages = input.get("messages", []) if isinstance(input, dict) else []

        # Pre-check: evaluate policy for each tool call in the last AI message
        last_msg = messages[-1] if messages else None
        if isinstance(last_msg, AIMessage) and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                tool_name = tc.get("name", "unknown")
                tool_args = tc.get("args", {})

                policy_result = self._policy.evaluate(tool_name, tool_args)

                if not policy_result.allowed:
                    # Tool is blocked by policy
                    logger.warning(
                        "Policy blocked tool '%s': %s",
                        tool_name, policy_result.reason,
                    )
                    denial_msg = ToolMessage(
                        content=(
                            f"[POLICY_BLOCKED] Tool '{tool_name}' was blocked: "
                            f"{policy_result.reason}. Try a different approach."
                        ),
                        tool_call_id=tc.get("id", str(uuid.uuid4())),
                        name=tool_name,
                        status="error",
                    )
                    return {"messages": [denial_msg]}

                if policy_result.requires_approval:
                    # Tool needs user approval — block it with a message that
                    # includes the approval info so the frontend can show the
                    # approval UI.  If the user approves, the tool is added to
                    # the session approved list and the next retry will pass.
                    request_id = str(uuid.uuid4())
                    logger.info(
                        "Tool '%s' requires approval (request_id=%s): %s",
                        tool_name, request_id, policy_result.reason,
                    )
                    denial_msg = ToolMessage(
                        content=(
                            f"[APPROVAL_REQUIRED] Tool '{tool_name}' requires user approval. "
                            f"Reason: {policy_result.reason}. "
                            f"Risk level: {policy_result.risk_level}. "
                            f"If approved, retry the tool call. "
                            f"[approval_request_id={request_id}]"
                        ),
                        tool_call_id=tc.get("id", str(uuid.uuid4())),
                        name=tool_name,
                        status="error",
                    )
                    return {"messages": [denial_msg]}

        # All checks passed — execute the tools normally
        out = self._inner.invoke(input, config, **kwargs)
        return truncate_tool_messages_payload(out, self._max_chars)

    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        """Async version — delegates to sync invoke (ToolNode is sync internally)."""
        return self.invoke(input, config, **kwargs)


def make_approval_tool_node(
    tools: list,
    policy: ToolPolicy | None = None,
) -> ApprovalToolRunnable:
    """Build an ApprovalToolRunnable with policy gate + truncation."""
    return ApprovalToolRunnable(tools, policy=policy)
