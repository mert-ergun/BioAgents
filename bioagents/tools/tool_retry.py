"""Retry transient tool failures before they become the agent's answer.

Only transient failures are retried; a wrong argument or an unwired capability fails
identically every time, so retrying would only delay the real message.
"""

from __future__ import annotations

import logging
import random
import re
import time
from typing import Any

from langchain_core.messages import ToolMessage

from bioagents.limits import (
    TOOL_MAX_ATTEMPTS,
    TOOL_RETRY_BASE_DELAY_SEC,
    TOOL_RETRY_MAX_DELAY_SEC,
)

logger = logging.getLogger(__name__)

# The call never reached a verdict about the request itself.
_TRANSIENT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bconnection (reset|refused|aborted|error)\b",
        r"\bconnectionerror\b",
        r"\bread timed out\b",
        r"\btimed? ?out\b",
        r"\btimeout(error)?\b",
        r"\btemporarily unavailable\b",
        r"\bservice unavailable\b",
        r"\bbad gateway\b",
        r"\bgateway time-?out\b",
        r"\binternal server error\b",
        r"\bremote end closed connection\b",
        r"\bserver disconnected\b",
        r"\bbroken pipe\b",
        r"\bname resolution\b",
        r"\btemporary failure in name resolution\b",
        r"\bssl(error|: )\b",
        r"\btoo many requests\b",
        r"\brate.?limit",
        r"\bthrottl",
        r"\bhttp (429|500|502|503|504)\b",
        r"\b(status|error) ?(code)?[: ]+(429|500|502|503|504)\b",
        r"\[(429|500|502|503|504)\]",
    )
)

_PERMANENT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bvalidation ?error\b",
        r"\binvalid (argument|parameter|input|value|smiles|sequence)\b",
        r"\bmissing (required )?(argument|parameter|field)\b",
        r"\bunexpected keyword argument\b",
        r"\bno such file or directory\b",
        r"\bfile not found\b",
        r"\bnot found: \b",
        r"\bhttp 40[0134]\b",
        r"\bcapability_unavailable\b",
        r"\bnot_implemented\b",
        r"\bis not installed\b",
        r"\bpolicy_blocked\b",
        r"\bapproval_required\b",
    )
)


def is_transient_error(text: str) -> bool:
    """True if ``text`` describes a failure that another attempt could clear."""
    if not text:
        return False
    if any(p.search(text) for p in _PERMANENT_PATTERNS):
        return False
    return any(p.search(text) for p in _TRANSIENT_PATTERNS)


def _message_text(msg: Any) -> str:
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return str(content)


def _tool_messages(payload: Any) -> list[ToolMessage]:
    if isinstance(payload, ToolMessage):
        return [payload]
    if isinstance(payload, dict):
        msgs = payload.get("messages")
        if isinstance(msgs, list):
            return [m for m in msgs if isinstance(m, ToolMessage)]
    if isinstance(payload, list):
        return [m for m in payload if isinstance(m, ToolMessage)]
    return []


def should_retry(payload: Any) -> tuple[bool, str]:
    """True if every tool message failed transiently.

    A batch with one success is left alone: re-running repeats that side effect.
    """
    messages = _tool_messages(payload)
    if not messages:
        return False, ""

    reasons = []
    for msg in messages:
        text = _message_text(msg)
        failed = getattr(msg, "status", "success") == "error" or '"status": "error"' in text
        if not failed or not is_transient_error(text):
            return False, ""
        reasons.append(text.strip()[:200])

    return True, reasons[0] if reasons else ""


def retry_delay(attempt: int) -> float:
    """Backoff before ``attempt`` (1-based index of the attempt about to be made)."""
    raw = TOOL_RETRY_BASE_DELAY_SEC * (2 ** max(0, attempt - 1))
    capped = min(raw, TOOL_RETRY_MAX_DELAY_SEC)
    return float(capped * (0.5 + random.random() / 2))  # nosec B311 - jitter, not crypto


def invoke_with_retry(
    invoke: Any,
    *args: Any,
    max_attempts: int | None = None,
    sleep: Any = time.sleep,
    **kwargs: Any,
) -> Any:
    """Call ``invoke``, retrying transient failures with exponential backoff.

    A final failure is returned annotated, not raised, so the agent can try another tool.
    """
    attempts = max_attempts if max_attempts is not None else TOOL_MAX_ATTEMPTS
    attempts = max(1, attempts)

    payload = invoke(*args, **kwargs)
    for attempt in range(1, attempts):
        retry, reason = should_retry(payload)
        if not retry:
            return payload
        delay = retry_delay(attempt)
        logger.warning(
            "Transient tool failure (attempt %d/%d), retrying in %.1fs: %s",
            attempt,
            attempts,
            delay,
            reason,
        )
        if delay > 0:
            sleep(delay)
        payload = invoke(*args, **kwargs)

    retry, reason = should_retry(payload)
    if retry and attempts > 1:
        return annotate_exhausted(payload, attempts)
    return payload


async def ainvoke_with_retry(
    ainvoke: Any,
    *args: Any,
    max_attempts: int | None = None,
    sleep: Any = None,
    **kwargs: Any,
) -> Any:
    """Async twin of :func:`invoke_with_retry`, so the event loop is not blocked."""
    import asyncio

    sleep = sleep or asyncio.sleep
    attempts = max(1, max_attempts if max_attempts is not None else TOOL_MAX_ATTEMPTS)

    payload = await ainvoke(*args, **kwargs)
    for attempt in range(1, attempts):
        retry, reason = should_retry(payload)
        if not retry:
            return payload
        delay = retry_delay(attempt)
        logger.warning(
            "Transient tool failure (attempt %d/%d), retrying in %.1fs: %s",
            attempt,
            attempts,
            delay,
            reason,
        )
        if delay > 0:
            await sleep(delay)
        payload = await ainvoke(*args, **kwargs)

    retry, _ = should_retry(payload)
    if retry and attempts > 1:
        return annotate_exhausted(payload, attempts)
    return payload


def annotate_exhausted(payload: Any, attempts: int) -> Any:
    """Mark a failure as having survived every attempt, so the agent stops re-calling."""
    note = (
        f" [BioAgents: this tool failed {attempts} times with a transient error. "
        "The service looks unavailable right now — use a different tool or report "
        "the failure; do not call this tool again for the same input.]"
    )

    def _annotate(msg: ToolMessage) -> ToolMessage:
        return ToolMessage(
            content=_message_text(msg) + note,
            tool_call_id=msg.tool_call_id,
            name=getattr(msg, "name", "") or "",
            status="error",
        )

    if isinstance(payload, ToolMessage):
        return _annotate(payload)
    if isinstance(payload, dict) and isinstance(payload.get("messages"), list):
        return {
            **payload,
            "messages": [
                _annotate(m) if isinstance(m, ToolMessage) else m for m in payload["messages"]
            ],
        }
    if isinstance(payload, list):
        return [_annotate(m) if isinstance(m, ToolMessage) else m for m in payload]
    return payload
