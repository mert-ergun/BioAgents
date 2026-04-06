"""Wrap chat models so a single invoke cannot block forever (thread pool + deadline)."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

import bioagents.limits as _limits

# Shared pool: avoids per-call executor overhead; invoke is still one future at a time per thread.
_pool = ThreadPoolExecutor(max_workers=32, thread_name_prefix="bioagents_llm_timeout")

# Global workflow deadline (epoch seconds).  When set, every invoke() will use
# min(static_timeout, time_remaining_to_deadline) so LLM calls cannot overshoot
# the experiment / streaming wall-clock budget.
_workflow_deadline: float | None = None


def set_workflow_deadline(deadline: float | None) -> None:
    """Set (or clear) a global wall-clock deadline that all LLM invokes must respect."""
    global _workflow_deadline
    _workflow_deadline = deadline


def _effective_timeout(static_timeout: float) -> float:
    """Return the tighter of *static_timeout* and the remaining workflow budget."""
    if _workflow_deadline is None:
        return static_timeout
    remaining = _workflow_deadline - time.monotonic()
    if remaining <= 0:
        return 0.1  # will fire immediately
    return min(static_timeout, remaining)


class _TimeoutBoundRunnable:
    """Runnable (e.g. ``ChatModel.bind()``) whose ``invoke`` respects the same wall-clock cap."""

    def __init__(self, runnable: Any, timeout_sec: float | None):
        object.__setattr__(self, "_runnable", runnable)
        object.__setattr__(self, "_timeout", timeout_sec)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        if self._timeout is None or self._timeout <= 0:
            return self._runnable.invoke(*args, **kwargs)
        effective = _effective_timeout(self._timeout)
        fut = _pool.submit(self._runnable.invoke, *args, **kwargs)
        try:
            return fut.result(timeout=effective)
        except FuturesTimeoutError as e:
            fut.cancel()
            raise TimeoutError(
                f"LLM invoke exceeded {effective:.0f}s "
                "(set BIOAGENTS_AGENT_LLM_INVOKE_TIMEOUT_SEC or disable with 0)"
            ) from e

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runnable, name)


class TimeoutBoundLLM:
    """Delegates to an underlying LLM but bounds `invoke` wall time."""

    def __init__(self, llm: Any, timeout_sec: float | None = None):
        object.__setattr__(self, "_llm", llm)
        object.__setattr__(
            self,
            "_timeout",
            timeout_sec if timeout_sec is not None else _limits.AGENT_LLM_INVOKE_TIMEOUT_SEC,
        )

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        if self._timeout is None or self._timeout <= 0:
            return self._llm.invoke(*args, **kwargs)
        effective = _effective_timeout(self._timeout)
        fut = _pool.submit(self._llm.invoke, *args, **kwargs)
        try:
            return fut.result(timeout=effective)
        except FuturesTimeoutError as e:
            fut.cancel()
            raise TimeoutError(
                f"LLM invoke exceeded {effective:.0f}s "
                "(set BIOAGENTS_AGENT_LLM_INVOKE_TIMEOUT_SEC or disable with 0)"
            ) from e

    def bind(self, **kwargs: Any) -> Any:
        """Preserve invoke timeout when callers use ``model.bind(stop=...)`` (e.g. smolagents)."""
        return _TimeoutBoundRunnable(self._llm.bind(**kwargs), self._timeout)

    def bind_tools(self, tools: Any) -> TimeoutBoundLLM:
        return TimeoutBoundLLM(self._llm.bind_tools(tools), self._timeout)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> TimeoutBoundLLM:
        return TimeoutBoundLLM(self._llm.with_structured_output(schema, **kwargs), self._timeout)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_llm", "_timeout"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._llm, name, value)

    def __or__(self, other: Any) -> Any:
        return self._llm | other

    def __ror__(self, other: Any) -> Any:
        return other | self._llm

    def __repr__(self) -> str:
        return f"TimeoutBoundLLM({self._llm!r}, timeout={self._timeout!r})"
