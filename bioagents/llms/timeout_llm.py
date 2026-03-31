"""Wrap chat models so a single invoke cannot block forever (thread pool + deadline)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

from bioagents.limits import AGENT_LLM_INVOKE_TIMEOUT_SEC

# Shared pool: avoids per-call executor overhead; invoke is still one future at a time per thread.
_pool = ThreadPoolExecutor(max_workers=32, thread_name_prefix="bioagents_llm_timeout")


class TimeoutBoundLLM:
    """Delegates to an underlying LLM but bounds `invoke` wall time."""

    def __init__(self, llm: Any, timeout_sec: float | None = None):
        object.__setattr__(self, "_llm", llm)
        object.__setattr__(
            self,
            "_timeout",
            timeout_sec if timeout_sec is not None else AGENT_LLM_INVOKE_TIMEOUT_SEC,
        )

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        if self._timeout is None or self._timeout <= 0:
            return self._llm.invoke(*args, **kwargs)
        fut = _pool.submit(self._llm.invoke, *args, **kwargs)
        try:
            return fut.result(timeout=self._timeout)
        except FuturesTimeoutError as e:
            fut.cancel()
            raise TimeoutError(
                f"LLM invoke exceeded {self._timeout}s "
                "(set BIOAGENTS_AGENT_LLM_INVOKE_TIMEOUT_SEC or disable with 0)"
            ) from e

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
