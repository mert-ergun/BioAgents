"""Rate limiter for managing API request limits."""

import time
from collections import deque
from threading import Lock
from typing import Any

from langchain_core.runnables import Runnable, RunnableConfig


class RateLimiter:
    """
    Thread-safe rate limiter using a sliding window approach.

    Tracks request timestamps and enforces a maximum number of requests
    per time window.
    """

    def __init__(self, max_requests: int, time_window: float):
        """
        Initialize the rate limiter.

        Args:
            max_requests: Maximum number of requests allowed in the time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = Lock()

    def acquire(self) -> None:
        """
        Acquire permission to make a request.

        This method blocks if the rate limit would be exceeded,
        waiting until a request slot becomes available.
        """
        while True:
            sleep_time = 0

            with self.lock:
                current_time = time.time()

                # Remove requests outside the time window
                while self.requests and self.requests[0] <= current_time - self.time_window:
                    self.requests.popleft()

                if len(self.requests) >= self.max_requests:
                    sleep_time = self.requests[0] + self.time_window - current_time
                    if sleep_time > 0:
                        print(
                            f"Rate limit reached. Waiting {sleep_time:.1f}s "
                            f"({len(self.requests)}/{self.max_requests} requests in window)..."
                        )
                    else:
                        continue
                else:
                    self.requests.append(time.time())
                    return

            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_stats(self) -> dict:
        """
        Get current rate limiter statistics.

        Returns:
            Dictionary with current request count and limit info
        """
        with self.lock:
            current_time = time.time()
            while self.requests and self.requests[0] <= current_time - self.time_window:
                self.requests.popleft()

            return {
                "current_requests": len(self.requests),
                "max_requests": self.max_requests,
                "time_window": self.time_window,
                "requests_remaining": self.max_requests - len(self.requests),
            }


class RateLimitedLLM:
    """
    Wrapper for LLMs that applies rate limiting.

    This class acts as a transparent proxy, delegating all calls to the
    underlying LLM while applying rate limiting to invoke operations.
    """

    def __init__(self, llm, rate_limiter: RateLimiter | None = None):
        """
        Initialize rate-limited LLM wrapper.

        Args:
            llm: The underlying LLM instance
            rate_limiter: Optional rate limiter. If None, no rate limiting is applied.
        """
        object.__setattr__(self, "_llm", llm)
        object.__setattr__(self, "_rate_limiter", rate_limiter)

    @property
    def llm(self):
        """Get the underlying LLM instance."""
        return self._llm

    @property
    def rate_limiter(self):
        """Get the rate limiter instance."""
        return self._rate_limiter

    def invoke(self, *args, **kwargs):
        """Invoke the LLM with rate limiting."""
        if self._rate_limiter:
            self._rate_limiter.acquire()
        return self._llm.invoke(*args, **kwargs)

    def bind_tools(self, tools):
        """
        Bind tools to the LLM.

        Returns the rate-limited wrapped version for tool binding,
        which is used in agent contexts where invoke() is called.
        """
        bound_llm = self._llm.bind_tools(tools)
        return RateLimitedLLM(bound_llm, self._rate_limiter)

    def with_structured_output(self, schema, **kwargs):
        """
        Add structured output to the LLM with rate limiting.

        Returns a rate-limited wrapper that's compatible with LangChain's
        Runnable interface, allowing it to work with pipe operators.
        """
        structured_llm = self._llm.with_structured_output(schema, **kwargs)
        return RateLimitedRunnable(structured_llm, self._rate_limiter)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying LLM."""
        return getattr(self._llm, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Delegate attribute setting to the underlying LLM."""
        if name in ("_llm", "_rate_limiter"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._llm, name, value)

    def __or__(self, other):
        """Support pipe operator for LangChain chaining."""
        # Delegate pipe operations to the underlying LLM
        return self._llm | other

    def __ror__(self, other):
        """Support reverse pipe operator for LangChain chaining."""
        # For cases where something is piped to us
        from langchain_core.runnables import RunnableSequence

        return RunnableSequence(other, self)

    def __repr__(self) -> str:
        """String representation."""
        return f"RateLimitedLLM({self._llm!r}, rate_limiter={self._rate_limiter!r})"


class RateLimitedRunnable(Runnable):
    """
    Rate-limited wrapper for LangChain Runnables (e.g., structured output).

    This class properly inherits from Runnable to be fully compatible with
    LangChain's pipe operator and RunnableSequence.
    """

    def __init__(self, runnable: Runnable, rate_limiter: RateLimiter | None = None):
        """
        Initialize rate-limited runnable wrapper.

        Args:
            runnable: The underlying Runnable instance
            rate_limiter: Optional rate limiter
        """
        super().__init__()
        self._runnable = runnable
        self._rate_limiter = rate_limiter

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs) -> Any:
        """Invoke the runnable with rate limiting."""
        if self._rate_limiter:
            self._rate_limiter.acquire()
        return self._runnable.invoke(input, config, **kwargs)

    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs) -> Any:
        """Async invoke with rate limiting."""
        if self._rate_limiter:
            self._rate_limiter.acquire()
        return await self._runnable.ainvoke(input, config, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying runnable."""
        return getattr(self._runnable, name)

    def __repr__(self) -> str:
        """String representation."""
        return f"RateLimitedRunnable({self._runnable!r})"
