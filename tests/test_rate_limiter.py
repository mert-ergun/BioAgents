"""Tests for rate limiter module."""

import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

from bioagents.llms.rate_limiter import RateLimitedLLM, RateLimiter


class TestRateLimiter:
    """Tests for the RateLimiter class."""

    def test_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(max_requests=10, time_window=60.0)
        assert limiter.max_requests == 10
        assert limiter.time_window == 60.0
        assert len(limiter.requests) == 0

    def test_acquire_within_limit(self):
        """Test acquiring requests within the rate limit."""
        limiter = RateLimiter(max_requests=5, time_window=1.0)

        # Should allow up to max_requests without blocking
        for _ in range(5):
            limiter.acquire()

        assert len(limiter.requests) == 5

    def test_acquire_blocks_when_limit_reached(self):
        """Test that acquire blocks when rate limit is reached."""
        limiter = RateLimiter(max_requests=3, time_window=1.0)

        # Fill up the rate limiter
        for _ in range(3):
            limiter.acquire()

        # The next request should block until the time window passes
        start_time = time.time()
        limiter.acquire()
        elapsed_time = time.time() - start_time

        # Should have waited close to 1 second
        assert elapsed_time >= 0.9  # Allow some tolerance

    def test_get_stats(self):
        """Test getting rate limiter statistics."""
        limiter = RateLimiter(max_requests=10, time_window=60.0)

        # Initially, no requests
        stats = limiter.get_stats()
        assert stats["current_requests"] == 0
        assert stats["max_requests"] == 10
        assert stats["time_window"] == 60.0
        assert stats["requests_remaining"] == 10

        # After some requests
        for _ in range(3):
            limiter.acquire()

        stats = limiter.get_stats()
        assert stats["current_requests"] == 3
        assert stats["requests_remaining"] == 7

    def test_thread_safety(self):
        """Test that rate limiter is thread-safe."""
        limiter = RateLimiter(max_requests=10, time_window=1.0)

        def make_request():
            limiter.acquire()

        # Make requests from multiple threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            for future in futures:
                future.result()

        stats = limiter.get_stats()
        assert stats["current_requests"] == 10


class TestRateLimitedLLM:
    """Tests for the RateLimitedLLM wrapper."""

    def test_initialization(self):
        """Test rate limited LLM initialization."""
        mock_llm = Mock()
        limiter = RateLimiter(max_requests=10, time_window=60.0)
        rate_limited_llm = RateLimitedLLM(mock_llm, limiter)

        assert rate_limited_llm.llm == mock_llm
        assert rate_limited_llm.rate_limiter == limiter

    def test_invoke_with_rate_limiting(self):
        """Test that invoke applies rate limiting."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = "response"

        limiter = RateLimiter(max_requests=2, time_window=1.0)
        rate_limited_llm = RateLimitedLLM(mock_llm, limiter)

        # First two calls should succeed quickly
        rate_limited_llm.invoke("test1")
        rate_limited_llm.invoke("test2")

        assert mock_llm.invoke.call_count == 2

        # Third call should be delayed
        start_time = time.time()
        rate_limited_llm.invoke("test3")
        elapsed_time = time.time() - start_time

        assert elapsed_time >= 0.9  # Should have waited
        assert mock_llm.invoke.call_count == 3

    def test_invoke_without_rate_limiting(self):
        """Test that invoke works without a rate limiter."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = "response"

        rate_limited_llm = RateLimitedLLM(mock_llm, rate_limiter=None)

        # Should invoke without any delay
        result = rate_limited_llm.invoke("test")
        assert result == "response"
        mock_llm.invoke.assert_called_once_with("test")

    def test_bind_tools(self):
        """Test binding tools to the LLM."""
        mock_llm = Mock()
        mock_bound_llm = Mock()
        mock_llm.bind_tools.return_value = mock_bound_llm

        limiter = RateLimiter(max_requests=10, time_window=60.0)
        rate_limited_llm = RateLimitedLLM(mock_llm, limiter)

        tools = [Mock(), Mock()]
        result = rate_limited_llm.bind_tools(tools)

        mock_llm.bind_tools.assert_called_once_with(tools)
        assert isinstance(result, RateLimitedLLM)
        assert result.llm == mock_bound_llm
        assert result.rate_limiter == limiter

    def test_with_structured_output(self):
        """Test adding structured output to the LLM."""
        mock_llm = Mock()
        mock_structured_llm = Mock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        limiter = RateLimiter(max_requests=10, time_window=60.0)
        rate_limited_llm = RateLimitedLLM(mock_llm, limiter)

        schema = {"type": "object"}
        result = rate_limited_llm.with_structured_output(schema)

        mock_llm.with_structured_output.assert_called_once()
        # Check that it returns a rate-limited runnable
        assert hasattr(result, "_runnable")
        assert hasattr(result, "_rate_limiter")

    def test_attribute_delegation(self):
        """Test that attributes are delegated to the underlying LLM."""
        mock_llm = Mock()
        mock_llm.some_attribute = "test_value"
        mock_llm.some_method = Mock(return_value="method_result")

        rate_limited_llm = RateLimitedLLM(mock_llm, None)

        # Attribute access should be delegated
        assert rate_limited_llm.some_attribute == "test_value"

        # Method calls should be delegated
        result = rate_limited_llm.some_method("arg")
        assert result == "method_result"
        mock_llm.some_method.assert_called_once_with("arg")

    def test_repr(self):
        """Test string representation."""
        mock_llm = Mock()
        limiter = RateLimiter(max_requests=10, time_window=60.0)
        rate_limited_llm = RateLimitedLLM(mock_llm, limiter)

        repr_str = repr(rate_limited_llm)
        assert "RateLimitedLLM" in repr_str
