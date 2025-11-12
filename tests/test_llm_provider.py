"""Tests for LLM provider module."""

import os
from unittest.mock import Mock, patch

import pytest

from bioagents.llms.llm_provider import (
    _get_or_create_rate_limiter,
    get_llm,
)
from bioagents.llms.rate_limiter import RateLimitedLLM


class TestGetOrCreateRateLimiter:
    """Tests for the _get_or_create_rate_limiter function."""

    def test_returns_none_when_disabled(self):
        """Test that None is returned when rate limiting is disabled."""
        result = _get_or_create_rate_limiter("test", 0)
        assert result is None

        result = _get_or_create_rate_limiter("test", -1)
        assert result is None

    def test_creates_rate_limiter(self):
        """Test creating a rate limiter."""
        result = _get_or_create_rate_limiter("test_provider", 10)
        assert result is not None
        assert result.max_requests == 10
        assert result.time_window == 60.0

    def test_returns_shared_instance(self):
        """Test that the same rate limiter instance is returned for same provider."""
        limiter1 = _get_or_create_rate_limiter("test_shared", 15)
        limiter2 = _get_or_create_rate_limiter("test_shared", 15)

        # Should be the exact same object
        assert limiter1 is limiter2


class TestGetLLM:
    """Tests for the get_llm function."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_openai_provider_default(self):
        """Test getting OpenAI LLM with defaults."""
        with patch("bioagents.llms.llm_provider.ChatOpenAI") as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm

            get_llm(provider="openai")

            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["model"] == "gpt-4o-mini"
            assert call_kwargs["temperature"] == 0.0
            assert call_kwargs["api_key"] == "test-key"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_openai_provider_custom_model(self):
        """Test getting OpenAI LLM with custom model."""
        with patch("bioagents.llms.llm_provider.ChatOpenAI") as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm

            get_llm(provider="openai", model="gpt-4", temperature=0.7)

            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["model"] == "gpt-4"
            assert call_kwargs["temperature"] == 0.7

    @patch.dict(os.environ, {}, clear=True)
    def test_openai_provider_missing_api_key(self):
        """Test that ValueError is raised when OPENAI_API_KEY is missing."""
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            get_llm(provider="openai")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_RATE_LIMIT": "5"}, clear=True)
    def test_openai_with_rate_limiting(self):
        """Test OpenAI LLM with rate limiting enabled."""
        with patch("bioagents.llms.llm_provider.ChatOpenAI") as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm

            result = get_llm(provider="openai")

            # Should return a rate-limited wrapper
            assert isinstance(result, RateLimitedLLM)
            assert result.llm == mock_llm

    def test_ollama_provider_default(self):
        """Test getting Ollama LLM with defaults."""
        with patch("bioagents.llms.llm_provider.ChatOpenAI") as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm

            get_llm(provider="ollama")

            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["model"] == "qwen3:14b"
            assert call_kwargs["temperature"] == 0.0
            assert call_kwargs["base_url"] == "http://localhost:11434/v1"
            assert call_kwargs["api_key"] == "ollama"

    @patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://custom:8000/v1"}, clear=True)
    def test_ollama_provider_custom_base_url(self):
        """Test Ollama with custom base URL."""
        with patch("bioagents.llms.llm_provider.ChatOpenAI") as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm

            get_llm(provider="ollama")

            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["base_url"] == "http://custom:8000/v1"

    @patch.dict(os.environ, {"OLLAMA_RATE_LIMIT": "10"}, clear=True)
    def test_ollama_with_rate_limiting(self):
        """Test Ollama LLM with rate limiting enabled."""
        with patch("bioagents.llms.llm_provider.ChatOpenAI") as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm

            result = get_llm(provider="ollama")

            # Should return a rate-limited wrapper
            assert isinstance(result, RateLimitedLLM)

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini-key"}, clear=True)
    def test_gemini_provider_default(self):
        """Test getting Gemini LLM with defaults."""
        with patch("bioagents.llms.llm_provider.ChatGoogleGenerativeAI") as mock_gemini:
            mock_llm = Mock()
            mock_gemini.return_value = mock_llm

            get_llm(provider="gemini")

            mock_gemini.assert_called_once()
            call_kwargs = mock_gemini.call_args[1]
            assert call_kwargs["model"] == "gemini-2.5-flash"
            assert call_kwargs["temperature"] == 0.0
            assert call_kwargs["google_api_key"] == "test-gemini-key"

    @patch.dict(os.environ, {}, clear=True)
    def test_gemini_provider_missing_api_key(self):
        """Test that ValueError is raised when GEMINI_API_KEY is missing."""
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            get_llm(provider="gemini")

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True)
    def test_gemini_default_rate_limiting(self):
        """Test that Gemini has default rate limiting."""
        with patch("bioagents.llms.llm_provider.ChatGoogleGenerativeAI") as mock_gemini:
            mock_llm = Mock()
            mock_gemini.return_value = mock_llm

            result = get_llm(provider="gemini")

            # Gemini should have default rate limiting of 8 requests/minute
            assert isinstance(result, RateLimitedLLM)

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_RATE_LIMIT": "15"}, clear=True)
    def test_gemini_custom_rate_limiting(self):
        """Test Gemini with custom rate limiting."""
        with patch("bioagents.llms.llm_provider.ChatGoogleGenerativeAI") as mock_gemini:
            mock_llm = Mock()
            mock_gemini.return_value = mock_llm

            result = get_llm(provider="gemini")

            assert isinstance(result, RateLimitedLLM)
            # Rate limiter should be set to 15 requests/minute
            assert result.rate_limiter.max_requests == 15

    @patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}, clear=True)
    def test_provider_from_env_var(self):
        """Test reading provider from environment variable."""
        with patch("bioagents.llms.llm_provider.ChatOpenAI") as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm

            get_llm()  # No provider specified

            mock_openai.assert_called_once()

    @patch.dict(os.environ, {"LLM_PROVIDER": "invalid_provider"}, clear=True)
    def test_invalid_provider_from_env(self):
        """Test that ValueError is raised for invalid provider in env var."""
        with pytest.raises(ValueError, match="Invalid LLM_PROVIDER"):
            get_llm()

    def test_invalid_provider_explicit(self):
        """Test that ValueError is raised for invalid provider."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            get_llm(provider="invalid")  # type: ignore

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_custom_temperature(self):
        """Test custom temperature setting."""
        with patch("bioagents.llms.llm_provider.ChatOpenAI") as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm

            get_llm(provider="openai", temperature=0.9)

            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["temperature"] == 0.9
