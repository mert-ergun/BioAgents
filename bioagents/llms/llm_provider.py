"""LLM provider configuration supporting OpenAI, Ollama, and Google Gemini."""

import os
from contextvars import ContextVar
from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from bioagents.limits import AGENT_LLM_INVOKE_TIMEOUT_SEC
from bioagents.llms.rate_limiter import RateLimitedLLM, RateLimiter
from bioagents.llms.timeout_llm import TimeoutBoundLLM
from bioagents.prompts.prompt_loader import get_prompt_llm_model

# Context var for per-request API key overrides (set by server when user provides keys)
_api_keys_override: ContextVar[dict[str, str] | None] = ContextVar(
    "api_keys_override", default=None
)


def set_api_keys_override(api_keys: dict[str, str] | None) -> None:
    """Set API key overrides for the current context (e.g., per-request from frontend)."""
    _api_keys_override.set(api_keys)


def _get_api_key(provider: str) -> str | None:
    """Get API key for provider, checking override first, then os.environ."""
    override = _api_keys_override.get()
    if override and provider in override and override[provider]:
        return override[provider]
    return None


def _wrap_llm_with_invoke_timeout(llm):
    """Bound wall time of each `invoke` (see BIOAGENTS_AGENT_LLM_INVOKE_TIMEOUT_SEC)."""
    if AGENT_LLM_INVOKE_TIMEOUT_SEC and AGENT_LLM_INVOKE_TIMEOUT_SEC > 0:
        return TimeoutBoundLLM(llm, AGENT_LLM_INVOKE_TIMEOUT_SEC)
    return llm


def _request_timeout_seconds(env_key: str, default: float | None) -> float | None:
    """Parse HTTP request timeout from env. Use ``0`` or empty to disable (no timeout)."""
    raw = os.getenv(env_key)
    if raw is None or raw.strip() == "":
        return default
    if raw.strip() == "0":
        return None
    return float(raw)


# Global rate limiters - shared across all LLM instances
_rate_limiters: dict[str, RateLimiter] = {}
_rate_limiter_initialized: dict[str, bool] = {}


def _get_or_create_rate_limiter(provider: str, requests_per_minute: int) -> RateLimiter | None:
    """
    Get or create a shared rate limiter for the provider.

    This ensures all LLM instances for the same provider share the same
    rate limiter, making the rate limiting effective across all agents.

    Args:
        provider: The provider name
        requests_per_minute: Max requests per minute

    Returns:
        Shared RateLimiter instance or None if rate limiting disabled
    """
    global _rate_limiters, _rate_limiter_initialized

    if requests_per_minute <= 0:
        return None

    key = f"{provider}_{requests_per_minute}"

    if key not in _rate_limiters:
        _rate_limiters[key] = RateLimiter(max_requests=requests_per_minute, time_window=60.0)
        if provider not in _rate_limiter_initialized:
            print(
                f"{provider.capitalize()} rate limiting active: {requests_per_minute} requests/minute"
            )
            _rate_limiter_initialized[provider] = True

    return _rate_limiters[key]


def get_llm(
    provider: Literal["openai", "ollama", "gemini"] | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    prompt_name: str | None = None,
):
    """
    Get a configured LLM instance with optional rate limiting.

    Args:
        provider: The LLM provider to use ('openai', 'ollama', or 'gemini').
                  If None, reads from LLM_PROVIDER env var (defaults to 'openai')
        model: The model name. If None, uses defaults or prompt metadata:
               - Prompt metadata (if prompt_name provided and models defined)
               - Otherwise provider defaults:
                 * OpenAI: 'gpt-5.1'
                 * Ollama: 'qwen3:14b'
                 * Gemini: 'gemini-2.5-flash'
        temperature: The temperature for generation (0.0 = deterministic)
        prompt_name: Optional prompt identifier used to look up provider-specific
                     model recommendations in the XML metadata

    Returns:
        A configured LLM instance, optionally wrapped with rate limiting

    Rate Limiting:
        Rate limiting can be configured via environment variables:
        - OPENAI_RATE_LIMIT: Max requests per minute for OpenAI (default: no limit)
        - GEMINI_RATE_LIMIT: Max requests per minute for Gemini (default: 8)
          Default is conservative to provide buffer below the 10 req/min free tier limit
        - OLLAMA_RATE_LIMIT: Max requests per minute for Ollama (default: no limit)

        Request timeouts (HTTP; avoids hanging forever on a stuck API call):
        - LLM_REQUEST_TIMEOUT: Default seconds for OpenAI and Gemini (default: 600).
          Set to ``0`` to disable.
        - GEMINI_REQUEST_TIMEOUT / OPENAI_REQUEST_TIMEOUT / OLLAMA_REQUEST_TIMEOUT:
          Override per provider.

        Additional safety limits (see ``bioagents.limits``):
        - BIOAGENTS_AGENT_LLM_INVOKE_TIMEOUT_SEC: Max seconds per ``invoke`` (default 180; 0 disables).
        - BIOAGENTS_RATE_LIMIT_MAX_WAIT_SEC: Max seconds to wait for an RPM slot (default 120; 0 disables).
    """
    if provider is None:
        provider_str = os.getenv("LLM_PROVIDER", "openai").lower()
        if provider_str not in ("openai", "ollama", "gemini"):
            raise ValueError(
                f"Invalid LLM_PROVIDER: {provider_str}. Must be one of: openai, ollama, gemini"
            )
        provider = provider_str  # type: ignore[assignment]
    if model is None and prompt_name:
        prompt_model = get_prompt_llm_model(prompt_name, provider)
        if prompt_model:
            model = prompt_model

    if provider == "openai":
        model = model or "gpt-5.1"
        api_key = _get_api_key("openai") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Add it in Settings or set the OPENAI_API_KEY environment variable."
            )

        openai_timeout = _request_timeout_seconds(
            "OPENAI_REQUEST_TIMEOUT",
            _request_timeout_seconds("LLM_REQUEST_TIMEOUT", 600.0),
        )
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            timeout=openai_timeout,
        )
        llm = _wrap_llm_with_invoke_timeout(llm)

        rate_limit = os.getenv("OPENAI_RATE_LIMIT")
        if rate_limit:
            requests_per_minute = int(rate_limit)
            rate_limiter = _get_or_create_rate_limiter("openai", requests_per_minute)
            if rate_limiter:
                return RateLimitedLLM(llm, rate_limiter)

        return llm

    elif provider == "ollama":
        model = model or "qwen3:14b"
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

        ollama_timeout = _request_timeout_seconds(
            "OLLAMA_REQUEST_TIMEOUT",
            _request_timeout_seconds("LLM_REQUEST_TIMEOUT", None),
        )
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key="ollama",  # Ollama doesn't require a real API key
            timeout=ollama_timeout,
        )
        llm = _wrap_llm_with_invoke_timeout(llm)

        rate_limit = os.getenv("OLLAMA_RATE_LIMIT")
        if rate_limit:
            requests_per_minute = int(rate_limit)
            rate_limiter = _get_or_create_rate_limiter("ollama", requests_per_minute)
            if rate_limiter:
                return RateLimitedLLM(llm, rate_limiter)

        return llm

    elif provider == "gemini":
        model = model or "gemini-2.5-flash"
        api_key = _get_api_key("gemini") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Add it in Settings or set the GEMINI_API_KEY environment variable."
            )

        gemini_timeout = _request_timeout_seconds(
            "GEMINI_REQUEST_TIMEOUT",
            _request_timeout_seconds("LLM_REQUEST_TIMEOUT", 600.0),
        )
        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key,
            timeout=gemini_timeout,
        )
        llm = _wrap_llm_with_invoke_timeout(llm)

        rate_limit = os.getenv("GEMINI_RATE_LIMIT", "8")
        requests_per_minute = int(rate_limit)

        rate_limiter = _get_or_create_rate_limiter("gemini", requests_per_minute)
        if rate_limiter:
            return RateLimitedLLM(llm, rate_limiter)

        return llm

    else:
        raise ValueError(f"Unsupported provider: {provider}")
