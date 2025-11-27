"""LLM provider configuration supporting OpenAI, Ollama, and Google Gemini."""

import os
from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from bioagents.llms.rate_limiter import RateLimitedLLM, RateLimiter
from bioagents.prompts.prompt_loader import get_prompt_llm_model

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
    """
    if provider is None:
        provider_str = os.getenv("LLM_PROVIDER", "openai").lower()
        if provider_str not in ("openai", "ollama", "gemini"):
            raise ValueError(f"Invalid LLM_PROVIDER: {provider_str}. Must be one of: openai, ollama, gemini")
        provider = provider_str  # type: ignore[assignment]
    if model is None and prompt_name:
        prompt_model = get_prompt_llm_model(prompt_name, provider)
        if prompt_model:
            model = prompt_model

    if provider == "openai":
        model = model or "gpt-5.1"
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
        )
        
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

        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key="ollama",  # Ollama doesn't require a real API key
        )
        
        rate_limit = os.getenv("OLLAMA_RATE_LIMIT")
        if rate_limit:
            requests_per_minute = int(rate_limit)
            rate_limiter = _get_or_create_rate_limiter("ollama", requests_per_minute)
            if rate_limiter:
                return RateLimitedLLM(llm, rate_limiter)

        return llm

    elif provider == "gemini":
        model = model or "gemini-2.5-flash"
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key,
        )
        
        rate_limit = os.getenv("GEMINI_RATE_LIMIT", "8")
        requests_per_minute = int(rate_limit)

        rate_limiter = _get_or_create_rate_limiter("gemini", requests_per_minute)
        if rate_limiter:
            return RateLimitedLLM(llm, rate_limiter)

        return llm

    else:
        raise ValueError(f"Unsupported provider: {provider}")
