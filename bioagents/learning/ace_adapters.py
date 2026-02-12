"""LLM adapter to convert BioAgents LLM to ACE's OpenAI client format."""

import contextlib
import os
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from openai import OpenAI

# Import Gemini adapter if needed
create_gemini_ace_client: Callable[[Any], Any] | None = None
with contextlib.suppress(ImportError):
    from bioagents.learning.gemini_adapter import create_gemini_ace_client


class BioAgentsLLMAdapter:
    """Adapter to convert BioAgents LangChain LLM to ACE's OpenAI client format."""

    def __init__(self, bioagents_llm: BaseChatModel):
        """
        Initialize adapter with BioAgents LLM.

        Args:
            bioagents_llm: LangChain LLM instance from BioAgents
        """
        self.bioagents_llm = bioagents_llm
        self._api_provider = self._detect_provider()
        self._model_name = self._extract_model_name()

    def _detect_provider(self) -> str:
        """Detect API provider from LLM instance."""
        actual_llm = self.bioagents_llm
        if hasattr(self.bioagents_llm, "llm"):
            actual_llm = self.bioagents_llm.llm

        llm_type = str(type(actual_llm)).lower()

        if "openai" in llm_type or "chatopenai" in llm_type:
            return "openai"
        elif "gemini" in llm_type or "google" in llm_type or "generativeai" in llm_type:
            return "gemini"
        elif "ollama" in llm_type:
            return "openai"
        else:
            return "openai"

    def _extract_model_name(self) -> str:
        """Extract model name from LLM instance."""
        actual_llm = self.bioagents_llm
        if hasattr(self.bioagents_llm, "llm"):
            actual_llm = self.bioagents_llm.llm

        model_name = None

        if hasattr(actual_llm, "model_name"):
            model_name = actual_llm.model_name
        elif hasattr(actual_llm, "model"):
            model_name = actual_llm.model

        if model_name and "gemini" in model_name.lower() and model_name.startswith("models/"):
            model_name = model_name[7:]

        return model_name or "gpt-4"

    def create_openai_client(self):
        """
        Create OpenAI-compatible client from BioAgents LLM config.

        Returns:
            OpenAI client instance, or Gemini adapter if using Gemini
        """
        actual_llm = self.bioagents_llm
        if hasattr(self.bioagents_llm, "llm"):
            actual_llm = self.bioagents_llm.llm

        llm_type_str = str(type(actual_llm)).lower()
        is_gemini = (
            "gemini" in llm_type_str or "google" in llm_type_str or "generativeai" in llm_type_str
        )

        if is_gemini and create_gemini_ace_client is not None:
            return create_gemini_ace_client(actual_llm)

        api_key = None
        base_url = None

        if hasattr(actual_llm, "openai_api_key"):
            api_key = actual_llm.openai_api_key
        elif hasattr(actual_llm, "api_key"):
            api_key = actual_llm.api_key
        elif hasattr(actual_llm, "google_api_key"):
            api_key = actual_llm.google_api_key

        if hasattr(actual_llm, "base_url"):
            base_url = actual_llm.base_url

        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")

        if not base_url:
            ollama_url = os.getenv("OLLAMA_BASE_URL")
            if ollama_url:
                base_url = ollama_url

        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        return OpenAI(**client_kwargs)

    def get_api_provider(self) -> str:
        """Get detected API provider."""
        return self._api_provider

    def get_model_name(self) -> str:
        """Get model name."""
        return self._model_name


def create_ace_client_from_bioagents_llm(bioagents_llm: BaseChatModel) -> tuple[OpenAI, str, str]:
    """
    Create ACE-compatible client from BioAgents LLM.

    Args:
        bioagents_llm: LangChain LLM instance from BioAgents

    Returns:
        Tuple of (OpenAI client, api_provider, model_name)
    """
    adapter = BioAgentsLLMAdapter(bioagents_llm)
    client = adapter.create_openai_client()
    return client, adapter.get_api_provider(), adapter.get_model_name()
