"""Tests for model provider API key loading and validation."""

import os
from unittest.mock import patch

import pytest

from bioagents.tools.provider_utils import get_provider_key_or_ask


class TestGetProviderKeyOrAskKnownProviders:
    """Tests for API key retrieval from environment for known providers."""

    @pytest.mark.parametrize(
        "provider,env_var,env_value",
        [
            ("Tamarind Bio", "TAMARIND_API_KEY", "tam-key-123"),
            ("NVIDIA BioNeMo", "NVIDIA_API_KEY", "nvidia-key-456"),
            ("Vertex AI", "VERTEX_API_KEY", "vertex-key-789"),
            ("Neurosnap", "NEUROSNAP_API_KEY", "neuro-key-abc"),
            ("Levitate Bio", "LEVITATE_API_KEY", "levitate-key-def"),
            ("Hugging Face", "HUGGINGFACE_API_KEY", "hf-token-xyz"),
            ("Hugging Face (Weights)", "HUGGINGFACE_API_KEY", "hf-weights-token"),
            ("EvolutionaryScale Forge", "EVOSCALE_API_KEY", "evoscale-key"),
            ("AWS SageMaker", "AWS_API_KEY", "aws-key"),
            ("Google (Official)", "GOOGLE_API_KEY", "google-key-123"),
        ],
    )
    def test_returns_key_when_present(self, provider, env_var, env_value):
        """Test that a configured API key is returned for each known provider."""
        with patch.dict(os.environ, {env_var: env_value}, clear=True):
            result = get_provider_key_or_ask(provider, "TestTool")

        assert result == env_value

    def test_hugging_face_variants_share_env_var(self):
        """Test that both Hugging Face provider names use HUGGINGFACE_API_KEY."""
        with patch.dict(os.environ, {"HUGGINGFACE_API_KEY": "shared-hf-key"}, clear=True):
            assert get_provider_key_or_ask("Hugging Face", "ESM2") == "shared-hf-key"
            assert (
                get_provider_key_or_ask("Hugging Face (Weights)", "ESM2") == "shared-hf-key"
            )


class TestGetProviderKeyOrAskMissingKeys:
    """Tests for missing or invalid API key handling."""

    @pytest.mark.parametrize(
        "provider,env_var",
        [
            ("Tamarind Bio", "TAMARIND_API_KEY"),
            ("NVIDIA BioNeMo", "NVIDIA_API_KEY"),
            ("Hugging Face", "HUGGINGFACE_API_KEY"),
            ("Google (Official)", "GOOGLE_API_KEY"),
            ("Levitate Bio", "LEVITATE_API_KEY"),
        ],
    )
    def test_missing_key_returns_engagement_signal(self, provider, env_var):
        """Test that missing keys return an engagement-pending error signal."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_provider_key_or_ask(provider, "ProteinMPNN")

        assert result.startswith("Error: Missing API key")
        assert "[ENGAGEMENT_PENDING]" in result
        assert env_var in result
        assert provider in result
        assert "ProteinMPNN" in result

    @pytest.mark.parametrize(
        "provider,env_var",
        [
            ("Hugging Face", "HUGGINGFACE_API_KEY"),
            ("Google (Official)", "GOOGLE_API_KEY"),
        ],
    )
    def test_empty_string_key_treated_as_missing(self, provider, env_var):
        """Test that empty string keys are treated as missing."""
        with patch.dict(os.environ, {env_var: ""}, clear=True):
            result = get_provider_key_or_ask(provider, "TestModel")

        assert result.startswith("Error: Missing API key")
        assert "[ENGAGEMENT_PENDING]" in result

    def test_engagement_signal_json_format(self):
        """Test that the engagement signal contains valid JSON metadata."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_provider_key_or_ask("Hugging Face", "esm2_t6_8M_UR50D")

        assert '"type": "api_key_request"' in result
        assert '"env_var": "HUGGINGFACE_API_KEY"' in result
        assert "Please enter your API key for Hugging Face" in result


class TestGetProviderKeyOrAskEdgeCases:
    """Tests for edge cases and unmapped providers."""

    def test_unknown_provider_returns_no_key_required(self):
        """Test that unmapped providers do not require an API key."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_provider_key_or_ask("OpenAI", "gpt-4-science")

        assert result == "NO_KEY_REQUIRED"

    def test_unregistered_provider_name(self):
        """Test behavior for completely unknown provider names."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_provider_key_or_ask("Unknown Provider Inc", "SomeTool")

        assert result == "NO_KEY_REQUIRED"

    def test_whitespace_only_key_is_returned_as_is(self):
        """Test that whitespace-only keys are returned without validation."""
        with patch.dict(os.environ, {"HUGGINGFACE_API_KEY": "   "}, clear=True):
            result = get_provider_key_or_ask("Hugging Face", "ESM2")

        assert result == "   "

    def test_key_not_leaked_in_error_message(self):
        """Test that error messages do not expose existing key values."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_provider_key_or_ask("Google (Official)", "Gemini")

        assert "google-key" not in result.lower()
        assert result.startswith("Error:")
