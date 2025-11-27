"""Tests for LangSmith configuration module."""

import os
import warnings
from unittest.mock import MagicMock, patch

import pytest

from bioagents.llms.langsmith_config import (
    get_langsmith_api_key,
    get_langsmith_callbacks,
    get_langsmith_config,
    get_langsmith_endpoint,
    get_langsmith_project,
    is_langsmith_enabled,
    print_langsmith_status,
    setup_langsmith_environment,
    validate_langsmith_config,
)


class TestIsLangSmithEnabled:
    """Tests for is_langsmith_enabled function."""

    @pytest.mark.parametrize("env_value", ["true", "1", "yes", "on", "TRUE", "On"])
    def test_enabled_values(self, env_value):
        """Test values that should enable LangSmith."""
        with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": env_value}, clear=True):
            assert is_langsmith_enabled() is True

    @pytest.mark.parametrize("env_value", ["false", "0", "no", "off", "FALSE", "junk", ""])
    def test_disabled_values(self, env_value):
        """Test values that should disable LangSmith."""
        with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": env_value}, clear=True):
            assert is_langsmith_enabled() is False

    def test_default_disabled(self):
        """Test that it is disabled by default if env var is missing."""
        with patch.dict(os.environ, {}, clear=True):
            assert is_langsmith_enabled() is False


class TestGetLangSmithApiKey:
    """Tests for get_langsmith_api_key function."""

    def test_langchain_api_key(self):
        """Test getting key from LANGCHAIN_API_KEY."""
        with patch.dict(os.environ, {"LANGCHAIN_API_KEY": "test-key"}, clear=True):
            assert get_langsmith_api_key() == "test-key"

    def test_langsmith_api_key(self):
        """Test getting key from LANGSMITH_API_KEY."""
        with patch.dict(os.environ, {"LANGSMITH_API_KEY": "test-key"}, clear=True):
            assert get_langsmith_api_key() == "test-key"

    def test_langchain_preference(self):
        """Test that LANGCHAIN_API_KEY takes precedence."""
        with patch.dict(
            os.environ,
            {"LANGCHAIN_API_KEY": "langchain-key", "LANGSMITH_API_KEY": "langsmith-key"},
            clear=True,
        ):
            assert get_langsmith_api_key() == "langchain-key"

    def test_no_key(self):
        """Test returning None when no key is set."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_langsmith_api_key() is None


class TestGetLangSmithProject:
    """Tests for get_langsmith_project function."""

    def test_default_project(self):
        """Test default project name."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_langsmith_project() == "bioagents"

    def test_langchain_project(self):
        """Test getting project from LANGCHAIN_PROJECT."""
        with patch.dict(os.environ, {"LANGCHAIN_PROJECT": "my-project"}, clear=True):
            assert get_langsmith_project() == "my-project"

    def test_langsmith_project(self):
        """Test getting project from LANGSMITH_PROJECT."""
        with patch.dict(os.environ, {"LANGSMITH_PROJECT": "my-project"}, clear=True):
            assert get_langsmith_project() == "my-project"


class TestGetLangSmithEndpoint:
    """Tests for get_langsmith_endpoint function."""

    def test_default_none(self):
        """Test default is None."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_langsmith_endpoint() is None

    def test_langchain_endpoint(self):
        """Test getting endpoint from LANGCHAIN_ENDPOINT."""
        url = "https://custom.endpoint"
        with patch.dict(os.environ, {"LANGCHAIN_ENDPOINT": url}, clear=True):
            assert get_langsmith_endpoint() == url

    def test_langsmith_endpoint(self):
        """Test getting endpoint from LANGSMITH_ENDPOINT."""
        url = "https://custom.endpoint"
        with patch.dict(os.environ, {"LANGSMITH_ENDPOINT": url}, clear=True):
            assert get_langsmith_endpoint() == url


class TestValidateLangSmithConfig:
    """Tests for validate_langsmith_config function."""

    def test_valid_when_disabled(self):
        """Test valid when tracing is disabled, even without key."""
        with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "false"}, clear=True):
            is_valid, error = validate_langsmith_config()
            assert is_valid is True
            assert error == ""

    def test_valid_with_key(self):
        """Test valid when enabled and key is present."""
        with patch.dict(
            os.environ,
            {"LANGCHAIN_TRACING_V2": "true", "LANGCHAIN_API_KEY": "test-key"},
            clear=True,
        ):
            is_valid, error = validate_langsmith_config()
            assert is_valid is True
            assert error == ""

    def test_invalid_missing_key(self):
        """Test invalid when enabled but key is missing."""
        with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"}, clear=True):
            is_valid, error = validate_langsmith_config()
            assert is_valid is False
            assert "API KEY" in error.upper()


class TestGetLangSmithConfig:
    """Tests for get_langsmith_config function."""

    def test_empty_when_disabled(self):
        """Test returns empty dict when disabled."""
        with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "false"}, clear=True):
            config = get_langsmith_config()
            assert config == {}

    def test_raises_when_invalid(self):
        """Test raises ValueError when enabled but missing key."""
        with (
            patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"}, clear=True),
            pytest.raises(ValueError, match="API key"),
        ):
            get_langsmith_config()

    def test_returns_config_when_valid(self):
        """Test returns correct config when valid."""
        with patch.dict(
            os.environ,
            {
                "LANGCHAIN_TRACING_V2": "true",
                "LANGCHAIN_API_KEY": "test-key",
                "LANGCHAIN_PROJECT": "test-project",
                "LANGCHAIN_ENDPOINT": "https://test.endpoint",
            },
            clear=True,
        ):
            config = get_langsmith_config()

            assert "configurable" in config
            assert "langsmith" in config["configurable"]
            assert config["configurable"]["langsmith"]["enabled"] is True
            assert config["configurable"]["langsmith"]["project"] == "test-project"

            # Should set env vars
            assert os.environ["LANGCHAIN_API_KEY"] == "test-key"
            assert os.environ["LANGCHAIN_PROJECT"] == "test-project"
            assert os.environ["LANGCHAIN_ENDPOINT"] == "https://test.endpoint"


class TestGetLangSmithCallbacks:
    """Tests for get_langsmith_callbacks function."""

    def test_none_when_disabled(self):
        """Test returns None when disabled."""
        with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "false"}, clear=True):
            assert get_langsmith_callbacks() is None

    def test_none_with_warning_when_invalid(self):
        """Test returns None and warns when enabled but invalid."""
        with (
            patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"}, clear=True),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            result = get_langsmith_callbacks()

            assert result is None
            assert len(w) > 0
            assert issubclass(w[-1].category, UserWarning)
            assert "API KEY" in str(w[-1].message).upper()

    def test_returns_tracer_when_valid(self):
        """Test returns tracer when valid."""
        with (
            patch.dict(
                os.environ,
                {"LANGCHAIN_TRACING_V2": "true", "LANGCHAIN_API_KEY": "test-key"},
                clear=True,
            ),
            patch("bioagents.llms.langsmith_config.LangChainTracer") as MockTracer,
        ):
            mock_tracer_instance = MagicMock()
            MockTracer.return_value = mock_tracer_instance

            callbacks = get_langsmith_callbacks()

            assert callbacks is not None
            assert len(callbacks) == 1
            assert callbacks[0] == mock_tracer_instance
            MockTracer.assert_called_once()


class TestSetupLangSmithEnvironment:
    """Tests for setup_langsmith_environment function."""

    def test_does_nothing_when_disabled(self):
        """Test does nothing when disabled."""
        with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "false"}, clear=True):
            setup_langsmith_environment()
            # Should not change anything
            assert os.environ.get("LANGCHAIN_TRACING_V2") == "false"

    def test_raises_when_invalid(self):
        """Test raises ValueError when enabled but invalid."""
        with (
            patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"}, clear=True),
            pytest.raises(ValueError, match="API key"),
        ):
            setup_langsmith_environment()

    def test_sets_env_vars_when_valid(self):
        """Test sets all environment variables when valid."""
        initial_env = {
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_API_KEY": "test-key",
            "LANGCHAIN_PROJECT": "test-project",
            "LANGCHAIN_ENDPOINT": "https://test.endpoint",
        }
        with patch.dict(os.environ, initial_env, clear=True):
            setup_langsmith_environment()

            # Check mirrored variables
            assert os.environ["LANGSMITH_API_KEY"] == "test-key"
            assert os.environ["LANGSMITH_PROJECT"] == "test-project"
            assert os.environ["LANGSMITH_ENDPOINT"] == "https://test.endpoint"
            assert os.environ["LANGCHAIN_TRACING_V2"] == "true"


class TestPrintLangSmithStatus:
    """Tests for print_langsmith_status function."""

    def test_print_when_disabled(self, capsys):
        """Test output when disabled."""
        with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "false"}, clear=True):
            print_langsmith_status()
            captured = capsys.readouterr()
            assert "Enabled: False" in captured.out
            assert "To enable LangSmith monitoring" in captured.out

    def test_print_when_enabled_and_valid(self, capsys):
        """Test output when enabled and valid."""
        with patch.dict(
            os.environ,
            {
                "LANGCHAIN_TRACING_V2": "true",
                "LANGCHAIN_API_KEY": "test-key",
                "LANGCHAIN_PROJECT": "test-project",
            },
            clear=True,
        ):
            print_langsmith_status()
            captured = capsys.readouterr()
            assert "Enabled: True" in captured.out
            assert "API Key: ✓ Set" in captured.out
            assert "Project: test-project" in captured.out
            assert "Configuration valid" in captured.out

    def test_print_when_enabled_but_invalid(self, capsys):
        """Test output when enabled but invalid (missing key)."""
        with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"}, clear=True):
            print_langsmith_status()
            captured = capsys.readouterr()
            assert "Enabled: True" in captured.out
            assert "API Key: ✗ Missing" in captured.out
            assert "Warning:" in captured.out
