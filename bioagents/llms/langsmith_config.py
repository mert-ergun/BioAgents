"""LangSmith configuration and tracing setup for monitoring agent execution.

This module provides robust LangSmith integration for monitoring the BioAgents
multi-agent system. It ensures reliable token usage tracking and provides
comprehensive observability through LangChain's monitoring platform.

Usage:
    # Enable LangSmith tracing (set LANGCHAIN_TRACING_V2=true in .env)
    from bioagents.llms.langsmith_config import get_langsmith_config, is_langsmith_enabled

    if is_langsmith_enabled():
        config = get_langsmith_config()
        # Use config in graph.stream() or graph.invoke()
"""

import os
import warnings
from typing import Any

from langchain_core.tracers import LangChainTracer


def is_langsmith_enabled() -> bool:
    """
    Check if LangSmith tracing is enabled via environment variables.

    Returns:
        True if LangSmith tracing should be enabled, False otherwise
    """
    tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower()
    return tracing_v2 in ("true", "1", "yes", "on")


def get_langsmith_api_key() -> str | None:
    """
    Get LangSmith API key from environment variables.

    Checks both LANGCHAIN_API_KEY and LANGSMITH_API_KEY (both are supported).

    Returns:
        API key string if found, None otherwise
    """
    # Check both variable names (LangChain supports both)
    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    return api_key


def get_langsmith_project() -> str:
    """
    Get LangSmith project name from environment variables.

    Returns:
        Project name string, defaults to 'bioagents' if not set
    """
    project = os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT")
    return project or "bioagents"


def get_langsmith_endpoint() -> str | None:
    """
    Get LangSmith API endpoint from environment variables.

    Returns:
        Endpoint URL string if set, None to use default
    """
    endpoint = os.getenv("LANGCHAIN_ENDPOINT") or os.getenv("LANGSMITH_ENDPOINT")
    return endpoint if endpoint else None


def validate_langsmith_config() -> tuple[bool, str]:
    """
    Validate LangSmith configuration.

    Returns:
        Tuple of (is_valid, error_message)
        If is_valid is True, error_message is empty
    """
    if not is_langsmith_enabled():
        return True, ""  # Not enabled, so no validation needed

    api_key = get_langsmith_api_key()
    if not api_key:
        return False, (
            "LangSmith tracing is enabled (LANGCHAIN_TRACING_V2=true) but "
            "LANGCHAIN_API_KEY or LANGSMITH_API_KEY is not set. "
            "Please set one of these environment variables with your LangSmith API key. "
            "Get your API key from: https://smith.langchain.com/settings"
        )

    return True, ""


def get_langsmith_config() -> dict[str, Any]:
    """
    Get LangSmith configuration dictionary for use with LangGraph.

    This creates a RunnableConfig with LangSmith callbacks that will:
    - Track all LLM calls with token usage
    - Monitor agent execution flow
    - Record tool calls and responses
    - Provide observability in LangSmith dashboard

    Returns:
        Dictionary with 'configurable' key containing LangSmith settings

    Raises:
        ValueError: If LangSmith is enabled but API key is missing
    """
    if not is_langsmith_enabled():
        return {}

    is_valid, error_msg = validate_langsmith_config()
    if not is_valid:
        raise ValueError(error_msg)

    # Set environment variables for LangChain to pick up
    # LangChain automatically reads these when creating tracers
    api_key = get_langsmith_api_key()
    if api_key:
        os.environ["LANGCHAIN_API_KEY"] = api_key

    project = get_langsmith_project()
    if project:
        os.environ["LANGCHAIN_PROJECT"] = project

    endpoint = get_langsmith_endpoint()
    if endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint

    # Return config dict for LangGraph
    # LangGraph will automatically use LangChainTracer when LANGCHAIN_TRACING_V2 is set
    return {
        "configurable": {
            "langsmith": {
                "enabled": True,
                "project": project,
            }
        }
    }


def get_langsmith_callbacks() -> list[LangChainTracer] | None:
    """
    Get LangSmith callback handlers for explicit use.

    This is useful when you need to explicitly pass callbacks to LangChain
    components. For LangGraph, the environment variables are usually sufficient.

    Returns:
        List of LangChainTracer callbacks if LangSmith is enabled, None otherwise
    """
    if not is_langsmith_enabled():
        return None

    is_valid, error_msg = validate_langsmith_config()
    if not is_valid:
        warnings.warn(error_msg, UserWarning, stacklevel=2)
        return None

    # Create tracer with explicit configuration
    tracer = LangChainTracer(
        project_name=get_langsmith_project(),
    )

    return [tracer]


def setup_langsmith_environment() -> None:
    """
    Set up LangSmith environment variables for automatic tracing.

    This function ensures all necessary environment variables are set
    so that LangChain components automatically use LangSmith tracing.
    Call this early in your application (e.g., in main.py after load_dotenv()).

    Raises:
        ValueError: If LangSmith is enabled but API key is missing
    """
    if not is_langsmith_enabled():
        return

    is_valid, error_msg = validate_langsmith_config()
    if not is_valid:
        raise ValueError(error_msg)

    # Ensure environment variables are set for LangChain auto-detection
    api_key = get_langsmith_api_key()
    if api_key:
        os.environ["LANGCHAIN_API_KEY"] = api_key
        # Also set LANGSMITH_API_KEY for compatibility
        os.environ["LANGSMITH_API_KEY"] = api_key

    project = get_langsmith_project()
    if project:
        os.environ["LANGCHAIN_PROJECT"] = project
        os.environ["LANGSMITH_PROJECT"] = project

    endpoint = get_langsmith_endpoint()
    if endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint
        os.environ["LANGSMITH_ENDPOINT"] = endpoint

    # Enable tracing v2
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    print(f"✓ LangSmith monitoring enabled (project: {project})")
    print("  View traces at: https://smith.langchain.com")


def print_langsmith_status() -> None:
    """Print current LangSmith configuration status."""
    enabled = is_langsmith_enabled()
    print("\n" + "=" * 60)
    print("LANGSMITH MONITORING STATUS")
    print("=" * 60)
    print(f"Enabled: {enabled}")

    if enabled:
        api_key = get_langsmith_api_key()
        project = get_langsmith_project()
        endpoint = get_langsmith_endpoint() or "https://api.smith.langchain.com (default)"

        print(f"API Key: {'✓ Set' if api_key else '✗ Missing'}")
        print(f"Project: {project}")
        print(f"Endpoint: {endpoint}")

        is_valid, error_msg = validate_langsmith_config()
        if not is_valid:
            print(f"\n⚠ Warning: {error_msg}")
        else:
            print("\n✓ Configuration valid")
            print("  View traces at: https://smith.langchain.com")
    else:
        print("\nTo enable LangSmith monitoring:")
        print("  1. Set LANGCHAIN_TRACING_V2=true in your .env file")
        print("  2. Set LANGCHAIN_API_KEY with your LangSmith API key")
        print("  3. (Optional) Set LANGCHAIN_PROJECT to organize runs")
        print("  4. Get your API key from: https://smith.langchain.com/settings")

    print("=" * 60 + "\n")
