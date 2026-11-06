"""Run selected-tool contract checks without live APIs or model calculations."""

import os

if __name__ == "__main__":
    os.environ["BIOAGENTS_TOOL_MODE"] = "selected"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    import pytest

    raise SystemExit(
        pytest.main(
            [
                "-q",
                "--no-cov",
                "--maxfail=5",
                "tests/test_modular_tool_selection.py",
                "tests/test_agent_manifest.py",
                "tests/test_prompt_loader.py",
                "tests/test_custom_tool_workspace.py",
                "tests/test_summary_agent.py",
                "tests/test_tool_result_status.py",
                "tests/test_tool_universe.py",
                "tests/test_tool_discovery.py",
                "tests/test_supervisor_completion.py",
                "tests/test_dependency_events.py",
                "tests/test_dependency_audit_coverage.py",
                "tests/test_dependency_inference.py",
            ]
        )
    )
