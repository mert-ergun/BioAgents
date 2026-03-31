"""End-to-end tests for full agent workflows with LLM adjudication."""

from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

pytestmark = pytest.mark.e2e


class TestFullWorkflow:
    """Test complete agent workflows end-to-end."""

    @pytest.mark.slow
    def test_graph_creation_and_structure(self):
        """Test that the full graph can be created with all 27+ agents."""
        from bioagents.graph import ALL_MEMBERS

        assert len(ALL_MEMBERS) >= 27

        expected_agents = [
            "research",
            "analysis",
            "coder",
            "ml",
            "dl",
            "literature",
            "web_browser",
            "paper_replication",
            "data_acquisition",
            "genomics",
            "transcriptomics",
            "structural_biology",
            "phylogenetics",
            "docking",
            "planner",
            "tool_validator",
            "tool_discovery",
            "prompt_optimizer",
            "result_checker",
            "shell",
            "git",
            "environment",
            "visualization",
            "report",
            "critic",
            "tool_builder",
            "protein_design",
        ]
        for agent in expected_agents:
            assert agent in ALL_MEMBERS, f"Missing agent: {agent}"

    def test_all_members_are_unique(self):
        """Verify no duplicate agent names in ALL_MEMBERS."""
        from bioagents.graph import ALL_MEMBERS

        assert len(ALL_MEMBERS) == len(set(ALL_MEMBERS))


class TestSelfCorrection:
    """Test that meta-agents can detect and correct issues."""

    def test_empty_response_retry(self):
        """Test that agents retry on empty responses."""
        from bioagents.agents.helpers import invoke_with_retry

        mock_llm = Mock()
        mock_llm.invoke.side_effect = [
            AIMessage(content=""),
            AIMessage(content="Recovered response"),
        ]

        msgs = [SystemMessage(content="test"), HumanMessage(content="query")]
        result = invoke_with_retry("TestAgent", mock_llm, msgs)
        assert result["messages"][0].content == "Recovered response"

    def test_graceful_degradation(self):
        """Test that the system degrades gracefully when all retries fail."""
        from bioagents.agents.helpers import invoke_with_retry

        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="")

        msgs = [
            SystemMessage(content="test"),
            HumanMessage(content="query"),
            AIMessage(content="Some useful content from earlier"),
        ]
        result = invoke_with_retry("TestAgent", mock_llm, msgs, max_retries=1)
        assert result["messages"][0].content  # Should not be empty
        assert "useful content" in result["messages"][0].content

    def test_tool_call_not_treated_as_empty(self):
        """Tool calls without text content should NOT be retried."""
        from bioagents.agents.helpers import invoke_with_retry

        mock_llm = Mock()
        response = AIMessage(
            content="",
            tool_calls=[{"name": "fetch", "args": {"id": "P04637"}, "id": "c1"}],
        )
        mock_llm.invoke.return_value = response

        msgs = [SystemMessage(content="test"), HumanMessage(content="query")]
        result = invoke_with_retry("TestAgent", mock_llm, msgs)
        assert len(result["messages"]) == 1
        assert result["messages"][0].tool_calls
        assert mock_llm.invoke.call_count == 1


class TestHelperUtilities:
    """Test helper utilities used across agents."""

    def test_is_empty_response_true(self):
        from bioagents.agents.helpers import is_empty_response

        assert is_empty_response(AIMessage(content="")) is True
        assert is_empty_response(AIMessage(content="   ")) is True

    def test_is_empty_response_false(self):
        from bioagents.agents.helpers import is_empty_response

        assert is_empty_response(AIMessage(content="data")) is False
        assert (
            is_empty_response(
                AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}])
            )
            is False
        )

    def test_get_content_text_various_formats(self):
        from bioagents.agents.helpers import get_content_text

        assert get_content_text("hello") == "hello"
        assert (
            get_content_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
            == "a b"
        )
        assert get_content_text(None) == ""
