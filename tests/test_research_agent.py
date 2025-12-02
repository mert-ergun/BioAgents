"""Comprehensive tests for the Research Agent with literature search capabilities."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from bioagents.agents.research_agent import create_research_agent


class TestResearchAgentCreation:
    """Tests for research agent creation and initialization."""

    @patch("bioagents.agents.research_agent.get_llm")
    def test_create_research_agent(self, mock_get_llm):
        """Test creating a research agent."""
        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_get_llm.return_value = mock_llm

        tools = [Mock()]
        agent = create_research_agent(tools)

        assert callable(agent)
        mock_get_llm.assert_called_once_with(prompt_name="research")
        mock_llm.bind_tools.assert_called_once_with(tools)

    @patch("bioagents.agents.research_agent.get_llm")
    def test_create_research_agent_with_multiple_tools(self, mock_get_llm):
        """Test creating research agent with multiple tools."""
        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_get_llm.return_value = mock_llm

        tools = [Mock(name=f"tool_{i}") for i in range(5)]
        agent = create_research_agent(tools)

        assert callable(agent)
        mock_llm.bind_tools.assert_called_once_with(tools)


class TestResearchAgentInvocation:
    """Tests for research agent invocation and message handling."""

    @patch("bioagents.agents.research_agent.get_llm")
    def test_research_agent_basic_invoke(self, mock_get_llm):
        """Test basic invocation of the research agent."""
        mock_llm = Mock()
        mock_bound_llm = Mock()
        mock_response = AIMessage(content="Research complete", name="Research")

        mock_llm.bind_tools = Mock(return_value=mock_bound_llm)
        mock_bound_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        tools = [Mock()]
        agent = create_research_agent(tools)

        state = {"messages": [HumanMessage(content="Search for p53 cancer papers")]}
        result = agent(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0] == mock_response
        mock_bound_llm.invoke.assert_called_once()

    @patch("bioagents.agents.research_agent.get_llm")
    def test_research_agent_includes_system_message(self, mock_get_llm):
        """Test that research agent includes system message."""
        mock_llm = Mock()
        mock_bound_llm = Mock()
        mock_response = AIMessage(content="Response")

        mock_llm.bind_tools = Mock(return_value=mock_bound_llm)
        mock_bound_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        tools = [Mock()]
        agent = create_research_agent(tools)

        state = {"messages": [HumanMessage(content="Test")]}
        agent(state)

        # Check that invoke was called with messages that include system message
        call_args = mock_bound_llm.invoke.call_args[0][0]
        assert len(call_args) >= 2  # System message + user message
        assert isinstance(call_args[0], SystemMessage)

    @patch("bioagents.agents.research_agent.get_llm")
    def test_research_agent_with_conversation_history(self, mock_get_llm):
        """Test research agent with multiple messages in conversation."""
        mock_llm = Mock()
        mock_bound_llm = Mock()
        mock_response = AIMessage(content="Follow-up research")

        mock_llm.bind_tools = Mock(return_value=mock_bound_llm)
        mock_bound_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        tools = [Mock()]
        agent = create_research_agent(tools)

        state = {
            "messages": [
                HumanMessage(content="Search for CRISPR papers"),
                AIMessage(content="Found 10 papers"),
                HumanMessage(content="Now search for gene editing ethics"),
            ]
        }
        result = agent(state)

        assert "messages" in result
        call_args = mock_bound_llm.invoke.call_args[0][0]
        # Should include system message + all conversation messages
        assert len(call_args) >= 4


class TestResearchAgentToolCalls:
    """Tests for research agent tool calling functionality."""

    @patch("bioagents.agents.research_agent.get_llm")
    def test_research_agent_literature_search_tool_call(self, mock_get_llm):
        """Test research agent making literature search tool call."""
        mock_llm = Mock()
        mock_bound_llm = Mock()

        mock_response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "tool_universe_find_tools",
                    "args": {"description": "search PubMed for articles", "limit": 5},
                    "id": "call_123",
                    "type": "tool_call",
                }
            ],
        )

        mock_llm.bind_tools = Mock(return_value=mock_bound_llm)
        mock_bound_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        tools = [Mock(name="tool_universe_find_tools")]
        agent = create_research_agent(tools)

        state = {"messages": [HumanMessage(content="Search for p53 papers")]}
        result = agent(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert hasattr(result["messages"][0], "tool_calls")
        assert len(result["messages"][0].tool_calls) == 1
        assert result["messages"][0].tool_calls[0]["name"] == "tool_universe_find_tools"

    @patch("bioagents.agents.research_agent.get_llm")
    def test_research_agent_multiple_tool_calls(self, mock_get_llm):
        """Test research agent making multiple tool calls."""
        mock_llm = Mock()
        mock_bound_llm = Mock()

        mock_response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "tool_universe_find_tools",
                    "args": {"description": "search PubMed", "limit": 5},
                    "id": "call_1",
                    "type": "tool_call",
                },
                {
                    "name": "tool_universe_call_tool",
                    "args": {
                        "tool_name": "PubMed_search_articles",
                        "arguments_json": '{"query": "p53 cancer"}',
                    },
                    "id": "call_2",
                    "type": "tool_call",
                },
            ],
        )

        mock_llm.bind_tools = Mock(return_value=mock_bound_llm)
        mock_bound_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        tools = [
            Mock(name="tool_universe_find_tools"),
            Mock(name="tool_universe_call_tool"),
        ]
        agent = create_research_agent(tools)

        state = {"messages": [HumanMessage(content="Search p53 in PubMed")]}
        result = agent(state)

        assert len(result["messages"][0].tool_calls) == 2

    @patch("bioagents.agents.research_agent.get_llm")
    def test_research_agent_uniprot_fetch(self, mock_get_llm):
        """Test research agent fetching UniProt data."""
        mock_llm = Mock()
        mock_bound_llm = Mock()

        mock_response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "fetch_uniprot_fasta",
                    "args": {"protein_id": "P04637"},
                    "id": "call_uniprot",
                    "type": "tool_call",
                }
            ],
        )

        mock_llm.bind_tools = Mock(return_value=mock_bound_llm)
        mock_bound_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        tools = [Mock(name="fetch_uniprot_fasta")]
        agent = create_research_agent(tools)

        state = {"messages": [HumanMessage(content="Fetch protein P04637")]}
        result = agent(state)

        assert result["messages"][0].tool_calls[0]["name"] == "fetch_uniprot_fasta"
        assert result["messages"][0].tool_calls[0]["args"]["protein_id"] == "P04637"


class TestResearchAgentLiteratureSearch:
    """Tests for literature search specific functionality."""

    @patch("bioagents.agents.research_agent.get_llm")
    def test_pubmed_search_workflow(self, mock_get_llm):
        """Test complete PubMed search workflow."""
        mock_llm = Mock()
        mock_bound_llm = Mock()

        # Simulate the workflow: find tools -> call tool
        responses = [
            # First call: find tools
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "tool_universe_find_tools",
                        "args": {"description": "search PubMed biomedical", "limit": 5},
                        "id": "call_find",
                        "type": "tool_call",
                    }
                ],
            ),
            # Second call: execute search
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "tool_universe_call_tool",
                        "args": {
                            "tool_name": "PubMed_search_articles",
                            "arguments_json": '{"query": "CRISPR", "max_results": 20}',
                        },
                        "id": "call_search",
                        "type": "tool_call",
                    }
                ],
            ),
        ]

        mock_llm.bind_tools = Mock(return_value=mock_bound_llm)
        mock_bound_llm.invoke = Mock(side_effect=responses)
        mock_get_llm.return_value = mock_llm

        tools = [
            Mock(name="tool_universe_find_tools"),
            Mock(name="tool_universe_call_tool"),
        ]
        agent = create_research_agent(tools)

        # First call
        state1 = {"messages": [HumanMessage(content="Search PubMed for CRISPR")]}
        result1 = agent(state1)
        assert result1["messages"][0].tool_calls[0]["name"] == "tool_universe_find_tools"

        # Second call (after tool result)
        state2 = {
            "messages": [
                HumanMessage(content="Search PubMed for CRISPR"),
                result1["messages"][0],
                ToolMessage(content="Found PubMed tool", tool_call_id="call_find"),
            ]
        }
        result2 = agent(state2)
        assert result2["messages"][0].tool_calls[0]["name"] == "tool_universe_call_tool"

    @patch("bioagents.agents.research_agent.get_llm")
    def test_multi_source_literature_search(self, mock_get_llm):
        """Test searching multiple literature sources."""
        mock_llm = Mock()
        mock_bound_llm = Mock()

        mock_response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "tool_universe_call_tool",
                    "args": {
                        "tool_name": "PubMed_search_articles",
                        "arguments_json": '{"query": "protein folding"}',
                    },
                    "id": "call_pubmed",
                    "type": "tool_call",
                },
                {
                    "name": "tool_universe_call_tool",
                    "args": {
                        "tool_name": "ArXiv_search_papers",
                        "arguments_json": '{"query": "protein folding ML"}',
                    },
                    "id": "call_arxiv",
                    "type": "tool_call",
                },
            ],
        )

        mock_llm.bind_tools = Mock(return_value=mock_bound_llm)
        mock_bound_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        tools = [Mock(name="tool_universe_call_tool")]
        agent = create_research_agent(tools)

        state = {"messages": [HumanMessage(content="Search PubMed and ArXiv for protein folding")]}
        result = agent(state)

        assert len(result["messages"][0].tool_calls) == 2
        tool_names = [tc["name"] for tc in result["messages"][0].tool_calls]
        assert all(name == "tool_universe_call_tool" for name in tool_names)


class TestResearchAgentErrorHandling:
    """Tests for error handling in research agent."""

    @patch("bioagents.agents.research_agent.get_llm")
    def test_research_agent_handles_empty_state(self, mock_get_llm):
        """Test research agent handles empty message state."""
        mock_llm = Mock()
        mock_bound_llm = Mock()
        mock_response = AIMessage(content="No messages to process")

        mock_llm.bind_tools = Mock(return_value=mock_bound_llm)
        mock_bound_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        tools = [Mock()]
        agent = create_research_agent(tools)

        state = {"messages": []}
        result = agent(state)

        assert "messages" in result
        mock_bound_llm.invoke.assert_called_once()

    @patch("bioagents.agents.research_agent.get_llm")
    def test_research_agent_handles_llm_error(self, mock_get_llm):
        """Test research agent handles LLM errors gracefully."""
        mock_llm = Mock()
        mock_bound_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_bound_llm)
        mock_bound_llm.invoke = Mock(side_effect=Exception("LLM Error"))
        mock_get_llm.return_value = mock_llm

        tools = [Mock()]
        agent = create_research_agent(tools)

        state = {"messages": [HumanMessage(content="Test")]}

        with pytest.raises(Exception) as exc_info:
            agent(state)
        assert "LLM Error" in str(exc_info.value)


class TestResearchAgentIntegration:
    """Integration tests for research agent with realistic scenarios."""

    @patch("bioagents.agents.research_agent.get_llm")
    def test_complete_literature_search_scenario(self, mock_get_llm):
        """Test complete literature search scenario from start to finish."""
        mock_llm = Mock()
        mock_bound_llm = Mock()

        # Simulate a complete workflow
        mock_response = AIMessage(
            content="Found 15 relevant papers on CRISPR gene editing",
            tool_calls=[
                {
                    "name": "tool_universe_find_tools",
                    "args": {"description": "search PubMed", "limit": 5},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        )

        mock_llm.bind_tools = Mock(return_value=mock_bound_llm)
        mock_bound_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        tools = [
            Mock(name="tool_universe_find_tools"),
            Mock(name="tool_universe_call_tool"),
        ]
        agent = create_research_agent(tools)

        state = {"messages": [HumanMessage(content="Search for recent CRISPR gene editing papers")]}
        result = agent(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        message = result["messages"][0]
        assert isinstance(message, AIMessage)
        assert "CRISPR" in message.content or len(message.tool_calls) > 0

    @patch("bioagents.agents.research_agent.get_llm")
    def test_combined_literature_and_data_retrieval(self, mock_get_llm):
        """Test scenario combining literature search and data retrieval."""
        mock_llm = Mock()
        mock_bound_llm = Mock()

        mock_response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "fetch_uniprot_fasta",
                    "args": {"protein_id": "P53_HUMAN"},
                    "id": "call_uniprot",
                    "type": "tool_call",
                },
                {
                    "name": "tool_universe_call_tool",
                    "args": {
                        "tool_name": "PubMed_search_articles",
                        "arguments_json": '{"query": "P53 cancer"}',
                    },
                    "id": "call_pubmed",
                    "type": "tool_call",
                },
            ],
        )

        mock_llm.bind_tools = Mock(return_value=mock_bound_llm)
        mock_bound_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        tools = [
            Mock(name="fetch_uniprot_fasta"),
            Mock(name="tool_universe_call_tool"),
        ]
        agent = create_research_agent(tools)

        state = {
            "messages": [
                HumanMessage(content="Get P53 protein sequence and search for related papers")
            ]
        }
        result = agent(state)

        assert len(result["messages"][0].tool_calls) == 2
        tool_names = {tc["name"] for tc in result["messages"][0].tool_calls}
        assert "fetch_uniprot_fasta" in tool_names
        assert "tool_universe_call_tool" in tool_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
