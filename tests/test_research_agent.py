"""Comprehensive tests for the Research Agent with literature search capabilities."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from bioagents.agents.research_agent import create_research_agent

_FINAL_RESEARCH_JSON = (
    '{"fetched_sequences":[],"literature_findings":"done","data_sources":[],'
    '"completeness":"full","next_steps":"","status":"success","error":null}'
)


def _build_mock_llm_for_sub_agent_flow(
    bound_llm, decomposer_content="1. Search for relevant data\n2. Analyze results"
):
    """
    Build a get_llm mock that handles the sub-agent architecture.

    get_llm is called with different prompt_name values:
      - "research" -> returns llm that has .bind_tools() (main agent)
      - "research_decomposer" -> returns decomposer LLM
      - "research_merger" -> returns merger LLM (when sub-agents finish without tools)

    Returns a mock get_llm function.
    """
    calls = {"count": 0}

    def mock_get_llm(prompt_name=None, **kwargs):
        calls["count"] += 1
        if prompt_name == "research_decomposer":
            # Decomposer: returns numbered list for parse_sub_tasks
            mock_decomposer = Mock()
            mock_decomposer.invoke = Mock(return_value=Mock(content=decomposer_content))
            return mock_decomposer
        elif prompt_name == "research_merger":
            # Merger: returns final synthesis
            mock_merger = Mock()
            mock_merger.invoke = Mock(return_value=AIMessage(content=_FINAL_RESEARCH_JSON))
            return mock_merger
        else:
            # Default: "research" or any other — return the base LLM with bind_tools
            return _mock_llm_with_bind_tools(bound_llm)

    return mock_get_llm


def _mock_llm_with_bind_tools(bound_llm):
    """Create a mock LLM that returns bound_llm from bind_tools."""
    mock_llm = Mock()
    mock_llm.bind_tools = Mock(return_value=bound_llm)
    return mock_llm


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
        """Test basic invocation of the research agent.

        In the sub-agent architecture, get_llm is called multiple times:
        research (main), research_decomposer (decompose), and sub-agent LLM calls.
        The sub-agents return AIMessages via create_retry_response.
        """
        # The bound LLM is used by sub-agents via create_retry_response
        mock_bound_llm = Mock()
        sub_agent_response = AIMessage(content="Found relevant research data")
        mock_bound_llm.invoke = Mock(return_value=sub_agent_response)

        mock_llm = _mock_llm_with_bind_tools(mock_bound_llm)

        def get_llm_side_effect(prompt_name=None, **kwargs):
            if prompt_name == "research_decomposer":
                mock_decomposer = Mock()
                mock_decomposer.invoke = Mock(
                    return_value=Mock(content="1. Search for relevant data")
                )
                return mock_decomposer
            elif prompt_name == "research_merger":
                mock_merger = Mock()
                mock_merger.invoke = Mock(return_value=AIMessage(content=_FINAL_RESEARCH_JSON))
                return mock_merger
            return mock_llm

        mock_get_llm.side_effect = get_llm_side_effect

        tools = [Mock()]
        agent = create_research_agent(tools)

        state = {"messages": [HumanMessage(content="Search for p53 cancer papers")]}
        result = agent(state)

        # The new architecture returns shared-memory-compatible dict
        assert "messages" in result
        assert "data" in result
        assert "tool_calls" in result
        assert "raw_output" in result
        assert len(result["messages"]) >= 1

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
        mock_bound_llm = Mock()
        sub_agent_response = AIMessage(content="Follow-up research data")
        mock_bound_llm.invoke = Mock(return_value=sub_agent_response)

        mock_llm = _mock_llm_with_bind_tools(mock_bound_llm)

        def get_llm_side_effect(prompt_name=None, **kwargs):
            if prompt_name == "research_decomposer":
                mock_decomposer = Mock()
                mock_decomposer.invoke = Mock(
                    return_value=Mock(
                        content="1. Search for CRISPR papers\n2. Search for gene editing ethics"
                    )
                )
                return mock_decomposer
            elif prompt_name == "research_merger":
                mock_merger = Mock()
                mock_merger.invoke = Mock(return_value=AIMessage(content=_FINAL_RESEARCH_JSON))
                return mock_merger
            return mock_llm

        mock_get_llm.side_effect = get_llm_side_effect

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
        assert "data" in result


class TestResearchAgentToolCalls:
    """Tests for research agent tool calling functionality."""

    @patch("bioagents.agents.research_agent.get_llm")
    def test_research_agent_literature_search_tool_call(self, mock_get_llm):
        """Test research agent making literature search tool call.

        In sub-agent architecture, the sub-agent LLM response includes tool_calls.
        These get aggregated and the combined AIMessage with tool_calls is returned.
        """
        mock_bound_llm = Mock()

        # Sub-agent returns a response with tool_calls
        sub_agent_response = AIMessage(
            content="Searching PubMed for articles",
            tool_calls=[
                {
                    "name": "tool_universe_find_tools",
                    "args": {"description": "search PubMed for articles", "limit": 5},
                    "id": "call_123",
                    "type": "tool_call",
                }
            ],
        )
        mock_bound_llm.invoke = Mock(return_value=sub_agent_response)

        mock_llm = _mock_llm_with_bind_tools(mock_bound_llm)

        def get_llm_side_effect(prompt_name=None, **kwargs):
            if prompt_name == "research_decomposer":
                mock_decomposer = Mock()
                mock_decomposer.invoke = Mock(
                    return_value=Mock(content="1. Search PubMed for articles about p53")
                )
                return mock_decomposer
            return mock_llm

        mock_get_llm.side_effect = get_llm_side_effect

        tools = [Mock(name="tool_universe_find_tools")]
        agent = create_research_agent(tools)

        state = {"messages": [HumanMessage(content="Search for p53 papers")]}
        result = agent(state)

        # Sub-agent had tool_calls -> aggregated result has tool_calls forwarded
        assert "messages" in result
        assert "tool_calls" in result
        assert "tool_universe_find_tools" in result["tool_calls"]

    @patch("bioagents.agents.research_agent.get_llm")
    def test_research_agent_multiple_tool_calls(self, mock_get_llm):
        """Test research agent making multiple tool calls."""
        mock_bound_llm = Mock()

        sub_agent_response = AIMessage(
            content="Searching multiple databases",
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
        mock_bound_llm.invoke = Mock(return_value=sub_agent_response)

        mock_llm = _mock_llm_with_bind_tools(mock_bound_llm)

        def get_llm_side_effect(prompt_name=None, **kwargs):
            if prompt_name == "research_decomposer":
                mock_decomposer = Mock()
                mock_decomposer.invoke = Mock(
                    return_value=Mock(content="1. Search PubMed for p53 cancer articles")
                )
                return mock_decomposer
            return mock_llm

        mock_get_llm.side_effect = get_llm_side_effect

        tools = [
            Mock(name="tool_universe_find_tools"),
            Mock(name="tool_universe_call_tool"),
        ]
        agent = create_research_agent(tools)

        state = {"messages": [HumanMessage(content="Search p53 in PubMed")]}
        result = agent(state)

        assert "tool_calls" in result
        assert result["tool_calls"].count("tool_universe_find_tools") >= 1
        assert result["tool_calls"].count("tool_universe_call_tool") >= 1

    @patch("bioagents.agents.research_agent.get_llm")
    def test_research_agent_uniprot_fetch(self, mock_get_llm):
        """Test research agent fetching UniProt data."""
        mock_bound_llm = Mock()

        sub_agent_response = AIMessage(
            content="Fetching UniProt data",
            tool_calls=[
                {
                    "name": "fetch_uniprot_fasta",
                    "args": {"protein_id": "P04637"},
                    "id": "call_uniprot",
                    "type": "tool_call",
                }
            ],
        )
        mock_bound_llm.invoke = Mock(return_value=sub_agent_response)

        mock_llm = _mock_llm_with_bind_tools(mock_bound_llm)

        def get_llm_side_effect(prompt_name=None, **kwargs):
            if prompt_name == "research_decomposer":
                mock_decomposer = Mock()
                mock_decomposer.invoke = Mock(
                    return_value=Mock(content="1. Fetch protein P04637 from UniProt")
                )
                return mock_decomposer
            return mock_llm

        mock_get_llm.side_effect = get_llm_side_effect

        tools = [Mock(name="fetch_uniprot_fasta")]
        agent = create_research_agent(tools)

        state = {"messages": [HumanMessage(content="Fetch protein P04637")]}
        result = agent(state)

        assert "tool_calls" in result
        assert "fetch_uniprot_fasta" in result["tool_calls"]


class TestResearchAgentLiteratureSearch:
    """Tests for literature search specific functionality."""

    @patch("bioagents.agents.research_agent.get_llm")
    def test_pubmed_search_workflow(self, mock_get_llm):
        """Test complete PubMed search workflow."""
        mock_bound_llm = Mock()

        # Sub-agent returns tool calls on first invoke
        responses = [
            # First invoke: sub-agent wants to find tools
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
            # Second invoke: sub-agent wants to call tool
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

        mock_bound_llm.invoke = Mock(side_effect=responses)
        mock_llm = _mock_llm_with_bind_tools(mock_bound_llm)

        def get_llm_side_effect(prompt_name=None, **kwargs):
            if prompt_name == "research_decomposer":
                mock_decomposer = Mock()
                mock_decomposer.invoke = Mock(
                    return_value=Mock(content="1. Search PubMed for CRISPR articles")
                )
                return mock_decomposer
            return mock_llm

        mock_get_llm.side_effect = get_llm_side_effect

        tools = [
            Mock(name="tool_universe_find_tools"),
            Mock(name="tool_universe_call_tool"),
        ]
        agent = create_research_agent(tools)

        # First call
        state1 = {"messages": [HumanMessage(content="Search PubMed for CRISPR")]}
        result1 = agent(state1)
        assert "tool_universe_find_tools" in result1["tool_calls"]

        # Second call (after tool result) — continues single-agent tool loop; mock one more pair
        mock_bound_llm.invoke = Mock(
            side_effect=[
                responses[1],
                AIMessage(content=_FINAL_RESEARCH_JSON),
            ]
        )
        state2 = {
            "messages": [
                HumanMessage(content="Search PubMed for CRISPR"),
                result1["messages"][0],
                ToolMessage(content="Found PubMed tool", tool_call_id="call_find_0"),
            ]
        }
        result2 = agent(state2)
        assert "tool_universe_call_tool" in result2["tool_calls"]

    @patch("bioagents.agents.research_agent.get_llm")
    def test_multi_source_literature_search(self, mock_get_llm):
        """Test searching multiple literature sources."""
        mock_bound_llm = Mock()

        sub_agent_response = AIMessage(
            content="Searching multiple sources",
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
        mock_bound_llm.invoke = Mock(return_value=sub_agent_response)

        mock_llm = _mock_llm_with_bind_tools(mock_bound_llm)

        def get_llm_side_effect(prompt_name=None, **kwargs):
            if prompt_name == "research_decomposer":
                mock_decomposer = Mock()
                mock_decomposer.invoke = Mock(
                    return_value=Mock(content="1. Search PubMed and ArXiv for protein folding")
                )
                return mock_decomposer
            return mock_llm

        mock_get_llm.side_effect = get_llm_side_effect

        tools = [Mock(name="tool_universe_call_tool")]
        agent = create_research_agent(tools)

        state = {"messages": [HumanMessage(content="Search PubMed and ArXiv for protein folding")]}
        result = agent(state)

        assert "tool_calls" in result
        assert result["tool_calls"].count("tool_universe_call_tool") >= 2


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
        mock_bound_llm.invoke.assert_not_called()

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
        result = agent(state)
        assert result["error"] == "LLM Error"


class TestResearchAgentIntegration:
    """Integration tests for research agent with realistic scenarios."""

    @patch("bioagents.agents.research_agent.get_llm")
    def test_complete_literature_search_scenario(self, mock_get_llm):
        """Test complete literature search scenario from start to finish."""
        mock_bound_llm = Mock()

        # Sub-agent returns tool calls
        sub_agent_response = AIMessage(
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
        mock_bound_llm.invoke = Mock(return_value=sub_agent_response)

        mock_llm = _mock_llm_with_bind_tools(mock_bound_llm)

        def get_llm_side_effect(prompt_name=None, **kwargs):
            if prompt_name == "research_decomposer":
                mock_decomposer = Mock()
                mock_decomposer.invoke = Mock(
                    return_value=Mock(content="1. Search for CRISPR gene editing papers")
                )
                return mock_decomposer
            return mock_llm

        mock_get_llm.side_effect = get_llm_side_effect

        tools = [
            Mock(name="tool_universe_find_tools"),
            Mock(name="tool_universe_call_tool"),
        ]
        agent = create_research_agent(tools)

        state = {"messages": [HumanMessage(content="Search for recent CRISPR gene editing papers")]}
        result = agent(state)

        assert "messages" in result
        assert "data" in result
        assert len(result["messages"]) >= 1
        message = result["messages"][0]
        assert isinstance(message, AIMessage)
        assert "CRISPR" in message.content or "tool_universe_find_tools" in result["tool_calls"]

    @patch("bioagents.agents.research_agent.get_llm")
    def test_combined_literature_and_data_retrieval(self, mock_get_llm):
        """Test scenario combining literature search and data retrieval."""
        mock_bound_llm = Mock()

        sub_agent_response = AIMessage(
            content="Fetching P53 data and searching literature",
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
        mock_bound_llm.invoke = Mock(return_value=sub_agent_response)

        mock_llm = _mock_llm_with_bind_tools(mock_bound_llm)

        def get_llm_side_effect(prompt_name=None, **kwargs):
            if prompt_name == "research_decomposer":
                mock_decomposer = Mock()
                mock_decomposer.invoke = Mock(
                    return_value=Mock(
                        content="1. Get P53 protein sequence\n2. Search for related papers"
                    )
                )
                return mock_decomposer
            return mock_llm

        mock_get_llm.side_effect = get_llm_side_effect

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

        assert "tool_calls" in result
        tool_names_used = set(result["tool_calls"])
        assert "fetch_uniprot_fasta" in tool_names_used
        assert "tool_universe_call_tool" in tool_names_used


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
