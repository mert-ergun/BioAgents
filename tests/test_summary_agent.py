"""Comprehensive tests for the Summary Agent with memory-based summary generation."""

from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from bioagents.agents.summary_agent import (
    assess_execution_complexity,
    create_summary_agent,
    format_memory_for_summary,
)


class TestSummaryAgentCreation:
    """Tests for summary agent creation and initialization."""

    @patch("bioagents.agents.summary_agent.get_llm")
    @patch("bioagents.agents.summary_agent.load_prompt")
    def test_create_summary_agent(self, mock_load_prompt, mock_get_llm):
        """Test creating a summary agent."""
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        mock_load_prompt.return_value = "Summary system prompt"

        agent = create_summary_agent()

        assert callable(agent)
        mock_get_llm.assert_called_once_with(prompt_name="summary")
        mock_load_prompt.assert_called_once_with("summary")

    @patch("bioagents.agents.summary_agent.get_llm")
    @patch("bioagents.agents.summary_agent.load_prompt")
    def test_create_summary_agent_returns_callable_node(self, mock_load_prompt, mock_get_llm):
        """Test that create_summary_agent returns a callable node function."""
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        mock_load_prompt.return_value = "Summary prompt"

        node = create_summary_agent()

        # Verify it's a callable function (the summary_node)
        assert callable(node)


class TestSummaryAgentWithMemory:
    """Tests for summary agent with populated memory (complex execution)."""

    @patch("bioagents.agents.summary_agent.get_llm")
    @patch("bioagents.agents.summary_agent.load_prompt")
    def test_summary_agent_with_populated_memory(self, mock_load_prompt, mock_get_llm):
        """Test summary agent processing with populated memory."""
        mock_llm = Mock()
        mock_response = AIMessage(content="Protein is a complex molecule", name="Summary")
        mock_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm
        mock_load_prompt.return_value = "Summary prompt"

        agent = create_summary_agent()

        # Simulate memory with successful research agent output
        memory = {
            "research": {
                "status": "success",
                "data": {"papers": ["Paper 1", "Paper 2"], "query": "What is protein?"},
                "raw_output": "Found 2 papers about proteins",
                "tool_calls": [],
            }
        }

        state = {
            "messages": [HumanMessage(content="What is protein?")],
            "memory": memory,
        }

        result = agent(state)

        # Verify result structure
        assert "data" in result
        assert "raw_output" in result
        assert "tool_calls" in result
        assert "error" in result

        # Verify response content
        assert result["data"]["summary"] == "Protein is a complex molecule"
        assert result["raw_output"] == "Protein is a complex molecule"
        assert result["tool_calls"] == []
        assert result["error"] is None

        # Verify LLM was invoked
        mock_llm.invoke.assert_called_once()

        # Verify system message was included in the call
        call_args = mock_llm.invoke.call_args[0][0]
        assert len(call_args) >= 2
        assert isinstance(call_args[0], SystemMessage)

    @patch("bioagents.agents.summary_agent.get_llm")
    @patch("bioagents.agents.summary_agent.load_prompt")
    def test_summary_agent_complex_execution_path(self, mock_load_prompt, mock_get_llm):
        """Test summary agent with complex multi-agent memory."""
        mock_llm = Mock()
        mock_response = AIMessage(
            content="Proteins are polymers of amino acids with complex structures",
            name="Summary",
        )
        mock_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm
        mock_load_prompt.return_value = "Summary prompt"

        agent = create_summary_agent()

        # Simulate memory with multiple successful agent outputs
        memory = {
            "research": {
                "status": "success",
                "data": {"papers": 5, "query": "What is protein?"},
                "raw_output": "Found papers on protein structure",
                "tool_calls": [{"name": "search_literature", "args": {}}],
            },
            "analysis": {
                "status": "success",
                "data": {"structure": "alpha-helix", "properties": ["polar", "charged"]},
                "raw_output": "Analyzed protein properties",
                "tool_calls": [{"name": "analyze_sequence", "args": {}}],
            },
        }

        state = {
            "messages": [HumanMessage(content="What is protein?")],
            "memory": memory,
        }

        result = agent(state)

        assert result["error"] is None
        assert "summary" in result["data"]

        # Verify system message includes complexity assessment
        call_args = mock_llm.invoke.call_args[0][0]
        system_message_content = call_args[0].content
        assert (
            "EXECUTION COMPLEXITY" in system_message_content
            or "complex" in system_message_content.lower()
        )

    @patch("bioagents.agents.summary_agent.get_llm")
    @patch("bioagents.agents.summary_agent.load_prompt")
    def test_summary_agent_query_what_is_protein(self, mock_load_prompt, mock_get_llm):
        """Test summary agent with the specific query: 'What is protein?'."""
        mock_llm = Mock()
        expected_response = (
            "Proteins are large, complex molecules made up of amino acids. "
            "They play crucial roles in virtually all biological processes, "
            "including catalyzing metabolic reactions, DNA replication, and cell signaling."
        )
        mock_response = AIMessage(content=expected_response, name="Summary")
        mock_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm
        mock_load_prompt.return_value = "Summary prompt"

        agent = create_summary_agent()

        memory = {
            "research": {
                "status": "success",
                "data": {
                    "query": "What is protein?",
                    "findings": [
                        "Proteins are amino acid polymers",
                        "Essential biological molecules",
                        "Multiple structural forms",
                    ],
                },
                "raw_output": "Research completed",
                "tool_calls": [],
            }
        }

        state = {
            "messages": [HumanMessage(content="What is protein?")],
            "memory": memory,
        }

        result = agent(state)

        assert result["error"] is None
        assert expected_response in result["data"]["summary"]
        assert result["raw_output"] == expected_response


class TestSummaryAgentEmptyMemory:
    """Tests for summary agent with empty or minimal memory."""

    @patch("bioagents.agents.summary_agent.get_llm")
    @patch("bioagents.agents.summary_agent.load_prompt")
    def test_summary_agent_empty_memory_direct_answer(self, mock_load_prompt, mock_get_llm):
        """Test summary agent with empty memory falls back to direct answer."""
        mock_llm = Mock()
        mock_response = AIMessage(content="Proteins are biological macromolecules", name="Summary")
        mock_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm
        mock_load_prompt.return_value = "Summary prompt"

        agent = create_summary_agent()

        # Empty memory triggers direct answer path
        state = {
            "messages": [HumanMessage(content="What is protein?")],
            "memory": {},
        }

        result = agent(state)

        assert result["error"] is None
        assert "summary" in result["data"]
        assert result["tool_calls"] == []

        # Verify direct-answer prompt was used
        call_args = mock_llm.invoke.call_args[0][0]
        system_message_content = call_args[0].content
        assert "directly answer" in system_message_content.lower()

    @patch("bioagents.agents.summary_agent.get_llm")
    @patch("bioagents.agents.summary_agent.load_prompt")
    def test_summary_agent_no_memory_key(self, mock_load_prompt, mock_get_llm):
        """Test summary agent when state has no memory key."""
        mock_llm = Mock()
        mock_response = AIMessage(content="Direct answer to protein question", name="Summary")
        mock_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm
        mock_load_prompt.return_value = "Summary prompt"

        agent = create_summary_agent()

        # State without memory key
        state = {"messages": [HumanMessage(content="What is protein?")]}

        result = agent(state)

        assert result["error"] is None
        assert "summary" in result["data"]

    @patch("bioagents.agents.summary_agent.get_llm")
    @patch("bioagents.agents.summary_agent.load_prompt")
    def test_summary_agent_memory_with_no_success_status(self, mock_load_prompt, mock_get_llm):
        """Test summary agent with memory entries that have no successful status."""
        mock_llm = Mock()
        mock_response = AIMessage(content="Direct answer", name="Summary")
        mock_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm
        mock_load_prompt.return_value = "Summary prompt"

        agent = create_summary_agent()

        # Memory with failed agent
        memory = {
            "research": {
                "status": "failed",
                "data": None,
                "error": "Tool not available",
            }
        }

        state = {
            "messages": [HumanMessage(content="What is protein?")],
            "memory": memory,
        }

        result = agent(state)

        assert result["error"] is None
        # Should fall back to direct answer when no success status
        call_args = mock_llm.invoke.call_args[0][0]
        system_message_content = call_args[0].content
        assert "directly answer" in system_message_content.lower()


class TestSummaryAgentErrorHandling:
    """Tests for summary agent error handling."""

    @patch("bioagents.agents.summary_agent.get_llm")
    @patch("bioagents.agents.summary_agent.load_prompt")
    def test_summary_agent_llm_invocation_error(self, mock_load_prompt, mock_get_llm):
        """Test summary agent gracefully handles LLM invocation errors."""
        mock_llm = Mock()
        mock_llm.invoke = Mock(side_effect=Exception("LLM Service unavailable"))
        mock_get_llm.return_value = mock_llm
        mock_load_prompt.return_value = "Summary prompt"

        agent = create_summary_agent()

        state = {
            "messages": [HumanMessage(content="What is protein?")],
            "memory": {"research": {"status": "success", "data": {}}},
        }

        result = agent(state)

        # Should return error in response, not raise exception
        assert result["error"] is not None
        assert "LLM Service unavailable" in result["error"]
        assert result["data"] == {}
        assert result["raw_output"] == ""

    @patch("bioagents.agents.summary_agent.get_llm")
    @patch("bioagents.agents.summary_agent.load_prompt")
    def test_summary_agent_missing_messages_key(self, mock_load_prompt, mock_get_llm):
        """Test summary agent handles state without messages key."""
        mock_llm = Mock()
        mock_response = AIMessage(content="Response", name="Summary")
        mock_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm
        mock_load_prompt.return_value = "Summary prompt"

        agent = create_summary_agent()

        # State without messages key
        state = {"memory": {}}

        result = agent(state)

        # Should handle gracefully
        assert result["error"] is None or isinstance(result["error"], (str, type(None)))


class TestSummaryAgentSystemMessage:
    """Tests for system message composition in summary agent."""

    @patch("bioagents.agents.summary_agent.get_llm")
    @patch("bioagents.agents.summary_agent.load_prompt")
    def test_summary_agent_includes_system_message(self, mock_load_prompt, mock_get_llm):
        """Test that summary agent includes system message."""
        mock_llm = Mock()
        mock_response = AIMessage(content="Response")
        mock_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm
        mock_load_prompt.return_value = "Summary system prompt"

        agent = create_summary_agent()

        state = {
            "messages": [HumanMessage(content="What is protein?")],
            "memory": {},
        }

        agent(state)

        # Check that invoke was called with messages that include system message
        call_args = mock_llm.invoke.call_args[0][0]
        assert len(call_args) >= 2  # System message + user message
        assert isinstance(call_args[0], SystemMessage)

    @patch("bioagents.agents.summary_agent.get_llm")
    @patch("bioagents.agents.summary_agent.load_prompt")
    def test_summary_agent_system_message_with_memory_context(self, mock_load_prompt, mock_get_llm):
        """Test system message includes memory context when available."""
        mock_llm = Mock()
        mock_response = AIMessage(content="Summary with context")
        mock_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm
        mock_load_prompt.return_value = "Summary prompt"

        agent = create_summary_agent()

        memory = {
            "analysis": {
                "status": "success",
                "data": {"result": "test_data"},
                "raw_output": "Analysis complete",
            }
        }

        state = {
            "messages": [HumanMessage(content="What is protein?")],
            "memory": memory,
        }

        agent(state)

        call_args = mock_llm.invoke.call_args[0][0]
        system_message_content = call_args[0].content
        # Should include memory context
        assert (
            "SHARED MEMORY" in system_message_content or "memory" in system_message_content.lower()
        )


class TestSummaryAgentComplexityAssessment:
    """Tests for complexity assessment function."""

    def test_assess_execution_complexity_simple(self):
        """Test complexity assessment for simple execution."""
        memory = {
            "research": {
                "status": "success",
                "data": {"result": "test"},
                "tool_calls": [],
            }
        }

        complexity = assess_execution_complexity(memory)

        assert complexity == "simple"

    def test_assess_execution_complexity_complex_multiple_agents(self):
        """Test complexity assessment for multiple agents."""
        memory = {
            "research": {
                "status": "success",
                "data": {"result": "test"},
                "tool_calls": [],
            },
            "analysis": {
                "status": "success",
                "data": {"result": "test"},
                "tool_calls": [],
            },
            "ml": {"status": "success", "data": {"result": "test"}, "tool_calls": []},
        }

        complexity = assess_execution_complexity(memory)

        assert complexity == "complex"

    def test_assess_execution_complexity_complex_with_tool_calls(self):
        """Test complexity assessment with tool calls."""
        memory = {
            "research": {
                "status": "success",
                "data": {"result": "test"},
                "tool_calls": [{"name": "search", "args": {}}],
            }
        }

        complexity = assess_execution_complexity(memory)

        assert complexity == "complex"


class TestSummaryAgentMemoryFormatting:
    """Tests for memory formatting function."""

    def test_format_memory_for_summary_single_agent(self):
        """Test formatting memory with single agent."""
        memory = {
            "research": {
                "status": "success",
                "data": {"papers": 5},
                "raw_output": "Found papers",
            }
        }

        formatted = format_memory_for_summary(memory)

        assert "RESEARCH" in formatted
        assert "papers" in formatted

    def test_format_memory_for_summary_multiple_agents(self):
        """Test formatting memory with multiple agents."""
        memory = {
            "research": {
                "status": "success",
                "data": {"papers": 5},
                "raw_output": "Found papers",
            },
            "analysis": {
                "status": "success",
                "data": {"properties": ["polar"]},
                "raw_output": "Analyzed",
            },
        }

        formatted = format_memory_for_summary(memory)

        assert "RESEARCH" in formatted
        assert "ANALYSIS" in formatted

    def test_format_memory_for_summary_excludes_failed_agents(self):
        """Test that formatting excludes failed agents."""
        memory = {
            "research": {
                "status": "success",
                "data": {"papers": 5},
                "raw_output": "Found papers",
            },
            "failed_agent": {"status": "failed", "data": None, "error": "Error"},
        }

        formatted = format_memory_for_summary(memory)

        assert "RESEARCH" in formatted
        assert "FAILED_AGENT" not in formatted

    def test_format_memory_for_summary_empty_memory(self):
        """Test formatting empty memory."""
        memory = {}

        formatted = format_memory_for_summary(memory)

        assert "No completed agent tasks" in formatted
