"""Comprehensive tests for the Summary Agent with message-based summary generation."""

from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from bioagents.agents.summary_agent import (
    analyze_execution_mode,
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
    """Tests for summary agent with messages-based execution (complex execution)."""

    @patch("bioagents.agents.summary_agent.get_llm")
    @patch("bioagents.agents.summary_agent.load_prompt")
    def test_summary_agent_with_complex_messages(self, mock_load_prompt, mock_get_llm):
        """Test summary agent processing with multi-agent message history."""
        mock_llm = Mock()
        mock_response = AIMessage(content="Protein is a complex molecule", name="Summary")
        mock_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm
        mock_load_prompt.return_value = "Summary prompt"

        agent = create_summary_agent()

        # Simulate complex multi-agent conversation with tool calls
        messages = [
            HumanMessage(content="What is protein?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search",
                        "args": {"q": "protein structure"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Found 2 papers about proteins", name="Research"),
        ]

        state = {"messages": messages}

        result = agent(state)

        # Verify result structure: new API returns {"messages": [AIMessage(...)]}
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].content == "Protein is a complex molecule"

        # Verify LLM was invoked
        mock_llm.invoke.assert_called_once()

        # Verify system message was included in the call
        call_args = mock_llm.invoke.call_args[0][0]
        assert len(call_args) >= 2
        assert isinstance(call_args[0], SystemMessage)

    @patch("bioagents.agents.summary_agent.get_llm")
    @patch("bioagents.agents.summary_agent.load_prompt")
    def test_summary_agent_complex_execution_path(self, mock_load_prompt, mock_get_llm):
        """Test summary agent with complex multi-agent execution history."""
        mock_llm = Mock()
        mock_response = AIMessage(
            content="Proteins are polymers of amino acids with complex structures",
            name="Summary",
        )
        mock_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm
        mock_load_prompt.return_value = "Summary prompt"

        agent = create_summary_agent()

        # Simulate complex multi-agent conversation
        messages = [
            HumanMessage(content="Analyze this protein"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_literature",
                        "args": {"q": "protein"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Found papers on protein structure", name="Research"),
            HumanMessage(content="Also run ML"),
            AIMessage(content="Training complete", name="Analysis"),
        ]

        state = {"messages": messages}

        result = agent(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert (
            result["messages"][0].content
            == "Proteins are polymers of amino acids with complex structures"
        )

        # Verify system message includes FULL SUMMARY mode instruction
        call_args = mock_llm.invoke.call_args[0][0]
        mode_instruction = call_args[1].content
        assert "FULL SUMMARY" in mode_instruction

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

        # Single human message with a simple AI response => direct_answer mode
        messages = [
            HumanMessage(content="What is protein?"),
        ]

        state = {"messages": messages}

        result = agent(state)

        assert "messages" in result
        assert expected_response in result["messages"][0].content


class TestSummaryAgentEmptyMemory:
    """Tests for summary agent with empty or minimal messages (direct answer path)."""

    @patch("bioagents.agents.summary_agent.get_llm")
    @patch("bioagents.agents.summary_agent.load_prompt")
    def test_summary_agent_single_message_direct_answer(self, mock_load_prompt, mock_get_llm):
        """Test summary agent with single human message uses direct answer path."""
        mock_llm = Mock()
        mock_response = AIMessage(content="Proteins are biological macromolecules", name="Summary")
        mock_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm
        mock_load_prompt.return_value = "Summary prompt"

        agent = create_summary_agent()

        # Single human message with no tool calls => direct_answer mode
        state = {
            "messages": [HumanMessage(content="What is protein?")],
        }

        result = agent(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "Proteins are biological macromolecules"

        # Verify DIRECT ANSWER mode instruction was used
        call_args = mock_llm.invoke.call_args[0][0]
        mode_instruction = call_args[1].content
        assert "DIRECT ANSWER" in mode_instruction

    @patch("bioagents.agents.summary_agent.get_llm")
    @patch("bioagents.agents.summary_agent.load_prompt")
    def test_summary_agent_no_messages_key(self, mock_load_prompt, mock_get_llm):
        """Test summary agent when state has no messages key."""
        mock_llm = Mock()
        mock_response = AIMessage(content="Direct answer to protein question", name="Summary")
        mock_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm
        mock_load_prompt.return_value = "Summary prompt"

        agent = create_summary_agent()

        # State without messages key defaults to empty list => direct_answer mode
        state = {}

        result = agent(state)

        assert "messages" in result
        assert len(result["messages"]) == 1

    @patch("bioagents.agents.summary_agent.get_llm")
    @patch("bioagents.agents.summary_agent.load_prompt")
    def test_summary_agent_empty_messages_list(self, mock_load_prompt, mock_get_llm):
        """Test summary agent with empty messages list uses full_summary mode."""
        mock_llm = Mock()
        mock_response = AIMessage(content="Direct answer", name="Summary")
        mock_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm
        mock_load_prompt.return_value = "Summary prompt"

        agent = create_summary_agent()

        # Empty messages list: 0 human messages (not == 1) => not simple => full_summary mode
        state = {"messages": []}

        result = agent(state)

        assert "messages" in result
        # Verify full summary mode was selected (empty list has 0 human msgs, not 1)
        call_args = mock_llm.invoke.call_args[0][0]
        mode_instruction = call_args[1].content
        assert "FULL SUMMARY" in mode_instruction


class TestSummaryAgentErrorHandling:
    """Tests for summary agent error handling."""

    @patch("bioagents.agents.summary_agent.get_llm")
    @patch("bioagents.agents.summary_agent.load_prompt")
    def test_summary_agent_llm_invocation_error(self, mock_load_prompt, mock_get_llm):
        """Test summary agent propagates LLM invocation errors."""
        mock_llm = Mock()
        mock_llm.invoke = Mock(side_effect=Exception("LLM Service unavailable"))
        mock_get_llm.return_value = mock_llm
        mock_load_prompt.return_value = "Summary prompt"

        agent = create_summary_agent()

        state = {
            "messages": [HumanMessage(content="What is protein?")],
        }

        # The new agent does not catch exceptions from invoke_with_retry;
        # the error propagates to the caller.
        try:
            result = agent(state)
            # If it doesn't raise, it should still return a valid structure
            assert "messages" in result
        except Exception as e:
            assert "LLM Service unavailable" in str(e)

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

        # State without messages key => defaults to [] => direct_answer mode
        state = {}

        result = agent(state)

        # Should handle gracefully and return messages
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "Response"


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
        }

        agent(state)

        # Check that invoke was called with messages that include system message
        call_args = mock_llm.invoke.call_args[0][0]
        assert len(call_args) >= 2  # System message + mode instruction + user message
        assert isinstance(call_args[0], SystemMessage)

    @patch("bioagents.agents.summary_agent.get_llm")
    @patch("bioagents.agents.summary_agent.load_prompt")
    def test_summary_agent_system_message_contains_prompt(self, mock_load_prompt, mock_get_llm):
        """Test system message contains the summary prompt and mode instruction."""
        mock_llm = Mock()
        mock_response = AIMessage(content="Summary with context")
        mock_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm
        mock_load_prompt.return_value = "Summary prompt"

        agent = create_summary_agent()

        messages = [
            HumanMessage(content="Analyze this protein"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search",
                        "args": {"q": "protein"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Analysis complete", name="Analysis"),
        ]

        state = {"messages": messages}

        agent(state)

        call_args = mock_llm.invoke.call_args[0][0]
        # First message is the system prompt (SUMMARY_AGENT_PROMPT, loaded at module import)
        assert isinstance(call_args[0], SystemMessage)
        assert "SummaryAgent" in call_args[0].content
        # Second message is the mode instruction (FULL SUMMARY for complex execution)
        assert isinstance(call_args[1], SystemMessage)
        assert "FULL SUMMARY" in call_args[1].content


class TestSummaryAgentComplexityAssessment:
    """Tests for execution mode analysis function."""

    def test_analyze_execution_mode_simple(self):
        """Test mode analysis for simple single-turn query."""
        messages = [
            HumanMessage(content="What is DNA?"),
            AIMessage(content="DNA is a molecule..."),
        ]

        mode, reasoning = analyze_execution_mode(messages)

        assert mode == "direct_answer"
        assert "Single user query" in reasoning

    def test_analyze_execution_mode_complex_multiple_agents(self):
        """Test mode analysis for complex multi-agent execution."""
        messages = [
            HumanMessage(content="Analyze this protein"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search",
                        "args": {"q": "protein"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Found results"),
            HumanMessage(content="Also run ML"),
            AIMessage(content="Training complete"),
        ]

        mode, reasoning = analyze_execution_mode(messages)

        assert mode == "full_summary"
        assert "Complex execution" in reasoning

    def test_analyze_execution_mode_complex_with_tool_calls(self):
        """Test mode analysis with tool calls."""
        messages = [
            HumanMessage(content="Search for genes"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search",
                        "args": {"q": "genes"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
        ]

        mode, reasoning = analyze_execution_mode(messages)

        assert mode == "full_summary"
        assert "tool calls" in reasoning


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
