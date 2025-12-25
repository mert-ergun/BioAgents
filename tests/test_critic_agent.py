"""Tests for the Critic Agent."""

from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from bioagents.agents.critic_agent import create_critic_agent


class TestCriticAgent:
    """Tests for the critic agent."""

    @patch("bioagents.agents.critic_agent.get_llm")
    def test_create_critic_agent(self, mock_get_llm):
        """Test creating a critic agent."""
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm

        agent = create_critic_agent()

        assert callable(agent)
        mock_get_llm.assert_called_once()

    @patch("bioagents.agents.critic_agent.get_llm")
    @patch("bioagents.agents.critic_agent.load_prompt")
    def test_critic_agent_invoke(self, mock_load_prompt, mock_get_llm):
        """Test invoking the critic agent."""
        mock_llm = Mock()
        mock_response = AIMessage(content="Validation Successful", name="Critic")
        mock_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm
        mock_load_prompt.return_value = "Critic system prompt"

        agent = create_critic_agent()

        state = {"messages": [HumanMessage(content="Validate the analysis of p53")]}
        result = agent(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0] == mock_response
        assert result["messages"][0].content == "Validation Successful"
        mock_llm.invoke.assert_called_once()

    @patch("bioagents.agents.critic_agent.get_llm")
    def test_critic_agent_includes_system_message(self, mock_get_llm):
        """Test that critic agent includes system message."""
        mock_llm = Mock()
        mock_response = AIMessage(content="Response")
        mock_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        agent = create_critic_agent()

        state = {"messages": [HumanMessage(content="Test")]}
        agent(state)

        # Check that invoke was called with messages that include system message
        call_args = mock_llm.invoke.call_args[0][0]
        assert len(call_args) >= 2  # System message + user message
        assert isinstance(call_args[0], SystemMessage)
        assert "specialized Critic Agent" in call_args[0].content
