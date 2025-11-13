"""Tests for agent modules."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from bioagents.agents.analysis_agent import create_analysis_agent
from bioagents.agents.report_agent import create_report_agent
from bioagents.agents.research_agent import create_research_agent
from bioagents.agents.supervisor_agent import RouteResponse, create_supervisor_agent


class TestResearchAgent:
    """Tests for the research agent."""

    @patch("bioagents.agents.research_agent.get_llm")
    def test_create_research_agent(self, mock_get_llm):
        """Test creating a research agent."""
        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_get_llm.return_value = mock_llm

        tools = [Mock()]
        agent = create_research_agent(tools)

        assert callable(agent)
        mock_get_llm.assert_called_once()
        mock_llm.bind_tools.assert_called_once_with(tools)

    @patch("bioagents.agents.research_agent.get_llm")
    def test_research_agent_invoke(self, mock_get_llm):
        """Test invoking the research agent."""
        mock_llm = Mock()
        mock_bound_llm = Mock()
        mock_response = AIMessage(content="Research complete", name="Research")

        mock_llm.bind_tools = Mock(return_value=mock_bound_llm)
        mock_bound_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        tools = [Mock()]
        agent = create_research_agent(tools)

        state = {"messages": [HumanMessage(content="Fetch protein P04637")]}
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
        assert hasattr(call_args[0], "__class__")


class TestAnalysisAgent:
    """Tests for the analysis agent."""

    @patch("bioagents.agents.analysis_agent.get_llm")
    def test_create_analysis_agent(self, mock_get_llm):
        """Test creating an analysis agent."""
        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_get_llm.return_value = mock_llm

        tools = [Mock()]
        agent = create_analysis_agent(tools)

        assert callable(agent)
        mock_get_llm.assert_called_once()
        mock_llm.bind_tools.assert_called_once_with(tools)

    @patch("bioagents.agents.analysis_agent.get_llm")
    def test_analysis_agent_invoke(self, mock_get_llm):
        """Test invoking the analysis agent."""
        mock_llm = Mock()
        mock_bound_llm = Mock()
        mock_response = AIMessage(content="Analysis complete", name="Analysis")

        mock_llm.bind_tools = Mock(return_value=mock_bound_llm)
        mock_bound_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        tools = [Mock()]
        agent = create_analysis_agent(tools)

        state = {"messages": [HumanMessage(content="Analyze sequence")]}
        result = agent(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0] == mock_response
        mock_bound_llm.invoke.assert_called_once()

    @patch("bioagents.agents.analysis_agent.get_llm")
    def test_analysis_agent_includes_system_message(self, mock_get_llm):
        """Test that analysis agent includes system message."""
        mock_llm = Mock()
        mock_bound_llm = Mock()
        mock_response = AIMessage(content="Response")

        mock_llm.bind_tools = Mock(return_value=mock_bound_llm)
        mock_bound_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        tools = [Mock()]
        agent = create_analysis_agent(tools)

        state = {"messages": [HumanMessage(content="Test")]}
        agent(state)

        # Check that invoke was called with messages that include system message
        call_args = mock_bound_llm.invoke.call_args[0][0]
        assert len(call_args) >= 2  # System message + user message


class TestReportAgent:
    """Tests for the report agent."""

    @patch("bioagents.agents.report_agent.get_llm")
    def test_create_report_agent(self, mock_get_llm):
        """Test creating a report agent."""
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm

        agent = create_report_agent()

        assert callable(agent)
        mock_get_llm.assert_called_once()

    @patch("bioagents.agents.report_agent.get_llm")
    def test_report_agent_invoke(self, mock_get_llm):
        """Test invoking the report agent."""
        mock_llm = Mock()
        mock_response = AIMessage(content="Report complete", name="Report")
        mock_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        agent = create_report_agent()

        state = {"messages": [HumanMessage(content="Create report")]}
        result = agent(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0] == mock_response
        mock_llm.invoke.assert_called_once()

    @patch("bioagents.agents.report_agent.get_llm")
    def test_report_agent_no_tools(self, mock_get_llm):
        """Test that report agent doesn't bind tools."""
        mock_llm = Mock()
        mock_response = AIMessage(content="Report")
        mock_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        agent = create_report_agent()
        state = {"messages": [HumanMessage(content="Test")]}
        agent(state)

        # Report agent shouldn't call bind_tools
        assert not hasattr(mock_llm, "bind_tools") or not mock_llm.bind_tools.called

    @patch("bioagents.agents.report_agent.get_llm")
    def test_report_agent_includes_system_message(self, mock_get_llm):
        """Test that report agent includes system message."""
        mock_llm = Mock()
        mock_response = AIMessage(content="Response")
        mock_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        agent = create_report_agent()

        state = {"messages": [HumanMessage(content="Test")]}
        agent(state)

        # Check that invoke was called with messages that include system message
        call_args = mock_llm.invoke.call_args[0][0]
        assert len(call_args) >= 2  # System message + user message


class TestSupervisorAgent:
    """Tests for the supervisor agent."""

    def test_route_response_model(self):
        """Test RouteResponse Pydantic model."""
        response = RouteResponse(next_agent="research", reasoning="Need to fetch data")

        assert response.next_agent == "research"
        assert response.reasoning == "Need to fetch data"

    def test_route_response_validation(self):
        """Test RouteResponse validates agent choices."""
        # Valid choices
        valid_agents = ["research", "analysis", "report", "FINISH"]
        for agent in valid_agents:
            response = RouteResponse(next_agent=agent, reasoning="Test")
            assert response.next_agent == agent

        # Invalid choice should raise validation error
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            RouteResponse(next_agent="invalid_agent", reasoning="Test")

    @patch("bioagents.agents.supervisor_agent.get_llm")
    def test_create_supervisor_agent(self, mock_get_llm):
        """Test creating a supervisor agent."""
        mock_llm = Mock()
        mock_structured_llm = Mock()
        mock_llm.with_structured_output = Mock(return_value=mock_structured_llm)
        mock_get_llm.return_value = mock_llm

        members = ["research", "analysis", "report"]
        agent = create_supervisor_agent(members)

        assert callable(agent)
        mock_get_llm.assert_called_once()
        mock_llm.with_structured_output.assert_called_once()

    @patch("bioagents.agents.supervisor_agent.get_llm")
    def test_supervisor_agent_invoke(self, mock_get_llm):
        """Test invoking the supervisor agent."""
        mock_llm = Mock()
        mock_structured_llm = Mock()
        mock_chain = Mock()

        # Create mock route response
        mock_route_response = RouteResponse(
            next_agent="research", reasoning="Need to fetch protein data"
        )

        mock_chain.invoke = Mock(return_value=mock_route_response)
        mock_llm.with_structured_output = Mock(return_value=mock_structured_llm)
        mock_get_llm.return_value = mock_llm

        # Patch the prompt creation to return our mock chain
        with patch(
            "bioagents.agents.supervisor_agent.ChatPromptTemplate.from_messages"
        ) as mock_prompt:
            mock_prompt.return_value.__or__ = Mock(return_value=mock_chain)

            members = ["research", "analysis", "report"]
            agent = create_supervisor_agent(members)

            state = {"messages": [HumanMessage(content="Fetch protein P04637")]}
            result = agent(state)

            assert "next" in result
            assert "reasoning" in result
            assert "messages" in result
            assert result["next"] == "research"
            assert result["reasoning"] == "Need to fetch protein data"
            assert result["messages"] == []

    @patch("bioagents.agents.supervisor_agent.get_llm")
    def test_supervisor_agent_finish_routing(self, mock_get_llm):
        """Test supervisor routing to FINISH."""
        mock_llm = Mock()
        mock_structured_llm = Mock()
        mock_chain = Mock()

        mock_route_response = RouteResponse(next_agent="FINISH", reasoning="Task complete")

        mock_chain.invoke = Mock(return_value=mock_route_response)
        mock_llm.with_structured_output = Mock(return_value=mock_structured_llm)
        mock_get_llm.return_value = mock_llm

        with patch(
            "bioagents.agents.supervisor_agent.ChatPromptTemplate.from_messages"
        ) as mock_prompt:
            mock_prompt.return_value.__or__ = Mock(return_value=mock_chain)

            members = ["research", "analysis", "report"]
            agent = create_supervisor_agent(members)

            state = {"messages": [HumanMessage(content="Done")]}
            result = agent(state)

            assert result["next"] == "FINISH"
            assert result["reasoning"] == "Task complete"

    @patch("bioagents.agents.supervisor_agent.get_llm")
    def test_supervisor_includes_all_members(self, mock_get_llm):
        """Test that supervisor is aware of all team members."""
        mock_llm = Mock()
        mock_structured_llm = Mock()
        mock_llm.with_structured_output = Mock(return_value=mock_structured_llm)
        mock_get_llm.return_value = mock_llm

        members = ["research", "analysis", "report", "custom_agent"]
        agent = create_supervisor_agent(members)

        # The agent should be created successfully
        assert callable(agent)


class TestAgentIntegration:
    """Integration tests for agent interactions."""

    @patch("bioagents.agents.research_agent.get_llm")
    def test_research_agent_with_tool_calls(self, mock_get_llm):
        """Test research agent with tool calls in response."""
        mock_llm = Mock()
        mock_bound_llm = Mock()

        # Create a response with tool calls
        mock_response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "fetch_uniprot_fasta",
                    "args": {"protein_id": "P04637"},
                    "id": "call_123",
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

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert hasattr(result["messages"][0], "tool_calls")
        assert len(result["messages"][0].tool_calls) == 1

    @patch("bioagents.agents.supervisor_agent.get_llm")
    def test_supervisor_sequential_routing(self, mock_get_llm):
        """Test supervisor routing through multiple agents."""
        mock_llm = Mock()
        mock_structured_llm = Mock()
        mock_chain = Mock()

        # Simulate routing decisions
        routing_sequence = [
            RouteResponse(next_agent="research", reasoning="Fetch data first"),
            RouteResponse(next_agent="analysis", reasoning="Now analyze the data"),
            RouteResponse(next_agent="report", reasoning="Synthesize findings"),
            RouteResponse(next_agent="FINISH", reasoning="All tasks complete"),
        ]

        mock_chain.invoke = Mock(side_effect=routing_sequence)
        mock_llm.with_structured_output = Mock(return_value=mock_structured_llm)
        mock_get_llm.return_value = mock_llm

        with patch(
            "bioagents.agents.supervisor_agent.ChatPromptTemplate.from_messages"
        ) as mock_prompt:
            mock_prompt.return_value.__or__ = Mock(return_value=mock_chain)

            members = ["research", "analysis", "report"]
            agent = create_supervisor_agent(members)

            # Simulate multiple calls
            state1 = {"messages": [HumanMessage(content="Start")]}
            result1 = agent(state1)
            assert result1["next"] == "research"

            state2 = {"messages": [HumanMessage(content="Continue")]}
            result2 = agent(state2)
            assert result2["next"] == "analysis"

            state3 = {"messages": [HumanMessage(content="Continue")]}
            result3 = agent(state3)
            assert result3["next"] == "report"

            state4 = {"messages": [HumanMessage(content="Continue")]}
            result4 = agent(state4)
            assert result4["next"] == "FINISH"
