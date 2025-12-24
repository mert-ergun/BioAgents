"""Tests for graph module."""

from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage

from bioagents.graph import (
    AgentState,
    agent_node,
    create_graph,
    route_supervisor,
    should_continue_to_tools,
)


class TestAgentState:
    """Tests for AgentState class."""

    def test_agent_state_creation(self):
        """Test creating an AgentState."""
        state = AgentState()
        assert isinstance(state, dict)

    def test_agent_state_with_messages(self):
        """Test AgentState with messages."""
        messages = [HumanMessage(content="Test")]
        state = AgentState(messages=messages)
        assert state["messages"] == messages

    def test_agent_state_with_next(self):
        """Test AgentState with next field."""
        state = AgentState(next="research")
        assert state["next"] == "research"

    def test_agent_state_with_reasoning(self):
        """Test AgentState with reasoning field."""
        state = AgentState(reasoning="Need to fetch data")
        assert state["reasoning"] == "Need to fetch data"


class TestAgentNode:
    """Tests for agent_node wrapper function."""

    def test_agent_node_basic(self):
        """Test basic agent_node functionality."""
        mock_agent = Mock()
        mock_response = AIMessage(content="Response")
        mock_agent.return_value = {"messages": [mock_response]}

        state = {"messages": [HumanMessage(content="Test")]}
        result = agent_node(state, mock_agent, "TestAgent")

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0].name == "TestAgent"
        mock_agent.assert_called_once_with(state)

    def test_agent_node_multiple_messages(self):
        """Test agent_node with multiple messages."""
        mock_agent = Mock()
        messages = [
            AIMessage(content="First response"),
            AIMessage(content="Second response"),
        ]
        mock_agent.return_value = {"messages": messages}

        state = {"messages": [HumanMessage(content="Test")]}
        result = agent_node(state, mock_agent, "TestAgent")

        assert len(result["messages"]) == 2
        for msg in result["messages"]:
            assert msg.name == "TestAgent"

    def test_agent_node_empty_messages(self):
        """Test agent_node with empty messages."""
        mock_agent = Mock()
        mock_agent.return_value = {"messages": []}

        state = {"messages": [HumanMessage(content="Test")]}
        result = agent_node(state, mock_agent, "TestAgent")

        assert result["messages"] == []


class TestShouldContinueToTools:
    """Tests for should_continue_to_tools conditional edge function."""

    def test_continue_with_tool_calls(self):
        """Test routing to tools when tool calls are present."""
        ai_message = AIMessage(
            content="",
            tool_calls=[
                {"name": "fetch_uniprot_fasta", "args": {"protein_id": "P04637"}, "id": "call_123"}
            ],
        )
        state = {"messages": [HumanMessage(content="Test"), ai_message]}

        result = should_continue_to_tools(state)

        assert result == "tools"

    def test_continue_without_tool_calls(self):
        """Test routing to supervisor when no tool calls."""
        ai_message = AIMessage(content="No tools needed")
        state = {"messages": [HumanMessage(content="Test"), ai_message]}

        result = should_continue_to_tools(state)

        assert result == "supervisor"

    def test_continue_with_empty_tool_calls(self):
        """Test routing with empty tool calls list."""
        ai_message = AIMessage(content="Done", tool_calls=[])
        state = {"messages": [HumanMessage(content="Test"), ai_message]}

        result = should_continue_to_tools(state)

        assert result == "supervisor"

    def test_continue_with_human_message(self):
        """Test routing when last message is not an AI message."""
        state = {"messages": [HumanMessage(content="Test")]}

        result = should_continue_to_tools(state)

        # HumanMessage doesn't have tool_calls attribute
        assert result == "supervisor"


class TestRouteSupervisor:
    """Tests for route_supervisor function."""

    def test_route_to_research(self):
        """Test routing to research agent."""
        state = {"next": "research", "messages": []}
        result = route_supervisor(state)
        assert result == "research"

    def test_route_to_analysis(self):
        """Test routing to analysis agent."""
        state = {"next": "analysis", "messages": []}
        result = route_supervisor(state)
        assert result == "analysis"

    def test_route_to_report(self):
        """Test routing to report agent."""
        state = {"next": "report", "messages": []}
        result = route_supervisor(state)
        assert result == "report"

    def test_route_to_finish(self):
        """Test routing to end."""
        state = {"next": "FINISH", "messages": []}
        result = route_supervisor(state)
        assert result == "end"

    def test_route_with_missing_next(self):
        """Test routing when next is not in state."""
        state = {"messages": []}
        result = route_supervisor(state)
        assert result == "end"  # Default to end when next is missing


class TestCreateGraph:
    """Tests for create_graph function."""

    @patch("bioagents.graph.create_supervisor_agent")
    @patch("bioagents.graph.create_research_agent")
    @patch("bioagents.graph.create_analysis_agent")
    @patch("bioagents.graph.create_report_agent")
    def test_create_graph_basic(self, mock_report, mock_analysis, mock_research, mock_supervisor):
        """Test basic graph creation."""
        # Setup mocks
        mock_supervisor.return_value = Mock()
        mock_research.return_value = Mock()
        mock_analysis.return_value = Mock()
        mock_report.return_value = Mock()

        graph = create_graph()

        assert graph is not None
        mock_supervisor.assert_called_once()
        mock_research.assert_called_once()
        mock_analysis.assert_called_once()
        mock_report.assert_called_once()

    @patch("bioagents.graph.create_supervisor_agent")
    @patch("bioagents.graph.create_research_agent")
    @patch("bioagents.graph.create_analysis_agent")
    @patch("bioagents.graph.create_report_agent")
    def test_create_graph_agent_creation_with_tools(
        self, mock_report, mock_analysis, mock_research, mock_supervisor
    ):
        """Test that agents are created with correct tools."""
        mock_supervisor.return_value = Mock()
        mock_research.return_value = Mock()
        mock_analysis.return_value = Mock()
        mock_report.return_value = Mock()

        create_graph()

        # Check research agent was created with ToolUniverse + UniProt tools
        research_call_args = mock_research.call_args[0][0]
        research_tool_names = {tool.name for tool in research_call_args}
        assert {
            "fetch_uniprot_fasta",
            "tool_universe_find_tools",
            "tool_universe_call_tool",
        }.issubset(research_tool_names)

        # Check analysis agent was created with correct tools
        analysis_call_args = mock_analysis.call_args[0][0]
        assert len(analysis_call_args) == 3  # molecular_weight, composition, pI

        # Check supervisor was created with correct members
        supervisor_call_args = mock_supervisor.call_args[0][0]
        assert "research" in supervisor_call_args
        assert "analysis" in supervisor_call_args
        assert "report" in supervisor_call_args

    @patch("bioagents.graph.create_supervisor_agent")
    @patch("bioagents.graph.create_research_agent")
    @patch("bioagents.graph.create_analysis_agent")
    @patch("bioagents.graph.create_report_agent")
    def test_create_graph_returns_compiled(
        self, mock_report, mock_analysis, mock_research, mock_supervisor
    ):
        """Test that create_graph returns a compiled graph."""
        mock_supervisor.return_value = Mock()
        mock_research.return_value = Mock()
        mock_analysis.return_value = Mock()
        mock_report.return_value = Mock()

        graph = create_graph()

        # Compiled graph should have invoke and stream methods
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "stream")


class TestGraphWorkflow:
    """Integration tests for graph workflow."""

    @patch("bioagents.graph.create_supervisor_agent")
    @patch("bioagents.graph.create_research_agent")
    @patch("bioagents.graph.create_analysis_agent")
    @patch("bioagents.graph.create_report_agent")
    def test_graph_workflow_simple(
        self, mock_report, mock_analysis, mock_research, mock_supervisor
    ):
        """Test a simple workflow through the graph."""
        # Create mock agents that return appropriate responses
        supervisor_responses = [
            {"next": "research", "reasoning": "Fetch data", "messages": []},
            {"next": "FINISH", "reasoning": "Done", "messages": []},
        ]

        research_response = {"messages": [AIMessage(content="Data fetched", name="Research")]}

        # Setup mocks with side effects for multiple calls
        mock_supervisor_agent = Mock(side_effect=supervisor_responses)
        mock_research_agent = Mock(return_value=research_response)

        mock_supervisor.return_value = mock_supervisor_agent
        mock_research.return_value = mock_research_agent
        mock_analysis.return_value = Mock()
        mock_report.return_value = Mock()

        graph = create_graph()

        # Test that graph was created
        assert graph is not None

    @patch("bioagents.graph.create_supervisor_agent")
    @patch("bioagents.graph.create_research_agent")
    @patch("bioagents.graph.create_analysis_agent")
    @patch("bioagents.graph.create_report_agent")
    def test_graph_entry_point(self, mock_report, mock_analysis, mock_research, mock_supervisor):
        """Test that graph has correct entry point."""
        mock_supervisor.return_value = Mock()
        mock_research.return_value = Mock()
        mock_analysis.return_value = Mock()
        mock_report.return_value = Mock()

        graph = create_graph()

        # The graph should be compiled and ready to use
        assert hasattr(graph, "get_graph")


class TestGraphEdges:
    """Tests for graph edges and routing."""

    def test_should_continue_to_tools_logic(self):
        """Test the logic of should_continue_to_tools function."""
        # Case 1: Has tool calls
        state_with_tools = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "test_tool", "args": {}, "id": "call_123"}],
                )
            ]
        }
        assert should_continue_to_tools(state_with_tools) == "tools"

        # Case 2: No tool calls
        state_without_tools = {"messages": [AIMessage(content="No tools")]}
        assert should_continue_to_tools(state_without_tools) == "supervisor"

    def test_route_supervisor_logic(self):
        """Test the logic of route_supervisor function."""
        # Test all valid routes
        assert route_supervisor({"next": "research"}) == "research"
        assert route_supervisor({"next": "analysis"}) == "analysis"
        assert route_supervisor({"next": "report"}) == "report"
        assert route_supervisor({"next": "FINISH"}) == "end"

        # Test default case
        assert route_supervisor({}) == "end"


class TestGraphStructure:
    """Tests for graph structure and node configuration."""

    @patch("bioagents.graph.create_supervisor_agent")
    @patch("bioagents.graph.create_research_agent")
    @patch("bioagents.graph.create_analysis_agent")
    @patch("bioagents.graph.create_report_agent")
    @patch("bioagents.graph.ToolNode")
    def test_tool_nodes_created(
        self, mock_tool_node, mock_report, mock_analysis, mock_research, mock_supervisor
    ):
        """Test that tool nodes are created for research and analysis."""
        mock_supervisor.return_value = Mock()
        mock_research.return_value = Mock()
        mock_analysis.return_value = Mock()
        mock_report.return_value = Mock()
        mock_tool_node.return_value = Mock()

        create_graph()

        # ToolNode should be called 4 times: research, analysis, tool_builder, protein_design
        assert mock_tool_node.call_count == 4

    @patch("bioagents.graph.create_supervisor_agent")
    @patch("bioagents.graph.create_research_agent")
    @patch("bioagents.graph.create_analysis_agent")
    @patch("bioagents.graph.create_report_agent")
    def test_partial_agent_node_wrapping(
        self, mock_report, mock_analysis, mock_research, mock_supervisor
    ):
        """Test that agents are wrapped with partial function for naming."""
        mock_supervisor.return_value = Mock()
        mock_research.return_value = Mock()
        mock_analysis.return_value = Mock()
        mock_report.return_value = Mock()

        # This should not raise any errors
        graph = create_graph()
        assert graph is not None
