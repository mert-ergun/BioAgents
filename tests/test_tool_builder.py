"""Tests for the Tool Builder Agent and Tool Registry."""

import json
from unittest.mock import MagicMock, patch

import pytest

from bioagents.tools.tool_registry import (
    ToolDefinition,
    ToolParameter,
    ToolRegistry,
)


class TestToolParameter:
    """Tests for ToolParameter dataclass."""

    def test_to_schema_basic(self):
        """Test basic schema generation."""
        param = ToolParameter(
            name="sequence",
            type="string",
            description="A protein sequence",
            required=True,
        )
        schema = param.to_schema()

        assert schema["type"] == "string"
        assert schema["description"] == "A protein sequence"
        assert "default" not in schema

    def test_to_schema_with_default(self):
        """Test schema with default value."""
        param = ToolParameter(
            name="limit",
            type="integer",
            description="Max results",
            required=False,
            default=10,
        )
        schema = param.to_schema()

        assert schema["default"] == 10

    def test_to_schema_with_enum(self):
        """Test schema with enum values."""
        param = ToolParameter(
            name="format",
            type="string",
            description="Output format",
            required=True,
            enum=["json", "csv", "tsv"],
        )
        schema = param.to_schema()

        assert schema["enum"] == ["json", "csv", "tsv"]


class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""

    @pytest.fixture
    def sample_tool_def(self):
        """Create a sample tool definition."""
        return ToolDefinition(
            name="test_tool",
            description="A test tool",
            category="testing",
            parameters=[
                ToolParameter(
                    name="input",
                    type="string",
                    description="Input data",
                    required=True,
                ),
                ToolParameter(
                    name="option",
                    type="boolean",
                    description="An option",
                    required=False,
                    default=False,
                ),
            ],
            return_type="dict",
            return_description="Result dictionary",
            code="def test_tool(input: str, option: bool = False) -> dict:\n    return {}",
        )

    def test_to_dict(self, sample_tool_def):
        """Test conversion to dictionary."""
        data = sample_tool_def.to_dict()

        assert data["name"] == "test_tool"
        assert data["category"] == "testing"
        assert len(data["parameters"]) == 2
        assert data["parameters"][0]["name"] == "input"

    def test_from_dict(self, sample_tool_def):
        """Test creation from dictionary."""
        data = sample_tool_def.to_dict()
        restored = ToolDefinition.from_dict(data)

        assert restored.name == sample_tool_def.name
        assert restored.description == sample_tool_def.description
        assert len(restored.parameters) == len(sample_tool_def.parameters)

    def test_get_schema(self, sample_tool_def):
        """Test JSON schema generation."""
        schema = sample_tool_def.get_schema()

        assert schema["type"] == "object"
        assert "input" in schema["properties"]
        assert "option" in schema["properties"]
        assert "input" in schema["required"]
        assert "option" not in schema["required"]

    def test_get_function_signature(self, sample_tool_def):
        """Test function signature generation."""
        sig = sample_tool_def.get_function_signature()

        assert "def test_tool(" in sig
        assert "input: str" in sig
        assert "option: bool = False" in sig

    def test_safe_name_conversion(self):
        """Test conversion of names to valid Python identifiers."""
        tool = ToolDefinition(
            name="My-Tool_123",
            description="Test",
            category="test",
            parameters=[],
            return_type="string",
            return_description="Test",
        )

        assert tool._safe_name() == "my_tool_123"

    def test_safe_name_starts_with_digit(self):
        """Test name that starts with digit."""
        tool = ToolDefinition(
            name="123_tool",
            description="Test",
            category="test",
            parameters=[],
            return_type="string",
            return_description="Test",
        )

        assert tool._safe_name() == "_123_tool"


class TestToolRegistry:
    """Tests for the Tool Registry."""

    @pytest.fixture
    def temp_registry(self, tmp_path):
        """Create a registry with a temporary directory."""
        return ToolRegistry(tools_dir=tmp_path)

    @pytest.fixture
    def sample_tool(self):
        """Create a sample tool definition."""
        return ToolDefinition(
            name="sample_tool",
            description="A sample tool for testing",
            category="testing",
            parameters=[
                ToolParameter(
                    name="value",
                    type="integer",
                    description="A numeric value",
                    required=True,
                ),
            ],
            return_type="integer",
            return_description="The doubled value",
            code='''def sample_tool(value: int) -> int:
    """Double the input value."""
    return value * 2
''',
        )

    def test_register_tool(self, temp_registry, sample_tool):
        """Test tool registration."""
        success = temp_registry.register_tool(sample_tool)

        assert success
        assert temp_registry.get_tool("sample_tool") is not None

    def test_register_tool_duplicate(self, temp_registry, sample_tool):
        """Test that duplicate registration fails without overwrite."""
        temp_registry.register_tool(sample_tool)
        success = temp_registry.register_tool(sample_tool, overwrite=False)

        assert not success

    def test_register_tool_overwrite(self, temp_registry, sample_tool):
        """Test that duplicate registration succeeds with overwrite."""
        temp_registry.register_tool(sample_tool)

        sample_tool.description = "Updated description"
        success = temp_registry.register_tool(sample_tool, overwrite=True)

        assert success
        assert temp_registry.get_tool("sample_tool").description == "Updated description"

    def test_unregister_tool(self, temp_registry, sample_tool):
        """Test tool unregistration."""
        temp_registry.register_tool(sample_tool)
        success = temp_registry.unregister_tool("sample_tool")

        assert success
        assert temp_registry.get_tool("sample_tool") is None

    def test_list_tools(self, temp_registry, sample_tool):
        """Test listing tools."""
        temp_registry.register_tool(sample_tool)

        tools = temp_registry.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "sample_tool"

    def test_list_tools_by_category(self, temp_registry, sample_tool):
        """Test filtering by category."""
        temp_registry.register_tool(sample_tool)

        # Add another tool with different category
        other_tool = ToolDefinition(
            name="other_tool",
            description="Another tool",
            category="genomics",
            parameters=[],
            return_type="string",
            return_description="Result",
        )
        temp_registry.register_tool(other_tool)

        testing_tools = temp_registry.list_tools(category="testing")
        assert len(testing_tools) == 1
        assert testing_tools[0].name == "sample_tool"

    def test_search_tools(self, temp_registry, sample_tool):
        """Test tool search."""
        temp_registry.register_tool(sample_tool)

        results = temp_registry.search_tools("sample testing", limit=5)

        assert len(results) > 0
        assert results[0][0].name == "sample_tool"
        assert results[0][1] > 0  # Score should be positive

    def test_load_tool_function(self, temp_registry, sample_tool):
        """Test dynamic function loading."""
        temp_registry.register_tool(sample_tool)

        func = temp_registry.load_tool_function("sample_tool")

        assert func is not None
        assert func(5) == 10  # 5 * 2 = 10

    def test_validate_tool(self, temp_registry, sample_tool):
        """Test tool validation."""
        temp_registry.register_tool(sample_tool)

        success = temp_registry.validate_tool("sample_tool", test_args={"value": 3})

        assert success
        tool = temp_registry.get_tool("sample_tool")
        assert tool.validated

    def test_record_usage(self, temp_registry, sample_tool):
        """Test usage recording."""
        temp_registry.register_tool(sample_tool)

        initial_count = temp_registry.get_tool("sample_tool").usage_count
        temp_registry.record_usage("sample_tool")

        assert temp_registry.get_tool("sample_tool").usage_count == initial_count + 1

    def test_get_categories(self, temp_registry, sample_tool):
        """Test getting unique categories."""
        temp_registry.register_tool(sample_tool)

        categories = temp_registry.get_categories()
        assert "testing" in categories

    def test_export_for_context(self, temp_registry, sample_tool):
        """Test exporting documentation."""
        temp_registry.register_tool(sample_tool)

        docs = temp_registry.export_for_context(["sample_tool"])

        assert "sample_tool" in docs
        assert "A sample tool for testing" in docs
        assert "value" in docs

    def test_persistence(self, tmp_path, sample_tool):
        """Test that registry persists across instances."""
        # Create first instance and add tool
        registry1 = ToolRegistry(tools_dir=tmp_path)
        registry1.register_tool(sample_tool)

        # Create second instance - should load the same tool
        registry2 = ToolRegistry(tools_dir=tmp_path)

        assert registry2.get_tool("sample_tool") is not None
        assert registry2.get_tool("sample_tool").description == sample_tool.description


class TestToolBuilderTools:
    """Tests for the Tool Builder Agent tools."""

    def test_list_custom_tools_returns_json(self):
        """Test that list_custom_tools returns valid JSON."""
        from bioagents.tools.tool_builder_tools import list_custom_tools

        result = list_custom_tools.invoke({})
        data = json.loads(result)

        assert "tools" in data
        assert "total" in data

    def test_search_custom_tools_returns_json(self):
        """Test that search_custom_tools returns valid JSON."""
        from bioagents.tools.tool_builder_tools import search_custom_tools

        result = search_custom_tools.invoke({"query": "protein analysis"})
        data = json.loads(result)

        assert "query" in data
        assert "results" in data


class TestToolBuilderWrappers:
    """Tests for the smolagents wrappers."""

    def test_get_custom_tool_wrappers(self):
        """Test that wrappers are created correctly."""
        from bioagents.tools.tool_builder_wrappers import get_custom_tool_wrappers

        tools = get_custom_tool_wrappers()

        assert len(tools) >= 4  # At least our 4 main tools
        names = [t.name for t in tools]
        assert "search_custom_tools" in names
        assert "list_custom_tools" in names
        assert "execute_custom_tool" in names

    def test_search_tool_forward(self):
        """Test the search tool forward method."""
        from bioagents.tools.tool_builder_wrappers import CustomToolSearchTool

        tool = CustomToolSearchTool()
        result = tool.forward(query="test", limit=3)

        data = json.loads(result)
        assert "query" in data
        assert data["query"] == "test"
        assert "results" in data

    def test_list_tool_forward(self):
        """Test the list tool forward method."""
        from bioagents.tools.tool_builder_wrappers import CustomToolListTool

        tool = CustomToolListTool()
        result = tool.forward()

        data = json.loads(result)
        assert "tools" in data
        assert "total" in data


class TestGraphIntegration:
    """Tests for graph integration."""

    @patch("bioagents.graph.create_supervisor_agent")
    @patch("bioagents.graph.create_research_agent")
    @patch("bioagents.graph.create_analysis_agent")
    @patch("bioagents.graph.create_report_agent")
    @patch("bioagents.graph.create_critic_agent")
    @patch("bioagents.graph.create_tool_builder_agent")
    @patch("bioagents.graph.create_protein_design_agent")
    @patch("bioagents.graph.create_coder_agent")
    def test_graph_includes_tool_builder(
        self,
        mock_coder,
        mock_protein,
        mock_builder,
        mock_critic,
        mock_report,
        mock_analysis,
        mock_research,
        mock_supervisor,
    ):
        """Test that the graph includes tool_builder node."""
        from bioagents.graph import create_graph

        # Mock all agent creations to avoid API calls
        mock_supervisor.return_value = MagicMock()
        mock_research.return_value = MagicMock()
        mock_analysis.return_value = MagicMock()
        mock_report.return_value = MagicMock()
        mock_critic.return_value = MagicMock()
        mock_builder.return_value = MagicMock()
        mock_protein.return_value = MagicMock()
        mock_coder.return_value = MagicMock()

        # This should not raise an exception
        graph = create_graph()

        # Check that the graph was created
        assert graph is not None


class TestToolBuilderAgent:
    """Tests for the Tool Builder Agent creation."""

    def test_create_tool_builder_agent(self):
        """Test agent creation."""
        from bioagents.agents.tool_builder_agent import create_tool_builder_agent

        # Mock the LLM to avoid API calls
        with patch("bioagents.agents.tool_builder_agent.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.bind_tools = MagicMock(return_value=mock_llm)
            mock_get_llm.return_value = mock_llm

            agent = create_tool_builder_agent()

            assert callable(agent)

    def test_get_tool_builder_tools(self):
        """Test getting all tool builder tools."""
        from bioagents.tools.tool_builder_tools import get_tool_builder_tools

        tools = get_tool_builder_tools()

        assert len(tools) >= 7  # We have at least 7 tools defined
        tool_names = [t.name for t in tools]

        assert "extract_tools_from_text" in tool_names
        assert "generate_tool_wrapper" in tool_names
        assert "register_custom_tool" in tool_names
        assert "validate_custom_tool" in tool_names
