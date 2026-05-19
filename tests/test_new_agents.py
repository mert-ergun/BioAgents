"""Tests for new agent modules added in Phase 0."""

from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage

# ---------------------------------------------------------------------------
# Helpers shared across agent tests
# ---------------------------------------------------------------------------


def _make_mock_llm(with_tools=True):
    """Create a mock LLM that returns a non-empty AIMessage."""
    mock_llm = Mock()
    mock_llm.invoke.return_value = AIMessage(content="Mock agent response")
    if with_tools:
        mock_llm.bind_tools.return_value = mock_llm
    return mock_llm


def _make_state(content="Test query"):
    return {"messages": [HumanMessage(content=content)]}


# =========================================================================
# 1. Literature Agent  (tools: get_literature_tools)
# =========================================================================


class TestLiteratureAgent:
    @patch("bioagents.agents.literature_agent.get_llm")
    @patch("bioagents.agents.literature_agent.get_literature_tools")
    def test_create_agent_returns_callable(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        mock_llm.return_value = _make_mock_llm()

        from bioagents.agents.literature_agent import create_literature_agent

        agent = create_literature_agent()
        assert callable(agent)

    @patch("bioagents.agents.literature_agent.get_llm")
    @patch("bioagents.agents.literature_agent.get_literature_tools")
    def test_node_returns_messages(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        mock_llm.return_value = _make_mock_llm()

        from bioagents.agents.literature_agent import create_literature_agent

        agent = create_literature_agent()
        result = agent(_make_state("Search for CRISPR papers"))
        assert "messages" in result
        assert len(result["messages"]) > 0

    @patch("bioagents.agents.literature_agent.get_llm")
    @patch("bioagents.agents.literature_agent.get_literature_tools")
    def test_retry_on_empty_response(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        llm = _make_mock_llm()
        llm.invoke.side_effect = [
            AIMessage(content=""),
            AIMessage(content="Recovered"),
        ]
        mock_llm.return_value = llm

        from bioagents.agents.literature_agent import create_literature_agent

        agent = create_literature_agent()
        result = agent(_make_state())
        assert result["messages"][0].content == "Recovered"
        assert llm.invoke.call_count == 2


# =========================================================================
# 2. Web Browser Agent  (tools: get_web_tools)
# =========================================================================


class TestWebBrowserAgent:
    @patch("bioagents.agents.web_browser_agent.get_llm")
    @patch("bioagents.agents.web_browser_agent.get_web_tools")
    def test_create_agent_returns_callable(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        mock_llm.return_value = _make_mock_llm()

        from bioagents.agents.web_browser_agent import create_web_browser_agent

        assert callable(create_web_browser_agent())

    @patch("bioagents.agents.web_browser_agent.get_llm")
    @patch("bioagents.agents.web_browser_agent.get_web_tools")
    def test_node_returns_messages(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        mock_llm.return_value = _make_mock_llm()

        from bioagents.agents.web_browser_agent import create_web_browser_agent

        result = create_web_browser_agent()(_make_state("Fetch docs"))
        assert "messages" in result
        assert len(result["messages"]) > 0

    @patch("bioagents.agents.web_browser_agent.get_llm")
    @patch("bioagents.agents.web_browser_agent.get_web_tools")
    def test_retry_on_empty_response(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        llm = _make_mock_llm()
        llm.invoke.side_effect = [AIMessage(content=""), AIMessage(content="OK")]
        mock_llm.return_value = llm

        from bioagents.agents.web_browser_agent import create_web_browser_agent

        result = create_web_browser_agent()(_make_state())
        assert result["messages"][0].content == "OK"


# =========================================================================
# 3. Paper Replication Agent  (tools: get_web_tools + get_git_tools)
# =========================================================================


class TestPaperReplicationAgent:
    @patch("bioagents.agents.paper_replication_agent.get_llm")
    @patch("bioagents.agents.paper_replication_agent.get_git_tools")
    @patch("bioagents.agents.paper_replication_agent.get_web_tools")
    def test_create_agent_returns_callable(self, mock_web, mock_git, mock_llm):
        mock_web.return_value = []
        mock_git.return_value = []
        mock_llm.return_value = _make_mock_llm()

        from bioagents.agents.paper_replication_agent import create_paper_replication_agent

        assert callable(create_paper_replication_agent())

    @patch("bioagents.agents.paper_replication_agent.get_llm")
    @patch("bioagents.agents.paper_replication_agent.get_git_tools")
    @patch("bioagents.agents.paper_replication_agent.get_web_tools")
    def test_node_returns_messages(self, mock_web, mock_git, mock_llm):
        mock_web.return_value = []
        mock_git.return_value = []
        mock_llm.return_value = _make_mock_llm()

        from bioagents.agents.paper_replication_agent import create_paper_replication_agent

        result = create_paper_replication_agent()(_make_state("Replicate paper"))
        assert "messages" in result
        assert len(result["messages"]) > 0

    @patch("bioagents.agents.paper_replication_agent.get_llm")
    @patch("bioagents.agents.paper_replication_agent.get_git_tools")
    @patch("bioagents.agents.paper_replication_agent.get_web_tools")
    def test_retry_on_empty_response(self, mock_web, mock_git, mock_llm):
        mock_web.return_value = []
        mock_git.return_value = []
        llm = _make_mock_llm()
        llm.invoke.side_effect = [AIMessage(content=""), AIMessage(content="Done")]
        mock_llm.return_value = llm

        from bioagents.agents.paper_replication_agent import create_paper_replication_agent

        result = create_paper_replication_agent()(_make_state())
        assert result["messages"][0].content == "Done"


# =========================================================================
# 4. Data Acquisition Agent  (tools: get_web_tools + get_file_tools)
# =========================================================================


class TestDataAcquisitionAgent:
    @patch("bioagents.agents.data_acquisition_agent.get_llm")
    @patch("bioagents.agents.data_acquisition_agent.get_file_tools")
    @patch("bioagents.agents.data_acquisition_agent.get_web_tools")
    def test_create_agent_returns_callable(self, mock_web, mock_file, mock_llm):
        mock_web.return_value = []
        mock_file.return_value = []
        mock_llm.return_value = _make_mock_llm()

        from bioagents.agents.data_acquisition_agent import create_data_acquisition_agent

        assert callable(create_data_acquisition_agent())

    @patch("bioagents.agents.data_acquisition_agent.get_llm")
    @patch("bioagents.agents.data_acquisition_agent.get_file_tools")
    @patch("bioagents.agents.data_acquisition_agent.get_web_tools")
    def test_node_returns_messages(self, mock_web, mock_file, mock_llm):
        mock_web.return_value = []
        mock_file.return_value = []
        mock_llm.return_value = _make_mock_llm()

        from bioagents.agents.data_acquisition_agent import create_data_acquisition_agent

        result = create_data_acquisition_agent()(_make_state("Download GEO"))
        assert "messages" in result
        assert len(result["messages"]) > 0

    @patch("bioagents.agents.data_acquisition_agent.get_llm")
    @patch("bioagents.agents.data_acquisition_agent.get_file_tools")
    @patch("bioagents.agents.data_acquisition_agent.get_web_tools")
    def test_retry_on_empty_response(self, mock_web, mock_file, mock_llm):
        mock_web.return_value = []
        mock_file.return_value = []
        llm = _make_mock_llm()
        llm.invoke.side_effect = [AIMessage(content=""), AIMessage(content="Got data")]
        mock_llm.return_value = llm

        from bioagents.agents.data_acquisition_agent import create_data_acquisition_agent

        result = create_data_acquisition_agent()(_make_state())
        assert result["messages"][0].content == "Got data"


# =========================================================================
# 5. Genomics Agent  (tools: get_genomics_tools)
# =========================================================================


class TestGenomicsAgent:
    @patch("bioagents.agents.genomics_agent.get_llm")
    @patch("bioagents.agents.genomics_agent.get_genomics_tools")
    def test_create_agent_returns_callable(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        mock_llm.return_value = _make_mock_llm()

        from bioagents.agents.genomics_agent import create_genomics_agent

        assert callable(create_genomics_agent())

    @patch("bioagents.agents.genomics_agent.get_llm")
    @patch("bioagents.agents.genomics_agent.get_genomics_tools")
    def test_node_returns_messages(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        mock_llm.return_value = _make_mock_llm()

        from bioagents.agents.genomics_agent import create_genomics_agent

        result = create_genomics_agent()(_make_state("Align sequences"))
        assert "messages" in result
        assert len(result["messages"]) > 0

    @patch("bioagents.agents.genomics_agent.get_llm")
    @patch("bioagents.agents.genomics_agent.get_genomics_tools")
    def test_retry_on_empty_response(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        llm = _make_mock_llm()
        llm.invoke.side_effect = [AIMessage(content=""), AIMessage(content="Aligned")]
        mock_llm.return_value = llm

        from bioagents.agents.genomics_agent import create_genomics_agent

        result = create_genomics_agent()(_make_state())
        assert result["messages"][0].content == "Aligned"


# =========================================================================
# 6. Transcriptomics Agent  (tools: get_transcriptomics_tools)
# =========================================================================


class TestTranscriptomicsAgent:
    @patch("bioagents.agents.transcriptomics_agent.get_llm")
    @patch("bioagents.agents.transcriptomics_agent.get_transcriptomics_tools")
    def test_create_agent_returns_callable(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        mock_llm.return_value = _make_mock_llm()

        from bioagents.agents.transcriptomics_agent import create_transcriptomics_agent

        assert callable(create_transcriptomics_agent())

    @patch("bioagents.agents.transcriptomics_agent.get_llm")
    @patch("bioagents.agents.transcriptomics_agent.get_transcriptomics_tools")
    def test_node_returns_messages(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        mock_llm.return_value = _make_mock_llm()

        from bioagents.agents.transcriptomics_agent import create_transcriptomics_agent

        result = create_transcriptomics_agent()(_make_state("DE analysis"))
        assert "messages" in result
        assert len(result["messages"]) > 0

    @patch("bioagents.agents.transcriptomics_agent.get_llm")
    @patch("bioagents.agents.transcriptomics_agent.get_transcriptomics_tools")
    def test_retry_on_empty_response(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        llm = _make_mock_llm()
        llm.invoke.side_effect = [AIMessage(content=""), AIMessage(content="DE done")]
        mock_llm.return_value = llm

        from bioagents.agents.transcriptomics_agent import create_transcriptomics_agent

        result = create_transcriptomics_agent()(_make_state())
        assert result["messages"][0].content == "DE done"


# =========================================================================
# 7. Structural Biology Agent  (tools: get_structural_tools)
# =========================================================================


class TestStructuralBiologyAgent:
    @patch("bioagents.agents.structural_biology_agent.get_llm")
    @patch("bioagents.agents.structural_biology_agent.get_structural_tools")
    def test_create_agent_returns_callable(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        mock_llm.return_value = _make_mock_llm()

        from bioagents.agents.structural_biology_agent import create_structural_biology_agent

        assert callable(create_structural_biology_agent())

    @patch("bioagents.agents.structural_biology_agent.get_llm")
    @patch("bioagents.agents.structural_biology_agent.get_structural_tools")
    def test_node_returns_messages(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        mock_llm.return_value = _make_mock_llm()

        from bioagents.agents.structural_biology_agent import create_structural_biology_agent

        result = create_structural_biology_agent()(_make_state("Predict structure"))
        assert "messages" in result
        assert len(result["messages"]) > 0

    @patch("bioagents.agents.structural_biology_agent.get_llm")
    @patch("bioagents.agents.structural_biology_agent.get_structural_tools")
    def test_retry_on_empty_response(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        llm = _make_mock_llm()
        llm.invoke.side_effect = [AIMessage(content=""), AIMessage(content="Structure predicted")]
        mock_llm.return_value = llm

        from bioagents.agents.structural_biology_agent import create_structural_biology_agent

        result = create_structural_biology_agent()(_make_state())
        assert result["messages"][0].content == "Structure predicted"


# =========================================================================
# 8. Phylogenetics Agent  (tools: get_genomics_tools)
# =========================================================================


class TestPhylogeneticsAgent:
    @patch("bioagents.agents.phylogenetics_agent.get_llm")
    @patch("bioagents.agents.phylogenetics_agent.get_genomics_tools")
    def test_create_agent_returns_callable(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        mock_llm.return_value = _make_mock_llm()

        from bioagents.agents.phylogenetics_agent import create_phylogenetics_agent

        assert callable(create_phylogenetics_agent())

    @patch("bioagents.agents.phylogenetics_agent.get_llm")
    @patch("bioagents.agents.phylogenetics_agent.get_genomics_tools")
    def test_node_returns_messages(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        mock_llm.return_value = _make_mock_llm()

        from bioagents.agents.phylogenetics_agent import create_phylogenetics_agent

        result = create_phylogenetics_agent()(_make_state("Build tree"))
        assert "messages" in result
        assert len(result["messages"]) > 0

    @patch("bioagents.agents.phylogenetics_agent.get_llm")
    @patch("bioagents.agents.phylogenetics_agent.get_genomics_tools")
    def test_retry_on_empty_response(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        llm = _make_mock_llm()
        llm.invoke.side_effect = [AIMessage(content=""), AIMessage(content="Tree built")]
        mock_llm.return_value = llm

        from bioagents.agents.phylogenetics_agent import create_phylogenetics_agent

        result = create_phylogenetics_agent()(_make_state())
        assert result["messages"][0].content == "Tree built"


# =========================================================================
# 9. Docking Agent  (no tools — uses invoke_with_retry)
# =========================================================================


class TestDockingAgent:
    @patch("bioagents.agents.docking_agent.get_llm")
    def test_create_agent_returns_callable(self, mock_llm):
        mock_llm.return_value = _make_mock_llm(with_tools=False)

        from bioagents.agents.docking_agent import create_docking_agent

        assert callable(create_docking_agent())

    @patch("bioagents.agents.docking_agent.get_llm")
    def test_node_returns_messages(self, mock_llm):
        mock_llm.return_value = _make_mock_llm(with_tools=False)

        from bioagents.agents.docking_agent import create_docking_agent

        result = create_docking_agent()(_make_state("Dock ligand"))
        assert "messages" in result
        assert len(result["messages"]) > 0

    @patch("bioagents.agents.docking_agent.get_llm")
    def test_retry_on_empty_response(self, mock_llm):
        llm = _make_mock_llm(with_tools=False)
        llm.invoke.side_effect = [AIMessage(content=""), AIMessage(content="Docked")]
        mock_llm.return_value = llm

        from bioagents.agents.docking_agent import create_docking_agent

        result = create_docking_agent()(_make_state())
        assert result["messages"][0].content == "Docked"

    @patch("bioagents.agents.docking_agent.get_llm")
    def test_fallback_on_all_empty(self, mock_llm):
        llm = _make_mock_llm(with_tools=False)
        llm.invoke.return_value = AIMessage(content="")
        mock_llm.return_value = llm

        from bioagents.agents.docking_agent import create_docking_agent

        result = create_docking_agent()(_make_state())
        assert len(result["messages"]) == 1
        assert result["messages"][0].content != ""


# =========================================================================
# 10. Planner Agent  (no tools — uses invoke_with_retry)
# =========================================================================


class TestPlannerAgent:
    @patch("bioagents.agents.planner_agent.get_llm")
    def test_create_agent_returns_callable(self, mock_llm):
        mock_llm.return_value = _make_mock_llm(with_tools=False)

        from bioagents.agents.planner_agent import create_planner_agent

        assert callable(create_planner_agent())

    @patch("bioagents.agents.planner_agent.get_llm")
    def test_node_returns_messages(self, mock_llm):
        mock_llm.return_value = _make_mock_llm(with_tools=False)

        from bioagents.agents.planner_agent import create_planner_agent

        result = create_planner_agent()(_make_state("Plan analysis"))
        assert "messages" in result
        assert len(result["messages"]) > 0

    @patch("bioagents.agents.planner_agent.get_llm")
    def test_retry_on_empty_response(self, mock_llm):
        llm = _make_mock_llm(with_tools=False)
        llm.invoke.side_effect = [AIMessage(content=""), AIMessage(content="Plan ready")]
        mock_llm.return_value = llm

        from bioagents.agents.planner_agent import create_planner_agent

        result = create_planner_agent()(_make_state())
        assert result["messages"][0].content == "Plan ready"


# =========================================================================
# 11. Tool Validator Agent  (no tools — uses invoke_with_retry)
# =========================================================================


class TestToolValidatorAgent:
    @patch("bioagents.agents.tool_validator_agent.get_llm")
    def test_create_agent_returns_callable(self, mock_llm):
        mock_llm.return_value = _make_mock_llm(with_tools=False)

        from bioagents.agents.tool_validator_agent import create_tool_validator_agent

        assert callable(create_tool_validator_agent())

    @patch("bioagents.agents.tool_validator_agent.get_llm")
    def test_node_returns_messages(self, mock_llm):
        mock_llm.return_value = _make_mock_llm(with_tools=False)

        from bioagents.agents.tool_validator_agent import create_tool_validator_agent

        result = create_tool_validator_agent()(_make_state("Validate tool call"))
        assert "messages" in result
        assert len(result["messages"]) > 0

    @patch("bioagents.agents.tool_validator_agent.get_llm")
    def test_retry_on_empty_response(self, mock_llm):
        llm = _make_mock_llm(with_tools=False)
        llm.invoke.side_effect = [AIMessage(content=""), AIMessage(content="Valid")]
        mock_llm.return_value = llm

        from bioagents.agents.tool_validator_agent import create_tool_validator_agent

        result = create_tool_validator_agent()(_make_state())
        assert result["messages"][0].content == "Valid"


# =========================================================================
# 12. Tool Discovery Agent  (tools: get_tool_builder_tools)
# =========================================================================


class TestToolDiscoveryAgent:
    @patch("bioagents.agents.tool_discovery_agent.get_llm")
    @patch("bioagents.agents.tool_discovery_agent.get_tool_builder_tools")
    def test_create_agent_returns_callable(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        mock_llm.return_value = _make_mock_llm()

        from bioagents.agents.tool_discovery_agent import create_tool_discovery_agent

        assert callable(create_tool_discovery_agent())

    @patch("bioagents.agents.tool_discovery_agent.get_llm")
    @patch("bioagents.agents.tool_discovery_agent.get_tool_builder_tools")
    def test_node_returns_messages(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        mock_llm.return_value = _make_mock_llm()

        from bioagents.agents.tool_discovery_agent import create_tool_discovery_agent

        result = create_tool_discovery_agent()(_make_state("Find BLAST tools"))
        assert "messages" in result
        assert len(result["messages"]) > 0

    @patch("bioagents.agents.tool_discovery_agent.get_llm")
    @patch("bioagents.agents.tool_discovery_agent.get_tool_builder_tools")
    def test_retry_on_empty_response(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        llm = _make_mock_llm()
        llm.invoke.side_effect = [AIMessage(content=""), AIMessage(content="Found tools")]
        mock_llm.return_value = llm

        from bioagents.agents.tool_discovery_agent import create_tool_discovery_agent

        result = create_tool_discovery_agent()(_make_state())
        assert result["messages"][0].content == "Found tools"


# =========================================================================
# 13. Prompt Optimizer Agent  (no tools — uses invoke_with_retry)
# =========================================================================


class TestPromptOptimizerAgent:
    @patch("bioagents.agents.prompt_optimizer_agent.get_llm")
    def test_create_agent_returns_callable(self, mock_llm):
        mock_llm.return_value = _make_mock_llm(with_tools=False)

        from bioagents.agents.prompt_optimizer_agent import create_prompt_optimizer_agent

        assert callable(create_prompt_optimizer_agent())

    @patch("bioagents.agents.prompt_optimizer_agent.get_llm")
    def test_node_returns_messages(self, mock_llm):
        mock_llm.return_value = _make_mock_llm(with_tools=False)

        from bioagents.agents.prompt_optimizer_agent import create_prompt_optimizer_agent

        result = create_prompt_optimizer_agent()(_make_state("Optimize prompts"))
        assert "messages" in result
        assert len(result["messages"]) > 0

    @patch("bioagents.agents.prompt_optimizer_agent.get_llm")
    def test_retry_on_empty_response(self, mock_llm):
        llm = _make_mock_llm(with_tools=False)
        llm.invoke.side_effect = [AIMessage(content=""), AIMessage(content="Optimized")]
        mock_llm.return_value = llm

        from bioagents.agents.prompt_optimizer_agent import create_prompt_optimizer_agent

        result = create_prompt_optimizer_agent()(_make_state())
        assert result["messages"][0].content == "Optimized"


# =========================================================================
# 14. Result Checker Agent  (no tools — uses invoke_with_retry)
# =========================================================================


class TestResultCheckerAgent:
    @patch("bioagents.agents.result_checker_agent.get_llm")
    def test_create_agent_returns_callable(self, mock_llm):
        mock_llm.return_value = _make_mock_llm(with_tools=False)

        from bioagents.agents.result_checker_agent import create_result_checker_agent

        assert callable(create_result_checker_agent())

    @patch("bioagents.agents.result_checker_agent.get_llm")
    def test_node_returns_messages(self, mock_llm):
        mock_llm.return_value = _make_mock_llm(with_tools=False)

        from bioagents.agents.result_checker_agent import create_result_checker_agent

        result = create_result_checker_agent()(_make_state("Check results"))
        assert "messages" in result
        assert len(result["messages"]) > 0

    @patch("bioagents.agents.result_checker_agent.get_llm")
    def test_retry_on_empty_response(self, mock_llm):
        llm = _make_mock_llm(with_tools=False)
        llm.invoke.side_effect = [AIMessage(content=""), AIMessage(content="Results OK")]
        mock_llm.return_value = llm

        from bioagents.agents.result_checker_agent import create_result_checker_agent

        result = create_result_checker_agent()(_make_state())
        assert result["messages"][0].content == "Results OK"

    @patch("bioagents.agents.result_checker_agent.get_llm")
    def test_fallback_on_all_empty(self, mock_llm):
        llm = _make_mock_llm(with_tools=False)
        llm.invoke.return_value = AIMessage(content="")
        mock_llm.return_value = llm

        from bioagents.agents.result_checker_agent import create_result_checker_agent

        result = create_result_checker_agent()(_make_state())
        assert result["messages"][0].content != ""


# =========================================================================
# 15. Shell Agent  (tools: get_shell_tools)
# =========================================================================


class TestShellAgent:
    @patch("bioagents.agents.shell_agent.get_llm")
    @patch("bioagents.agents.shell_agent.get_shell_tools")
    def test_create_agent_returns_callable(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        mock_llm.return_value = _make_mock_llm()

        from bioagents.agents.shell_agent import create_shell_agent

        assert callable(create_shell_agent())

    @patch("bioagents.agents.shell_agent.get_llm")
    @patch("bioagents.agents.shell_agent.get_shell_tools")
    def test_node_returns_messages(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        mock_llm.return_value = _make_mock_llm()

        from bioagents.agents.shell_agent import create_shell_agent

        result = create_shell_agent()(_make_state("Run command"))
        assert "messages" in result
        assert len(result["messages"]) > 0

    @patch("bioagents.agents.shell_agent.get_llm")
    @patch("bioagents.agents.shell_agent.get_shell_tools")
    def test_retry_on_empty_response(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        llm = _make_mock_llm()
        llm.invoke.side_effect = [AIMessage(content=""), AIMessage(content="Executed")]
        mock_llm.return_value = llm

        from bioagents.agents.shell_agent import create_shell_agent

        result = create_shell_agent()(_make_state())
        assert result["messages"][0].content == "Executed"


# =========================================================================
# 16. Git Agent  (tools: get_git_tools)
# =========================================================================


class TestGitAgent:
    @patch("bioagents.agents.git_agent.get_llm")
    @patch("bioagents.agents.git_agent.get_git_tools")
    def test_create_agent_returns_callable(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        mock_llm.return_value = _make_mock_llm()

        from bioagents.agents.git_agent import create_git_agent

        assert callable(create_git_agent())

    @patch("bioagents.agents.git_agent.get_llm")
    @patch("bioagents.agents.git_agent.get_git_tools")
    def test_node_returns_messages(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        mock_llm.return_value = _make_mock_llm()

        from bioagents.agents.git_agent import create_git_agent

        result = create_git_agent()(_make_state("Clone repo"))
        assert "messages" in result
        assert len(result["messages"]) > 0

    @patch("bioagents.agents.git_agent.get_llm")
    @patch("bioagents.agents.git_agent.get_git_tools")
    def test_retry_on_empty_response(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        llm = _make_mock_llm()
        llm.invoke.side_effect = [AIMessage(content=""), AIMessage(content="Cloned")]
        mock_llm.return_value = llm

        from bioagents.agents.git_agent import create_git_agent

        result = create_git_agent()(_make_state())
        assert result["messages"][0].content == "Cloned"


# =========================================================================
# 17. Environment Agent  (tools: get_environment_tools)
# =========================================================================


class TestEnvironmentAgent:
    @patch("bioagents.agents.environment_agent.get_llm")
    @patch("bioagents.agents.environment_agent.get_environment_tools")
    def test_create_agent_returns_callable(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        mock_llm.return_value = _make_mock_llm()

        from bioagents.agents.environment_agent import create_environment_agent

        assert callable(create_environment_agent())

    @patch("bioagents.agents.environment_agent.get_llm")
    @patch("bioagents.agents.environment_agent.get_environment_tools")
    def test_node_returns_messages(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        mock_llm.return_value = _make_mock_llm()

        from bioagents.agents.environment_agent import create_environment_agent

        result = create_environment_agent()(_make_state("Setup env"))
        assert "messages" in result
        assert len(result["messages"]) > 0

    @patch("bioagents.agents.environment_agent.get_llm")
    @patch("bioagents.agents.environment_agent.get_environment_tools")
    def test_retry_on_empty_response(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        llm = _make_mock_llm()
        llm.invoke.side_effect = [AIMessage(content=""), AIMessage(content="Env ready")]
        mock_llm.return_value = llm

        from bioagents.agents.environment_agent import create_environment_agent

        result = create_environment_agent()(_make_state())
        assert result["messages"][0].content == "Env ready"


# =========================================================================
# 18. Visualization Agent  (tools: get_visualization_tools)
# =========================================================================


class TestVisualizationAgent:
    @patch("bioagents.agents.visualization_agent.get_llm")
    @patch("bioagents.agents.visualization_agent.get_visualization_tools")
    def test_create_agent_returns_callable(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        mock_llm.return_value = _make_mock_llm()

        from bioagents.agents.visualization_agent import create_visualization_agent

        assert callable(create_visualization_agent())

    @patch("bioagents.agents.visualization_agent.get_llm")
    @patch("bioagents.agents.visualization_agent.get_visualization_tools")
    def test_node_returns_messages(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        mock_llm.return_value = _make_mock_llm()

        from bioagents.agents.visualization_agent import create_visualization_agent

        result = create_visualization_agent()(_make_state("Create heatmap"))
        assert "messages" in result
        assert len(result["messages"]) > 0

    @patch("bioagents.agents.visualization_agent.get_llm")
    @patch("bioagents.agents.visualization_agent.get_visualization_tools")
    def test_retry_on_empty_response(self, mock_tools, mock_llm):
        mock_tools.return_value = []
        llm = _make_mock_llm()
        llm.invoke.side_effect = [AIMessage(content=""), AIMessage(content="Plot created")]
        mock_llm.return_value = llm

        from bioagents.agents.visualization_agent import create_visualization_agent

        result = create_visualization_agent()(_make_state())
        assert result["messages"][0].content == "Plot created"
