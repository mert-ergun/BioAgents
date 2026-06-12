"""Tests for new tool modules added in Phase 0."""

import json
from unittest.mock import Mock, patch

# =========================================================================
# Shell Tools
# =========================================================================


class TestShellTools:
    def test_get_shell_tools_returns_list(self):
        from bioagents.tools.shell_tools import get_shell_tools

        tools = get_shell_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_tools_have_names_and_descriptions(self):
        from bioagents.tools.shell_tools import get_shell_tools

        for tool in get_shell_tools():
            assert hasattr(tool, "name")
            assert tool.name
            assert hasattr(tool, "description")
            assert tool.description

    def test_expected_tool_names(self):
        from bioagents.tools.shell_tools import get_shell_tools

        names = {t.name for t in get_shell_tools()}
        assert "run_shell_command" in names
        assert "install_python_package" in names
        assert "check_installed_packages" in names

    @patch("bioagents.tools.shell_tools.get_sandbox")
    def test_run_shell_command(self, mock_get_sandbox):
        mock_sandbox = Mock()
        mock_sandbox.run_command.return_value = {
            "success": True,
            "stdout": "hello\n",
            "stderr": "",
            "returncode": 0,
        }
        mock_get_sandbox.return_value = mock_sandbox

        from bioagents.tools.shell_tools import run_shell_command

        result = json.loads(run_shell_command.invoke({"command": "echo hello"}))
        assert result["success"] is True
        assert "hello" in result["stdout"]


# =========================================================================
# Git Tools
# =========================================================================


class TestGitTools:
    def test_get_git_tools_returns_list(self):
        from bioagents.tools.git_tools import get_git_tools

        tools = get_git_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_tools_have_names_and_descriptions(self):
        from bioagents.tools.git_tools import get_git_tools

        for tool in get_git_tools():
            assert hasattr(tool, "name")
            assert tool.name
            assert hasattr(tool, "description")
            assert tool.description

    def test_expected_tool_names(self):
        from bioagents.tools.git_tools import get_git_tools

        names = {t.name for t in get_git_tools()}
        assert "git_clone_repo" in names
        assert "list_repo_files" in names
        assert "read_repo_file" in names
        assert "git_checkout_branch" in names

    @patch("bioagents.tools.git_tools.get_sandbox")
    def test_git_clone_repo(self, mock_get_sandbox):
        mock_sandbox = Mock()
        mock_sandbox.git_clone.return_value = {
            "success": True,
            "stdout": "Cloning...",
            "stderr": "",
        }
        mock_get_sandbox.return_value = mock_sandbox

        from bioagents.tools.git_tools import git_clone_repo

        result = git_clone_repo.invoke({"repo_url": "https://github.com/user/repo.git"})
        assert "Successfully cloned" in result


# =========================================================================
# Web Tools
# =========================================================================


class TestWebTools:
    def test_get_web_tools_returns_list(self):
        from bioagents.tools.web_tools import get_web_tools

        tools = get_web_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_tools_have_names_and_descriptions(self):
        from bioagents.tools.web_tools import get_web_tools

        for tool in get_web_tools():
            assert hasattr(tool, "name")
            assert tool.name
            assert hasattr(tool, "description")
            assert tool.description

    def test_expected_tool_names(self):
        from bioagents.tools.web_tools import get_web_tools

        names = {t.name for t in get_web_tools()}
        assert "fetch_url_content" in names
        assert "search_google_scholar" in names
        assert "download_file_from_url" in names

    @patch("bioagents.tools.web_tools.requests")
    def test_fetch_url_content(self, mock_requests):
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body>Hello World</body></html>"
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.raise_for_status = Mock()
        mock_requests.get.return_value = mock_resp

        from bioagents.tools.web_tools import fetch_url_content

        result = fetch_url_content.invoke({"url": "https://example.com"})
        assert "Hello World" in result


# =========================================================================
# Literature Tools
# =========================================================================


class TestLiteratureTools:
    def test_get_literature_tools_returns_list(self):
        from bioagents.tools.literature_tools import get_literature_tools

        tools = get_literature_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_tools_have_names_and_descriptions(self):
        from bioagents.tools.literature_tools import get_literature_tools

        for tool in get_literature_tools():
            assert hasattr(tool, "name")
            assert tool.name
            assert hasattr(tool, "description")
            assert tool.description

    def test_expected_tool_names(self):
        from bioagents.tools.literature_tools import get_literature_tools

        names = {t.name for t in get_literature_tools()}
        assert "search_pubmed" in names
        assert "search_arxiv" in names
        assert "search_biorxiv" in names
        assert "fetch_paper_metadata" in names


# =========================================================================
# Genomics Tools
# =========================================================================


class TestGenomicsTools:
    def test_get_genomics_tools_returns_list(self):
        from bioagents.tools.genomics_tools import get_genomics_tools

        tools = get_genomics_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_tools_have_names_and_descriptions(self):
        from bioagents.tools.genomics_tools import get_genomics_tools

        for tool in get_genomics_tools():
            assert hasattr(tool, "name")
            assert tool.name
            assert hasattr(tool, "description")
            assert tool.description

    def test_expected_tool_names(self):
        from bioagents.tools.genomics_tools import get_genomics_tools

        names = {t.name for t in get_genomics_tools()}
        assert "run_blast_search" in names
        assert "parse_fasta_file" in names
        assert "reverse_complement" in names
        assert "translate_dna" in names
        assert "calculate_gc_content" in names

    def test_reverse_complement(self):
        from bioagents.tools.genomics_tools import reverse_complement

        result = reverse_complement.invoke({"sequence": "ATGC"})
        assert result == "GCAT"

    def test_reverse_complement_invalid(self):
        from bioagents.tools.genomics_tools import reverse_complement

        result = reverse_complement.invoke({"sequence": "ATXG"})
        assert "Error" in result

    def test_translate_dna(self):
        from bioagents.tools.genomics_tools import translate_dna

        result = translate_dna.invoke({"dna_sequence": "ATGGCC"})
        assert result == "MA"

    def test_calculate_gc_content(self):
        from bioagents.tools.genomics_tools import calculate_gc_content

        result = json.loads(calculate_gc_content.invoke({"sequence": "AATTGGCC"}))
        assert result["gc_content"] == "50.00%"
        assert result["total_length"] == 8


# =========================================================================
# Transcriptomics Tools
# =========================================================================


class TestTranscriptomicsTools:
    def test_get_transcriptomics_tools_returns_list(self):
        from bioagents.tools.transcriptomics_tools import get_transcriptomics_tools

        tools = get_transcriptomics_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_tools_have_names_and_descriptions(self):
        from bioagents.tools.transcriptomics_tools import get_transcriptomics_tools

        for tool in get_transcriptomics_tools():
            assert hasattr(tool, "name")
            assert tool.name
            assert hasattr(tool, "description")
            assert tool.description

    def test_expected_tool_names(self):
        from bioagents.tools.transcriptomics_tools import get_transcriptomics_tools

        names = {t.name for t in get_transcriptomics_tools()}
        assert "run_differential_expression" in names
        assert "run_gene_set_enrichment" in names
        assert "normalize_expression_data" in names


# =========================================================================
# Visualization Tools
# =========================================================================


class TestVisualizationTools:
    def test_get_visualization_tools_returns_list(self):
        from bioagents.tools.visualization_tools import get_visualization_tools

        tools = get_visualization_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_tools_have_names_and_descriptions(self):
        from bioagents.tools.visualization_tools import get_visualization_tools

        for tool in get_visualization_tools():
            assert hasattr(tool, "name")
            assert tool.name
            assert hasattr(tool, "description")
            assert tool.description

    def test_expected_tool_names(self):
        from bioagents.tools.visualization_tools import get_visualization_tools

        names = {t.name for t in get_visualization_tools()}
        assert "create_bar_chart" in names
        assert "create_heatmap" in names
        assert "create_scatter_plot" in names
        assert "create_volcano_plot" in names


# =========================================================================
# File Tools
# =========================================================================


class TestFileTools:
    def test_get_file_tools_returns_list(self):
        from bioagents.tools.file_tools import get_file_tools

        tools = get_file_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_tools_have_names_and_descriptions(self):
        from bioagents.tools.file_tools import get_file_tools

        for tool in get_file_tools():
            assert hasattr(tool, "name")
            assert tool.name
            assert hasattr(tool, "description")
            assert tool.description

    def test_expected_tool_names(self):
        from bioagents.tools.file_tools import get_file_tools

        names = {t.name for t in get_file_tools()}
        assert "read_local_file" in names
        assert "write_local_file" in names
        assert "list_local_directory" in names
        assert "get_file_info" in names

    @patch("bioagents.tools.file_tools.get_sandbox")
    def test_read_local_file(self, mock_get_sandbox):
        mock_sandbox = Mock()
        mock_sandbox.read_file.return_value = "file content here"
        mock_get_sandbox.return_value = mock_sandbox

        from bioagents.tools.file_tools import read_local_file

        result = read_local_file.invoke({"file_path": "test.txt"})
        assert result == "file content here"

    @patch("bioagents.tools.file_tools.get_sandbox")
    def test_write_local_file(self, mock_get_sandbox):
        mock_sandbox = Mock()
        mock_sandbox.write_file.return_value = "/sandbox/test.txt"
        mock_get_sandbox.return_value = mock_sandbox

        from bioagents.tools.file_tools import write_local_file

        result = write_local_file.invoke({"file_path": "test.txt", "content": "hello"})
        assert "Successfully wrote" in result


# =========================================================================
# Environment Tools
# =========================================================================


class TestEnvironmentTools:
    def test_get_environment_tools_returns_list(self):
        from bioagents.tools.environment_tools import get_environment_tools

        tools = get_environment_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_tools_have_names_and_descriptions(self):
        from bioagents.tools.environment_tools import get_environment_tools

        for tool in get_environment_tools():
            assert hasattr(tool, "name")
            assert tool.name
            assert hasattr(tool, "description")
            assert tool.description

    def test_expected_tool_names(self):
        from bioagents.tools.environment_tools import get_environment_tools

        names = {t.name for t in get_environment_tools()}
        assert "create_virtual_environment" in names
        assert "install_requirements" in names
        assert "check_gpu_available" in names
        assert "get_system_info" in names
