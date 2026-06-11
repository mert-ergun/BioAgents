"""Tests for LitSense literature search tool."""

import json
from unittest.mock import Mock, patch

import requests

from bioagents.tools.literature_tools import HTTP_TIMEOUT, get_literature_tools, search_litsense


class TestSearchLitsenseToolMetadata:
    """Tests for search_litsense tool registration and metadata."""

    def test_tool_name_and_description(self):
        """Test that the tool has expected name and description."""
        assert search_litsense.name == "search_litsense"
        assert hasattr(search_litsense, "description")
        assert "litsense" in search_litsense.description.lower()

    def test_included_in_literature_tools(self):
        """Test that search_litsense is exported via get_literature_tools."""
        names = {t.name for t in get_literature_tools()}
        assert "search_litsense" in names


class TestSearchLitsenseSuccess:
    """Tests for successful LitSense API responses."""

    @patch("bioagents.tools.literature_tools.requests.get")
    def test_successful_search_returns_json(self, mock_get):
        """Test parsing a successful LitSense API response."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = [
            {
                "pmid": "12345678",
                "text": "TP53 mutations are common in cancer.",
                "score": 0.95,
            },
            {
                "pmid": "87654321",
                "text": "The p53 protein regulates cell cycle.",
                "score": 0.88,
            },
        ]
        mock_get.return_value = mock_response

        result = search_litsense.invoke({"query": "p53 cancer", "max_results": 5})
        parsed = json.loads(result)

        assert len(parsed) == 2
        assert parsed[0]["pmid"] == "12345678"
        assert parsed[0]["sentence"] == "TP53 mutations are common in cancer."
        assert parsed[0]["score"] == 0.95
        assert parsed[1]["pmid"] == "87654321"

    @patch("bioagents.tools.literature_tools.requests.get")
    def test_api_url_and_params(self, mock_get):
        """Test that the correct LitSense 2.0 endpoint and params are used."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        search_litsense.invoke({"query": "CRISPR gene editing", "max_results": 3})

        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert (
            call_args[0][0]
            == "https://www.ncbi.nlm.nih.gov/research/litsense2-api/api/sentences/"
        )
        assert call_args[1]["params"] == {"query": "CRISPR gene editing", "rerank": "true"}
        assert call_args[1]["timeout"] == HTTP_TIMEOUT

    @patch("bioagents.tools.literature_tools.requests.get")
    def test_max_results_clamped_to_valid_range(self, mock_get):
        """Test that max_results is clamped between 1 and 30."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = [{"pmid": "1", "text": "sentence", "score": 0.5}] * 35
        mock_get.return_value = mock_response

        result = search_litsense.invoke({"query": "test", "max_results": 100})
        parsed = json.loads(result)
        assert len(parsed) == 30

        mock_get.reset_mock()
        mock_response.json.return_value = [{"pmid": "1", "text": "sentence", "score": 0.5}]
        mock_get.return_value = mock_response

        result = search_litsense.invoke({"query": "test", "max_results": 0})
        parsed = json.loads(result)
        assert len(parsed) == 1

    @patch("bioagents.tools.literature_tools.requests.get")
    def test_missing_fields_use_defaults(self, mock_get):
        """Test that missing API fields are handled with empty defaults."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = [{}]
        mock_get.return_value = mock_response

        result = search_litsense.invoke({"query": "test"})
        parsed = json.loads(result)

        assert parsed[0]["pmid"] == ""
        assert parsed[0]["sentence"] == ""
        assert parsed[0]["score"] == ""


class TestSearchLitsenseEmptyResults:
    """Tests for LitSense searches that return no matches."""

    @patch("bioagents.tools.literature_tools.requests.get")
    def test_empty_api_response(self, mock_get):
        """Test message when LitSense returns no results."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        result = search_litsense.invoke({"query": "nonexistent_xyz_query_12345"})

        assert isinstance(result, str)
        assert "No LitSense results found" in result
        assert "nonexistent_xyz_query_12345" in result


class TestSearchLitsenseErrorHandling:
    """Tests for LitSense error handling."""

    @patch("bioagents.tools.literature_tools.requests.get")
    def test_http_error(self, mock_get):
        """Test HTTP error handling."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "Server error", response=mock_response
        )
        mock_get.return_value = mock_response

        result = search_litsense.invoke({"query": "p53"})

        assert "Error searching LitSense" in result

    @patch("bioagents.tools.literature_tools.requests.get")
    def test_timeout_error(self, mock_get):
        """Test timeout error handling."""
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

        result = search_litsense.invoke({"query": "p53"})

        assert "Error searching LitSense" in result
        assert "timed out" in result.lower()

    @patch("bioagents.tools.literature_tools.requests.get")
    def test_connection_error(self, mock_get):
        """Test connection error handling."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")

        result = search_litsense.invoke({"query": "p53"})

        assert "Error searching LitSense" in result

    @patch("bioagents.tools.literature_tools.requests.get")
    def test_json_decode_error(self, mock_get):
        """Test handling of invalid JSON responses."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response

        result = search_litsense.invoke({"query": "p53"})

        assert "Error searching LitSense" in result
