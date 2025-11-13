"""Tests for proteomics tools."""

from unittest.mock import Mock, patch

import requests

from bioagents.tools.proteomics_tools import fetch_uniprot_fasta


class TestFetchUniprotFasta:
    """Tests for the fetch_uniprot_fasta tool."""

    def test_fetch_valid_protein(self):
        """Test fetching a valid protein sequence."""
        result = fetch_uniprot_fasta.invoke({"protein_id": "P04637"})

        assert isinstance(result, str)
        assert result.startswith(">")
        assert "P04637" in result or "P53_HUMAN" in result
        assert len(result) > 100  # Should have sequence data

    def test_fetch_invalid_protein(self):
        """Test fetching an invalid protein ID."""
        result = fetch_uniprot_fasta.invoke({"protein_id": "INVALID_ID_12345"})

        assert isinstance(result, str)
        assert "Error" in result or "not found" in result.lower()

    def test_fetch_alternate_protein_id(self):
        """Test fetching with alternate protein ID format."""
        result = fetch_uniprot_fasta.invoke({"protein_id": "P53_HUMAN"})

        assert isinstance(result, str)
        assert len(result) > 0

    @patch("bioagents.tools.proteomics_tools.requests.get")
    def test_fetch_successful_response(self, mock_get):
        """Test successful API response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = ">sp|P12345|TEST_HUMAN Test protein\nMASLKGFVP"
        mock_get.return_value = mock_response

        result = fetch_uniprot_fasta.invoke({"protein_id": "P12345"})

        assert result == ">sp|P12345|TEST_HUMAN Test protein\nMASLKGFVP"
        mock_get.assert_called_once()
        assert "P12345" in mock_get.call_args[0][0]

    @patch("bioagents.tools.proteomics_tools.requests.get")
    def test_fetch_404_error(self, mock_get):
        """Test 404 error handling."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )
        mock_get.return_value = mock_response

        result = fetch_uniprot_fasta.invoke({"protein_id": "NOTFOUND"})

        assert "Error" in result
        assert "not found" in result.lower()
        assert "NOTFOUND" in result

    @patch("bioagents.tools.proteomics_tools.requests.get")
    def test_fetch_timeout_error(self, mock_get):
        """Test timeout error handling."""
        mock_get.side_effect = requests.exceptions.Timeout("Timeout occurred")

        result = fetch_uniprot_fasta.invoke({"protein_id": "P12345"})

        assert "Error" in result
        assert "Timeout" in result

    @patch("bioagents.tools.proteomics_tools.requests.get")
    def test_fetch_connection_error(self, mock_get):
        """Test connection error handling."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")

        result = fetch_uniprot_fasta.invoke({"protein_id": "P12345"})

        assert "Error" in result

    @patch("bioagents.tools.proteomics_tools.requests.get")
    def test_fetch_http_error_500(self, mock_get):
        """Test 500 server error handling."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "Server error", response=mock_response
        )
        mock_get.return_value = mock_response

        result = fetch_uniprot_fasta.invoke({"protein_id": "P12345"})

        assert "Error" in result

    @patch("bioagents.tools.proteomics_tools.requests.get")
    def test_timeout_parameter(self, mock_get):
        """Test that timeout parameter is used."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = ">sp|P12345|TEST\nMASL"
        mock_get.return_value = mock_response

        fetch_uniprot_fasta.invoke({"protein_id": "P12345"})

        # Verify timeout is set to 10 seconds
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["timeout"] == 10

    def test_tool_metadata(self):
        """Test tool has proper metadata."""
        assert fetch_uniprot_fasta.name == "fetch_uniprot_fasta"
        assert hasattr(fetch_uniprot_fasta, "description")
        assert "protein" in fetch_uniprot_fasta.description.lower()

    def test_strip_whitespace(self):
        """Test that response text is stripped of extra whitespace."""
        with patch("bioagents.tools.proteomics_tools.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "  >sp|P12345|TEST\nMASL  \n\n"
            mock_get.return_value = mock_response

            result = fetch_uniprot_fasta.invoke({"protein_id": "P12345"})

            # Should be stripped
            assert result == ">sp|P12345|TEST\nMASL"
