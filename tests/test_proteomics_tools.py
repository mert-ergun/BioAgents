"""Tests for proteomics tools."""

from bioagents.tools.proteomics_tools import fetch_uniprot_fasta


def test_fetch_uniprot_fasta_valid():
    """Test fetching a valid protein sequence."""
    result = fetch_uniprot_fasta.invoke({"protein_id": "P04637"})

    assert isinstance(result, str)
    assert result.startswith(">")
    assert "P04637" in result or "P53_HUMAN" in result
    assert len(result) > 100  # Should have sequence data


def test_fetch_uniprot_fasta_invalid():
    """Test fetching an invalid protein ID."""
    result = fetch_uniprot_fasta.invoke({"protein_id": "INVALID_ID_12345"})

    assert isinstance(result, str)
    assert "Error" in result or "not found" in result.lower()


def test_fetch_uniprot_fasta_alternate_id():
    """Test fetching with alternate protein ID format."""
    result = fetch_uniprot_fasta.invoke({"protein_id": "P53_HUMAN"})

    assert isinstance(result, str)
    assert len(result) > 0
