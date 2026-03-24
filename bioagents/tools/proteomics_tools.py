"""Proteomics tools for fetching protein data from UniProt."""

import requests
from langchain_core.tools import tool


def fetch_uniprot_fasta_impl(protein_id: str, timeout: int = 10) -> str:
    """
    Fetch the FASTA sequence for a protein from UniProt (plain function for reuse).

    Args:
        protein_id: The UniProt protein identifier (e.g., 'P53_HUMAN' or 'P04637')
        timeout: HTTP timeout in seconds.

    Returns:
        The FASTA sequence as a string, or an error message if the fetch fails.
    """
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.fasta"

        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        return response.text.strip()

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return f"Error: Protein '{protein_id}' not found in UniProt."
        return f"Error fetching protein data: {e!s}"

    except Exception as e:
        return f"Error: {e!s}"


@tool
def fetch_uniprot_fasta(protein_id: str) -> str:
    """
    Fetch the FASTA sequence for a protein from UniProt.

    Args:
        protein_id: The UniProt protein identifier (e.g., 'P53_HUMAN' or 'P04637')

    Returns:
        The FASTA sequence as a string, or an error message if the fetch fails.
    """
    return fetch_uniprot_fasta_impl(protein_id)
