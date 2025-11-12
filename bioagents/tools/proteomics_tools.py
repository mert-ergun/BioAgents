"""Proteomics tools for fetching protein data from UniProt."""

import requests
from langchain_core.tools import tool


@tool
def fetch_uniprot_fasta(protein_id: str) -> str:
    """
    Fetch the FASTA sequence for a protein from UniProt.

    Args:
        protein_id: The UniProt protein identifier (e.g., 'P53_HUMAN' or 'P04637')

    Returns:
        The FASTA sequence as a string, or an error message if the fetch fails.
    """
    try:
        # UniProt REST API endpoint for FASTA format
        url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.fasta"

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        return response.text.strip()

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return f"Error: Protein '{protein_id}' not found in UniProt."
        return f"Error fetching protein data: {e!s}"

    except Exception as e:
        return f"Error: {e!s}"
