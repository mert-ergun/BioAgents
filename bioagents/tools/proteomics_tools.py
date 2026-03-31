"""Proteomics tools for fetching protein data from UniProt."""

import json

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


@tool
def download_uniprot_flat_file(accession: str, output_path: str) -> str:
    """Download full UniProtKB entry in plain text (`.txt`) format to the workspace.

    Use this for full UniProt entries instead of fetch_url_content: the flat file
    is often very large (many MB of references); this tool saves it to disk and
    returns a short JSON summary so the model is not flooded with text.

    Args:
        accession: UniProt accession (e.g. P04637).
        output_path: Path under the sandbox workspace (e.g. p53_uniprot_entry.txt).

    Returns:
        JSON with status, path, size, and a short preview of the first lines.
    """
    try:
        from bioagents.sandbox.sandbox_manager import get_sandbox

        accession = accession.strip()
        if not accession:
            return json.dumps({"status": "error", "message": "accession is empty"})
        url = f"https://rest.uniprot.org/uniprotkb/{accession}.txt"
        headers = {"User-Agent": "BioAgents/1.0 (data acquisition)"}
        response = requests.get(url, headers=headers, timeout=120)
        response.raise_for_status()
        body = response.text
        sandbox = get_sandbox()
        full_path = sandbox.write_file(output_path, body)
        preview = "\n".join(body.splitlines()[:40])
        if len(preview) > 1500:
            preview = preview[:1500] + "\n..."
        return json.dumps(
            {
                "status": "success",
                "accession": accession,
                "url": url,
                "file_path": str(full_path),
                "bytes_written": len(body),
                "preview_lines": preview,
            },
            indent=2,
        )
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return json.dumps(
                {"status": "error", "message": f"UniProt accession '{accession}' not found."}
            )
        return json.dumps({"status": "error", "message": str(e)})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
