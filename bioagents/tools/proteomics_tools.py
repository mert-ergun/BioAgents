"""Proteomics tools for fetching protein data from UniProt."""

import json
from typing import Literal

import requests
from langchain_core.tools import tool

from bioagents.tools.provider_utils import get_provider_key_or_ask


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

        return str(response.text.strip())

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


ESM3Provider = Literal["EvolutionaryScale Forge", "AWS SageMaker", "NVIDIA BioNeMo"]
SaProtProvider = Literal["Tamarind Bio", "Hugging Face"]
ESM2Provider = Literal["Local (HuggingFace)", "NVIDIA BioNeMo", "Hugging Face", "Tamarind Bio"]


def _unavailable_hosted_model(
    model: str, provider: str, sequence: str, local_alternative: str
) -> str:
    """Report honestly that a hosted model could not be run.

    Hosted-only models have no local implementation in this deployment. Rather than
    returning a success string for work that never happened, this surfaces exactly
    what is missing and which real tool to use instead.

    Deliberately does NOT prompt the user for an API key: no client is wired up, so
    supplying a key would still not run anything. Asking for one would imply a
    capability that does not exist.
    """
    return json.dumps(
        {
            "status": "error",
            "error_type": "not_implemented",
            "model": model,
            "provider": provider,
            "sequence_length": len(sequence.strip()) if sequence else 0,
            "message": (
                f"{model} has no local implementation and no client for provider "
                f"'{provider}' is wired up in this deployment. NO computation was performed."
            ),
            "use_instead": local_alternative,
            "do_not": (
                "Do not report or estimate results for this model. If you need these "
                "numbers, call the suggested local tool or tell the user it is unavailable."
            ),
        },
        indent=2,
    )


@tool
def run_esm3(sequence: str, provider: ESM3Provider = "EvolutionaryScale Forge") -> str:
    """Run the ESM-3 (98B) model for protein representation via a hosted provider.

    ESM-3 is not runnable locally; it requires a hosted provider and an API key.
    This tool does NOT fall back to a smaller model silently — if the provider is
    not configured it returns an error telling you to use `run_esm_embedding`
    (real, local ESM-2/ESM-1b) instead.

    Args:
        sequence: Protein sequence in single-letter amino acid code.
        provider: Hosted provider to call. Requires the matching API key in the environment.

    Returns:
        JSON with the provider response, or status='error' explaining what is missing.
        Never reports success for a computation that did not run.
    """
    return _unavailable_hosted_model(
        model="ESM-3 (98B)",
        provider=provider,
        sequence=sequence,
        local_alternative="run_esm_embedding (local ESM-2/ESM-1b, no API key required)",
    )


@tool
def run_saprot(sequence: str, provider: SaProtProvider = "Tamarind Bio") -> str:
    """Run SaProt for structure-aware protein embeddings via a hosted provider.

    SaProt needs structure tokens (Foldseek 3Di) alongside the sequence and is served
    by a hosted provider here. If the provider is not configured this returns an error
    rather than a fabricated result.

    Args:
        sequence: Protein sequence in single-letter amino acid code.
        provider: Hosted provider to call. Requires the matching API key in the environment.

    Returns:
        JSON with the provider response, or status='error' explaining what is missing.
        Never reports success for a computation that did not run.
    """
    return _unavailable_hosted_model(
        model="SaProt",
        provider=provider,
        sequence=sequence,
        local_alternative="run_esm_embedding (local sequence-only embeddings)",
    )


@tool
def run_esm2(sequence: str, provider: ESM2Provider = "Local (HuggingFace)") -> str:
    """Compute real ESM-2 protein embeddings. Runs locally — no API key needed.

    This delegates to the local ESM-2 implementation and returns actual embedding
    numbers computed from your sequence.

    Args:
        sequence: Protein sequence in single-letter amino acid code.
        provider: Execution backend. 'Local (HuggingFace)' runs ESM-2 on this machine
            and is the default; hosted providers require their API key.

    Returns:
        JSON with status, model, device, embedding_dim and the mean-pooled embedding
        vector. Returns status='error' on failure — never a fake success string.
    """
    from bioagents.tools.esm_tools import run_esm_embedding

    if provider != "Local (HuggingFace)":
        key = get_provider_key_or_ask(provider, "ESM-2")
        if "[ENGAGEMENT_PENDING]" in key:
            return key
    return str(run_esm_embedding.invoke({"sequence": sequence, "model": "esm2"}))
