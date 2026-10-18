"""Proteomics tools for fetching protein data from UniProt."""

import json
from typing import Literal

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


def _with_fallback_note(raw: str, note: str) -> str:
    """Attach a model-substitution note to an ESM tool's JSON result."""
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return raw
    if isinstance(payload, dict):
        payload["fallback_note"] = note
        return json.dumps(payload)
    return raw


@tool
def run_esm3(sequence: str, provider: ESM3Provider = "EvolutionaryScale Forge") -> str:
    """Protein embedding for an ESM-3 request; runs public ESM-2 locally as the fallback.

    ESM-3 is gated (EvolutionaryScale licence / Forge API) and no client for it is
    wired up in this deployment. Instead of failing, this computes a real embedding
    with the public ESM-2 650M model (no API key or HF token needed) and says so in
    'fallback_note'. The numbers are ESM-2 numbers, never ESM-3 numbers.

    Args:
        sequence: Protein sequence in single-letter amino acid code.
        provider: Hosted provider that was requested; recorded in the note only.

    Returns:
        JSON with status, model, device, embedding_dim, mean_embedding and a
        'fallback_note' naming the substitution, or status='error' explaining exactly
        what is missing. Never reports success for a computation that did not run.
    """
    from bioagents.tools.esm_tools import run_esm_embedding

    note = (
        f"ESM-3 via '{provider}' is gated and no client for it is available here, so "
        "NO ESM-3 computation was performed. Fell back to the public "
        "facebook/esm2_t33_650M_UR50D model run locally (no token required); the "
        "embedding below is an ESM-2 embedding."
    )
    raw = str(run_esm_embedding.invoke({"sequence": sequence, "model": "esm2"}))
    return _with_fallback_note(raw, note)


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
    """Compute real ESM-2 protein embeddings. Runs locally — no API key or HF token needed.

    ESM-2 is a public model, so every provider value runs the same local ESM-2 650M
    computation; a hosted provider name is only recorded in a note (no hosted client
    is wired up, and none is needed).

    Args:
        sequence: Protein sequence in single-letter amino acid code.
        provider: Requested backend. 'Local (HuggingFace)' is the default; other values
            still run locally and add a 'fallback_note' saying so.

    Returns:
        JSON with status, model, device, embedding_dim and the mean-pooled embedding
        vector. Returns status='error' on failure — never a fake success string.
    """
    from bioagents.tools.esm_tools import run_esm_embedding

    raw = str(run_esm_embedding.invoke({"sequence": sequence, "model": "esm2"}))
    if provider != "Local (HuggingFace)":
        return _with_fallback_note(
            raw,
            f"Provider '{provider}' has no client here; ESM-2 is public, so it was run "
            "locally instead (no API key or HF token required).",
        )
    return raw
