"""
uniprot_search_proteins - Search UniProtKB via the UniProt REST API.

Category: database

Queries https://rest.uniprot.org/uniprotkb/search using only the Python
standard library (urllib). Supports arbitrary UniProt query syntax
(e.g. 'taxonomy_id:10244 AND reviewed:true'), field selection, TSV/JSON
output formats, optional saving to file, and retries with exponential
backoff on transient network errors.
"""

import json
import time
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Any

from bioagents.tools.http_client import read_https_text

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
DEFAULT_FIELDS = "accession,id,protein_name,gene_names,organism_name,length,sequence"
DEFAULT_FORMAT = "tsv"
MAX_PAGE_SIZE = 500
DEFAULT_MAX_RESULTS = 100
DEFAULT_TIMEOUT = 60
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 2.0


def uniprot_search_proteins(
    query: str,
    output_file: str | None = None,
    fields: str | None = None,
    max_results: int | None = None,
    format: str = DEFAULT_FORMAT,
) -> dict[str, Any]:
    """Search UniProtKB for protein entries via the UniProt REST API.

    Args:
        query: UniProt search query, e.g. 'taxonomy_id:10244 AND reviewed:true'.
        output_file: optional path to save results (TSV or JSON).
        fields: comma-separated UniProt REST fields to retrieve.
        max_results: maximum number of entries (UniProt page-size cap 500).
        format: 'tsv' or 'json'.

    Returns:
        dict with status, query, format, n_results, results (list of dicts)
        and output_file when saved.

    Raises:
        ValueError: empty query or invalid format.
        RuntimeError: persistent network failure.

    Example:
        uniprot_search_proteins('taxonomy_id:10244 AND reviewed:true',
                                max_results=10)
    """
    if not query or not str(query).strip():
        raise ValueError("query must be a non-empty UniProt search string")
    fmt = (format or DEFAULT_FORMAT).lower()
    if fmt not in ("tsv", "json"):
        raise ValueError("format must be tsv or json")
    n = int(max_results) if max_results else DEFAULT_MAX_RESULTS
    n = max(1, min(n, MAX_PAGE_SIZE))
    flds = fields or DEFAULT_FIELDS

    params = {"query": query, "format": fmt, "fields": flds, "size": str(n)}
    url = UNIPROT_SEARCH_URL + "?" + urllib.parse.urlencode(params)

    last_err = None
    raw = None
    for attempt in range(1, DEFAULT_RETRIES + 1):
        try:
            raw = read_https_text(
                url,
                allowed_hosts={"rest.uniprot.org"},
                headers={"User-Agent": "BioAgents/1.0"},
                timeout=DEFAULT_TIMEOUT,
            )
            break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = exc
            if attempt < DEFAULT_RETRIES:
                time.sleep(DEFAULT_BACKOFF**attempt)
    if raw is None:
        raise RuntimeError(f"UniProt search failed after {DEFAULT_RETRIES:d} attempts: {last_err}")

    results: list[dict[str, Any]] = []
    if fmt == "tsv":
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        if lines:
            header = lines[0].split("\t")
            for ln in lines[1:]:
                parts = ln.split("\t")
                row = {}
                for i, col in enumerate(header):
                    row[col] = parts[i] if i < len(parts) else ""
                results.append(row)
    else:
        data = json.loads(raw)
        if isinstance(data, dict):
            results = data.get("results", [])
        elif isinstance(data, list):
            results = data

    out_path = None
    if output_file:
        out_path = output_file
        parent = str(Path(out_path).absolute().parent)
        if parent:
            Path(parent).mkdir(parents=True, exist_ok=True)
        with Path(out_path).open("w", encoding="utf-8") as fh:
            if fmt == "tsv":
                fh.write(raw)
            else:
                json.dump(results, fh, indent=2)

    return {
        "status": "success",
        "query": query,
        "format": fmt,
        "n_results": len(results),
        "results": results,
        "output_file": out_path,
    }
