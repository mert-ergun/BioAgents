"""
mutation_literature_comparison - Validate computed point mutations against
UniProt variant/mutagenesis annotations and PubMed literature.

Category: database

For each input mutation {position, wild_type, mutant, fitness_ratio, llr}:
 1. Fetch UniProtKB features (Natural variant, Mutagenesis) for the accession.
 2. Match feature records with the same start position, matching wild-type
    residue, and the queried mutant residue in alternatives.
 3. Optionally search PubMed via NCBI E-utilities (esearch + esummary) using
    "<gene> AND <WT><pos><mut>" notation to find literature reports.
 4. Write a comparison table (JSON + CSV) to output_dir and return it.

Stdlib-only implementation (urllib / csv / json); retries with backoff.
"""

import csv
import json
import logging
import time
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Any

from bioagents.tools.http_client import read_https_text

logger = logging.getLogger(__name__)

_UNIPROT_API = "https://rest.uniprot.org/uniprotkb/{acc}.json"
_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
_FEATURE_TYPES = ("Natural variant", "Mutagenesis")


def _fetch_json(
    url: str, params: dict[str, Any] | None = None, timeout: int = 60, retries: int = 3
) -> dict[str, Any]:
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    last_exc: Any = None
    for attempt in range(retries):
        try:
            text = read_https_text(
                url,
                allowed_hosts={"rest.uniprot.org", "eutils.ncbi.nlm.nih.gov"},
                headers={"Accept": "application/json"},
                timeout=timeout,
            )
            result = json.loads(text)
            if not isinstance(result, dict):
                raise ValueError("API JSON response must be an object")
            return result
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            OSError,
            ValueError,
        ) as exc:
            last_exc = exc
            logger.warning("HTTP attempt %d/%d failed: %s", attempt + 1, retries, exc)
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"HTTP request failed after {retries:d} attempts: {last_exc}")


def mutation_literature_comparison(
    mutations: list[dict[str, Any]],
    accession: str = "P24941",
    gene: str = "CDK2",
    output_dir: str = ".",
    include_pubmed: bool = True,
    max_pubmed_results: int = 5,
    timeout: int = 60,
    retries: int = 3,
) -> dict[str, Any]:
    """Check computed point mutations against UniProt variant/mutagenesis
    annotations and PubMed literature.

    Args:
        mutations: list of dicts with keys position (int), wild_type (1-letter),
            mutant (1-letter); optional keys fitness_ratio, llr.
        accession: UniProtKB accession to query (default P24941, human CDK2).
        gene: gene symbol used in PubMed queries (default CDK2).
        output_dir: directory for mutation_literature_comparison.json/.csv.
        include_pubmed: also run PubMed E-utilities searches (default True).
        max_pubmed_results: max PMIDs per mutation (default 5).
        timeout: HTTP timeout seconds (default 60).
        retries: retries on transient network errors (default 3).

    Returns:
        dict with status, accession, n_mutations, comparison rows
        (mutation string, fitness_ratio, llr, n_uniprot_records,
        uniprot_records with description and xrefs, pubmed_hits with PMIDs
        and titles), and output file paths.

    Raises:
        ValueError: malformed mutations input.
        RuntimeError: persistent network failure.

    Example:
        mutation_literature_comparison(
            [{"position": 33, "wild_type": "K", "mutant": "H",
              "fitness_ratio": 0.42, "llr": -0.87}],
            accession="P24941", gene="CDK2", output_dir="results")
    """
    if isinstance(mutations, str):
        mutations = json.loads(mutations)
    if not isinstance(mutations, list) or not mutations:
        raise ValueError("mutations must be a non-empty list of mutation dicts")
    norm: list[dict[str, Any]] = []
    for m in mutations:
        if not isinstance(m, dict):
            raise ValueError("each mutation must be a dict")
        for key in ("position", "wild_type", "mutant"):
            if key not in m:
                raise ValueError("each mutation needs 'position', 'wild_type', 'mutant'")
        norm.append(
            {
                "position": int(m["position"]),
                "wild_type": str(m["wild_type"]).upper()[:1],
                "mutant": str(m["mutant"]).upper()[:1],
                "fitness_ratio": m.get("fitness_ratio"),
                "llr": m.get("llr"),
            }
        )

    data = _fetch_json(_UNIPROT_API.format(acc=accession), timeout=timeout, retries=retries)
    feats = [f for f in data.get("features", []) if f.get("type") in _FEATURE_TYPES]

    rows: list[dict[str, Any]] = []
    for m in norm:
        mut_str = f"{m['wild_type']}{m['position']:d}{m['mutant']}"
        db_hits: list[dict[str, Any]] = []
        for f in feats:
            start = ((f.get("location") or {}).get("start") or {}).get("value")
            if start != m["position"]:
                continue
            wt = f.get("wildType")
            if wt and wt != m["wild_type"]:
                continue
            alts = f.get("alternatives") or [f.get("alternativeSequence")]
            if m["mutant"] not in alts:
                continue
            db_hits.append(
                {
                    "type": f.get("type"),
                    "description": f.get("description"),
                    "xrefs": [
                        "{}:{}".format(r.get("database"), r.get("id"))
                        for r in f.get("crossReferences", [])
                        if r.get("id")
                    ],
                }
            )
        pubmed: list[dict[str, Any]] = []
        if include_pubmed:
            term = f"{gene} AND {mut_str}"
            try:
                es = _fetch_json(
                    _ESEARCH,
                    {"db": "pubmed", "term": term, "retmode": "json", "retmax": max_pubmed_results},
                    timeout=timeout,
                    retries=retries,
                )
                pmids = (es.get("esearchresult") or {}).get("idlist", [])
                if pmids:
                    summ = _fetch_json(
                        _ESUMMARY,
                        {"db": "pubmed", "id": ",".join(pmids), "retmode": "json"},
                        timeout=timeout,
                        retries=retries,
                    )
                    for pmid in pmids:
                        rec = (summ.get("result") or {}).get(str(pmid), {})
                        pubmed.append(
                            {"pmid": pmid, "title": rec.get("title"), "pubdate": rec.get("pubdate")}
                        )
            except Exception as exc:  # record but do not fail the whole comparison
                logger.error("PubMed search failed for %s: %s", mut_str, exc)
                pubmed.append({"error": str(exc)})
        rows.append(
            {
                "mutation": mut_str,
                "position": m["position"],
                "wild_type": m["wild_type"],
                "mutant": m["mutant"],
                "fitness_ratio": m["fitness_ratio"],
                "llr": m["llr"],
                "n_uniprot_records": len(db_hits),
                "uniprot_records": db_hits,
                "n_pubmed_hits": len([p for p in pubmed if "pmid" in p]),
                "pubmed_hits": pubmed,
            }
        )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    json_path = str(Path(output_dir).joinpath("mutation_literature_comparison.json"))
    with Path(json_path).open("w") as fh:
        json.dump(
            {
                "status": "success",
                "accession": accession,
                "gene": gene,
                "n_mutations": len(rows),
                "comparison": rows,
            },
            fh,
            indent=2,
        )
    csv_path = str(Path(output_dir).joinpath("mutation_literature_comparison.csv"))
    with Path(csv_path).open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "mutation",
                "position",
                "wild_type",
                "mutant",
                "fitness_ratio",
                "llr",
                "n_uniprot_records",
                "uniprot_record_types",
                "uniprot_descriptions",
                "uniprot_xrefs",
                "n_pubmed_hits",
                "pubmed_pmids",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["mutation"],
                    r["position"],
                    r["wild_type"],
                    r["mutant"],
                    r["fitness_ratio"],
                    r["llr"],
                    r["n_uniprot_records"],
                    ";".join(h["type"] for h in r["uniprot_records"]),
                    ";".join(str(h.get("description")) for h in r["uniprot_records"]),
                    ";".join(x for h in r["uniprot_records"] for x in h["xrefs"]),
                    r["n_pubmed_hits"],
                    ";".join(p["pmid"] for p in r["pubmed_hits"] if "pmid" in p),
                ]
            )
    return {
        "status": "success",
        "accession": accession,
        "gene": gene,
        "n_mutations": len(rows),
        "comparison": rows,
        "output_json": json_path,
        "output_csv": csv_path,
    }
