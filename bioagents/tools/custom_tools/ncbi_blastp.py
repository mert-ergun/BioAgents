"""
ncbi_blastp - Run a protein BLAST (blastp) search via the NCBI BLAST URL API.

Category: genomics

Submits a protein sequence to https://blast.ncbi.nlm.nih.gov/Blast.cgi
(CMD=Put), polls until the search is READY, retrieves XML results, parses
hits, and aggregates the best hit per organism (virus species) with percent
identity, query coverage, bit score and e-value. Uses only the Python
urllib for HTTP and defusedxml for parsing remote XML safely.
"""

import contextlib
import json
import time
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Any

from defusedxml import ElementTree as ET
from defusedxml.common import DefusedXmlException

from bioagents.tools.http_client import read_https_text

BLAST_URL = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
DEFAULT_DATABASE = "nr"
DEFAULT_EXPECT = "1e-5"
DEFAULT_HITLIST_SIZE = 500
DEFAULT_POLL_INTERVAL = 10
DEFAULT_MAX_WAIT = 900
DEFAULT_TIMEOUT = 60
VALID_DATABASES = ("nr", "swissprot", "refseq_protein", "pdb")
VALID_AA = set("ACDEFGHIKLMNPQRSTVWYUBZXJO-*")


def _http_post(url: str, data: dict[str, str], timeout: int = DEFAULT_TIMEOUT) -> str:
    return read_https_text(
        url,
        allowed_hosts={"blast.ncbi.nlm.nih.gov"},
        data=urllib.parse.urlencode(data).encode("utf-8"),
        headers={"User-Agent": "BioAgents/1.0"},
        timeout=timeout,
    )


def _http_get(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    return read_https_text(
        url,
        allowed_hosts={"blast.ncbi.nlm.nih.gov"},
        headers={"User-Agent": "BioAgents/1.0"},
        timeout=timeout,
    )


def _parse_blast_xml(xml_text: str, rid: str):
    try:
        # NCBI can declare its public DTD; never resolve external references or entities.
        return ET.fromstring(xml_text, forbid_entities=True, forbid_external=True)
    except (ET.ParseError, DefusedXmlException) as exc:
        raise RuntimeError(f"Failed to parse NCBI BLAST XML for job {rid}: {exc}") from exc


def ncbi_blastp(
    sequence: str,
    target_name: str | None = None,
    database: str = DEFAULT_DATABASE,
    entrez_query: str | None = None,
    hitlist_size: int = DEFAULT_HITLIST_SIZE,
    expect: str = DEFAULT_EXPECT,
    output_file: str | None = None,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    max_wait: int = DEFAULT_MAX_WAIT,
) -> dict[str, Any]:
    """Run blastp against an NCBI BLAST database and parse the results.

    Args:
        sequence: protein sequence (raw string, whitespace tolerated).
        target_name: free-text label embedded in the output.
        database: nr, swissprot, refseq_protein or pdb.
        entrez_query: Entrez filter, e.g. 'viruses[orgn]'.
        hitlist_size: maximum number of aligned hits to return.
        expect: e-value threshold.
        output_file: optional path to save the full JSON result.
        poll_interval: seconds between status polls (NCBI minimum ~10 s).
        max_wait: maximum seconds to wait for job completion.

    Returns:
        dict with status, rid, database, entrez_query, query_length,
        target_name, n_hits, n_organisms, organisms (best hit per organism),
        hits (all parsed hits) and output_file when saved.

    Raises:
        ValueError: invalid sequence or database.
        RuntimeError: submission, polling or parsing failure.

    Example:
        ncbi_blastp(seq, target_name='F13L', database='nr',
                    entrez_query='viruses[orgn]', output_file='f13l.json')
    """
    seq = "".join(str(sequence).split()).upper()
    if not seq or not seq.replace("-", ""):
        raise ValueError("sequence must be a non-empty protein sequence")
    bad = [c for c in seq if c not in VALID_AA]
    if bad:
        raise ValueError(f"sequence contains non-amino-acid characters: {sorted(set(bad))}")
    db = (database or DEFAULT_DATABASE).lower()
    if db not in VALID_DATABASES:
        raise ValueError(f"database must be one of {VALID_DATABASES}")

    params: dict[str, str] = {
        "CMD": "Put",
        "PROGRAM": "blastp",
        "DATABASE": db,
        "QUERY": seq,
        "HITLIST_SIZE": str(int(hitlist_size)),
        "EXPECT": str(expect),
    }
    if entrez_query:
        params["ENTREZ_QUERY"] = entrez_query

    put_resp = _http_post(BLAST_URL, params)
    rid = None
    for line in put_resp.splitlines():
        if line.startswith("    RID = "):
            rid = line.split("=", 1)[1].strip()
            break
    if not rid:
        raise RuntimeError("Could not obtain RID from NCBI BLAST submission")

    interval = max(10, int(poll_interval))
    waited = 0
    status = "UNKNOWN"
    while waited < max_wait:
        time.sleep(interval)
        waited += interval
        info = _http_get(
            BLAST_URL
            + "?"
            + urllib.parse.urlencode({"CMD": "Get", "FORMAT_OBJECT": "SearchInfo", "RID": rid})
        )
        for line in info.splitlines():
            if line.startswith("      Status="):
                status = line.split("=", 1)[1].strip()
                break
        if status == "READY":
            break
        if status in ("FAILED", "UNKNOWN"):
            raise RuntimeError(f"NCBI BLAST job {rid} status: {status}")
    if status != "READY":
        raise RuntimeError(
            f"NCBI BLAST job {rid} not READY after {max_wait}s (last status {status})"
        )

    xml_text = _http_get(
        BLAST_URL
        + "?"
        + urllib.parse.urlencode(
            {
                "CMD": "Get",
                "FORMAT_TYPE": "XML",
                "RID": rid,
                "ALIGNMENTS": str(int(hitlist_size)),
                "DESCRIPTIONS": str(int(hitlist_size)),
            }
        )
    )

    root = _parse_blast_xml(xml_text, rid)

    hits: list[dict[str, Any]] = []
    query_len = 0
    for iteration in root.iter("Iteration"):
        qlen_el = iteration.find("Iteration_query-len")
        if qlen_el is not None and qlen_el.text:
            with contextlib.suppress(ValueError):
                query_len = int(qlen_el.text)
        for hit in iteration.iter("Hit"):
            hit_def = (hit.findtext("Hit_def") or "").strip()
            hit_id = (hit.findtext("Hit_id") or "").strip()
            hit_acc = (hit.findtext("Hit_accession") or "").strip()
            organism = ""
            if hit_def.endswith("]") and "[" in hit_def:
                organism = hit_def[hit_def.rindex("[") + 1 : -1].strip()
            best = None
            for hsp in hit.iter("Hsp"):
                identities = int(hsp.findtext("Hsp_identity") or 0)
                positives = int(hsp.findtext("Hsp_positives") or 0)
                align_len = int(hsp.findtext("Hsp_align-len") or 1)
                q_from = int(hsp.findtext("Hsp_query-from") or 1)
                q_to = int(hsp.findtext("Hsp_query-to") or 1)
                bits = float(hsp.findtext("Hsp_bit-score") or 0)
                ev = float(hsp.findtext("Hsp_evalue") or 0)
                score = {
                    "identity_pct": round(100.0 * identities / max(align_len, 1), 2),
                    "positives_pct": round(100.0 * positives / max(align_len, 1), 2),
                    "query_coverage_pct": round(
                        100.0 * (abs(q_to - q_from) + 1) / max(query_len, 1), 2
                    ),
                    "align_len": align_len,
                    "bit_score": bits,
                    "evalue": ev,
                }
                if best is None or score["bit_score"] > best["bit_score"]:
                    best = score
            if best is None:
                continue
            rec: dict[str, Any] = {
                "hit_id": hit_id,
                "hit_accession": hit_acc,
                "hit_def": hit_def,
                "organism": organism,
            }
            rec.update(best)
            hits.append(rec)

    organisms: dict[str, dict[str, Any]] = {}
    for h in hits:
        org = h.get("organism") or "unknown"
        cur = organisms.get(org)
        if cur is None or h["bit_score"] > cur["bit_score"]:
            organisms[org] = h

    result = {
        "status": "success",
        "rid": rid,
        "database": db,
        "entrez_query": entrez_query,
        "query_length": query_len,
        "target_name": target_name,
        "n_hits": len(hits),
        "n_organisms": len(organisms),
        "organisms": organisms,
        "hits": hits,
    }

    if output_file:
        parent = str(Path(str(Path(output_file).absolute())).parent)
        if parent:
            Path(parent).mkdir(parents=True, exist_ok=True)
        with Path(output_file).open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        result["output_file"] = output_file

    return result
