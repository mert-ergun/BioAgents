"""Genomics tools for sequence analysis and BLAST searches."""

import contextlib
import hashlib
import json
import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as ET  # nosec B405

import requests
from langchain_core.tools import tool

from bioagents.limits import (
    BLAST_BACKOFF_BASE_SEC,
    BLAST_BACKOFF_MAX_SEC,
    BLAST_POLL_INTERVAL_SEC,
    BLAST_TIMEOUT_SEC,
)
from bioagents.sandbox.time_budget import active_cell_timeout_sec, in_sandbox_cell, tool_cell_budget

logger = logging.getLogger(__name__)

HTTP_TIMEOUT = 30
BLAST_API = "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi"

CODON_TABLE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

COMPLEMENT = {
    "A": "T",
    "T": "A",
    "G": "C",
    "C": "G",
    "a": "t",
    "t": "a",
    "g": "c",
    "c": "g",
    "N": "N",
    "n": "n",
}


# NCBI BLAST URL API: one submit per 10 s, poll SearchInfo <=1/min per RID, send tool=/email=.
# https://blast.ncbi.nlm.nih.gov/doc/blast-help/developerinfo.html

BLAST_TOOL_ID = "bioagents"
BLAST_MIN_SUBMIT_INTERVAL_SEC = 10.0
BLAST_RID_TTL_SEC = 20 * 3600.0  # NCBI keeps results ~24 h
BLAST_WORKFLOW_MARGIN_SEC = 15.0
BLAST_MAX_HITS = 50
_RATE_LIMIT_MARKERS = ("cpu usage limit", "rate limit", "too many requests", "exceeded the")
_RID_RE = re.compile(r"^\s*RID = (\S+)", re.MULTILINE)
_RTOE_RE = re.compile(r"^\s*RTOE = (\d+)", re.MULTILINE)
_STATUS_RE = re.compile(r"Status=(\w+)")

# Indirection points so tests can drive a fake clock.
_now = time.monotonic
_sleep = time.sleep

_rid_cache: dict[str, tuple[str, float]] = {}
_rid_cache_lock = threading.Lock()
_submit_lock = threading.Lock()
_last_submit_at: list[float] = [float("-inf")]


class BlastRateLimitedError(Exception):
    """NCBI asked us to slow down (HTTP 429 or a CPU/rate-limit message)."""


class BlastTransientError(Exception):
    """Retryable server-side failure (HTTP 5xx, connection reset)."""


class BlastBudgetExhaustedError(Exception):
    """The time budget ran out while waiting to retry."""


def _ncbi_identity_params() -> dict[str, str]:
    params = {"tool": os.getenv("NCBI_TOOL", BLAST_TOOL_ID)}
    email = os.getenv("NCBI_EMAIL", "").strip()
    if email:
        params["email"] = email
    api_key = os.getenv("NCBI_API_KEY", "").strip()
    if api_key:
        params["api_key"] = api_key
    return params


def _is_rate_limited(status_code: int, text: str) -> bool:
    if status_code == 429:
        return True
    lowered = text[:4000].lower()
    return any(marker in lowered for marker in _RATE_LIMIT_MARKERS)


def _blast_http(method: str, params: dict, deadline: float) -> str:
    """One HTTP call to Blast.cgi, classified into rate-limit / transient / ok."""
    remaining = deadline - _now()
    if remaining <= 0:
        raise BlastBudgetExhaustedError("time budget exhausted before HTTP request")
    timeout = max(5.0, min(float(HTTP_TIMEOUT) * 2, remaining))
    full = {**params, **_ncbi_identity_params()}
    try:
        if method == "POST":
            resp = requests.post(BLAST_API, data=full, timeout=timeout)
        else:
            resp = requests.get(BLAST_API, params=full, timeout=timeout)
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise BlastTransientError(f"network error: {exc}") from exc
    text = resp.text or ""
    if _is_rate_limited(resp.status_code, text):
        raise BlastRateLimitedError(f"HTTP {resp.status_code}: NCBI rate limit")
    if resp.status_code >= 500:
        raise BlastTransientError(f"HTTP {resp.status_code} from NCBI")
    resp.raise_for_status()
    return text


def _with_backoff(method: str, params: dict, deadline: float, state: dict) -> str:
    """Retry rate-limited/transient calls with exponential backoff inside the budget."""
    delay = BLAST_BACKOFF_BASE_SEC
    while True:
        try:
            return _blast_http(method, params, deadline)
        except (BlastRateLimitedError, BlastTransientError) as exc:
            state["backoffs"] = state.get("backoffs", 0) + 1
            state["last_error"] = str(exc)
            if deadline - _now() <= delay:
                raise BlastBudgetExhaustedError(str(exc)) from exc
            logger.warning("BLAST %s: %s — backing off %.0fs", params.get("CMD"), exc, delay)
            _sleep(delay)
            delay = min(delay * 2, BLAST_BACKOFF_MAX_SEC)


def _blast_cache_key(program: str, database: str, query: str, opts: dict) -> str:
    raw = json.dumps([program, database, query, sorted(opts.items())], default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _cached_rid(key: str) -> str | None:
    with _rid_cache_lock:
        entry = _rid_cache.get(key)
        if entry and _now() - entry[1] < BLAST_RID_TTL_SEC:
            return entry[0]
        _rid_cache.pop(key, None)
        return None


def _remember_rid(key: str, rid: str) -> None:
    with _rid_cache_lock:
        _rid_cache[key] = (rid, _now())


def _forget_rid(key: str) -> None:
    with _rid_cache_lock:
        _rid_cache.pop(key, None)


def _submit_blast(put_params: dict, deadline: float, state: dict) -> tuple[str, int]:
    """CMD=Put once; returns (RID, RTOE seconds). Honors the 10 s submit spacing."""
    with _submit_lock:
        wait = _last_submit_at[0] + BLAST_MIN_SUBMIT_INTERVAL_SEC - _now()
        if wait > 0:
            _sleep(wait)
        text = _with_backoff("POST", put_params, deadline, state)
        _last_submit_at[0] = _now()
    rid_match = _RID_RE.search(text)
    if not rid_match:
        snippet = re.sub(r"\s+", " ", text)[:300]
        raise ValueError(f"NCBI returned no RID. Response: {snippet}")
    rtoe_match = _RTOE_RE.search(text)
    return rid_match.group(1), int(rtoe_match.group(1)) if rtoe_match else 0


def _poll_blast(rid: str, first_wait: float, deadline: float, state: dict) -> str:
    """Poll SearchInfo until READY/FAILED/UNKNOWN or the budget can't fit another poll.

    Returns 'READY', 'NO_HITS', 'FAILED', 'UNKNOWN' or 'TIMEOUT'.
    """
    wait = max(first_wait, BLAST_POLL_INTERVAL_SEC)
    while True:
        # >= not >: landing exactly on the deadline leaves no budget for the poll itself,
        # which would surface as a confusing "rate limited" error instead of a timeout.
        if _now() + wait >= deadline:
            return "TIMEOUT"
        _sleep(wait)
        text = _with_backoff(
            "GET", {"CMD": "Get", "RID": rid, "FORMAT_OBJECT": "SearchInfo"}, deadline, state
        )
        state["polls"] = state.get("polls", 0) + 1
        match = _STATUS_RE.search(text)
        status = match.group(1).upper() if match else "WAITING"
        if status == "READY":
            return "NO_HITS" if "ThereAreHits=no" in text else "READY"
        if status in ("FAILED", "UNKNOWN"):
            return status
        wait = BLAST_POLL_INTERVAL_SEC


def _parse_blast_xml(xml_text: str, max_hits: int) -> list[dict]:
    root = ET.fromstring(xml_text)  # nosec B314
    results = []
    for hit in root.findall(".//Hit")[:max_hits]:
        hsp = hit.find(".//Hsp")

        def _hsp(tag: str, _hsp_el=hsp) -> str:
            el = _hsp_el.find(tag) if _hsp_el is not None else None
            return (el.text or "") if el is not None else ""

        def _hit(tag: str, _hit_el=hit) -> str:
            el = _hit_el.find(tag)
            return (el.text or "") if el is not None else ""

        pct_identity = ""
        with contextlib.suppress(ValueError, ZeroDivisionError):
            pct_identity = f"{int(_hsp('Hsp_identity')) / int(_hsp('Hsp_align-len')) * 100:.1f}%"
        results.append(
            {
                "description": _hit("Hit_def"),
                "accession": _hit("Hit_accession"),
                "length": _hit("Hit_len"),
                "evalue": _hsp("Hsp_evalue"),
                "bit_score": _hsp("Hsp_bit-score"),
                "identity": pct_identity,
            }
        )
    return results


def _in_sandbox_cell() -> bool:
    """True when this code is running inside the coder sandbox's Jupyter kernel.

    The sandbox kills a cell at its deadline and restarts the kernel, so a BLAST still
    polling at that point dies with its RID unreported — the task3 failure. Detect the
    kernel so the budget can be capped below the cell limit and the RID always makes it
    back to the agent. See ``bioagents.sandbox.time_budget.in_sandbox_cell``.
    """
    return in_sandbox_cell()


def _blast_budget(timeout_sec: float) -> float:
    budget = BLAST_TIMEOUT_SEC if timeout_sec <= 0 else min(timeout_sec, BLAST_TIMEOUT_SEC)
    if _in_sandbox_cell():
        # BLAST_CELL_BUDGET_SEC, clamped strictly below the active cell deadline.
        budget = min(budget, tool_cell_budget("blast"))
    try:
        from bioagents.llms import timeout_llm

        wf_deadline = timeout_llm._workflow_deadline
    except Exception:
        wf_deadline = None
    if wf_deadline is not None:
        budget = min(budget, wf_deadline - time.monotonic() - BLAST_WORKFLOW_MARGIN_SEC)
    return float(max(0.0, budget))


def _timeout_hint(rid: str, elapsed: float, in_cell: bool) -> str:
    """Honest, actionable timeout text. The job is alive at NCBI; resume, never resubmit."""
    hint = (
        f"BLAST job {rid} is still running at NCBI after {elapsed:.0f}s (time budget "
        f"exhausted). No results yet — do NOT report partial or substitute homologs, and "
        f"do NOT resubmit. Call run_blast_search again with rid='{rid}' to resume it."
    )
    if in_cell:
        hint += (
            f" This ran inside the coder sandbox, whose cells are killed at "
            f"{active_cell_timeout_sec():.0f}s, so the budget was capped to keep the RID "
            "recoverable. Do not script BLAST here: hand this RID to the genomics agent's "
            "run_blast_search, which has the full budget."
        )
    return hint


def blast_search(
    sequence: str = "",
    database: str = "nr",
    program: str = "blastp",
    max_hits: int = 10,
    expect: float = 10.0,
    entrez_query: str = "",
    rid: str = "",
    timeout_sec: float = 0.0,
) -> dict:
    """Run (or resume) a remote NCBI BLAST search. Always returns a structured dict.

    ``status`` is one of: success, no_hits, timeout, failed, error. Every outcome
    that has an RID carries it, so a timeout can be resumed with ``rid=...``.
    """
    started = _now()
    in_cell = _in_sandbox_cell()
    deadline = started + _blast_budget(timeout_sec)
    max_hits = min(max(int(max_hits), 1), BLAST_MAX_HITS)
    body_lines = [ln for ln in (sequence or "").splitlines() if not ln.lstrip().startswith(">")]
    query = re.sub(r"\s+", "", "".join(body_lines))
    opts = {"hits": max_hits, "expect": expect, "entrez": entrez_query.strip()}
    key = _blast_cache_key(program, database, query, opts)
    state: dict = {}
    current_rid = rid.strip() or None
    reused = bool(current_rid)

    def outcome(status: str, **extra) -> dict:
        body = {
            "status": status,
            "rid": current_rid,
            "program": program,
            "database": database,
            "elapsed_sec": round(_now() - started, 1),
            "budget_sec": round(deadline - started, 1),
            "polls": state.get("polls", 0),
            "rate_limit_backoffs": state.get("backoffs", 0),
            "reused_rid": reused,
            "cell_bounded": in_cell,
        }
        body.update(extra)
        return body

    if not current_rid and len(query) < 10:
        return outcome("error", error="A query sequence of at least 10 residues is required.")
    if not current_rid:
        current_rid = _cached_rid(key)
        reused = bool(current_rid)

    try:
        rtoe = 0
        if not current_rid:
            put = {
                "CMD": "Put",
                "PROGRAM": program,
                "DATABASE": database,
                "QUERY": query,
                "HITLIST_SIZE": max_hits,
                "EXPECT": expect,
                "FORMAT_TYPE": "XML",
            }
            if opts["entrez"]:
                put["ENTREZ_QUERY"] = opts["entrez"]
            current_rid, rtoe = _submit_blast(put, deadline, state)
            if query:
                _remember_rid(key, current_rid)

        status = _poll_blast(current_rid, float(rtoe), deadline, state)
        if status == "TIMEOUT":
            return outcome("timeout", error=_timeout_hint(current_rid, _now() - started, in_cell))
        if status in ("FAILED", "UNKNOWN"):
            _forget_rid(key)
            reason = "failed on the NCBI server" if status == "FAILED" else "RID unknown/expired"
            return outcome("failed", error=f"BLAST job {current_rid} {reason}.")
        if status == "NO_HITS":
            return outcome("no_hits", hits=[])

        xml_text = _with_backoff(
            "GET",
            {"CMD": "Get", "RID": current_rid, "FORMAT_TYPE": "XML", "HITLIST_SIZE": max_hits},
            deadline,
            state,
        )
        hits = _parse_blast_xml(xml_text, max_hits)
        return outcome("success" if hits else "no_hits", hits=hits, hit_count=len(hits))
    except BlastBudgetExhaustedError as exc:
        if not current_rid:
            return outcome(
                "timeout",
                error=(
                    f"BLAST time budget exhausted before any job was submitted ({exc}). "
                    "Nothing ran — report this as not attempted and retry later."
                ),
            )
        stalled = (
            f" NCBI was rate limiting or unavailable ({exc})." if state.get("backoffs") else ""
        )
        return outcome(
            "timeout", error=_timeout_hint(current_rid, _now() - started, in_cell) + stalled
        )
    except Exception as exc:  # parsing / HTTP 4xx / unexpected
        logger.warning("BLAST search error (rid=%s): %s", current_rid, exc)
        return outcome("error", error=f"Error running BLAST search: {exc}")


def blast_protein_search_compat(**arguments) -> dict:
    """ToolUniverse ``BLAST_protein_search``-compatible adapter over :func:`blast_search`.

    Accepts the TU argument names (sequence, database, expect, hitlist_size, blast_type)
    so the TU tool name can be routed to this bounded implementation instead of
    Biopython's unbounded ``NCBIWWW.qblast``.

    Called from sandbox code the budget is capped below the cell kill timer, so the
    caller always gets a status + RID back instead of losing the job to a kernel restart.
    """
    return blast_search(
        sequence=arguments.get("sequence", ""),
        database=arguments.get("database", "nr"),
        program=arguments.get("blast_type", "blastp"),
        max_hits=arguments.get("hitlist_size", BLAST_MAX_HITS),
        expect=arguments.get("expect", 10.0),
        entrez_query=arguments.get("entrez_query", ""),
        rid=arguments.get("rid", ""),
        timeout_sec=arguments.get("timeout_sec", 0.0),
    )


@tool
def run_blast_search(
    sequence: str = "",
    database: str = "nr",
    program: str = "blastp",
    max_hits: int = 10,
    expect: float = 10.0,
    entrez_query: str = "",
    rid: str = "",
    timeout_sec: float = 0.0,
) -> str:
    """Run or resume an NCBI BLAST search (remote URL API) with a bounded time budget.

    Remote BLAST is slow (minutes; nr can take 10-20 min). This tool submits ONE job,
    polls it once per minute, and returns within the budget (default 1200 s, env
    BIOAGENTS_BLAST_TIMEOUT_SEC). On timeout it returns status 'timeout' with the RID —
    call again with rid=<RID> to resume; never resubmit the same sequence and never
    launch parallel BLAST jobs or fall back to Biopython qblast.

    A 'timeout' is NOT a result: report it as unfinished with its RID. Never substitute
    UniProt/keyword lookups, stubs or remembered homologs for BLAST hits and present them
    as if BLAST had run.

    Faster searches: database='swissprot' or 'refseq_protein' instead of 'nr', a small
    max_hits, and an entrez_query such as 'Viruses[Organism]' or 'txid10244[ORGN]'.

    Args:
        sequence: Query sequence (protein or nucleotide). Optional when rid is given.
        database: 'nr', 'nt', 'swissprot', 'pdb', 'refseq_protein', ...
        program: 'blastp', 'blastn', 'blastx', 'tblastn', 'tblastx'.
        max_hits: Maximum hits to return (1-50).
        expect: E-value threshold.
        entrez_query: Optional Entrez restriction (e.g. organism/taxid) to cut runtime.
        rid: Existing NCBI Request ID to resume instead of submitting.
        timeout_sec: Optional smaller budget in seconds (0 = configured default).

    Returns:
        JSON with status (success|no_hits|timeout|failed|error), rid, hits, timing, and
        cell_bounded (true when a sandbox cell limit shortened the budget).
    """
    return json.dumps(
        blast_search(
            sequence=sequence,
            database=database,
            program=program,
            max_hits=max_hits,
            expect=expect,
            entrez_query=entrez_query,
            rid=rid,
            timeout_sec=timeout_sec,
        ),
        indent=2,
    )


@tool
def parse_fasta_file(file_path: str) -> str:
    """Parse a FASTA file and return sequence information.

    Args:
        file_path: Path to the FASTA file (in sandbox or absolute).

    Returns:
        JSON string with sequence headers, lengths, and first 100 characters.
    """
    try:
        from bioagents.sandbox.sandbox_manager import get_sandbox

        sandbox = get_sandbox()
        content = sandbox.read_file(file_path)

        sequences = []
        current_header = ""
        current_seq: list[str] = []

        for line in content.strip().split("\n"):
            line = line.strip()
            if line.startswith(">"):
                if current_header:
                    seq = "".join(current_seq)
                    sequences.append(
                        {
                            "header": current_header,
                            "length": len(seq),
                            "preview": seq[:100],
                        }
                    )
                current_header = line[1:].strip()
                current_seq = []
            elif line:
                current_seq.append(line)

        if current_header:
            seq = "".join(current_seq)
            sequences.append(
                {
                    "header": current_header,
                    "length": len(seq),
                    "preview": seq[:100],
                }
            )

        if not sequences:
            return "No sequences found in the file. Ensure it is in FASTA format."
        return json.dumps(
            {
                "num_sequences": len(sequences),
                "sequences": sequences[:100],
            },
            indent=2,
        )
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found."
    except Exception as e:
        return f"Error parsing FASTA file: {e}"


@tool
def reverse_complement(sequence: str) -> str:
    """Compute the reverse complement of a DNA sequence.

    Args:
        sequence: DNA sequence (A, T, G, C characters). Case insensitive.

    Returns:
        The reverse complement sequence, or an error message.
    """
    try:
        seq = sequence.strip().replace(" ", "").replace("\n", "")
        invalid = set(seq.upper()) - {"A", "T", "G", "C", "N"}
        if invalid:
            return (
                f"Error: Invalid DNA characters found: {invalid}. Only A, T, G, C, N are allowed."
            )
        rc = "".join(COMPLEMENT.get(base, "N") for base in reversed(seq))
        return rc
    except Exception as e:
        return f"Error computing reverse complement: {e}"


@tool
def translate_dna(dna_sequence: str) -> str:
    """Translate a DNA sequence into a protein sequence using the standard codon table.

    The sequence is translated in reading frame 1 (from the first nucleotide).

    Args:
        dna_sequence: DNA sequence to translate. Must be A/T/G/C characters.

    Returns:
        The translated protein sequence (single-letter amino acids). Stop codons
        are shown as '*'.
    """
    try:
        seq = dna_sequence.strip().replace(" ", "").replace("\n", "").upper()
        invalid = set(seq) - {"A", "T", "G", "C", "N"}
        if invalid:
            return f"Error: Invalid DNA characters: {invalid}"

        protein: list[str] = []
        for i in range(0, len(seq) - 2, 3):
            codon = seq[i : i + 3]
            if "N" in codon:
                protein.append("X")
            else:
                protein.append(CODON_TABLE.get(codon, "X"))

        return "".join(protein)
    except Exception as e:
        return f"Error translating DNA: {e}"


@tool
def calculate_gc_content(sequence: str) -> str:
    """Calculate the GC content of a DNA or RNA sequence.

    Args:
        sequence: DNA or RNA sequence (A, T/U, G, C characters).

    Returns:
        GC content as a percentage with sequence statistics.
    """
    try:
        seq = sequence.strip().replace(" ", "").replace("\n", "").upper()
        seq = seq.replace("U", "T")
        valid = {"A", "T", "G", "C", "N"}
        invalid = set(seq) - valid
        if invalid:
            return f"Error: Invalid characters found: {invalid}"

        if not seq:
            return "Error: Empty sequence provided."

        gc_count = seq.count("G") + seq.count("C")
        total = len(seq)
        gc_pct = (gc_count / total) * 100

        return json.dumps(
            {
                "gc_content": f"{gc_pct:.2f}%",
                "gc_count": gc_count,
                "at_count": seq.count("A") + seq.count("T"),
                "n_count": seq.count("N"),
                "total_length": total,
            },
            indent=2,
        )
    except Exception as e:
        return f"Error calculating GC content: {e}"


def get_genomics_tools() -> list:
    """Return all genomics tools."""
    return [
        run_blast_search,
        parse_fasta_file,
        reverse_complement,
        translate_dna,
        calculate_gc_content,
    ]
