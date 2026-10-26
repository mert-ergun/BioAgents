"""PubMed abstract search wrapper using NCBI E-utilities (stdlib only).

Searches PubMed for article PMIDs matching a query, fetches titles and
abstracts, and saves a structured JSON summary plus a plain-text abstracts
file for downstream text mining.
"""

import json
import re
import time
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Any

from bioagents.tools.http_client import read_https_text

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MAX_RETRIES = 3
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}


def _http_get_with_retry(url: str, params: dict[str, str]) -> str:
    """Perform a GET request with retries and exponential backoff.

    Retries up to MAX_RETRIES times on URLError or retryable HTTP errors
    (5xx and 429), sleeping with exponential backoff between attempts.

    Args:
        url: Base URL for the request.
        params: Query parameters to URL-encode and append.

    Returns:
        The response body decoded as UTF-8 text.

    Raises:
        RuntimeError: If all retry attempts fail.
    """
    full_url = url + "?" + urllib.parse.urlencode(params)
    last_error: Exception = RuntimeError("Request not attempted")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return read_https_text(
                full_url,
                allowed_hosts={"eutils.ncbi.nlm.nih.gov"},
                headers={"User-Agent": "pubmed_search_abstracts-wrapper/1.0"},
                timeout=60,
            )
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in RETRYABLE_HTTP_CODES:
                raise RuntimeError(
                    f"NCBI E-utilities returned HTTP {exc.code} for {url}: {exc.reason}"
                ) from exc
        except urllib.error.URLError as exc:
            last_error = exc

        if attempt < MAX_RETRIES:
            time.sleep(2**attempt)

    raise RuntimeError(f"Request failed after {MAX_RETRIES} attempts: {url} ({last_error})")


def pubmed_search_abstracts(
    query: str,
    max_results: int = 10,
    output_dir: str = ".",
    sort: str = "relevance",
) -> dict:
    """Search PubMed for abstracts matching a query.

    Performs an NCBI E-utilities esearch (db=pubmed) followed by an efetch
    of titles and abstracts. Saves two files into output_dir:
      - <query_slug>_results.json : structured summary {query, count, results}
        where each result has pmid, title and pub_year (parsed heuristically
        from the record header).
      - <query_slug>_abstracts.txt : full raw efetch text output.

    Args:
        query: PubMed search query, e.g. 'monkeypox virus protein targets'.
        max_results: Maximum number of PMIDs to fetch (1-200).
        output_dir: Directory for output files (created if missing).
        sort: esearch sort order: 'relevance', 'pub_date' or 'Author'.

    Returns:
        dict with keys: status ('success'), query, n_results,
        results (list of dicts with pmid/title/pub_year),
        abstracts_file, json_file.

    Raises:
        ValueError: If arguments are invalid.
        RuntimeError: On NCBI errors or if no articles match the query.

    Example:
        >>> res = pubmed_search_abstracts("monkeypox virus protein targets",
        ...                               max_results=5, output_dir="out")
        >>> res["status"]
        'success'
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("'query' must be a non-empty string")
    if not isinstance(max_results, int) or not (1 <= max_results <= 200):
        raise ValueError("'max_results' must be an integer in [1, 200]")
    if sort not in ("relevance", "pub_date", "Author"):
        raise ValueError("'sort' must be one of: relevance, pub_date, Author")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", query.strip())[:60]

    # --- Step 1: esearch -------------------------------------------------
    esearch = _http_get_with_retry(
        f"{EUTILS_BASE}/esearch.fcgi",
        {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": str(max_results),
            "sort": sort,
        },
    )
    esearch_json = json.loads(esearch)
    idlist = esearch_json.get("esearchresult", {}).get("idlist", [])
    if not idlist:
        raise RuntimeError(f"No PubMed articles found for query: '{query}'")

    time.sleep(0.4)  # respect NCBI 3 req/s limit

    # --- Step 2: efetch abstracts ----------------------------------------
    abstracts_text = _http_get_with_retry(
        f"{EUTILS_BASE}/efetch.fcgi",
        {
            "db": "pubmed",
            "id": ",".join(idlist),
            "rettype": "abstract",
            "retmode": "text",
        },
    )

    # --- Step 3: parse records -------------------------------------------
    results: list[dict[str, Any]] = []
    blocks = re.split(r"\n\s*\n", abstracts_text.strip())
    for block in blocks:
        pmid = None
        title = None
        pub_year = None
        lines = [line for line in block.strip().splitlines() if line.strip()]
        if not lines:
            continue
        for line in lines:
            stripped = line.strip()
            if re.fullmatch(r"\d{6,9}", stripped):
                pmid = stripped
                continue
            if pub_year is None:
                year_match = re.search(r"\b(19|20)\d{2}\b", line)
                if year_match and len(line) > 20:
                    pub_year = year_match.group(0)
            if title is None and pmid is not None and len(stripped) > 15:
                title = stripped
        if pmid:
            results.append({"pmid": pmid, "title": title, "pub_year": pub_year})

    json_file = str(Path(output_dir).joinpath(f"{slug}_results.json"))
    abstracts_file = str(Path(output_dir).joinpath(f"{slug}_abstracts.txt"))

    with Path(json_file).open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "query": query,
                " count": len(results),
                "results": results,
            },
            handle,
            indent=2,
        )
    with Path(abstracts_file).open("w", encoding="utf-8") as handle:
        handle.write(abstracts_text)

    return {
        "status": "success",
        "query": query,
        "n_results": len(results),
        "results": results,
        "abstracts_file": abstracts_file,
        "json_file": json_file,
    }
