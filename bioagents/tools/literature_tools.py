"""Literature search tools for PubMed, arXiv, and bioRxiv."""

import json
import xml.etree.ElementTree as ET  # nosec B405

import requests
from langchain_core.tools import tool

HTTP_TIMEOUT = 30
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


@tool
def search_pubmed(query: str, max_results: int = 10) -> str:
    """Search PubMed for biomedical literature using NCBI E-utilities.

    Args:
        query: Search query (supports PubMed search syntax).
        max_results: Maximum number of results to return (default 10, max 50).

    Returns:
        JSON string with article titles, authors, journal, year, PMID, and DOI.
    """
    try:
        max_results = min(max(max_results, 1), 50)

        search_url = f"{NCBI_BASE}/esearch.fcgi"
        search_params: dict[str, str | int] = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        search_resp = requests.get(search_url, params=search_params, timeout=HTTP_TIMEOUT)
        search_resp.raise_for_status()
        search_data = search_resp.json()

        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return f"No PubMed results found for: '{query}'"

        fetch_url = f"{NCBI_BASE}/efetch.fcgi"
        fetch_params: dict[str, str] = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml",
        }
        fetch_resp = requests.get(fetch_url, params=fetch_params, timeout=HTTP_TIMEOUT)
        fetch_resp.raise_for_status()

        root = ET.fromstring(fetch_resp.text)  # nosec B314
        articles = []
        for article_elem in root.findall(".//PubmedArticle"):
            medline = article_elem.find(".//MedlineCitation")
            if medline is None:
                continue
            pmid_elem = medline.find("PMID")
            pmid = pmid_elem.text if pmid_elem is not None else ""

            art = medline.find(".//Article")
            if art is None:
                continue
            title_elem = art.find("ArticleTitle")
            title = title_elem.text if title_elem is not None else ""

            authors = []
            for author in art.findall(".//Author"):
                last = author.find("LastName")
                fore = author.find("ForeName")
                if last is not None:
                    name = last.text
                    if fore is not None:
                        name = f"{fore.text} {name}"
                    authors.append(name)

            journal_elem = art.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else ""

            year_elem = art.find(".//PubDate/Year")
            year = year_elem.text if year_elem is not None else ""

            doi = ""
            for eid in article_elem.findall(".//ArticleId"):
                if eid.get("IdType") == "doi":
                    doi = eid.text or ""
                    break

            abstract_parts = art.findall(".//Abstract/AbstractText")
            abstract = " ".join(p.text or "" for p in abstract_parts)[:500]

            articles.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "authors": authors[:5],
                    "journal": journal,
                    "year": year,
                    "doi": doi,
                    "abstract": abstract,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                }
            )

        return json.dumps(articles, indent=2)
    except Exception as e:
        return f"Error searching PubMed: {e}"


@tool
def search_arxiv(query: str, max_results: int = 5) -> str:
    """Search arXiv for preprints in physics, math, CS, and biology.

    Args:
        query: Search query string.
        max_results: Maximum number of results (default 5, max 30).

    Returns:
        JSON string with paper titles, authors, abstract, date, and links.
    """
    try:
        max_results = min(max(max_results, 1), 30)
        url = "http://export.arxiv.org/api/query"
        params: dict[str, str | int] = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        response = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        response.raise_for_status()

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(response.text)  # nosec B314
        entries = root.findall("atom:entry", ns)

        results = []
        for entry in entries:
            title = entry.find("atom:title", ns)
            summary = entry.find("atom:summary", ns)
            published = entry.find("atom:published", ns)
            link = entry.find("atom:id", ns)

            authors = []
            for author in entry.findall("atom:author", ns):
                name = author.find("atom:name", ns)
                if name is not None and name.text:
                    authors.append(name.text)

            categories = []
            for cat in entry.findall("{http://arxiv.org/schemas/atom}primary_category"):
                term = cat.get("term", "")
                if term:
                    categories.append(term)

            results.append(
                {
                    "title": (title.text or "").strip().replace("\n", " ")
                    if title is not None
                    else "",
                    "authors": authors[:5],
                    "abstract": ((summary.text or "").strip()[:500]) if summary is not None else "",
                    "published": (published.text or "")[:10] if published is not None else "",
                    "url": link.text if link is not None else "",
                    "categories": categories,
                }
            )

        if not results:
            return f"No arXiv results found for: '{query}'"
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error searching arXiv: {e}"


@tool
def search_biorxiv(query: str, max_results: int = 5) -> str:
    """Search bioRxiv for biology preprints.

    Args:
        query: Search query string.
        max_results: Maximum number of results (default 5, max 30).

    Returns:
        JSON string with paper titles, authors, abstract, date, DOI, and category.
    """
    try:
        max_results = min(max(max_results, 1), 30)
        url = "https://api.biorxiv.org/details/biorxiv/2020-01-01/3000-01-01/0/json"
        response = requests.get(url, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        query_lower = query.lower()
        query_terms = query_lower.split()
        results = []

        for paper in data.get("collection", []):
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            text = f"{title} {abstract}".lower()
            if all(term in text for term in query_terms):
                results.append(
                    {
                        "title": title,
                        "authors": paper.get("authors", ""),
                        "abstract": abstract[:500],
                        "date": paper.get("date", ""),
                        "doi": paper.get("doi", ""),
                        "category": paper.get("category", ""),
                        "url": f"https://doi.org/{paper.get('doi', '')}",
                    }
                )
            if len(results) >= max_results:
                break

        if not results:
            return f"No bioRxiv results found for: '{query}'. Note: bioRxiv API has limited search. Try broader terms."
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error searching bioRxiv: {e}"


@tool
def fetch_paper_metadata(doi: str) -> str:
    """Fetch metadata for a paper using its DOI via the CrossRef API.

    Args:
        doi: The DOI of the paper (e.g. '10.1038/s41586-021-03819-2').

    Returns:
        JSON string with title, authors, journal, year, abstract, and citation count.
    """
    try:
        doi = doi.strip().removeprefix("https://doi.org/").removeprefix("http://doi.org/")
        url = f"https://api.crossref.org/works/{doi}"
        headers = {"User-Agent": "BioAgents/1.0 (mailto:bioagents@research.org)"}
        response = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        data = response.json().get("message", {})

        authors = []
        for author in data.get("author", []):
            name = f"{author.get('given', '')} {author.get('family', '')}".strip()
            if name:
                authors.append(name)

        title_parts = data.get("title", [])
        title = title_parts[0] if title_parts else ""

        container = data.get("container-title", [])
        journal = container[0] if container else ""

        published = data.get("published-print", data.get("published-online", {}))
        date_parts = published.get("date-parts", [[]])
        year = str(date_parts[0][0]) if date_parts and date_parts[0] else ""

        abstract = data.get("abstract", "")
        if abstract:
            import re

            abstract = re.sub(r"<[^>]+>", "", abstract)[:500]

        return json.dumps(
            {
                "doi": doi,
                "title": title,
                "authors": authors[:10],
                "journal": journal,
                "year": year,
                "abstract": abstract,
                "citation_count": data.get("is-referenced-by-count", 0),
                "url": data.get("URL", ""),
                "type": data.get("type", ""),
            },
            indent=2,
        )
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return f"Error: DOI '{doi}' not found in CrossRef."
        return f"Error fetching metadata: HTTP {e.response.status_code}"
    except Exception as e:
        return f"Error fetching paper metadata: {e}"


def get_literature_tools() -> list:
    """Return all literature search tools."""
    return [search_pubmed, search_arxiv, search_biorxiv, fetch_paper_metadata]
