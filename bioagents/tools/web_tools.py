"""Web tools for fetching URLs, searching, and downloading files."""

import json
import os
import re
from urllib.parse import urlparse

import requests
from langchain_core.tools import tool

HTTP_TIMEOUT = 30


@tool
def fetch_url_content(url: str) -> str:
    """Fetch and return the text content of a URL.

    Args:
        url: The URL to fetch content from.

    Returns:
        The text content of the page (HTML stripped to plain text if possible),
        or an error message.
    """
    lower = url.lower()
    if "rest.uniprot.org" in lower and ".txt" in lower and "uniprotkb" in lower:
        return (
            "UniProt full-text flat files are very large. Do not fetch them with this tool. "
            "Use download_uniprot_flat_file(accession, output_path) instead."
        )
    try:
        headers = {"User-Agent": "BioAgents/1.0 (research tool)"}
        response = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "html" in content_type:
            text = re.sub(r"<script[^>]*>.*?</script>", "", response.text, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
        else:
            text = response.text

        if len(text) > 50000:
            text = text[:50000] + "\n\n... [truncated] ..."
        return text
    except requests.exceptions.Timeout:
        return f"Error: Request to '{url}' timed out after {HTTP_TIMEOUT}s."
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP {e.response.status_code} fetching '{url}'."
    except Exception as e:
        return f"Error fetching URL: {e}"


@tool
def search_google_scholar(query: str, num_results: int = 5) -> str:
    """Search Google Scholar for academic papers.

    Uses the Semantic Scholar API as a freely accessible alternative.

    Args:
        query: The search query string.
        num_results: Maximum number of results to return (default 5, max 20).

    Returns:
        JSON string with paper titles, authors, year, citation count, and URLs.
    """
    try:
        num_results = min(max(num_results, 1), 20)
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": num_results,
            "fields": "title,authors,year,citationCount,url,abstract,externalIds",
        }
        response = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        results = []
        for paper in data.get("data", []):
            authors = [a.get("name", "") for a in paper.get("authors", [])]
            results.append(
                {
                    "title": paper.get("title", ""),
                    "authors": authors[:5],
                    "year": paper.get("year"),
                    "citations": paper.get("citationCount", 0),
                    "url": paper.get("url", ""),
                    "doi": (paper.get("externalIds") or {}).get("DOI", ""),
                    "abstract": (paper.get("abstract") or "")[:300],
                }
            )
        if not results:
            return f"No results found for query: '{query}'"
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error searching academic papers: {e}"


@tool
def download_file_from_url(url: str, output_path: str = "") -> str:
    """Download a file from a URL to the sandbox workspace.

    Args:
        url: The URL of the file to download.
        output_path: Optional output file path in the sandbox. If empty,
                     the filename is inferred from the URL.

    Returns:
        The path where the file was saved, or an error message.
    """
    try:
        from bioagents.sandbox.sandbox_manager import get_sandbox

        sandbox = get_sandbox()

        if not output_path:
            parsed = urlparse(url)
            output_path = os.path.basename(parsed.path) or "downloaded_file"

        result = sandbox.download_file(url, output_path)
        if result["success"]:
            saved = sandbox.workdir / sandbox._normalize_path(output_path)
            return f"File downloaded successfully to: {saved}"
        return f"Failed to download file: {result['stderr']}"
    except Exception as e:
        return f"Error downloading file: {e}"


def get_web_tools() -> list:
    """Return all web tools."""
    return [fetch_url_content, search_google_scholar, download_file_from_url]
