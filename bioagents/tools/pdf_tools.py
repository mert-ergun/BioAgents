"""PDF tools for downloading, parsing, and extracting text from PDFs and webpages.

This module provides:
1. PDF text extraction via PyMuPDF (primary) with spaCy-layout fallback
2. LangChain @tool functions for agent integration
3. ToolUniverse integration for webpage-to-text extraction
"""

import logging
from pathlib import Path

from langchain_core.tools import tool

# Primary PDF library: PyMuPDF (lightweight, fast, no external deps)
try:
    import pymupdf

    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# Fallback PDF library: spaCy-layout (heavier, better layout analysis)
try:
    import spacy
    from spacy_layout import spaCyLayout

    HAS_SPACY_LAYOUT = True
except ImportError:
    HAS_SPACY_LAYOUT = False

# ToolUniverse wrapper for webpage extraction
from bioagents.tools.tool_universe import DEFAULT_WRAPPER

logger = logging.getLogger(__name__)


def _extract_with_pymupdf(pdf_path: str) -> str:
    """Extract text from PDF using PyMuPDF. Returns markdown-formatted text."""
    doc = pymupdf.open(pdf_path)
    pages: list[str] = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append(f"## Page {page_num + 1}\n\n{text}")

    doc.close()
    return "\n\n".join(pages)


def _extract_with_spacy_layout(pdf_path: str) -> str:
    """Extract text from PDF using spaCy-layout. Returns markdown."""
    nlp = spacy.blank("en")
    layout = spaCyLayout(nlp)
    doc = layout(pdf_path)
    doc = nlp(doc)
    markdown = doc._.markdown
    return str(markdown) if markdown else ""


# ============================================================================
# SECTION 1: LangChain @tool Functions
# ============================================================================


@tool
def fetch_webpage_as_pdf_text(url: str, timeout: int = 30) -> str:
    """Fetch a web page and extract its readable text, including JS-rendered pages.

    Use this to read documentation, articles or database pages that plain HTTP fetching
    cannot render. For PDFs already on disk use extract_pdf_text_spacy_layout instead.

    Args:
        url: Full URL of the page to fetch, including the scheme (https://...).
        timeout: Seconds to wait for the page to load before giving up (default 30).

    Returns:
        The extracted page text as a plain string. On failure returns a string starting
        with "Error fetching webpage" that names the URL and the underlying cause.
    """
    try:
        result = DEFAULT_WRAPPER.execute_tool(
            tool_name="get_webpage_text_from_url",
            arguments={"url": url, "timeout": int(timeout)},
        )
    except Exception as e:
        logger.error(f"Error fetching webpage as PDF: {e}")
        return f"Error fetching webpage '{url}': {e!s}"

    # The browser renderer is an optional ToolUniverse extra. Most pages do not need JS,
    # so fall back to a plain HTTP fetch rather than returning nothing — an agent that
    # cannot read a page tends to proceed on recalled knowledge instead.
    if _browser_unavailable(result):
        logger.info("Browser renderer unavailable; falling back to plain HTTP for %s", url)
        from bioagents.tools.web_tools import fetch_url_content

        fallback = str(fetch_url_content.invoke({"url": url}))
        if fallback and not fallback.lower().startswith("error"):
            return f"[Fetched without JS rendering — dynamic content may be missing]\n{fallback}"
        return (
            f"Error fetching webpage '{url}': the browser renderer is not installed and "
            f"the plain HTTP fallback also failed. Install the renderer with "
            f"`pip install 'tooluniverse[browser]' && playwright install chromium`. "
            f"Fallback result: {fallback}"
        )

    return result


def _browser_unavailable(result: str) -> bool:
    """Detect ToolUniverse reporting that page rendering needs a browser it does not have."""
    if not isinstance(result, str):
        return False
    lowered = result.lower()
    return "needs a browser" in lowered or "playwright" in lowered


@tool
def extract_pdf_text_spacy_layout(local_pdf_path: str) -> str:
    """Extract the full text and layout structure from a PDF file already on disk.

    Uses PyMuPDF as the primary extraction engine, falling back to spaCy-layout when
    available. Use this to read a downloaded paper's Methods section, supplementary
    data, or any local PDF. To read a web page instead, use fetch_webpage_as_pdf_text.

    Args:
        local_pdf_path: Path to the PDF file, absolute or relative to the project root.

    Returns:
        The extracted document text as a plain string, with layout structure preserved
        where the engine can detect it. If the file does not exist or cannot be parsed,
        returns a string starting with "Error:" that explains which step failed.
    """
    # Validate file exists
    path = Path(local_pdf_path)
    if not path.exists():
        return f"Error: File not found at '{local_pdf_path}'"

    if path.suffix.lower() != ".pdf":
        return f"Error: File '{local_pdf_path}' is not a PDF."

    # Try PyMuPDF first (lightweight, always available in Docker)
    if HAS_PYMUPDF:
        try:
            logger.info(f"Extracting PDF text with PyMuPDF: {local_pdf_path}")
            text = _extract_with_pymupdf(local_pdf_path)
            if text.strip():
                return text
            logger.warning("PyMuPDF extracted no text, trying spaCy-layout fallback...")
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")

    # Fallback to spaCy-layout (better layout analysis but heavy)
    if HAS_SPACY_LAYOUT:
        try:
            logger.info(f"Extracting PDF text with spaCy-layout: {local_pdf_path}")
            text = _extract_with_spacy_layout(local_pdf_path)
            if text.strip():
                return text
            return "Warning: PDF processed but no extractable text found."
        except Exception as e:
            logger.error(f"spaCy-layout extraction failed: {e}")
            return f"Error processing PDF with spaCy-layout: {e!s}"

    # No library available
    return (
        "Error: No PDF extraction library available. "
        "Install pymupdf (`pip install pymupdf`) or spacy + spacy-layout."
    )
