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
    """
    Fetch a webpage and extract text using ToolUniverse.
    Supports JS-rendered pages.
    """
    try:
        result = DEFAULT_WRAPPER.execute_tool(
            tool_name="get_webpage_text_from_url",
            arguments={"url": url, "timeout": int(timeout)},
        )
        return result
    except Exception as e:
        logger.error(f"Error fetching webpage as PDF: {e}")
        return f"Error fetching webpage '{url}': {e!s}"


@tool
def extract_pdf_text_spacy_layout(local_pdf_path: str) -> str:
    """Extract text and layout information from a local PDF file.

    Uses PyMuPDF (fast, lightweight) as the primary extraction engine,
    falling back to spaCy-layout if available. Accepts absolute or
    relative file paths.

    Args:
        local_pdf_path: Path to the PDF file (absolute or relative).
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
