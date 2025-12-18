"""PDF tools for downloading, parsing, and extracting text from PDFs and webpages.

This module now only provides:
1. PDF processing via spaCy-layout
2. LangChain @tool functions for agent integration
3. ToolUniverse integration for webpage-to-text extraction
"""

import json
import logging
from typing import Optional

from langchain_core.tools import tool

# Required PDF/NLP libraries
try:
    import requests
    import spacy
    from spacy_layout import spaCyLayout
    HAS_PDF_LIBRARIES = True
except ImportError:
    HAS_PDF_LIBRARIES = False

# ToolUniverse wrapper for webpage extraction
from bioagents.tools.tool_universe import DEFAULT_WRAPPER

logger = logging.getLogger(__name__)


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
            arguments={"url": url, "timeout": int(timeout)}
        )
        return result
    except Exception as e:
        logger.error(f"Error fetching webpage as PDF: {e}")
        return f"Error fetching webpage '{url}': {e!s}"


@tool
def extract_pdf_text_spacy_layout(local_pdf_path: str) -> str:
    """
    Extract text and layout information from a local PDF using spaCy-layout.
    """
    if not HAS_PDF_LIBRARIES:
        return "Error: Required libraries not installed ('spacy', 'spacy_layout')."

    try:
        logger.info(f"Processing PDF with spaCy-layout: {local_pdf_path}")
        nlp = spacy.blank("en")
        layout = spaCyLayout(nlp)

        doc = layout(local_pdf_path)
        doc = nlp(doc)  # Apply NLP pipeline

        markdown = doc._.markdown

        if not markdown or not markdown.strip():
            return "Warning: PDF processed but no extractable text found."

        return markdown

    except Exception as e:
        logger.error(f"Error extracting PDF with spaCy-layout: {e}")
        return f"Error processing PDF with spaCy-layout: {e!s}"
