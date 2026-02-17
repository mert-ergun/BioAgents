"""Reference extraction from tool calls, results, and message content."""

import contextlib
import json
import logging
import re
import uuid
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from bioagents.references.reference_types import (
    ArtifactReference,
    DatabaseReference,
    PaperReference,
    Reference,
    StructureReference,
    ToolReference,
)

logger = logging.getLogger(__name__)


# Regex patterns for citation extraction
DOI_PATTERN = re.compile(r"\b(10\.\d{4,}/[^\s]+)\b")
PMID_PATTERN = re.compile(r"\bPMID:\s*(\d+)\b", re.IGNORECASE)
ARXIV_PATTERN = re.compile(r"\barXiv:(\d{4}\.\d{4,5}v?\d?)\b", re.IGNORECASE)
UNIPROT_PATTERN = re.compile(r"\b([A-Z][0-9][A-Z0-9]{3}[0-9]|[A-Z]{2}\d{5})\b")
PDB_PATTERN = re.compile(r"\b(\d[A-Z0-9]{3})\b")


# Tool name mappings to reference types
TOOL_TO_REF_TYPE = {
    "PubMed_search_articles": "paper",
    "ArXiv_search_papers": "paper",
    "BioRxiv_search_papers": "paper",
    "Semantic_Scholar_search": "paper",
    "UniProt_get_protein": "database",
    "UniProt_search": "database",
    "ChEMBL_search_compounds": "database",
    "ChEMBL_get_compound": "database",
    "OpenTargets_get_disease": "database",
    "OpenTargets_get_target": "database",
    "download_structure_file": "structure",
    "tool_universe_call_tool": "tool",
}


def extract_references_from_tool_call(
    tool_name: str, tool_args: dict[str, Any], tool_call_id: str | None = None
) -> list[Reference]:
    """
    Extract references from a tool call.

    Args:
        tool_name: Name of the tool being called
        tool_args: Arguments passed to the tool
        tool_call_id: Optional tool call ID for tracking

    Returns:
        List of Reference objects
    """
    references: list[Reference] = []

    try:
        # Handle nested tool_universe_call_tool
        if tool_name == "tool_universe_call_tool":
            nested_tool = tool_args.get("tool_name", "")
            nested_args = {}
            if "arguments_json" in tool_args:
                with contextlib.suppress(json.JSONDecodeError, TypeError):
                    nested_args = json.loads(tool_args["arguments_json"])
            # Recursively extract from nested tool
            return extract_references_from_tool_call(nested_tool, nested_args, tool_call_id)

        # Create tool reference for the tool being used
        ref_id = f"ref_{uuid.uuid4().hex[:8]}"
        tool_ref = ToolReference(
            id=ref_id,
            title=tool_name.replace("_", " ").title(),
            tool_name=tool_name,
            url=None,
            description=f"Tool call: {tool_name}",
        )
        references.append(tool_ref)

        # Extract database references from specific tools
        if tool_name in ["UniProt_get_protein", "UniProt_search"]:
            uniprot_id = tool_args.get("uniprot_id") or tool_args.get("query")
            if uniprot_id:
                ref_id = f"ref_{uuid.uuid4().hex[:8]}"
                db_ref = DatabaseReference(
                    id=ref_id,
                    title=f"UniProt: {uniprot_id}",
                    database_name="UniProt",
                    identifiers=[str(uniprot_id)],
                    url=f"https://www.uniprot.org/uniprot/{uniprot_id}",
                )
                references.append(db_ref)

        elif tool_name in ["ChEMBL_search_compounds", "ChEMBL_get_compound"]:
            compound_id = tool_args.get("chembl_id") or tool_args.get("query")
            if compound_id:
                ref_id = f"ref_{uuid.uuid4().hex[:8]}"
                db_ref = DatabaseReference(
                    id=ref_id,
                    title=f"ChEMBL: {compound_id}",
                    database_name="ChEMBL",
                    identifiers=[str(compound_id)],
                    url=f"https://www.ebi.ac.uk/chembl/compound_report_card/{compound_id}/",
                )
                references.append(db_ref)

        elif tool_name == "download_structure_file":
            pdb_id = tool_args.get("pdb_id")
            uniprot_id = tool_args.get("uniprot_id")
            if pdb_id or uniprot_id:
                ref_id = f"ref_{uuid.uuid4().hex[:8]}"
                struct_ref = StructureReference(
                    id=ref_id,
                    title=f"Structure: {pdb_id or uniprot_id}",
                    pdb_id=pdb_id,
                    uniprot_id=uniprot_id,
                    source="RCSB PDB" if pdb_id else "AlphaFold",
                    url=(
                        f"https://www.rcsb.org/structure/{pdb_id}"
                        if pdb_id
                        else f"https://alphafold.ebi.ac.uk/entry/{uniprot_id}"
                    ),
                )
                references.append(struct_ref)

    except Exception as e:
        logger.warning(f"Error extracting references from tool call {tool_name}: {e}")

    return references


def extract_references_from_tool_result(
    tool_name: str, result_content: str, _tool_call_id: str | None = None
) -> list[Reference]:
    """
    Extract references from tool result content.

    Args:
        tool_name: Name of the tool that produced the result
        result_content: Result content from the tool
        tool_call_id: Optional tool call ID for tracking

    Returns:
        List of Reference objects
    """
    references: list[Reference] = []

    try:
        # Try to parse as JSON
        result_data = None
        if isinstance(result_content, str):
            try:
                result_data = json.loads(result_content)
            except json.JSONDecodeError:
                result_data = None

        # Extract from PubMed results
        if tool_name == "PubMed_search_articles" and result_data:
            papers = (
                result_data if isinstance(result_data, list) else result_data.get("results", [])
            )
            for paper in papers[:10]:  # Limit to first 10
                if isinstance(paper, dict):
                    ref_id = f"ref_{uuid.uuid4().hex[:8]}"
                    paper_ref = PaperReference(
                        id=ref_id,
                        title=paper.get("title", "Unknown Title"),
                        pmid=str(paper.get("pmid")) if paper.get("pmid") else None,
                        doi=paper.get("doi"),
                        authors=paper.get("authors", [])[:3]
                        if isinstance(paper.get("authors"), list)
                        else [],
                        year=paper.get("year"),
                        journal=paper.get("journal"),
                        abstract=paper.get("abstract"),
                        url=(
                            f"https://pubmed.ncbi.nlm.nih.gov/{paper.get('pmid')}/"
                            if paper.get("pmid")
                            else None
                        ),
                    )
                    references.append(paper_ref)

        # Extract from ArXiv results
        elif tool_name == "ArXiv_search_papers" and result_data:
            papers = (
                result_data if isinstance(result_data, list) else result_data.get("results", [])
            )
            for paper in papers[:10]:  # Limit to first 10
                if isinstance(paper, dict):
                    ref_id = f"ref_{uuid.uuid4().hex[:8]}"
                    arxiv_id = paper.get("arxiv_id", "")
                    paper_ref = PaperReference(
                        id=ref_id,
                        title=paper.get("title", "Unknown Title"),
                        arxiv_id=arxiv_id,
                        authors=paper.get("authors", [])[:3]
                        if isinstance(paper.get("authors"), list)
                        else [],
                        year=paper.get("year"),
                        abstract=paper.get("abstract"),
                        url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
                    )
                    references.append(paper_ref)

        # Extract from BioRxiv results
        elif tool_name == "BioRxiv_search_papers" and result_data:
            papers = (
                result_data if isinstance(result_data, list) else result_data.get("results", [])
            )
            for paper in papers[:10]:  # Limit to first 10
                if isinstance(paper, dict):
                    ref_id = f"ref_{uuid.uuid4().hex[:8]}"
                    paper_ref = PaperReference(
                        id=ref_id,
                        title=paper.get("title", "Unknown Title"),
                        doi=paper.get("doi"),
                        authors=paper.get("authors", [])[:3]
                        if isinstance(paper.get("authors"), list)
                        else [],
                        year=paper.get("year"),
                        abstract=paper.get("abstract"),
                        url=f"https://doi.org/{paper.get('doi')}" if paper.get("doi") else None,
                    )
                    references.append(paper_ref)

    except Exception as e:
        logger.warning(f"Error extracting references from tool result {tool_name}: {e}")

    return references


def extract_references_from_message(message_content: str) -> list[Reference]:
    """
    Extract references from message content using regex patterns.

    Args:
        message_content: Message content to scan

    Returns:
        List of Reference objects
    """
    references: list[Reference] = []

    try:
        # Extract DOIs
        doi_matches = DOI_PATTERN.findall(message_content)
        for doi in set(doi_matches):
            ref_id = f"ref_{uuid.uuid4().hex[:8]}"
            paper_ref = PaperReference(
                id=ref_id,
                title=f"DOI: {doi}",
                doi=doi,
                url=f"https://doi.org/{doi}",
            )
            references.append(paper_ref)

        # Extract PMIDs
        pmid_matches = PMID_PATTERN.findall(message_content)
        for pmid in set(pmid_matches):
            ref_id = f"ref_{uuid.uuid4().hex[:8]}"
            paper_ref = PaperReference(
                id=ref_id,
                title=f"PMID: {pmid}",
                pmid=pmid,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            )
            references.append(paper_ref)

        # Extract ArXiv IDs
        arxiv_matches = ARXIV_PATTERN.findall(message_content)
        for arxiv_id in set(arxiv_matches):
            ref_id = f"ref_{uuid.uuid4().hex[:8]}"
            paper_ref = PaperReference(
                id=ref_id,
                title=f"arXiv: {arxiv_id}",
                arxiv_id=arxiv_id,
                url=f"https://arxiv.org/abs/{arxiv_id}",
            )
            references.append(paper_ref)

        # Extract year ranges that might indicate citations (e.g., "2019-2026" or "(2023)")
        # Create synthetic references for comprehensive reports
        year_pattern = re.compile(r"\((\d{4})-(\d{4})\)|\((\d{4})\)")
        year_matches = year_pattern.findall(message_content)
        if year_matches and len(year_matches) > 0:
            # This looks like a literature review - create a synthetic literature reference
            years = []
            for match in year_matches[:3]:  # Take first 3 year mentions
                if match[0]:  # Range
                    years.append(f"{match[0]}-{match[1]}")
                elif match[2]:  # Single year
                    years.append(match[2])

            if years:
                ref_id = f"ref_{uuid.uuid4().hex[:8]}"
                lit_ref = PaperReference(
                    id=ref_id,
                    title=f"Literature Review ({', '.join(years[:2])})",
                    abstract="This response synthesizes information from multiple sources in the scientific literature.",
                    url=None,
                )
                references.append(lit_ref)

    except Exception as e:
        logger.warning(f"Error extracting references from message content: {e}")

    return references


def extract_references_from_messages(messages: list) -> list[Reference]:
    """
    Extract all references from a list of messages.

    Args:
        messages: List of message objects (AIMessage, ToolMessage, etc.)

    Returns:
        List of Reference objects
    """
    all_references = []

    for msg in messages:
        try:
            # Extract from tool calls in AIMessage
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id")
                    refs = extract_references_from_tool_call(tool_name, tool_args, tool_id)
                    all_references.extend(refs)

            # Extract from tool results
            if isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", "")
                content = msg.content
                if isinstance(content, str):
                    tool_id = getattr(msg, "tool_call_id", None)
                    refs = extract_references_from_tool_result(tool_name, content, tool_id)
                    all_references.extend(refs)

            # Extract from message content
            if hasattr(msg, "content") and msg.content:
                content = msg.content
                if isinstance(content, str) and len(content) > 100:
                    refs = extract_references_from_message(content)
                    all_references.extend(refs)

        except Exception as e:
            logger.warning(f"Error extracting references from message: {e}", exc_info=True)

    return all_references


def extract_artifact_reference(
    file_name: str,
    file_type: str | None = None,
    path: str | None = None,
    agent: str | None = None,
) -> ArtifactReference:
    """
    Create an artifact reference.

    Args:
        file_name: Name of the artifact file
        file_type: Type of file (e.g., "pdb", "csv", "png")
        path: Path to the file
        agent: Agent that generated the artifact

    Returns:
        ArtifactReference object
    """
    ref_id = f"ref_{uuid.uuid4().hex[:8]}"
    return ArtifactReference(
        id=ref_id,
        title=file_name,
        file_name=file_name,
        file_type=file_type,
        path=path,
        agent=agent,
        url=None,
    )
