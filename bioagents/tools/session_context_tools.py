"""Tool for agents to retrieve detailed outputs from previous session turns.

Provides `retrieve_previous_context` — a tool that all agents can call to
pull specific results from earlier conversation turns on demand, instead of
having everything pushed into their context window.

The session stores a *catalog* (short index cards) that is always injected,
and *full content* that agents can retrieve selectively via this tool.
"""

import logging
import re
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Maximum number of characters returned per retrieval call.
_MAX_RETRIEVAL_CHARS = 4000

# Minimum keyword-overlap score to be considered a match.
_MIN_OVERLAP_SCORE = 1


def _tokenize(text: str) -> set[str]:
    """Split text into lowercase word tokens, stripping punctuation."""
    return {
        w
        for w in re.split(r"\W+", text.lower())
        if len(w) > 1  # skip single-char tokens
    }


def _fuzzy_overlap(query_tokens: set[str], target_tokens: set[str]) -> float:
    """Count matches between two token sets, including prefix/stem matches.

    "bind" matches "binding", "dock" matches "docking", etc.
    Exact matches count as 1.0, prefix matches count as 0.7.
    """
    score = 0.0
    for qt in query_tokens:
        # Exact match
        if qt in target_tokens:
            score += 1.0
            continue
        # Prefix/stem match: query token is a prefix of a target token
        # (e.g. "bind" → "binding") or vice versa
        for tt in target_tokens:
            if len(qt) >= 3 and len(tt) >= 3 and (tt.startswith(qt) or qt.startswith(tt)):
                score += 0.7
                break
    return score


def _score_entry(query_tokens: set[str], entry: dict[str, Any]) -> float:
    """Score a node-output entry against tokenized query.

    scoring factors:
    - fuzzy keyword overlap between query and index/content
    - bonus if the entry's agent name appears in the query
    """
    index_tokens = _tokenize(entry.get("index", ""))
    agent_tokens = _tokenize(entry.get("agent", ""))
    content_tokens = _tokenize(entry.get("content", "")[:500])

    # Base overlap: query vs index (fuzzy)
    score = _fuzzy_overlap(query_tokens, index_tokens)

    # Agent name match bonus
    if query_tokens & agent_tokens:
        score += 2

    # Content overlap (weaker signal, fuzzy)
    score += _fuzzy_overlap(query_tokens, content_tokens) * 0.5

    return score


def search_node_outputs(
    query: str,
    node_outputs_by_turn: list[dict[str, dict[str, Any]]],
    turn_number: int | None = None,
    agent_name: str | None = None,
) -> str:
    """Search stored node outputs and return matching full content.

    Args:
        query: Natural language description of what the agent needs.
        node_outputs_by_turn: List of dicts, one per turn. Each dict maps
            ``"{agent_name}_{idx}"`` to ``{"index": str, "content": str, "agent": str}``.
        turn_number: Optional 1-indexed turn to search. If None, searches all turns.
        agent_name: Optional agent name to filter by.

    Returns:
        Concatenated full content of matching entries, or a "not found" message.
    """
    if not node_outputs_by_turn:
        return "No previous session context is available."

    query_tokens = _tokenize(query)
    if not query_tokens:
        return "Please provide a more descriptive query about what you need."

    # Determine which turns to search
    if turn_number is not None:
        turn_idx = turn_number - 1  # convert to 0-indexed
        if turn_idx < 0 or turn_idx >= len(node_outputs_by_turn):
            return f"Turn {turn_number} does not exist. Available turns: 1-{len(node_outputs_by_turn)}."
        turns_to_search = [(turn_number, node_outputs_by_turn[turn_idx])]
    else:
        turns_to_search = list(enumerate(node_outputs_by_turn, start=1))

    # Score every entry
    scored: list[tuple[float, int, str, str]] = []  # (score, turn, key, content)
    for turn_num, entries in turns_to_search:
        for key, entry in entries.items():
            # Optional agent name filter
            if agent_name and agent_name.lower() not in entry.get("agent", "").lower():
                continue

            score = _score_entry(query_tokens, entry)
            if score >= _MIN_OVERLAP_SCORE:
                scored.append((score, turn_num, key, entry.get("content", "")))

    if not scored:
        # Fallback: if agent_name was specified, return everything from that agent
        if agent_name:
            fallback_parts: list[str] = []
            for turn_num, entries in turns_to_search:
                for _key, entry in entries.items():
                    if agent_name.lower() in entry.get("agent", "").lower():
                        fallback_parts.append(
                            f"[Turn {turn_num} — {entry.get('agent', 'unknown')}]\n"
                            f"{entry.get('content', '')[:_MAX_RETRIEVAL_CHARS]}"
                        )
            if fallback_parts:
                return "\n\n".join(fallback_parts)

        return (
            "No matching previous context found for your query. "
            "Check the [PREVIOUS SESSION CATALOG] for available entries."
        )

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Build response within budget
    parts: list[str] = []
    total_chars = 0
    for _score, turn_num, key, content in scored:
        entry_text = f"[Turn {turn_num} — {key}]\n{content[:_MAX_RETRIEVAL_CHARS]}"
        if total_chars + len(entry_text) > _MAX_RETRIEVAL_CHARS:
            # Truncate last entry to fit
            remaining = _MAX_RETRIEVAL_CHARS - total_chars
            if remaining > 100:
                parts.append(entry_text[:remaining] + "\n... [truncated]")
            break
        parts.append(entry_text)
        total_chars += len(entry_text)

    return "\n\n".join(parts)


def create_retrieve_previous_context_tool(
    node_outputs_by_turn: list[dict[str, dict[str, Any]]],
):
    """Factory that creates a `retrieve_previous_context` tool bound to session data.

    Args:
        node_outputs_by_turn: The session's accumulated node outputs.

    Returns:
        A LangChain ``@tool`` that agents can invoke.
    """

    @tool
    def retrieve_previous_context(
        query: str,
        turn_number: int | None = None,
        agent_name: str | None = None,
    ) -> str:
        """Retrieve detailed outputs from previous turns in this session.

        Use this when you need specific results, analysis, or data from earlier
        work. Check the [PREVIOUS SESSION CATALOG] to see what is available.

        Args:
            query: Describe what you need, e.g. "structure analysis results"
                or "docking binding energy scores".
            turn_number: Which turn to search (1-indexed). Leave empty to search all.
            agent_name: Filter to a specific agent, e.g. "StructuralBiology".
        """
        return search_node_outputs(query, node_outputs_by_turn, turn_number, agent_name)

    return retrieve_previous_context
