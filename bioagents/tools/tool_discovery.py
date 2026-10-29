"""Discovery helpers for ``tool_universe_find_tools``: local tools, metadata audit, merging.

Three problems made keyword-only discovery change what experiments measured:

- tools shipped with BioAgents (``file_tools.py``) were not in the search index, so
  agents rebuilt trivial capabilities such as listing a directory at runtime;
- SDK registry metadata was passed through unchecked, so an entry whose name,
  description and parameters disagree (``NvidiaNIM_boltz2`` documented as a chemical
  component lookup with a required ``comp_id``) looked unverifiable rather than wrong;
- results from different finders were not merged, so one weak finder decided alone.

Everything here is deterministic and offline: no network, model or LLM is used.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

SOURCE_CUSTOM_REGISTRY = "custom_registry"
SOURCE_LOCAL = "local"
SOURCE_SDK = "sdk"
SOURCE_CATALOG = "catalog"

# A local tool needs this many distinct query concepts in its name/description.
_LOCAL_MIN_MATCHES = 2
# Every file tool's name has these; the verb or object in the name is what decides.
_GENERIC_LOCAL_NAME_TOKENS = frozenset({"local", "file", "get"})

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
        "via",
        "using",
        "this",
        "that",
        "these",
        "those",
        "all",
        "any",
        "some",
        "my",
        "our",
        "your",
        "its",
        "their",
    ]
)

# Query words agents actually used, mapped onto the vocabulary of the local tools.
_SYNONYMS: dict[str, str] = {
    "save": "write",
    "store": "write",
    "dump": "write",
    "export": "write",
    "persist": "write",
    "load": "read",
    "open": "read",
    "view": "read",
    "show": "read",
    "folder": "directory",
    "dir": "directory",
    "directories": "directory",
    "folders": "directory",
    "files": "file",
    "ls": "list",
    "listing": "list",
    "enumerate": "list",
    "exist": "exists",
    "present": "exists",
    "verify": "check",
    "metadata": "info",
    "size": "info",
    "stat": "info",
}

# Tool families whose purpose implies a sequence/structure/molecule input. A tool in
# one of these families with no such parameter has metadata that contradicts its name.
_STRUCTURE_TOOL_MARKERS = (
    "boltz",
    "alphafold",
    "cofold",
    "chai",
    "esmfold",
    "rosettafold",
    "openfold",
    "diffdock",
    "vina",
)
_STRUCTURE_PARAM_HINTS = (
    "seq",
    "smiles",
    "ligand",
    "protein",
    "structure",
    "pdb",
    "receptor",
    "molecule",
    "chain",
    "input",
    "fasta",
    "msa",
    "complex",
    "sdf",
    "mol",
)

METADATA_MISMATCH_NOTE = (
    "Entries with non-empty 'metadata_warnings' have registry metadata that is "
    "internally inconsistent (name, description and parameters disagree). The tool may "
    "still work, but verify what it does before relying on it, and say so in your report."
)


def _split_identifier(text: str) -> list[str]:
    """Split snake_case / camelCase / digits into lowercase word tokens."""
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", spaced)
    return [tok for tok in re.split(r"[^A-Za-z]+", spaced.lower()) if tok]


def _tokens(text: str) -> set[str]:
    """Meaningful, synonym-normalised tokens of free text or an identifier."""
    out: set[str] = set()
    for tok in _split_identifier(text):
        if tok in _STOPWORDS or len(tok) < 2:
            continue
        out.add(_SYNONYMS.get(tok, tok))
    return out


def _related(a: str, b: str) -> bool:
    """Whether two tokens are equal or share a stem of at least four letters."""
    if a == b:
        return True
    stem = min(len(a), len(b), 5)
    return stem >= 4 and a[:stem] == b[:stem]


# ---------------------------------------------------------------------------
# Local tools
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _local_tool_objects() -> dict[str, Any]:
    """Name -> LangChain tool for utilities shipped with BioAgents."""
    try:
        from bioagents.tools.file_tools import get_file_tools
    except Exception as exc:  # pragma: no cover - import-time environment issue
        logger.warning("Local file tools are unavailable for discovery: %s", exc)
        return {}
    return {tool_obj.name: tool_obj for tool_obj in get_file_tools()}


def _local_tool_entry(tool_obj: Any) -> dict[str, Any]:
    """Describe a LangChain tool in the find_tools result format."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    try:
        schema = tool_obj.args_schema.model_json_schema() if tool_obj.args_schema else {}
    except Exception:  # pragma: no cover - defensive
        schema = {}
    for pname, pschema in (schema.get("properties") or {}).items():
        properties[pname] = {
            "type": pschema.get("type", "string"),
            "description": pschema.get("description", ""),
        }
    required = list(schema.get("required") or [])
    description = (tool_obj.description or "").strip()
    return {
        "name": tool_obj.name,
        "description": _summary(description),
        "parameter": {"type": "object", "properties": properties, "required": required},
        "required": required,
        "source": SOURCE_LOCAL,
        "validated": True,
        "how_to_call": (
            "Shipped with BioAgents. Call it directly if it is in your tool list, or via "
            f"tool_universe_call_tool(tool_name='{tool_obj.name}', arguments_json=...)."
        ),
    }


def _summary(description: str) -> str:
    """First paragraph of a tool docstring: its purpose, without usage detail."""
    return description.strip().split("\n\n")[0].strip()


def get_local_tool(name: str) -> Any | None:
    """Return the shipped LangChain tool called ``name``, if any."""
    return _local_tool_objects().get(name)


def search_local_tools(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Rank shipped local tools against ``query``; only clear matches are returned."""
    query_tokens = _tokens(query)
    if not query_tokens:
        return []
    scored: list[tuple[int, str, dict[str, Any]]] = []
    for name, tool_obj in _local_tool_objects().items():
        name_tokens = _tokens(name)
        text_tokens = name_tokens | _tokens(_summary(tool_obj.description or ""))
        matches = {q for q in query_tokens if any(_related(q, t) for t in text_tokens)}
        distinctive = name_tokens - _GENERIC_LOCAL_NAME_TOKENS
        name_matches = {q for q in matches if any(_related(q, t) for t in distinctive)}
        if len(matches) < _LOCAL_MIN_MATCHES or not name_matches:
            continue
        score = len(matches) + 2 * len(name_matches)
        scored.append((score, name, tool_obj))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [_local_tool_entry(tool_obj) for _, _, tool_obj in scored[: max(1, limit)]]


# ---------------------------------------------------------------------------
# Metadata audit
# ---------------------------------------------------------------------------


def _schema_parts(entry: dict[str, Any]) -> tuple[dict[str, Any], set[str], bool]:
    """Return (properties, required names, schema_is_malformed) for a tool entry."""
    param = entry.get("parameter")
    if param is None:
        param = entry.get("parameters")
    if param is None:
        param = {}
    if not isinstance(param, dict):
        return {}, set(), True

    if isinstance(param.get("properties"), dict):
        properties = param["properties"]
    elif "type" in param and param.get("type") == "object":
        properties = {}
    else:
        # Legacy flat form: {"arg": {"type": ...}, ...}
        properties = {k: v for k, v in param.items() if k not in {"type", "required"}}

    required: set[str] = set()
    for source in (entry.get("required"), param.get("required")):
        if isinstance(source, list):
            required.update(str(r) for r in source)
    for pname, pschema in properties.items():
        if isinstance(pschema, dict) and pschema.get("required") is True:
            required.add(str(pname))
    return properties, required, False


def audit_tool_metadata(entry: dict[str, Any]) -> list[str]:
    """Check that a tool's name, description and parameter schema agree.

    Heuristics are deliberately conservative: each warning names a concrete,
    checkable inconsistency. An empty list means no inconsistency was detected, not
    that the tool is known to work.
    """
    warnings: list[str] = []
    name = str(entry.get("name") or "")
    description = str(entry.get("description") or "").strip()

    if not name:
        warnings.append("metadata mismatch: tool has no name")
    if not description:
        warnings.append("metadata mismatch: empty description")

    properties, required, malformed = _schema_parts(entry)
    if malformed:
        warnings.append("metadata mismatch: parameter schema is not an object")
    for pname, pschema in properties.items():
        if not isinstance(pschema, dict):
            warnings.append(f"metadata mismatch: parameter '{pname}' has no schema object")
    for missing in sorted(r for r in required if r not in properties):
        warnings.append(
            f"metadata mismatch: required parameter '{missing}' is not defined "
            "in the parameter schema"
        )

    name_tokens = {t for t in _tokens(name) if len(t) >= 3}
    desc_tokens = _tokens(description)
    if (
        name_tokens
        and len(desc_tokens) >= 4
        and not any(_related(n, d) for n in name_tokens for d in desc_tokens)
    ):
        warnings.append(
            "metadata mismatch: the description shares no terms with the tool name; "
            "it may describe a different tool"
        )

    lowered = name.lower()
    marker = next((m for m in _STRUCTURE_TOOL_MARKERS if m in lowered), None)
    if marker and properties:
        param_text = " ".join(str(p).lower() for p in properties)
        if not any(hint in param_text for hint in _STRUCTURE_PARAM_HINTS):
            warnings.append(
                f"metadata mismatch: '{name}' looks like a {marker} structure/docking "
                f"tool but none of its parameters ({', '.join(sorted(properties))}) "
                "take a sequence, structure or molecule"
            )
    return warnings


def annotate_entry(entry: dict[str, Any], default_source: str) -> dict[str, Any]:
    """Return a copy of ``entry`` with ``source``, ``metadata_warnings`` and ``validated``."""
    annotated = dict(entry)
    source = annotated.setdefault("source", default_source)
    if source == SOURCE_CATALOG:
        # Catalogue rows carry only a name and a link: nothing to cross-check, and
        # nothing to vouch for either.
        annotated["metadata_warnings"] = []
        annotated["validated"] = False
        return annotated

    warnings = audit_tool_metadata(annotated)
    annotated["metadata_warnings"] = warnings
    if source == SOURCE_CUSTOM_REGISTRY:
        # Runtime-built tools also need to have passed the registry's own validation.
        annotated["validated"] = bool(annotated.get("validated")) and not warnings
    else:
        annotated["validated"] = not warnings
    return annotated


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def merge_ranked(groups: list[tuple[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    """Merge finder result lists, deduplicating by name.

    Tools found by several finders come first (agreement is the strongest signal),
    then the remaining hits are interleaved round-robin so a single finder cannot
    crowd the others out of a short result list. Each entry records ``matched_by``.
    """
    by_name: dict[str, dict[str, Any]] = {}
    order: dict[str, tuple[int, int]] = {}
    for label, hits in groups:
        for rank, hit in enumerate(hits):
            hit_name = str(hit.get("name") or "")
            if not hit_name:
                continue
            if hit_name not in by_name:
                by_name[hit_name] = {**hit, "matched_by": []}
                order[hit_name] = (rank, len(order))
            else:
                best_rank, first_seen = order[hit_name]
                order[hit_name] = (min(best_rank, rank), first_seen)
            if label not in by_name[hit_name]["matched_by"]:
                by_name[hit_name]["matched_by"].append(label)

    def sort_key(hit_name: str) -> tuple[int, int, int]:
        rank, first_seen = order[hit_name]
        agreement = len(by_name[hit_name]["matched_by"])
        return (-min(agreement, 2), rank, first_seen)

    return [by_name[n] for n in sorted(by_name, key=sort_key)]
