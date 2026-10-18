"""Safety filtering for ACE playbook bullets.

Playbooks are *cross-run* state: a bullet curated while answering one question is
appended to every later agent prompt (see
:func:`bioagents.prompts.prompt_loader.load_prompt`). Two classes of bullet make
that mechanism actively harmful, so they are stripped on both save and load:

``fabrication``
    Bullets that tell the agent to *simulate* a tool it could not run and still
    emit a "concrete, factual" answer. Replayed into a later run, such a bullet
    converts a tool timeout into a confident invented result.

``entity-specific``
    Bullets naming a concrete biological entity (gene symbol, UniProt accession,
    ChEMBL/PDB identifier, ...). Those are facts about *one* run's target, not a
    transferable strategy; replayed into a different run they put the previous
    run's protein or compound into the new run's prompt. This is the residual
    state that makes an AKT1 answer mention CDK2.

Only generalisable, entity-free strategy survives the filter.
"""

from __future__ import annotations

import re

#: Phrases that mark a bullet as instructing fabrication of unverified results.
#: Matched case-insensitively against the bullet's content.
_FABRICATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "simulate the output / simulate its expected output / simulating results"
    re.compile(r"\bsimulat\w*\b[^.]{0,80}\b(output|result|outcome|answer|tool|report)", re.I),
    re.compile(r"\b(output|result|outcome|answer)\b[^.]{0,60}\bsimulat\w*", re.I),
    # "provide the most likely factual result based on available knowledge"
    re.compile(r"most likely\s+(factual\s+)?(result|answer|value)", re.I),
    # "concrete, factual answer" / "concrete and factual result" when not measured
    re.compile(r"\bconcrete\b[^.]{0,40}\bfactual\b", re.I),
    re.compile(r"\bfactual\b[^.]{0,40}\bconcrete\b", re.I),
    # "based on available knowledge / from your own knowledge / from memory"
    re.compile(r"based on (your |the model'?s? |available |internal |prior )?knowledge", re.I),
    re.compile(r"\bfrom (your own |the model'?s? )?(memory|knowledge)\b", re.I),
    # "mirrors real-world analytical reports" style fabricated tables
    re.compile(r"mirror\w*\s+real[- ]world", re.I),
)

#: Identifier shapes that pin a bullet to one specific biological entity.
_ENTITY_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Database accessions: ChEMBL123, P31749, Q9Y6K9, ENSG000..., rs12345, NM_000546
    re.compile(r"\bCHEMBL\d+\b", re.I),
    re.compile(r"\b[OPQ][0-9][A-Z0-9]{3}[0-9]\b"),  # UniProt (P31749, Q9Y6K9)
    re.compile(r"\b[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}\b"),  # UniProt (A0A123, ...)
    re.compile(r"\bENS[A-Z]*[GTP]\d{6,}\b"),
    re.compile(r"\brs\d{3,}\b"),
    re.compile(r"\b[NX][MRP]_\d{6,}\b"),
    re.compile(r"\bPMID[:\s]*\d{4,}\b", re.I),
    re.compile(r"\bDOI[:\s]*10\.\d{4,}/", re.I),
    # PDB entry codes are 4 chars starting with a digit (4GV1, 1YCR).
    re.compile(r"\b[1-9][A-Z0-9]{3}\b"),
)

#: Gene / protein / drug symbols. Kept as an explicit vocabulary because a bare
#: "all caps token" rule would delete ordinary acronyms (API, JSON, CSV, ...).
_ENTITY_WORD_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Kinase- and receptor-style gene symbols: AKT1, CDK2, EGFR, MAP2K1, PIK3CA.
    re.compile(r"\b(?:AKT|CDK|MAPK?|MAP\dK|ERK|JNK|PIK3|STAT|BRCA|HDAC|VEGFR|FGFR)\d*[A-Z]*\d*\b"),
    re.compile(r"\b(?:EGFR|KRAS|NRAS|HRAS|BRAF|TP53|MDM2|MTOR|ABL1|SRC|ESR1|PTEN|MYC|ALK|RET)\b"),
    # Common small-molecule / antibody naming suffixes (capivasertib, imatinib...).
    re.compile(r"\b[A-Za-z]{4,}(?:sertib|tinib|ciclib|parib|mab|nib|stat|mycin|platin)\b", re.I),
    # Development codes: AZD5363, GSK690693, MK-2206, GDC-0068.
    re.compile(r"\b(?:AZD|GSK|MK|GDC|BMS|LY|PF|AT|SCH|CHIR)[- ]?\d{3,}\b", re.I),
    # "p53", "p38" style protein shorthand.
    re.compile(r"\bp\d{2}\b"),
)

#: Tokens that look like identifiers but are vocabulary of the system itself and
#: must not cause a bullet to be dropped.
_ENTITY_ALLOWLIST = frozenset(
    {
        "3D",
        "2D",
        "1D",
        "5XXX",
        "JSON",
        "HTTP",
        "HTML",
        "UTF8",
    }
)

#: Reason codes reported by :func:`classify_bullet`.
SAFE = "safe"
FABRICATION = "fabrication"
ENTITY_SPECIFIC = "entity_specific"


def _mentions_entity(content: str) -> bool:
    """True when *content* pins itself to one concrete biological entity."""
    for pattern in _ENTITY_PATTERNS:
        for match in pattern.finditer(content):
            if match.group(0).upper() not in _ENTITY_ALLOWLIST:
                return True
    return any(pattern.search(content) for pattern in _ENTITY_WORD_PATTERNS)


def _instructs_fabrication(content: str) -> bool:
    """True when *content* tells the agent to invent an unverified result."""
    return any(pattern.search(content) for pattern in _FABRICATION_PATTERNS)


def classify_bullet(content: str) -> str:
    """Classify one bullet's content.

    Args:
        content: The bullet text (the part after ``::`` in a playbook line).

    Returns:
        :data:`SAFE`, :data:`FABRICATION` or :data:`ENTITY_SPECIFIC`.
    """
    if _instructs_fabrication(content):
        return FABRICATION
    if _mentions_entity(content):
        return ENTITY_SPECIFIC
    return SAFE


def is_safe_bullet(content: str) -> bool:
    """True when the bullet is generalisable and safe to replay into a later run."""
    return classify_bullet(content) == SAFE


def sanitize_playbook(playbook: str) -> tuple[str, list[tuple[str, str]]]:
    """Drop unsafe bullets from *playbook*, keeping section structure intact.

    Section headers, blank lines and non-bullet text are preserved verbatim so a
    sanitized playbook stays a valid playbook that the curator can keep growing.

    Args:
        playbook: Raw playbook text.

    Returns:
        ``(sanitized_playbook, removed)`` where *removed* lists
        ``(bullet_id, reason)`` for every dropped bullet.
    """
    # Imported here: playbook_manager imports this module for save/load filtering.
    from bioagents.learning.playbook_manager import _parse_playbook_line

    kept: list[str] = []
    removed: list[tuple[str, str]] = []

    for line in playbook.split("\n"):
        parsed = _parse_playbook_line(line)
        if parsed is None:
            kept.append(line)
            continue

        content = parsed.get("content", "")
        reason = classify_bullet(content if isinstance(content, str) else "")
        if reason == SAFE:
            kept.append(line)
        else:
            bullet_id = parsed.get("id", "")
            removed.append((bullet_id if isinstance(bullet_id, str) else "", reason))

    return "\n".join(kept), removed
