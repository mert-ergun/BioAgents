"""Tool policy engine — filters and evaluates tool calls for safety and relevance.

Prevents agents from calling unrelated tools, tools requiring unavailable API keys,
and provides an approval gate mechanism for flagged tools.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (env-tunable)
# ---------------------------------------------------------------------------

_STRICTNESS = os.getenv("BIOAGENTS_TOOL_POLICY_STRICTNESS", "moderate").lower()
_EXTRA_CATS_RAW = os.getenv("BIOAGENTS_TOOL_POLICY_EXTRA_CATEGORIES", "")
_EXTRA_CATS = frozenset(c.strip().lower() for c in _EXTRA_CATS_RAW.split(",") if c.strip())


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolPolicyResult:
    """Outcome of a tool policy evaluation."""

    allowed: bool
    reason: str
    requires_approval: bool
    category: str = ""
    risk_level: Literal["none", "low", "medium", "high"] = "none"


@dataclass
class ToolPolicyStats:
    """Running statistics for the current session."""

    auto_approved: int = 0
    user_approved: int = 0
    blocked: int = 0
    filtered_at_discovery: int = 0


# ---------------------------------------------------------------------------
# Category and keyword definitions
# ---------------------------------------------------------------------------

# ToolUniverse section names that are relevant to BioAgents
BIO_RELEVANT_SECTIONS: frozenset[str] = frozenset({
    "bioinformatics", "genomics", "proteomics", "structural biology",
    "cheminformatics", "single cell", "drug discovery", "molecular docking",
    "phylogenetics", "transcriptomics", "visualization", "scientific computing",
    "literature", "machine learning", "protein design", "molecular dynamics",
    "sequence analysis", "structural biology tools", "genomics tools",
    "cheminformatics tools", "single cell tools", "bioinformatics core tools",
    "packages tools",
})

# Keywords in tool names that indicate bio-relevance
BIO_KEYWORDS_IN_NAME: frozenset[str] = frozenset({
    "protein", "gene", "genome", "sequence", "alignment", "blast",
    "pdb", "structure", "mol", "chem", "drug", "docking", "pharmac",
    "bio", "uniprot", "alphafold", "pdb", "rcsb", "ensembl",
    "phylo", "transcript", "rnaseq", "scRNA", "single_cell",
    "cell", "mutation", "variant", "snp", "pathway", "metabol",
    "neuro", "brain", "disease", "clinical", "phenotype",
    "opentarget", "chembl", "pubchem", "drugbank",
})

# Keywords that clearly indicate unrelated tools
UNRELATED_KEYWORDS: frozenset[str] = frozenset({
    "earthquake", "seismic", "weather", "climate", "geology",
    "astronomy", "telescope", "nasa", "satellite", "orbit",
    "stock", "finance", "trading", "cryptocurrency", "bitcoin",
    "social_media", "twitter", "instagram", "facebook",
    "game", "gaming", "sports", "recipe", "cooking",
    "real_estate", "housing", "movie", "music_streaming",
})

# Tool name patterns suggesting API key requirement
API_KEY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"REST$", r"_API$", r"Cloud", r"Google", r"AWS",
        r"Azure", r"OpenAI", r"Anthropic", r"Stripe",
        r"SendGrid", r"Twilio", r"Slack",
    ]
]

# Known services that require API keys
API_KEY_SERVICES: frozenset[str] = frozenset({
    "Google", "AWS", "Azure", "OpenAI", "Anthropic",
    "Stripe", "SendGrid", "Twilio", "Slack",
    "Firebase", "MongoDB_Atlas",
})

# Tools that are always safe (local computation or search-only)
ALWAYS_SAFE_TOOLS: frozenset[str] = frozenset({
    "tool_universe_find_tools",
    "fetch_uniprot_fasta",
    "fetch_alphafold_structure",
    "fetch_pdb_structure",
    "download_structure_file",
    "calculate_molecular_weight",
    "analyze_amino_acid_composition",
    "calculate_isoelectric_point",
    "extract_pdf_text_spacy_layout",
    "fetch_webpage_as_pdf_text",
    "download_uniprot_flat_file",
})


# ---------------------------------------------------------------------------
# ToolPolicy
# ---------------------------------------------------------------------------

class ToolPolicy:
    """Evaluates tool calls against safety and relevance rules.

    Three strictness levels control how aggressively tools are filtered:

    - **strict**: Only core bioinformatics tools are auto-approved.
      Everything else requires user approval.
    - **moderate** (default): Bio-relevant tools are auto-approved.
      Clearly unrelated tools are blocked. Ambiguous tools require approval.
    - **permissive**: All tools are allowed. Only API-key tools require approval.
    """

    def __init__(
        self,
        strictness: Literal["strict", "moderate", "permissive"] | None = None,
        extra_categories: frozenset[str] | None = None,
        available_api_keys: set[str] | None = None,
        stats: ToolPolicyStats | None = None,
    ):
        self.strictness = strictness or _STRICTNESS  # type: ignore[assignment]
        self.extra_categories = extra_categories or _EXTRA_CATS
        self.available_api_keys = available_api_keys or set()
        self.stats = stats or ToolPolicyStats()
        self._approved_this_session: set[str] = set()

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> ToolPolicyResult:
        """Evaluate whether a tool call should be allowed.

        Returns a ``ToolPolicyResult`` with:
        - ``allowed``: whether the tool can execute
        - ``reason``: human-readable explanation
        - ``requires_approval``: whether user must approve before execution
        - ``category``: detected category of the tool
        - ``risk_level``: estimated risk
        """
        if not tool_name or not tool_name.strip():
            return ToolPolicyResult(
                allowed=False,
                reason="Empty tool name",
                requires_approval=False,
                category="invalid",
                risk_level="high",
            )

        tool_name = tool_name.strip()

        # 1. Always-safe tools pass through
        if tool_name in ALWAYS_SAFE_TOOLS:
            self.stats.auto_approved += 1
            return ToolPolicyResult(
                allowed=True,
                reason="Always-safe local tool",
                requires_approval=False,
                category="local",
                risk_level="none",
            )

        # 2. Previously approved in this session
        if tool_name in self._approved_this_session:
            self.stats.auto_approved += 1
            return ToolPolicyResult(
                allowed=True,
                reason="Previously approved this session",
                requires_approval=False,
                category="session_approved",
                risk_level="none",
            )

        # 3. Check if tool requires API key
        requires_api_key = self._detect_api_key_requirement(tool_name)
        if requires_api_key:
            key_available = self._check_api_key_available(tool_name)
            if not key_available:
                self.stats.blocked += 1
                return ToolPolicyResult(
                    allowed=False,
                    reason=f"Tool requires API key that is not configured: {self._extract_service_name(tool_name)}",
                    requires_approval=False,
                    category="api_key_missing",
                    risk_level="high",
                )
            # Key is available but tool still needs approval (it hits external services).
            # user_approved is incremented later when mark_approved() is called.
            return ToolPolicyResult(
                allowed=True,
                reason="External API tool — API key available",
                requires_approval=True,
                category="api_tool",
                risk_level="medium",
            )

        # 4. Relevance check (only in strict and moderate modes)
        if self.strictness != "permissive":
            relevance = self._assess_relevance(tool_name)

            if self.strictness == "strict":
                if relevance == "unrelated":
                    self.stats.blocked += 1
                    return ToolPolicyResult(
                        allowed=False,
                        reason="Tool is unrelated to bioinformatics (strict mode)",
                        requires_approval=False,
                        category="unrelated",
                        risk_level="high",
                    )
                if relevance == "related":
                    self.stats.auto_approved += 1
                    return ToolPolicyResult(
                        allowed=True,
                        reason="Related tool (strict mode)",
                        requires_approval=False,
                        category="related",
                        risk_level="low",
                    )
                # "core" passes through
                self.stats.auto_approved += 1
                return ToolPolicyResult(
                    allowed=True,
                    reason="Core tool",
                    requires_approval=False,
                    category="core",
                    risk_level="none",
                )

            # moderate mode
            if relevance == "unrelated":
                self.stats.blocked += 1
                return ToolPolicyResult(
                    allowed=False,
                    reason="Tool appears unrelated to this project",
                    requires_approval=False,
                    category="unrelated",
                    risk_level="high",
                )

        # Permissive mode or moderate with non-unrelated tool
        self.stats.auto_approved += 1
        return ToolPolicyResult(
            allowed=True,
            reason="Auto-approved",
            requires_approval=False,
            category="approved",
            risk_level="none",
        )

    # ------------------------------------------------------------------
    # Discovery filtering
    # ------------------------------------------------------------------

    def filter_find_results(self, tools: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
        """Filter tool_universe_find_tools results by relevance.

        Removes clearly unrelated tools from discovery results so agents
        never see them in the first place.
        """
        if self.strictness == "permissive":
            return tools[:limit]

        filtered: list[dict[str, Any]] = []
        blocked_count = 0

        for tool_entry in tools:
            name = tool_entry.get("name", "")
            section = tool_entry.get("section", "")
            description = tool_entry.get("description", "")

            # Check if tool name contains unrelated keywords
            name_lower = name.lower()
            section_lower = section.lower()
            desc_lower = description.lower()

            is_unrelated = False

            # Check against unrelated keywords
            for kw in UNRELATED_KEYWORDS:
                if kw in name_lower or kw in section_lower:
                    is_unrelated = True
                    break

            # In strict mode, also check description
            if self.strictness == "strict" and not is_unrelated:
                relevance = self._assess_relevance(name, description)
                if relevance == "unrelated":
                    is_unrelated = True

            if is_unrelated:
                blocked_count += 1
                logger.debug("Filtered unrelated tool from discovery: %s", name)
                continue

            filtered.append(tool_entry)
            if len(filtered) >= limit:
                break

        if blocked_count > 0:
            self.stats.filtered_at_discovery += blocked_count

        return filtered

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def mark_approved(self, tool_name: str) -> None:
        """Mark a tool as approved for the rest of this session."""
        self._approved_this_session.add(tool_name)
        self.stats.user_approved += 1

    def reset_session(self) -> None:
        """Reset session-specific state (approved tools, stats)."""
        self._approved_this_session.clear()
        self.stats = ToolPolicyStats()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_api_key_requirement(self, tool_name: str) -> bool:
        """Heuristic: does this tool likely need an API key?"""
        name_upper = tool_name.upper()
        for service in API_KEY_SERVICES:
            if service.upper() in name_upper:
                return True
        for pattern in API_KEY_PATTERNS:
            if pattern.search(tool_name):
                return True
        return False

    def _check_api_key_available(self, tool_name: str) -> bool:
        """Check if the likely-required API key is configured."""
        service = self._extract_service_name(tool_name)
        if service == "Google":
            return bool(self.available_api_keys & {"GOOGLE_API_KEY", "GCP_API_KEY", "GOOGLE_CLOUD_API_KEY"})
        if service == "AWS":
            return bool(self.available_api_keys & {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"})
        if service == "Azure":
            return bool(self.available_api_keys & {"AZURE_API_KEY", "AZURE_SUBSCRIPTION_KEY"})
        if service == "OpenAI":
            return bool(self.available_api_keys & {"OPENAI_API_KEY"})
        if service == "Anthropic":
            return bool(self.available_api_keys & {"ANTHROPIC_API_KEY"})
        # Generic: if any key with the service name exists
        return any(service.upper() in k.upper() for k in self.available_api_keys)

    def _extract_service_name(self, tool_name: str) -> str:
        """Extract the likely service name from a tool name."""
        for service in API_KEY_SERVICES:
            if service.lower() in tool_name.lower():
                return service
        # Try pattern matching
        for pattern in API_KEY_PATTERNS:
            m = pattern.search(tool_name)
            if m:
                return m.group().strip("_")
        return "unknown"

    def _assess_relevance(self, tool_name: str, description: str = "") -> Literal["core", "related", "unrelated"]:
        """Assess how relevant a tool is to BioAgents.

        Returns:
            "core" — clearly bioinformatics/computational biology
            "related" — tangentially useful (ML, visualization, etc.)
            "unrelated" — clearly not relevant
        """
        name_lower = tool_name.lower()
        desc_lower = description.lower()

        # Check for clearly unrelated
        for kw in UNRELATED_KEYWORDS:
            if kw in name_lower:
                return "unrelated"

        # Check for core bio keywords in name
        for kw in BIO_KEYWORDS_IN_NAME:
            if kw in name_lower:
                return "core"

        # Check description if available
        if desc_lower:
            for kw in BIO_KEYWORDS_IN_NAME:
                if kw in desc_lower:
                    return "core"
            for kw in UNRELATED_KEYWORDS:
                if kw in desc_lower:
                    return "unrelated"

        # Default: in moderate mode, ambiguous tools are "related"
        return "related"


# ---------------------------------------------------------------------------
# Module-level singleton (used by default)
# ---------------------------------------------------------------------------

DEFAULT_POLICY = ToolPolicy()


def get_default_policy() -> ToolPolicy:
    """Return the module-level default policy instance."""
    return DEFAULT_POLICY
