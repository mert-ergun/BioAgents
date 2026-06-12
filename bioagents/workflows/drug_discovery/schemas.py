"""Shared schemas and port-type tags for the drug discovery workflow family.

All segments in the reference guide ("program brief", "target dossier", etc.) flow as
plain dicts on workflow edges. We expose them as ``TypedDict`` shapes for readability
and expose canonical port-type tag strings so ``types_compatible`` in
:mod:`bioagents.workflows.schemas` gates edge wiring.
"""

from __future__ import annotations

from typing import Any, TypedDict

PROGRAM_BRIEF_TAG = "program_brief"
TARGET_DOSSIER_TAG = "target_dossier"
STRUCTURE_PACKAGE_TAG = "structure_package"
SERIES_PACKAGE_TAG = "series_package"
GATE_VERDICT_TAG = "gate_verdict"
DECISION_LOG_TAG = "decision_log"

# Verdict vocabulary used by every gate node.
VERDICT_PASS = "pass"  # nosec B105 - verdict label, not a credential
VERDICT_BORDERLINE = "borderline"
VERDICT_FAIL = "fail"

# Promotion tiers from the reference guide section 2.7.
TIER_REJECT = "reject"
TIER_HIT = "hit"
TIER_OPTIMIZABLE_HIT = "optimizable_hit"
TIER_LEAD = "lead"
TIER_DE_RISKED_LEAD = "de_risked_lead"


class ProgramBrief(TypedDict, total=False):
    """Captures how a program enters the workflow (doc section 1.3)."""

    entry_mode: str  # "disease_first" | "target_first" | "molecule_first"
    objective: str
    disease_id: str
    disease_label: str
    target_uniprot: str
    target_symbol: str
    seed_smiles: list[str]
    constraints: dict[str, Any]


class TargetDossier(TypedDict, total=False):
    """Target-research product (doc section 1.3)."""

    uniprot_id: str
    gene_symbol: str
    protein_name: str
    organism: str
    function_summary: str
    pathways: list[dict[str, Any]]
    interactors: list[dict[str, Any]]
    structures: list[dict[str, Any]]
    known_ligands: list[dict[str, Any]]
    tractability_notes: str


class StructurePackage(TypedDict, total=False):
    """Structure readiness output (doc section 1.4 - "Structure readiness")."""

    kind: str  # "holo" | "apo" | "predicted" | "none"
    pdb_id: str
    alphafold_id: str
    pocket_definition: dict[str, Any]
    native_ligand_smiles: str
    preparation_notes: str
    confidence_metric: float


class SeriesPackage(TypedDict, total=False):
    """Ligand series seed + generated analogues (doc section 1.3/1.4)."""

    seed_smiles: list[str]
    analogues: list[dict[str, Any]]  # [{smiles, origin, score, ...}]
    objective: str
    design_lane: str
    library_prep_notes: str


class GateVerdict(TypedDict):
    """One pass/fail decision attached to a candidate (doc section 2.6)."""

    gate: str
    verdict: str  # pass | borderline | fail
    reason_code: str
    reason_text: str
    metrics: dict[str, Any]
    thresholds: dict[str, Any]
    candidate_id: str  # molecule / series identifier or "__global__"


class DecisionLogEntry(TypedDict, total=False):
    """One row in the audit-trail decision log (doc section 2.7)."""

    candidate_id: str
    tier: str
    gates: list[GateVerdict]
    summary: str


def default_policy_thresholds() -> dict[str, dict[str, Any]]:
    """Default thresholds from the reference guide section 2.6.

    UI and ``run_custom_workflow`` can override any of these per-node via the
    ``thresholds`` param on each gate node.
    """
    return {
        "structure_readiness": {
            "accept_kinds": ["holo"],
            "borderline_kinds": ["apo", "predicted"],
        },
        "docking_validation": {
            "pass_rmsd_max": 2.0,
            "borderline_rmsd_max": 3.0,
            "require_key_contacts": True,
        },
        "boltz_hit_screen": {
            "pass_binder_probability": 0.70,
            "borderline_binder_probability": 0.50,
        },
        "consensus_structural_review": {
            "min_agreeing_methods": 2,
        },
        "early_admet": {
            "max_major_liabilities": 0,
            "borderline_major_liabilities": 1,
        },
        "off_target_tier1": {
            "max_critical_flags": 0,
            "borderline_critical_flags": 1,
        },
        "off_target_tier2": {
            "required_intended_margin": 0.1,
        },
        "retrosynthesis": {
            "min_route_score": 0.5,
            "max_step_count": 8,
        },
    }
