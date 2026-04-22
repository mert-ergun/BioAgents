"""Drug-discovery scenario preset builders (Scenarios A, B, C, D).

Each scenario is a saved workflow template that wires upstream sources into the
shared downstream pipeline (design → parallel docking + Boltz → ADMET →
off-target → retrosynthesis → gates → decision log) described in doc section 2.

Input conventions (fed via ``initial_inputs`` on the source nodes):
  Scenario A (disease-first): ``disease_id``, ``target_uniprot``
  Scenario B (target-first):  ``uniprot_id``
  Scenario C (molecule-first): ``smiles``, ``target_uniprot``
  Scenario D (hit/lead optimization): ``uniprot_id`` (seeds auto-fetched)

All scenarios converge into the same gates/decision log. A gate fed with empty
data emits a benign borderline verdict so the log stays complete.
"""

from __future__ import annotations

from typing import Any

from bioagents.workflows.drug_discovery.admet_nodes import (
    AdmetLabStubNode,
    AdmetSarStubNode,
    EarlyAdmetTriageNode,
)
from bioagents.workflows.drug_discovery.design_nodes import (
    Reinvent4StubNode,
)
from bioagents.workflows.drug_discovery.fetch_nodes import (
    BindingDbLigandsForTargetNode,
    ChemblActivitiesForTargetNode,
    KeggPathwaysForGeneNode,
    OpenTargetsDiseaseTargetsNode,
    PdbEntriesForUniprotNode,
    ProgramBriefEntryNode,
    ReactomePathwaysForGeneNode,
    StringInteractorsNode,
    SwissTargetPredictionNode,
    TargetDossierAssembleNode,
    UniprotRecordNode,
)
from bioagents.workflows.drug_discovery.gate_nodes import (
    BoltzHitScreenGateNode,
    ConsensusGateNode,
    DecisionLogNode,
    DockingValidationGateNode,
    EarlyAdmetGateNode,
    OffTargetTier1GateNode,
    OffTargetTier2GateNode,
    RetrosynthesisGateNode,
    StructureReadinessGateNode,
)
from bioagents.workflows.drug_discovery.ligand_nodes import (
    LigandLibraryPrepNode,
    SeriesFromChemblActivitiesNode,
    SmilesStandardizerNode,
    SwissAdmeHeuristicNode,
)
from bioagents.workflows.drug_discovery.offtarget_nodes import (
    BlastSimilarityStubNode,
    FoldseekStubNode,
    OffTargetTier1AggregateNode,
    OffTargetTier2RefineStubNode,
    ProbisStubNode,
    SeaSearchStubNode,
)
from bioagents.workflows.drug_discovery.retrosynth_nodes import (
    AiZynthFinderStubNode,
    RetrosynthRouteReviewNode,
)
from bioagents.workflows.drug_discovery.structure_nodes import (
    BoltzComplexStubNode,
    ConsensusStructuralReviewNode,
    PlipInteractionsStubNode,
    RedockingValidationNode,
    StructureReadinessNode,
    VinaDockingStubNode,
)
from bioagents.workflows.graph import WorkflowGraph

# Standard node ids used throughout the scenarios so the API can reliably
# surface decision-log outputs to the UI.
DECISION_LOG_NODE_ID = "decision_log"
DOSSIER_NODE_ID = "target_dossier"


# ---------------------------------------------------------------------------
# Shared downstream pipeline
# ---------------------------------------------------------------------------


def _wire_downstream(
    g: WorkflowGraph,
    *,
    series_id: str,
    structure_id: str,
    uniprot_source_id: str,
    n_candidates: int = 6,
    thresholds: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Attach design, evaluation, gates, and decision log downstream.

    The caller has already added and wired the ``series_id`` node (emits
    ``series_package``), ``structure_id`` node (emits ``structure_package``),
    and ``uniprot_source_id`` (emits ``uniprot_id``).
    """
    th = thresholds or {}

    # Design expansion (ligand-based; SeriesPackage in → SeriesPackage out)
    g.add_node("design", Reinvent4StubNode(n_candidates=n_candidates))
    g.add_edge(series_id, "design", {"series_package": "series_package"})

    # Docking lane
    g.add_node("vina", VinaDockingStubNode())
    g.add_edge("design", "vina", {"series_package": "series_package"})
    g.add_edge(structure_id, "vina", {"structure_package": "structure_package"})

    g.add_node("plip", PlipInteractionsStubNode())
    g.add_edge("vina", "plip", {"docking_results": "docking_results"})

    g.add_node("redock_val", RedockingValidationNode())
    g.add_edge("vina", "redock_val", {"docking_results": "docking_results"})
    g.add_edge(structure_id, "redock_val", {"structure_package": "structure_package"})

    # Boltz lane
    g.add_node("boltz", BoltzComplexStubNode())
    g.add_edge("design", "boltz", {"series_package": "series_package"})
    g.add_edge(structure_id, "boltz", {"structure_package": "structure_package"})

    # Consensus
    g.add_node("consensus", ConsensusStructuralReviewNode())
    g.add_edge("vina", "consensus", {"docking_results": "docking_results"})
    g.add_edge("boltz", "consensus", {"boltz_results": "boltz_results"})
    g.add_edge(structure_id, "consensus", {"structure_package": "structure_package"})

    # ADMET
    g.add_node("swissadme", SwissAdmeHeuristicNode())
    g.add_edge("design", "swissadme", {"series_package": "series_package"})
    g.add_node("admet_triage", EarlyAdmetTriageNode())
    g.add_edge("swissadme", "admet_triage", {"admet_report": "admet_report"})
    g.add_edge("design", "admet_triage", {"series_package": "series_package"})
    g.add_node("admetlab", AdmetLabStubNode())
    g.add_edge("design", "admetlab", {"series_package": "series_package"})
    g.add_node("admetsar", AdmetSarStubNode())
    g.add_edge("design", "admetsar", {"series_package": "series_package"})

    # Off-target Tier 1 lanes — fetch_sequence=False to keep scenarios fast/offline-safe
    g.add_node("blast", BlastSimilarityStubNode(fetch_sequence=False))
    g.add_edge(uniprot_source_id, "blast", {"uniprot_id": "uniprot_id"})
    g.add_node("foldseek", FoldseekStubNode())
    g.add_edge(uniprot_source_id, "foldseek", {"uniprot_id": "uniprot_id"})
    g.add_node("probis", ProbisStubNode())
    g.add_edge(uniprot_source_id, "probis", {"uniprot_id": "uniprot_id"})
    g.add_node("sea", SeaSearchStubNode())
    g.add_edge("design", "sea", {"series_package": "series_package"})
    g.add_node("ot1", OffTargetTier1AggregateNode())
    g.add_edge("blast", "ot1", {"neighbors": "blast_neighbors"})
    g.add_edge("foldseek", "ot1", {"neighbors": "foldseek_neighbors"})
    g.add_edge("probis", "ot1", {"neighbors": "probis_neighbors"})
    g.add_edge("sea", "ot1", {"neighbors": "sea_neighbors"})

    # Off-target Tier 2 refinement
    g.add_node("ot2", OffTargetTier2RefineStubNode())
    g.add_edge("ot1", "ot2", {"offtarget_panel": "offtarget_panel"})
    g.add_edge("vina", "ot2", {"docking_results": "docking_results"})
    g.add_edge("design", "ot2", {"series_package": "series_package"})

    # Retrosynthesis
    g.add_node("aizynth", AiZynthFinderStubNode())
    g.add_edge("design", "aizynth", {"series_package": "series_package"})
    g.add_node("retro_review", RetrosynthRouteReviewNode())
    g.add_edge("aizynth", "retro_review", {"routes": "routes"})

    # Gates (all receive per-gate threshold overrides if provided)
    g.add_node(
        "gate_structure",
        StructureReadinessGateNode(th.get("structure_readiness")),
    )
    g.add_edge(structure_id, "gate_structure", {"structure_package": "structure_package"})
    g.add_node("gate_docking", DockingValidationGateNode(th.get("docking_validation")))
    g.add_edge("redock_val", "gate_docking", {"validation": "validation"})
    g.add_node("gate_boltz", BoltzHitScreenGateNode(th.get("boltz_hit_screen")))
    g.add_edge("boltz", "gate_boltz", {"boltz_results": "boltz_results"})
    g.add_node("gate_consensus", ConsensusGateNode(th.get("consensus_structural_review")))
    g.add_edge("consensus", "gate_consensus", {"consensus": "consensus"})
    g.add_node("gate_admet", EarlyAdmetGateNode(th.get("early_admet")))
    g.add_edge("admet_triage", "gate_admet", {"triage": "triage"})
    g.add_node("gate_ot1", OffTargetTier1GateNode(th.get("off_target_tier1")))
    g.add_edge("ot1", "gate_ot1", {"offtarget_panel": "offtarget_panel"})
    g.add_node("gate_ot2", OffTargetTier2GateNode(th.get("off_target_tier2")))
    g.add_edge("ot2", "gate_ot2", {"tier2_report": "tier2_report"})
    g.add_node("gate_retro", RetrosynthesisGateNode(th.get("retrosynthesis")))
    g.add_edge("retro_review", "gate_retro", {"route_summary": "route_summary"})

    # Decision log aggregator
    g.add_node(DECISION_LOG_NODE_ID, DecisionLogNode())
    g.add_edge("gate_structure", DECISION_LOG_NODE_ID, {"verdicts": "structure_readiness"})
    g.add_edge("gate_docking", DECISION_LOG_NODE_ID, {"verdicts": "docking_validation"})
    g.add_edge("gate_boltz", DECISION_LOG_NODE_ID, {"verdicts": "boltz_hit_screen"})
    g.add_edge("gate_consensus", DECISION_LOG_NODE_ID, {"verdicts": "consensus"})
    g.add_edge("gate_admet", DECISION_LOG_NODE_ID, {"verdicts": "early_admet"})
    g.add_edge("gate_ot1", DECISION_LOG_NODE_ID, {"verdicts": "offtarget_tier1"})
    g.add_edge("gate_ot2", DECISION_LOG_NODE_ID, {"verdicts": "offtarget_tier2"})
    g.add_edge("gate_retro", DECISION_LOG_NODE_ID, {"verdicts": "retrosynthesis"})


def _wire_target_research(
    g: WorkflowGraph,
    *,
    uniprot_source_id: str,
    include_pdb: bool = True,
    include_bindingdb: bool = False,
) -> tuple[str, str]:
    """Wire UniProt + Reactome/KEGG/STRING + PDB + ChEMBL → TargetDossier + PDB list.

    Returns (pdb_entries_node_id, chembl_activities_node_id).
    """
    # Pathways + interactors use gene_symbol emitted by UniprotRecordNode.
    g.add_node("reactome", ReactomePathwaysForGeneNode())
    g.add_edge(uniprot_source_id, "reactome", {"gene_symbol": "gene_symbol"})
    g.add_node("kegg", KeggPathwaysForGeneNode())
    g.add_edge(uniprot_source_id, "kegg", {"gene_symbol": "gene_symbol"})
    g.add_node("string", StringInteractorsNode())
    g.add_edge(uniprot_source_id, "string", {"gene_symbol": "gene_symbol"})

    # ChEMBL activities for the target (used as a seed source for series).
    g.add_node("chembl_acts", ChemblActivitiesForTargetNode(max_rows=10))
    g.add_edge(uniprot_source_id, "chembl_acts", {"uniprot_id": "uniprot_id"})

    if include_bindingdb:
        g.add_node("bindingdb", BindingDbLigandsForTargetNode(max_rows=10))
        g.add_edge(uniprot_source_id, "bindingdb", {"uniprot_id": "uniprot_id"})

    # PDB entries for the target.
    if include_pdb:
        g.add_node("pdb_entries", PdbEntriesForUniprotNode(max_entries=5))
        g.add_edge(uniprot_source_id, "pdb_entries", {"uniprot_id": "uniprot_id"})
    else:
        g.add_node("pdb_entries", PdbEntriesForUniprotNode(max_entries=1))
        g.add_edge(uniprot_source_id, "pdb_entries", {"uniprot_id": "uniprot_id"})

    # Target dossier aggregator (informational).
    # Reactome is used for pathways — KEGG output is kept inside its own node.
    g.add_node(DOSSIER_NODE_ID, TargetDossierAssembleNode())
    g.add_edge(
        uniprot_source_id,
        DOSSIER_NODE_ID,
        {"uniprot_record": "uniprot_record"},
    )
    g.add_edge("reactome", DOSSIER_NODE_ID, {"pathways": "pathways"})
    g.add_edge("string", DOSSIER_NODE_ID, {"interactors": "interactors"})
    g.add_edge("pdb_entries", DOSSIER_NODE_ID, {"pdb_entries": "pdb_entries"})
    g.add_edge("chembl_acts", DOSSIER_NODE_ID, {"activities": "activities"})

    return ("pdb_entries", "chembl_acts")


# ---------------------------------------------------------------------------
# Scenario A: Disease-first
# ---------------------------------------------------------------------------


def build_scenario_a_disease_first(opts: dict[str, Any]) -> WorkflowGraph:
    """Disease-first: disease_id → target panel + full pipeline for target_uniprot."""
    n_candidates = max(1, min(32, int(opts.get("n_candidates", 6))))
    thresholds = opts.get("thresholds") or {}

    g = WorkflowGraph()
    # Entry + target panel (informational)
    g.add_node(
        "brief",
        ProgramBriefEntryNode(
            entry_mode="disease_first",
            objective=str(opts.get("objective", "")),
        ),
    )
    g.add_node("disease_targets", OpenTargetsDiseaseTargetsNode(max_targets=5))
    g.add_edge("brief", "disease_targets", {"program_brief": "program_brief"})

    # Target-centric sources (user-supplied target_uniprot)
    g.add_node("uni_record", UniprotRecordNode())
    _pdb_id, chembl_id = _wire_target_research(
        g, uniprot_source_id="uni_record", include_bindingdb=False
    )

    # Series from ChEMBL activities → LigandLibraryPrep
    g.add_node("series_seeds", SeriesFromChemblActivitiesNode(max_seeds=5))
    g.add_edge(chembl_id, "series_seeds", {"activities": "activities"})
    g.add_node(
        "series",
        LigandLibraryPrepNode(
            design_lane="ligand_based",
            objective=str(opts.get("objective", "")),
        ),
    )
    g.add_edge("series_seeds", "series", {"seed_smiles": "seed_smiles"})

    # Structure readiness from PDB entries
    g.add_node("structure", StructureReadinessNode())
    g.add_edge("pdb_entries", "structure", {"pdb_entries": "pdb_entries"})

    _wire_downstream(
        g,
        series_id="series",
        structure_id="structure",
        uniprot_source_id="uni_record",
        n_candidates=n_candidates,
        thresholds=thresholds,
    )
    return g


# ---------------------------------------------------------------------------
# Scenario B: Target-first
# ---------------------------------------------------------------------------


def build_scenario_b_target_first(opts: dict[str, Any]) -> WorkflowGraph:
    """Target-first: uniprot_id → full pipeline using ChEMBL seeds + PDB structure."""
    n_candidates = max(1, min(32, int(opts.get("n_candidates", 6))))
    thresholds = opts.get("thresholds") or {}

    g = WorkflowGraph()
    g.add_node(
        "brief",
        ProgramBriefEntryNode(
            entry_mode="target_first",
            objective=str(opts.get("objective", "")),
        ),
    )
    g.add_node("uni_record", UniprotRecordNode())
    _pdb_id, chembl_id = _wire_target_research(
        g, uniprot_source_id="uni_record", include_bindingdb=True
    )

    # Series from ChEMBL activities
    g.add_node("series_seeds", SeriesFromChemblActivitiesNode(max_seeds=5))
    g.add_edge(chembl_id, "series_seeds", {"activities": "activities"})
    g.add_node(
        "series",
        LigandLibraryPrepNode(
            design_lane="ligand_based",
            objective=str(opts.get("objective", "")),
        ),
    )
    g.add_edge("series_seeds", "series", {"seed_smiles": "seed_smiles"})

    # Structure
    g.add_node("structure", StructureReadinessNode())
    g.add_edge("pdb_entries", "structure", {"pdb_entries": "pdb_entries"})

    _wire_downstream(
        g,
        series_id="series",
        structure_id="structure",
        uniprot_source_id="uni_record",
        n_candidates=n_candidates,
        thresholds=thresholds,
    )
    return g


# ---------------------------------------------------------------------------
# Scenario C: Molecule-first
# ---------------------------------------------------------------------------


def build_scenario_c_molecule_first(opts: dict[str, Any]) -> WorkflowGraph:
    """Molecule-first: smiles + target_uniprot → predicted targets + pipeline."""
    n_candidates = max(1, min(32, int(opts.get("n_candidates", 6))))
    thresholds = opts.get("thresholds") or {}

    g = WorkflowGraph()
    g.add_node(
        "brief",
        ProgramBriefEntryNode(
            entry_mode="molecule_first",
            objective=str(opts.get("objective", "")),
        ),
    )
    # Standardize the seed SMILES
    g.add_node("seed_std", SmilesStandardizerNode())
    # Predict ligand-based targets (informational)
    g.add_node("swiss_targets", SwissTargetPredictionNode(max_targets=5))
    g.add_edge("seed_std", "swiss_targets", {"standardized_smiles": "smiles"})

    # Series from the standardized seed — use LigandLibraryPrep with list input.
    # We need an adapter: seed_std emits standardized_smiles as str, but
    # LigandLibraryPrep wants seed_smiles as list. We wrap via a tiny dedicated
    # adapter node — but to avoid introducing yet another node, we feed
    # ``series`` directly from ``initial_inputs`` and chain SmilesStandardizerNode
    # into the design stage via a second connection.
    # For V1, we keep it simple: run LigandLibraryPrep as a source taking
    # ``seed_smiles`` list from ``initial_inputs`` (same list as the user's SMILES).
    g.add_node(
        "series",
        LigandLibraryPrepNode(
            design_lane="ligand_based",
            objective=str(opts.get("objective", "")),
        ),
    )

    # Target-side fetches (user-supplied target_uniprot)
    g.add_node("uni_record", UniprotRecordNode())
    _wire_target_research(g, uniprot_source_id="uni_record", include_bindingdb=False)

    # Structure
    g.add_node("structure", StructureReadinessNode())
    g.add_edge("pdb_entries", "structure", {"pdb_entries": "pdb_entries"})

    _wire_downstream(
        g,
        series_id="series",
        structure_id="structure",
        uniprot_source_id="uni_record",
        n_candidates=n_candidates,
        thresholds=thresholds,
    )
    return g


# ---------------------------------------------------------------------------
# Scenario D: Hit/Lead optimization loop (fixed N rounds)
# ---------------------------------------------------------------------------


def build_scenario_d_optimization_loop(opts: dict[str, Any]) -> WorkflowGraph:
    """Optimization loop: uniprot_id → ChEMBL seeds → unrolled N-round design.

    The workflow executor lacks native looping; we unroll a fixed number of
    optimization rounds as a linear chain of design nodes. Each round expands
    the previous SeriesPackage; the final round feeds the shared downstream
    evaluation + gates + decision log.
    """
    n_candidates = max(1, min(32, int(opts.get("n_candidates", 6))))
    rounds = max(1, min(4, int(opts.get("rounds", 2))))
    thresholds = opts.get("thresholds") or {}

    g = WorkflowGraph()
    g.add_node(
        "brief",
        ProgramBriefEntryNode(
            entry_mode="target_first",
            objective=str(opts.get("objective", "hit_lead_optimization")),
        ),
    )
    g.add_node("uni_record", UniprotRecordNode())
    _pdb_id, chembl_id = _wire_target_research(
        g, uniprot_source_id="uni_record", include_bindingdb=True
    )

    # Initial series from ChEMBL activities.
    g.add_node("series_seeds", SeriesFromChemblActivitiesNode(max_seeds=5))
    g.add_edge(chembl_id, "series_seeds", {"activities": "activities"})
    g.add_node(
        "series",
        LigandLibraryPrepNode(
            design_lane="ligand_based",
            objective=str(opts.get("objective", "hit_lead_optimization")),
        ),
    )
    g.add_edge("series_seeds", "series", {"seed_smiles": "seed_smiles"})

    # Unroll N optimization rounds; each round expands the previous series.
    prev = "series"
    for i in range(rounds):
        nid = f"opt_round_{i}"
        g.add_node(nid, Reinvent4StubNode(n_candidates=n_candidates, seed=i + 1))
        g.add_edge(prev, nid, {"series_package": "series_package"})
        prev = nid

    # Structure
    g.add_node("structure", StructureReadinessNode())
    g.add_edge("pdb_entries", "structure", {"pdb_entries": "pdb_entries"})

    _wire_downstream(
        g,
        series_id=prev,
        structure_id="structure",
        uniprot_source_id="uni_record",
        n_candidates=n_candidates,
        thresholds=thresholds,
    )
    return g


# ---------------------------------------------------------------------------
# Scenario metadata (exposed to the frontend)
# ---------------------------------------------------------------------------


SCENARIO_FORM_FIELDS: dict[str, list[dict[str, Any]]] = {
    "dd_scenario_disease_first": [
        {
            "key": "disease_id",
            "label": "EFO disease identifier",
            "example": "EFO_0000311",
            "type": "string",
            "required": True,
        },
        {
            "key": "target_uniprot",
            "label": "Target UniProt accession",
            "example": "P04637",
            "type": "string",
            "required": True,
        },
    ],
    "dd_scenario_target_first": [
        {
            "key": "uniprot_id",
            "label": "Target UniProt accession",
            "example": "P04637",
            "type": "string",
            "required": True,
        },
    ],
    "dd_scenario_molecule_first": [
        {
            "key": "seed_smiles",
            "label": "Seed SMILES (comma-separated for multiple)",
            "example": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "type": "string",
            "required": True,
        },
        {
            "key": "target_uniprot",
            "label": "Intended target UniProt accession",
            "example": "P23219",
            "type": "string",
            "required": True,
        },
    ],
    "dd_scenario_optimization_loop": [
        {
            "key": "uniprot_id",
            "label": "Target UniProt accession",
            "example": "P00533",
            "type": "string",
            "required": True,
        },
    ],
}


# Maps a scenario id → list of (form_key, node_id, node_input_key, transform) tuples
# where transform can be "str", "upper", "smiles_list" or "first_smiles".
SCENARIO_INPUT_MAPPING: dict[str, list[tuple[str, str, str, str]]] = {
    "dd_scenario_disease_first": [
        ("disease_id", "brief", "primary_id", "str"),
        ("target_uniprot", "uni_record", "uniprot_id", "upper"),
    ],
    "dd_scenario_target_first": [
        ("uniprot_id", "brief", "primary_id", "upper"),
        ("uniprot_id", "uni_record", "uniprot_id", "upper"),
    ],
    "dd_scenario_molecule_first": [
        ("seed_smiles", "brief", "primary_id", "first_smiles"),
        ("seed_smiles", "seed_std", "smiles", "first_smiles"),
        ("seed_smiles", "series", "seed_smiles", "smiles_list"),
        ("target_uniprot", "uni_record", "uniprot_id", "upper"),
    ],
    "dd_scenario_optimization_loop": [
        ("uniprot_id", "brief", "primary_id", "upper"),
        ("uniprot_id", "uni_record", "uniprot_id", "upper"),
    ],
}


def build_scenario_initial_inputs(
    scenario_id: str, form_values: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Translate user-facing form values into per-node ``initial_inputs``."""
    mapping = SCENARIO_INPUT_MAPPING.get(scenario_id)
    if not mapping:
        raise ValueError(f"Unknown scenario id: {scenario_id!r}")
    out: dict[str, dict[str, Any]] = {}
    for form_key, node_id, node_input, transform in mapping:
        raw = form_values.get(form_key)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            continue
        if transform == "upper":
            value: Any = str(raw).strip().upper()
        elif transform == "smiles_list":
            if isinstance(raw, list):
                value = [str(s).strip() for s in raw if str(s).strip()]
            else:
                value = [s.strip() for s in str(raw).replace(";", ",").split(",") if s.strip()]
        elif transform == "first_smiles":
            if isinstance(raw, list):
                value = str(raw[0]).strip() if raw else ""
            else:
                first = str(raw).replace(";", ",").split(",")[0]
                value = first.strip()
        else:
            value = str(raw).strip()
        out.setdefault(node_id, {})[node_input] = value
    return out


# Backwards-compatible "source spec" used by :class:`PresetEntry`. The list is
# purely informational here since drug-discovery scenarios use form fields; we
# still populate a minimal single-source entry so the generic preset catalog
# treats these presets consistently.
SCENARIO_SOURCE_SPECS: dict[str, list[dict[str, Any]]] = {
    sid: [
        {
            "node_id": SCENARIO_INPUT_MAPPING[sid][0][1],
            "input_key": SCENARIO_INPUT_MAPPING[sid][0][2],
            "label": SCENARIO_FORM_FIELDS[sid][0]["label"],
        }
    ]
    for sid in SCENARIO_FORM_FIELDS
}


SCENARIO_OPTIONS_HELP: dict[str, dict[str, Any]] = {
    "dd_scenario_disease_first": {
        "objective": {"default": "", "type": "str"},
        "n_candidates": {"default": 6, "min": 1, "max": 32},
    },
    "dd_scenario_target_first": {
        "objective": {"default": "", "type": "str"},
        "n_candidates": {"default": 6, "min": 1, "max": 32},
    },
    "dd_scenario_molecule_first": {
        "objective": {"default": "", "type": "str"},
        "n_candidates": {"default": 6, "min": 1, "max": 32},
    },
    "dd_scenario_optimization_loop": {
        "objective": {"default": "hit_lead_optimization", "type": "str"},
        "n_candidates": {"default": 6, "min": 1, "max": 32},
        "rounds": {"default": 2, "min": 1, "max": 4},
    },
}


SCENARIO_INFO: dict[str, dict[str, Any]] = {
    "dd_scenario_disease_first": {
        "name": "Scenario A — Disease-first",
        "description": (
            "Start from a disease (EFO id). Rank targets via Open Targets and "
            "run the full small-molecule pipeline for the user's selected "
            "target UniProt. Produces a ranked target panel + decision log."
        ),
        "builder": build_scenario_a_disease_first,
    },
    "dd_scenario_target_first": {
        "name": "Scenario B — Target-first",
        "description": (
            "Start from a known target UniProt. Fetches UniProt + Reactome + "
            "STRING + ChEMBL + PDB, seeds a ligand series from ChEMBL "
            "activities, and runs design → docking + Boltz → gates."
        ),
        "builder": build_scenario_b_target_first,
    },
    "dd_scenario_molecule_first": {
        "name": "Scenario C — Molecule-first",
        "description": (
            "Start from a seed SMILES + intended target UniProt. Predicts "
            "ligand-based targets, expands the series, and runs the shared "
            "structural/ADMET/off-target/retrosynthesis gates."
        ),
        "builder": build_scenario_c_molecule_first,
    },
    "dd_scenario_optimization_loop": {
        "name": "Scenario D — Hit/Lead optimization loop",
        "description": (
            "Unrolled N-round optimization: ChEMBL seeds → repeated REINVENT4 "
            "expansion → parallel docking + Boltz → gates → decision log."
        ),
        "builder": build_scenario_d_optimization_loop,
    },
}


__all__ = [
    "DECISION_LOG_NODE_ID",
    "DOSSIER_NODE_ID",
    "SCENARIO_FORM_FIELDS",
    "SCENARIO_INFO",
    "SCENARIO_INPUT_MAPPING",
    "SCENARIO_OPTIONS_HELP",
    "SCENARIO_SOURCE_SPECS",
    "build_scenario_a_disease_first",
    "build_scenario_b_target_first",
    "build_scenario_c_molecule_first",
    "build_scenario_d_optimization_loop",
    "build_scenario_initial_inputs",
]
