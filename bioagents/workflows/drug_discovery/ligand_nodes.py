"""Ligand / chemistry nodes backed by RDKit where possible."""

from __future__ import annotations

from typing import Any, ClassVar

from bioagents.workflows.drug_discovery._rdkit_utils import (
    compute_descriptors,
    pains_alert_count,
    rdkit_available,
    safe_mol_from_smiles,
    standardize_smiles,
    synthetic_complexity_score,
)
from bioagents.workflows.drug_discovery.schemas import SERIES_PACKAGE_TAG
from bioagents.workflows.node import WorkflowNode
from bioagents.workflows.schemas import NodeMetadata


class SmilesStandardizerNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "dd_smiles_standardizer"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="SMILES standardizer (RDKit)",
            description="Canonicalize a SMILES string; returns empty when invalid.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"smiles": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"standardized_smiles": "str", "valid": "bool"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        raw = str(inputs.get("smiles", "")).strip()
        canon = standardize_smiles(raw) if raw else None
        return {"standardized_smiles": canon or "", "valid": canon is not None}


class LigandLibraryPrepNode(WorkflowNode):
    """Assemble a SeriesPackage: standardize seeds and attach basic descriptors."""

    workflow_type_id: ClassVar[str] = "dd_ligand_library_prep"

    def __init__(self, design_lane: str = "ligand_based", objective: str = "") -> None:
        self._design_lane = design_lane.strip() or "ligand_based"
        self._objective = objective.strip()

    @property
    def params(self) -> dict[str, Any]:
        return {"design_lane": self._design_lane, "objective": self._objective}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Ligand library prep",
            description="Standardize a list of seed SMILES and compute basic descriptors.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"seed_smiles": "list"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"series_package": SERIES_PACKAGE_TAG}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        raw = inputs.get("seed_smiles") or []
        if isinstance(raw, str):
            raw = [raw]
        seeds: list[str] = []
        analogues: list[dict[str, Any]] = []
        for s in raw:
            if not isinstance(s, str) or not s.strip():
                continue
            canon = standardize_smiles(s.strip()) or s.strip()
            seeds.append(canon)
            analogues.append(
                {
                    "smiles": canon,
                    "origin": "seed",
                    "descriptors": compute_descriptors(canon),
                }
            )
        pkg: dict[str, Any] = {
            "seed_smiles": seeds,
            "analogues": analogues,
            "objective": self._objective,
            "design_lane": self._design_lane,
            "library_prep_notes": "RDKit canonicalization" if rdkit_available() else "no RDKit",
        }
        return {"series_package": pkg}


class SeriesFromChemblActivitiesNode(WorkflowNode):
    """Adapter: build ``seed_smiles`` from a ChEMBL activities list."""

    workflow_type_id: ClassVar[str] = "dd_series_from_activities"

    def __init__(self, max_seeds: int = 10) -> None:
        self._max = max(1, min(50, int(max_seeds)))

    @property
    def params(self) -> dict[str, Any]:
        return {"max_seeds": self._max}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Seeds from ChEMBL activities",
            description="Extract canonical SMILES seeds from ChEMBL activity rows.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"activities": "list"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"seed_smiles": "list"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        rows = inputs.get("activities") or []
        seeds: list[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            s = row.get("canonical_smiles")
            if isinstance(s, str) and s.strip():
                seeds.append(s.strip())
                if len(seeds) >= self._max:
                    break
        return {"seed_smiles": seeds}


class SwissAdmeHeuristicNode(WorkflowNode):
    """Local Lipinski/Veber-style ADMET triage computed via RDKit descriptors.

    Labeled as heuristic in node metadata — not the real SwissADME API.
    """

    workflow_type_id: ClassVar[str] = "dd_swissadme_heuristic"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="SwissADME heuristic (RDKit)",
            description="Lipinski + Veber + PAINS flags from RDKit descriptors.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"series_package": SERIES_PACKAGE_TAG}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"admet_report": "dict"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        pkg = inputs.get("series_package") or {}
        analogues = pkg.get("analogues") if isinstance(pkg, dict) else None
        if not isinstance(analogues, list):
            analogues = []
        rows: list[dict[str, Any]] = []
        for a in analogues:
            if not isinstance(a, dict):
                continue
            smiles = a.get("smiles")
            if not isinstance(smiles, str):
                continue
            desc = a.get("descriptors") or compute_descriptors(smiles)
            if not desc:
                rows.append(
                    {
                        "smiles": smiles,
                        "liabilities": ["rdkit_unavailable"],
                        "major_liabilities": 1,
                        "descriptors": {},
                    }
                )
                continue
            liabilities: list[str] = []
            if desc["molecular_weight"] > 500:
                liabilities.append("mw_over_500")
            if desc["logp"] > 5:
                liabilities.append("logp_over_5")
            if desc["hbd"] > 5:
                liabilities.append("hbd_over_5")
            if desc["hba"] > 10:
                liabilities.append("hba_over_10")
            if desc["tpsa"] > 140:
                liabilities.append("tpsa_over_140")
            if desc["rotatable_bonds"] > 10:
                liabilities.append("rotbonds_over_10")
            pains = pains_alert_count(smiles)
            if pains:
                liabilities.append(f"pains_alerts:{pains}")
            # Major liabilities (Ro5 + PAINS).
            major = [
                lb
                for lb in liabilities
                if lb.startswith(("mw_", "logp_", "hbd_", "hba_", "pains_"))
            ]
            rows.append(
                {
                    "smiles": smiles,
                    "descriptors": desc,
                    "liabilities": liabilities,
                    "major_liabilities": len(major),
                    "pains_alerts": pains,
                }
            )
        worst = max((r["major_liabilities"] for r in rows), default=0)
        return {
            "admet_report": {
                "rows": rows,
                "worst_major_liabilities": int(worst),
                "total_evaluated": len(rows),
            }
        }


class RascoreHeuristicNode(WorkflowNode):
    """Synthesizability prescreen using an RDKit-based complexity heuristic."""

    workflow_type_id: ClassVar[str] = "dd_rascore_heuristic"

    @property
    def params(self) -> dict[str, Any]:
        return {}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="RAscore (heuristic)",
            description="Heuristic synthesizability prescreen; 0=hard, 1=easy.",
            version="1.0.0",
            category="model",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"series_package": SERIES_PACKAGE_TAG}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"prescreen": "dict", "stub": "bool"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        pkg = inputs.get("series_package") or {}
        rows: list[dict[str, Any]] = []
        if isinstance(pkg, dict):
            for a in pkg.get("analogues") or []:
                if not isinstance(a, dict):
                    continue
                s = a.get("smiles")
                if not isinstance(s, str):
                    continue
                rows.append({"smiles": s, "score": synthetic_complexity_score(s)})
        scores: list[float] = [float(r["score"]) for r in rows]
        worst: float = min(scores, default=0.0)
        best: float = max(scores, default=0.0)
        return {
            "prescreen": {
                "rows": rows,
                "worst_score": worst,
                "best_score": best,
            },
            "stub": True,
        }


def _safe_smi(s: str) -> str:
    mol = safe_mol_from_smiles(s)
    if mol is None:
        return s
    from rdkit import Chem

    try:
        smi: str = Chem.MolToSmiles(mol, canonical=True)
        return smi
    except Exception:
        return s


LIGAND_NODES: list[type[WorkflowNode]] = [
    SmilesStandardizerNode,
    LigandLibraryPrepNode,
    SeriesFromChemblActivitiesNode,
    SwissAdmeHeuristicNode,
    RascoreHeuristicNode,
]
