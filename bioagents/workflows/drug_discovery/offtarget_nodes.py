"""Off-target analysis nodes.

Tier 1 uses deterministic stubs (seeded from UniProt + SMILES) with a real
UniProt sequence fetch when possible. Tier 2 re-runs the stub docking lane on
the top off-targets and computes a margin vs. the intended target.
"""

from __future__ import annotations

from typing import Any, ClassVar

import requests

from bioagents.workflows.drug_discovery._rdkit_utils import stable_seed
from bioagents.workflows.drug_discovery.schemas import SERIES_PACKAGE_TAG
from bioagents.workflows.node import WorkflowNode
from bioagents.workflows.schemas import NodeMetadata


def _fetch_uniprot_sequence(uniprot_id: str, timeout: int = 15) -> str:
    acc = (uniprot_id or "").strip().upper()
    if not acc:
        return ""
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        lines = [ln for ln in r.text.splitlines() if ln and not ln.startswith(">")]
        return "".join(lines)
    except Exception:
        return ""


_STUB_OFFTARGETS = [
    ("CYP3A4", "P08684"),
    ("hERG", "Q12809"),
    ("CYP2D6", "P10635"),
    ("MAO-B", "P27338"),
    ("AChE", "P22303"),
    ("SRC", "P12931"),
    ("ABL1", "P00519"),
    ("HDAC1", "Q13547"),
]


def _stub_offtarget_rows(
    intended_uniprot: str, smiles: str, limit: int = 6
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sym, acc in _STUB_OFFTARGETS[:limit]:
        if acc == (intended_uniprot or "").upper():
            continue
        score = (stable_seed(smiles, acc, "otp") % 1000) / 1000.0
        rows.append(
            {
                "symbol": sym,
                "uniprot_id": acc,
                "score": round(score, 3),
                "source": "stub",
            }
        )
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


class BlastSimilarityStubNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "dd_blast_similarity_stub"

    def __init__(self, fetch_sequence: bool = True, timeout: int = 15) -> None:
        self._fetch = bool(fetch_sequence)
        self._timeout = timeout

    @property
    def params(self) -> dict[str, Any]:
        return {"fetch_sequence": self._fetch, "timeout": self._timeout}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="BLAST similarity (stub)",
            description="Stubbed sequence-homology off-target neighbors; real UniProt fetch included.",
            version="1.0.0",
            category="model",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"uniprot_id": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"neighbors": "list", "stub": "bool"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        acc = str(inputs.get("uniprot_id", "")).strip().upper()
        seq = _fetch_uniprot_sequence(acc, timeout=self._timeout) if self._fetch else ""
        rows = _stub_offtarget_rows(acc, seq or acc, limit=6)
        for r in rows:
            r["method"] = "blast_stub"
        return {"neighbors": rows, "stub": True}


class FoldseekStubNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "dd_foldseek_stub"

    @property
    def params(self) -> dict[str, Any]:
        return {}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Foldseek neighbors (stub)",
            description="Structural-similarity off-target stubs.",
            version="1.0.0",
            category="model",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"uniprot_id": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"neighbors": "list", "stub": "bool"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        acc = str(inputs.get("uniprot_id", "")).strip().upper()
        rows = _stub_offtarget_rows(acc, acc + "fold", limit=5)
        for r in rows:
            r["method"] = "foldseek_stub"
        return {"neighbors": rows, "stub": True}


class ProbisStubNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "dd_probis_stub"

    @property
    def params(self) -> dict[str, Any]:
        return {}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="ProBiS pocket neighbors (stub)",
            description="Local pocket-similarity off-target stubs.",
            version="1.0.0",
            category="model",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"uniprot_id": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"neighbors": "list", "stub": "bool"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        acc = str(inputs.get("uniprot_id", "")).strip().upper()
        rows = _stub_offtarget_rows(acc, acc + "pocket", limit=4)
        for r in rows:
            r["method"] = "probis_stub"
        return {"neighbors": rows, "stub": True}


class SeaSearchStubNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "dd_sea_search_stub"

    @property
    def params(self) -> dict[str, Any]:
        return {}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="SEA ligand similarity (stub)",
            description="Ligand-based off-target neighbors via chemistry similarity.",
            version="1.0.0",
            category="model",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"series_package": SERIES_PACKAGE_TAG}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"neighbors": "list", "stub": "bool"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        pkg = inputs.get("series_package") or {}
        seeds = pkg.get("seed_smiles") if isinstance(pkg, dict) else None
        key = (seeds or ["x"])[0] if isinstance(seeds, list) else "x"
        rows = _stub_offtarget_rows("", str(key) + "sea", limit=6)
        for r in rows:
            r["method"] = "sea_stub"
        return {"neighbors": rows, "stub": True}


class OffTargetTier1AggregateNode(WorkflowNode):
    """Combine Tier 1 lanes and flag candidates with multi-lane support."""

    workflow_type_id: ClassVar[str] = "dd_offtarget_tier1_aggregate"

    def __init__(self, critical_threshold: float = 0.6) -> None:
        self._critical = float(critical_threshold)

    @property
    def params(self) -> dict[str, Any]:
        return {"critical_threshold": self._critical}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Off-target Tier 1 aggregate",
            description="Collapse BLAST/Foldseek/ProBiS/SEA neighbors into a unified panel.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {
            "blast_neighbors": "list",
            "foldseek_neighbors": "list",
            "probis_neighbors": "list",
            "sea_neighbors": "list",
        }

    @property
    def output_schema(self) -> dict[str, str]:
        return {"offtarget_panel": "dict"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        lanes = {
            "blast": inputs.get("blast_neighbors") or [],
            "foldseek": inputs.get("foldseek_neighbors") or [],
            "probis": inputs.get("probis_neighbors") or [],
            "sea": inputs.get("sea_neighbors") or [],
        }
        by_target: dict[str, dict[str, Any]] = {}
        for lane_name, rows in lanes.items():
            for r in rows:
                if not isinstance(r, dict):
                    continue
                key = r.get("uniprot_id") or r.get("symbol")
                if not key:
                    continue
                slot = by_target.setdefault(
                    key,
                    {
                        "symbol": r.get("symbol"),
                        "uniprot_id": r.get("uniprot_id"),
                        "lanes": set(),
                        "max_score": 0.0,
                    },
                )
                slot["lanes"].add(lane_name)
                score = float(r.get("score") or 0.0)
                if score > slot["max_score"]:
                    slot["max_score"] = score
        panel: list[dict[str, Any]] = []
        critical: list[dict[str, Any]] = []
        for slot in by_target.values():
            lanes_list = sorted(slot["lanes"])
            lane_count: int = len(lanes_list)
            max_score: float = round(float(slot["max_score"]), 3)
            entry: dict[str, Any] = {
                "symbol": slot.get("symbol"),
                "uniprot_id": slot.get("uniprot_id"),
                "supporting_lanes": lanes_list,
                "lane_count": lane_count,
                "max_score": max_score,
            }
            panel.append(entry)
            if lane_count >= 2 and max_score >= self._critical:
                critical.append(entry)
        panel.sort(key=lambda e: (e["lane_count"], e["max_score"]), reverse=True)
        return {
            "offtarget_panel": {
                "rows": panel,
                "critical_flags": critical,
                "critical_threshold": self._critical,
            }
        }


class OffTargetTier2RefineStubNode(WorkflowNode):
    """Rerun a stub docking lane against top off-targets to compute a margin."""

    workflow_type_id: ClassVar[str] = "dd_offtarget_tier2_refine_stub"

    def __init__(self, top_k: int = 3) -> None:
        self._top_k = max(1, min(10, int(top_k)))

    @property
    def params(self) -> dict[str, Any]:
        return {"top_k": self._top_k}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Off-target Tier 2 refine (stub)",
            description="Stubbed docking/Boltz refinement over top off-targets.",
            version="1.0.0",
            category="model",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {
            "offtarget_panel": "dict",
            "docking_results": "list",
            "series_package": SERIES_PACKAGE_TAG,
        }

    @property
    def output_schema(self) -> dict[str, str]:
        return {"tier2_report": "dict", "stub": "bool"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        panel = inputs.get("offtarget_panel") or {}
        rows = panel.get("rows") if isinstance(panel, dict) else None
        top = []
        if isinstance(rows, list):
            top = rows[: self._top_k]
        intended = {
            p.get("smiles"): float(p.get("affinity_kcal_mol") or 0.0)
            for p in (inputs.get("docking_results") or [])
            if isinstance(p, dict) and p.get("smiles")
        }
        analyses: list[dict[str, Any]] = []
        for off in top:
            acc = off.get("uniprot_id") if isinstance(off, dict) else None
            for smi, intended_score in intended.items():
                off_score = -5.0 - ((stable_seed(smi, acc or "off", "tier2") % 1000) / 1000.0) * 3.0
                margin = round(off_score - intended_score, 3)
                analyses.append(
                    {
                        "smiles": smi,
                        "offtarget": off,
                        "intended_affinity_kcal_mol": intended_score,
                        "offtarget_affinity_kcal_mol": round(off_score, 3),
                        "intended_margin_kcal_mol": margin,
                    }
                )
        return {
            "tier2_report": {
                "rows": analyses,
                "top_offtargets": top,
            },
            "stub": True,
        }


OFFTARGET_NODES: list[type[WorkflowNode]] = [
    BlastSimilarityStubNode,
    FoldseekStubNode,
    ProbisStubNode,
    SeaSearchStubNode,
    OffTargetTier1AggregateNode,
    OffTargetTier2RefineStubNode,
]
