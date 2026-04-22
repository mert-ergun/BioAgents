"""Deterministic stub proposal-engine nodes.

Each node emulates a design family from the reference guide section 1.4A. They
produce schema-faithful SeriesPackage extensions (adding analogues derived from
RDKit-based deterministic transformations) while being clearly flagged as stubs.
"""

from __future__ import annotations

import random
from typing import Any, ClassVar

from bioagents.workflows.drug_discovery._rdkit_utils import (
    compute_descriptors,
    enumerate_methyl_analogues,
    stable_seed,
    standardize_smiles,
)
from bioagents.workflows.drug_discovery.schemas import SERIES_PACKAGE_TAG
from bioagents.workflows.node import WorkflowNode
from bioagents.workflows.schemas import NodeMetadata


def _extend_pkg_with_analogues(
    series: dict[str, Any],
    origin: str,
    generator_name: str,
    candidates: list[str],
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)  # nosec B311 - stub deterministic sampling
    existing = {a.get("smiles") for a in series.get("analogues", []) if isinstance(a, dict)}
    added: list[dict[str, Any]] = []
    for smi in candidates:
        if not smi or smi in existing:
            continue
        existing.add(smi)
        added.append(
            {
                "smiles": smi,
                "origin": origin,
                "generator": generator_name,
                "score": round(rng.random(), 4),
                "descriptors": compute_descriptors(smi),
            }
        )
    out = dict(series)
    out["analogues"] = list(series.get("analogues", [])) + added
    return out


class _DesignStubBase(WorkflowNode):
    """Shared scaffolding for stub design nodes."""

    _origin: ClassVar[str] = "stub"
    _generator_label: ClassVar[str] = "stub"

    def __init__(self, n_candidates: int = 6, seed: int = 1) -> None:
        self._n = max(1, min(32, int(n_candidates)))
        self._seed = int(seed)

    @property
    def params(self) -> dict[str, Any]:
        return {"n_candidates": self._n, "seed": self._seed}

    @property
    def input_schema(self) -> dict[str, str]:
        return {"series_package": SERIES_PACKAGE_TAG}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"series_package": SERIES_PACKAGE_TAG, "stub": "bool"}

    def _candidates(self, seeds: list[str]) -> list[str]:
        """Default: deterministic methyl-walk enumeration from seeds."""
        out: list[str] = []
        if not seeds:
            return out
        per_seed = max(1, self._n // len(seeds))
        for seed_smi in seeds:
            out.extend(enumerate_methyl_analogues(seed_smi, per_seed + 1))
        # Drop duplicates while preserving order.
        seen: set[str] = set()
        uniq: list[str] = []
        for s in out:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        return uniq[: self._n]

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        pkg = inputs.get("series_package") or {}
        if not isinstance(pkg, dict):
            pkg = {}
        seeds_raw = pkg.get("seed_smiles") or []
        seeds = [s for s in seeds_raw if isinstance(s, str) and s.strip()]
        candidates = self._candidates(seeds)
        hash_seed = stable_seed(self._generator_label, self._seed, *seeds)
        enriched = _extend_pkg_with_analogues(
            pkg, self._origin, self._generator_label, candidates, hash_seed
        )
        enriched.setdefault("design_lane", self._generator_label)
        return {"series_package": enriched, "stub": True}


class Reinvent4StubNode(_DesignStubBase):
    workflow_type_id: ClassVar[str] = "dd_reinvent4_stub"
    _origin: ClassVar[str] = "reinvent4_stub"
    _generator_label: ClassVar[str] = "reinvent4"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="REINVENT4 (stub)",
            description="Deterministic stub: generates analogues from seed SMILES.",
            version="1.0.0",
            category="model",
        )


class ChemTSv2StubNode(_DesignStubBase):
    workflow_type_id: ClassVar[str] = "dd_chemtsv2_stub"
    _origin: ClassVar[str] = "chemtsv2_stub"
    _generator_label: ClassVar[str] = "chemtsv2"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="ChemTSv2 (stub)",
            description="Deterministic stub: goal-directed ligand generation.",
            version="1.0.0",
            category="model",
        )


class StonedSelfiesStubNode(_DesignStubBase):
    workflow_type_id: ClassVar[str] = "dd_stoned_selfies_stub"
    _origin: ClassVar[str] = "stoned_stub"
    _generator_label: ClassVar[str] = "stoned_selfies"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="STONED-SELFIES (stub)",
            description="Deterministic stub: local molecule edits around seeds.",
            version="1.0.0",
            category="model",
        )


class MegaMolBartStubNode(_DesignStubBase):
    workflow_type_id: ClassVar[str] = "dd_megamolbart_stub"
    _origin: ClassVar[str] = "megamolbart_stub"
    _generator_label: ClassVar[str] = "megamolbart"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="MegaMolBART (stub)",
            description="Deterministic stub: SMILES infilling proposals.",
            version="1.0.0",
            category="model",
        )


class Pocket2MolStubNode(_DesignStubBase):
    workflow_type_id: ClassVar[str] = "dd_pocket2mol_stub"
    _origin: ClassVar[str] = "pocket2mol_stub"
    _generator_label: ClassVar[str] = "pocket2mol"

    def __init__(self, n_candidates: int = 6, seed: int = 1) -> None:
        super().__init__(n_candidates=n_candidates, seed=seed)

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Pocket2Mol (stub)",
            description="Deterministic stub: pocket-conditioned 3D proposals.",
            version="1.0.0",
            category="model",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"series_package": SERIES_PACKAGE_TAG, "structure_package": "structure_package"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        pkg = inputs.get("series_package") or {}
        struct = inputs.get("structure_package") or {}
        seeds: list[str] = []
        if isinstance(pkg, dict):
            seeds.extend(s for s in pkg.get("seed_smiles") or [] if isinstance(s, str))
        native = struct.get("native_ligand_smiles") if isinstance(struct, dict) else None
        if isinstance(native, str) and native.strip():
            seeds.append(standardize_smiles(native.strip()) or native.strip())
        seeds = [s for s in seeds if s]
        if not seeds:
            seeds = ["c1ccccc1"]
        candidates = self._candidates(seeds)
        hash_seed = stable_seed(
            self._generator_label,
            self._seed,
            *seeds,
            (struct.get("pdb_id") if isinstance(struct, dict) else None) or "",
        )
        enriched = _extend_pkg_with_analogues(
            pkg if isinstance(pkg, dict) else {},
            self._origin,
            self._generator_label,
            candidates,
            hash_seed,
        )
        enriched.setdefault("design_lane", "target_based")
        return {"series_package": enriched, "stub": True}


class TargetDiffStubNode(Pocket2MolStubNode):
    workflow_type_id: ClassVar[str] = "dd_targetdiff_stub"
    _origin: ClassVar[str] = "targetdiff_stub"
    _generator_label: ClassVar[str] = "targetdiff"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="TargetDiff (stub)",
            description="Deterministic stub: diffusion-based pocket-conditioned design.",
            version="1.0.0",
            category="model",
        )


class AutoGrow4StubNode(_DesignStubBase):
    workflow_type_id: ClassVar[str] = "dd_autogrow4_stub"
    _origin: ClassVar[str] = "autogrow4_stub"
    _generator_label: ClassVar[str] = "autogrow4"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="AutoGrow4 (stub)",
            description="Deterministic stub: docking-guided evolutionary growth.",
            version="1.0.0",
            category="model",
        )


DESIGN_NODES: list[type[WorkflowNode]] = [
    Reinvent4StubNode,
    ChemTSv2StubNode,
    StonedSelfiesStubNode,
    MegaMolBartStubNode,
    Pocket2MolStubNode,
    TargetDiffStubNode,
    AutoGrow4StubNode,
]
