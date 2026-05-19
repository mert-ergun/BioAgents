"""Retrosynthesis nodes (stub AiZynthFinder + real route review logic)."""

from __future__ import annotations

import random
from typing import Any, ClassVar

from bioagents.workflows.drug_discovery._rdkit_utils import (
    stable_seed,
    synthetic_complexity_score,
)
from bioagents.workflows.drug_discovery.schemas import SERIES_PACKAGE_TAG
from bioagents.workflows.node import WorkflowNode
from bioagents.workflows.schemas import NodeMetadata


class AiZynthFinderStubNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "dd_aizynthfinder_stub"

    def __init__(self, max_routes: int = 3) -> None:
        self._max = max(1, min(10, int(max_routes)))

    @property
    def params(self) -> dict[str, Any]:
        return {"max_routes": self._max}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="AiZynthFinder (stub)",
            description="Deterministic retrosynthesis route proposals.",
            version="1.0.0",
            category="model",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"series_package": SERIES_PACKAGE_TAG}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"routes": "list", "stub": "bool"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        pkg = inputs.get("series_package") or {}
        analogues = pkg.get("analogues") if isinstance(pkg, dict) else None
        out: list[dict[str, Any]] = []
        for a in analogues or []:
            if not isinstance(a, dict):
                continue
            smi = a.get("smiles")
            if not isinstance(smi, str):
                continue
            rng = random.Random(stable_seed(smi, "aizynth"))  # nosec B311 - stub deterministic scoring
            ease = synthetic_complexity_score(smi)
            routes: list[dict[str, Any]] = []
            for i in range(self._max):
                steps = max(1, int(rng.uniform(2, 8) + (1 - ease) * 4))
                score = round(max(0.0, min(1.0, ease - (i * 0.12) + rng.uniform(-0.05, 0.05))), 3)
                routes.append(
                    {
                        "route_index": i,
                        "step_count": steps,
                        "route_score": score,
                        "purchasable_precursors": score > 0.45,
                    }
                )
            out.append({"smiles": smi, "routes": routes})
        return {"routes": out, "stub": True}


class RetrosynthRouteReviewNode(WorkflowNode):
    """Keep the best route per analogue and produce a flat summary."""

    workflow_type_id: ClassVar[str] = "dd_retrosynth_route_review"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Retrosynthesis route review",
            description="Select the best route per analogue and summarize.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"routes": "list"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"route_summary": "dict"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        per_mol = inputs.get("routes") or []
        rows: list[dict[str, Any]] = []
        for entry in per_mol:
            if not isinstance(entry, dict):
                continue
            smi = entry.get("smiles")
            routes = entry.get("routes") or []
            if not isinstance(routes, list) or not routes:
                rows.append(
                    {
                        "smiles": smi,
                        "best_route": None,
                        "route_score": 0.0,
                        "step_count": None,
                        "purchasable_precursors": False,
                    }
                )
                continue
            best = max(
                (r for r in routes if isinstance(r, dict)),
                key=lambda r: float(r.get("route_score") or 0.0),
                default=None,
            )
            if best is None:
                continue
            rows.append(
                {
                    "smiles": smi,
                    "best_route": best,
                    "route_score": float(best.get("route_score") or 0.0),
                    "step_count": int(best.get("step_count") or 0),
                    "purchasable_precursors": bool(best.get("purchasable_precursors")),
                }
            )
        return {"route_summary": {"rows": rows}}


RETROSYNTH_NODES: list[type[WorkflowNode]] = [
    AiZynthFinderStubNode,
    RetrosynthRouteReviewNode,
]
