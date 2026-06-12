"""ADMET triage + expanded ADMET stub nodes."""

from __future__ import annotations

import random
from typing import Any, ClassVar

from bioagents.workflows.drug_discovery._rdkit_utils import (
    pains_alert_count,
    stable_seed,
)
from bioagents.workflows.drug_discovery.schemas import SERIES_PACKAGE_TAG
from bioagents.workflows.node import WorkflowNode
from bioagents.workflows.schemas import NodeMetadata


class EarlyAdmetTriageNode(WorkflowNode):
    """Combine Lipinski/Veber (from heuristic SwissADME) with PAINS to flag liabilities."""

    workflow_type_id: ClassVar[str] = "dd_early_admet_triage"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Early ADMET triage",
            description="Count major PK/tox liabilities per analogue.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"admet_report": "dict", "series_package": SERIES_PACKAGE_TAG}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"triage": "dict"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        report = inputs.get("admet_report") or {}
        rows_in = report.get("rows") if isinstance(report, dict) else None
        out_rows: list[dict[str, Any]] = []
        if isinstance(rows_in, list):
            for r in rows_in:
                if not isinstance(r, dict):
                    continue
                smi = r.get("smiles")
                if not isinstance(smi, str):
                    continue
                majors = int(r.get("major_liabilities") or 0)
                out_rows.append(
                    {
                        "smiles": smi,
                        "major_liabilities": majors,
                        "liabilities": list(r.get("liabilities") or []),
                        "pains_alerts": int(r.get("pains_alerts") or pains_alert_count(smi)),
                    }
                )
        worst = max((r["major_liabilities"] for r in out_rows), default=0)
        return {
            "triage": {
                "rows": out_rows,
                "worst_major_liabilities": int(worst),
                "total_evaluated": len(out_rows),
            }
        }


class AdmetLabStubNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "dd_admetlab_stub"

    @property
    def params(self) -> dict[str, Any]:
        return {}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="ADMETlab (stub)",
            description="Deterministic expanded ADMET endpoint screen.",
            version="1.0.0",
            category="model",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"series_package": SERIES_PACKAGE_TAG}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"admet_expanded": "dict", "stub": "bool"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        pkg = inputs.get("series_package") or {}
        rows: list[dict[str, Any]] = []
        endpoints = (
            "caco2_permeability",
            "hia",
            "herg_liability",
            "cyp3a4_inhibition",
            "ames_mutagenicity",
        )
        for a in pkg.get("analogues") or [] if isinstance(pkg, dict) else []:
            if not isinstance(a, dict):
                continue
            smi = a.get("smiles")
            if not isinstance(smi, str):
                continue
            rng = random.Random(stable_seed(smi, "admetlab"))  # nosec B311 - stub deterministic scoring
            result = {ep: round(rng.random(), 3) for ep in endpoints}
            majors = sum(1 for v in result.values() if v > 0.75)
            rows.append({"smiles": smi, "endpoints": result, "major_liabilities": majors})
        return {
            "admet_expanded": {"rows": rows, "endpoints": list(endpoints)},
            "stub": True,
        }


class AdmetSarStubNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "dd_admetsar_stub"

    @property
    def params(self) -> dict[str, Any]:
        return {}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="admetSAR (stub)",
            description="Deterministic complementary ADMET profile (stub).",
            version="1.0.0",
            category="model",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"series_package": SERIES_PACKAGE_TAG}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"admet_complementary": "dict", "stub": "bool"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        pkg = inputs.get("series_package") or {}
        rows: list[dict[str, Any]] = []
        for a in pkg.get("analogues") or [] if isinstance(pkg, dict) else []:
            if not isinstance(a, dict):
                continue
            smi = a.get("smiles")
            if not isinstance(smi, str):
                continue
            rng = random.Random(stable_seed(smi, "admetsar"))  # nosec B311 - stub deterministic scoring
            rows.append(
                {
                    "smiles": smi,
                    "oral_bioavailability": round(rng.random(), 3),
                    "blood_brain_barrier": round(rng.random(), 3),
                    "hepatotoxicity": round(rng.random(), 3),
                }
            )
        return {"admet_complementary": {"rows": rows}, "stub": True}


ADMET_NODES: list[type[WorkflowNode]] = [
    EarlyAdmetTriageNode,
    AdmetLabStubNode,
    AdmetSarStubNode,
]
