"""Gate nodes + decision-log aggregator.

Every gate node encodes one row from the reference-guide policy table (section
2.6) and emits a list of :class:`GateVerdict` entries. The terminal
:class:`DecisionLogNode` collects all verdicts and computes promotion tiers
(doc section 2.7).
"""

from __future__ import annotations

from typing import Any, ClassVar

from bioagents.workflows.drug_discovery.schemas import (
    DECISION_LOG_TAG,
    STRUCTURE_PACKAGE_TAG,
    TIER_DE_RISKED_LEAD,
    TIER_HIT,
    TIER_LEAD,
    TIER_OPTIMIZABLE_HIT,
    TIER_REJECT,
    VERDICT_BORDERLINE,
    VERDICT_FAIL,
    VERDICT_PASS,
    default_policy_thresholds,
)
from bioagents.workflows.node import WorkflowNode
from bioagents.workflows.schemas import NodeMetadata

_GLOBAL = "__global__"


def _verdict(
    *,
    gate: str,
    verdict: str,
    reason_code: str,
    reason_text: str,
    metrics: dict[str, Any],
    thresholds: dict[str, Any],
    candidate_id: str = _GLOBAL,
) -> dict[str, Any]:
    return {
        "gate": gate,
        "verdict": verdict,
        "reason_code": reason_code,
        "reason_text": reason_text,
        "metrics": metrics,
        "thresholds": thresholds,
        "candidate_id": candidate_id,
    }


class _GateBase(WorkflowNode):
    """Shared plumbing: thresholds param + list-of-verdicts output."""

    gate_name: ClassVar[str] = "gate"

    def __init__(self, thresholds: dict[str, Any] | None = None) -> None:
        defaults = default_policy_thresholds().get(self.gate_name, {})
        merged = {**defaults}
        if thresholds:
            merged.update(thresholds)
        self._thresholds = merged

    @property
    def params(self) -> dict[str, Any]:
        return {"thresholds": self._thresholds}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"verdicts": "list"}


class StructureReadinessGateNode(_GateBase):
    workflow_type_id: ClassVar[str] = "dd_gate_structure_readiness"
    gate_name: ClassVar[str] = "structure_readiness"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Gate: structure readiness",
            description="Holo > apo > predicted > none promotion policy (doc 2.6).",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"structure_package": STRUCTURE_PACKAGE_TAG}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        pkg = inputs.get("structure_package") or {}
        kind = str(pkg.get("kind", "none"))
        accept = self._thresholds.get("accept_kinds", ["holo"])
        borderline = self._thresholds.get("borderline_kinds", ["apo", "predicted"])
        if kind in accept:
            v, code, text = (
                VERDICT_PASS,
                "holo_available",
                "Experimental ligand-bound structure available.",
            )
        elif kind in borderline:
            v, code, text = (
                VERDICT_BORDERLINE,
                f"{kind}_available",
                f"Usable {kind} structure — escalate review.",
            )
        else:
            v, code, text = VERDICT_FAIL, "no_structure", "No reliable pocket definition."
        return {
            "verdicts": [
                _verdict(
                    gate=self.gate_name,
                    verdict=v,
                    reason_code=code,
                    reason_text=text,
                    metrics={"kind": kind, "confidence_metric": pkg.get("confidence_metric")},
                    thresholds=self._thresholds,
                )
            ]
        }


class DockingValidationGateNode(_GateBase):
    workflow_type_id: ClassVar[str] = "dd_gate_docking_validation"
    gate_name: ClassVar[str] = "docking_validation"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Gate: docking validation",
            description="Redocking RMSD policy (doc 2.6).",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"validation": "dict"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        val = inputs.get("validation") or {}
        pass_rmsd = float(self._thresholds.get("pass_rmsd_max", 2.0))
        borderline_rmsd = float(self._thresholds.get("borderline_rmsd_max", 3.0))
        require_contacts = bool(self._thresholds.get("require_key_contacts", True))
        if not val.get("has_native_ligand"):
            return {
                "verdicts": [
                    _verdict(
                        gate=self.gate_name,
                        verdict=VERDICT_BORDERLINE,
                        reason_code="no_native_ligand",
                        reason_text="No native ligand present; redocking validation skipped.",
                        metrics=val,
                        thresholds=self._thresholds,
                    )
                ]
            }
        rmsd = float(val.get("rmsd") or 0.0)
        ok_contacts = bool(val.get("key_contacts_recovered"))
        if rmsd <= pass_rmsd and (ok_contacts or not require_contacts):
            v, code, text = (
                VERDICT_PASS,
                "rmsd_within_pass",
                f"RMSD {rmsd:.2f} Å within pass threshold.",
            )
        elif rmsd <= borderline_rmsd:
            v, code, text = (
                VERDICT_BORDERLINE,
                "rmsd_borderline",
                f"RMSD {rmsd:.2f} Å in borderline band.",
            )
        else:
            v, code, text = VERDICT_FAIL, "rmsd_exceeds", f"RMSD {rmsd:.2f} Å exceeds threshold."
        if not ok_contacts and require_contacts and v == VERDICT_PASS:
            v, code, text = (
                VERDICT_BORDERLINE,
                "contacts_not_recovered",
                "Key contacts not recovered.",
            )
        return {
            "verdicts": [
                _verdict(
                    gate=self.gate_name,
                    verdict=v,
                    reason_code=code,
                    reason_text=text,
                    metrics={"rmsd": rmsd, "key_contacts_recovered": ok_contacts},
                    thresholds=self._thresholds,
                )
            ]
        }


class BoltzHitScreenGateNode(_GateBase):
    workflow_type_id: ClassVar[str] = "dd_gate_boltz_hit_screen"
    gate_name: ClassVar[str] = "boltz_hit_screen"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Gate: Boltz hit screen",
            description="affinity_probability_binary thresholds (doc 2.6).",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"boltz_results": "list"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        rows = inputs.get("boltz_results") or []
        pass_p = float(self._thresholds.get("pass_binder_probability", 0.70))
        border_p = float(self._thresholds.get("borderline_binder_probability", 0.50))
        verdicts: list[dict[str, Any]] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            smi = r.get("smiles")
            if not isinstance(smi, str):
                continue
            p = float(r.get("affinity_probability_binary") or 0.0)
            if p >= pass_p:
                v, code, text = (
                    VERDICT_PASS,
                    "binder_prob_pass",
                    f"Boltz probability {p:.2f} ≥ {pass_p:.2f}.",
                )
            elif p >= border_p:
                v, code, text = (
                    VERDICT_BORDERLINE,
                    "binder_prob_borderline",
                    f"Boltz probability {p:.2f} in borderline band.",
                )
            else:
                v, code, text = (
                    VERDICT_FAIL,
                    "binder_prob_fail",
                    f"Boltz probability {p:.2f} < {border_p:.2f}.",
                )
            verdicts.append(
                _verdict(
                    gate=self.gate_name,
                    verdict=v,
                    reason_code=code,
                    reason_text=text,
                    metrics={"affinity_probability_binary": p},
                    thresholds=self._thresholds,
                    candidate_id=smi,
                )
            )
        return {"verdicts": verdicts}


class ConsensusGateNode(_GateBase):
    workflow_type_id: ClassVar[str] = "dd_gate_consensus"
    gate_name: ClassVar[str] = "consensus_structural_review"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Gate: consensus structural review",
            description="Require agreement across docking/Boltz/crystal (doc 2.6).",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"consensus": "list"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        rows = inputs.get("consensus") or []
        min_agree = int(self._thresholds.get("min_agreeing_methods", 2))
        verdicts: list[dict[str, Any]] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            smi = r.get("smiles")
            if not isinstance(smi, str):
                continue
            agree = int(r.get("agreeing_methods") or 0)
            if agree >= min_agree:
                v, code, text = (
                    VERDICT_PASS,
                    "consensus_ok",
                    f"{agree} methods agree (≥ {min_agree}).",
                )
            elif agree == 1:
                v, code, text = (
                    VERDICT_BORDERLINE,
                    "consensus_partial",
                    "One method supports the hypothesis.",
                )
            else:
                v, code, text = (
                    VERDICT_FAIL,
                    "consensus_conflict",
                    "No methods support the hypothesis.",
                )
            verdicts.append(
                _verdict(
                    gate=self.gate_name,
                    verdict=v,
                    reason_code=code,
                    reason_text=text,
                    metrics={"agreeing_methods": agree},
                    thresholds=self._thresholds,
                    candidate_id=smi,
                )
            )
        return {"verdicts": verdicts}


class EarlyAdmetGateNode(_GateBase):
    workflow_type_id: ClassVar[str] = "dd_gate_early_admet"
    gate_name: ClassVar[str] = "early_admet"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Gate: early ADMET",
            description="Major-liability policy (doc 2.6).",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"triage": "dict"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        tri = inputs.get("triage") or {}
        rows = tri.get("rows") if isinstance(tri, dict) else []
        max_ok = int(self._thresholds.get("max_major_liabilities", 0))
        border_ok = int(self._thresholds.get("borderline_major_liabilities", 1))
        verdicts: list[dict[str, Any]] = []
        for r in rows or []:
            if not isinstance(r, dict):
                continue
            smi = r.get("smiles")
            if not isinstance(smi, str):
                continue
            majors = int(r.get("major_liabilities") or 0)
            if majors <= max_ok:
                v, code, text = VERDICT_PASS, "admet_clean", "No major ADMET liabilities."
            elif majors <= border_ok:
                v, code, text = (
                    VERDICT_BORDERLINE,
                    "admet_borderline",
                    f"{majors} major liability(ies) — review.",
                )
            else:
                v, code, text = VERDICT_FAIL, "admet_fail", f"{majors} major liabilities."
            verdicts.append(
                _verdict(
                    gate=self.gate_name,
                    verdict=v,
                    reason_code=code,
                    reason_text=text,
                    metrics={"major_liabilities": majors, "liabilities": r.get("liabilities")},
                    thresholds=self._thresholds,
                    candidate_id=smi,
                )
            )
        return {"verdicts": verdicts}


class OffTargetTier1GateNode(_GateBase):
    workflow_type_id: ClassVar[str] = "dd_gate_offtarget_tier1"
    gate_name: ClassVar[str] = "off_target_tier1"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Gate: off-target Tier 1",
            description="Count critical off-target flags across lanes (doc 2.6).",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"offtarget_panel": "dict"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        panel = inputs.get("offtarget_panel") or {}
        critical = panel.get("critical_flags") if isinstance(panel, dict) else []
        n = len(critical or [])
        max_ok = int(self._thresholds.get("max_critical_flags", 0))
        border_ok = int(self._thresholds.get("borderline_critical_flags", 1))
        if n <= max_ok:
            v, code, text = (
                VERDICT_PASS,
                "offtarget_clean",
                "No critical off-target flagged by multiple lanes.",
            )
        elif n <= border_ok:
            v, code, text = (
                VERDICT_BORDERLINE,
                "offtarget_concern",
                f"{n} concerning off-target(s).",
            )
        else:
            v, code, text = (
                VERDICT_FAIL,
                "offtarget_critical",
                f"{n} critical off-target(s) supported by ≥2 lanes.",
            )
        return {
            "verdicts": [
                _verdict(
                    gate=self.gate_name,
                    verdict=v,
                    reason_code=code,
                    reason_text=text,
                    metrics={"critical_count": n, "critical_flags": critical or []},
                    thresholds=self._thresholds,
                )
            ]
        }


class OffTargetTier2GateNode(_GateBase):
    workflow_type_id: ClassVar[str] = "dd_gate_offtarget_tier2"
    gate_name: ClassVar[str] = "off_target_tier2"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Gate: off-target Tier 2",
            description="Intended-target margin after refinement (doc 2.6).",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"tier2_report": "dict"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        report = inputs.get("tier2_report") or {}
        rows = report.get("rows") if isinstance(report, dict) else []
        margin_req = float(self._thresholds.get("required_intended_margin", 0.1))
        per_smiles: dict[str, list[float]] = {}
        for r in rows or []:
            if not isinstance(r, dict):
                continue
            smi = r.get("smiles")
            margin = float(r.get("intended_margin_kcal_mol") or 0.0)
            if isinstance(smi, str):
                per_smiles.setdefault(smi, []).append(margin)
        verdicts: list[dict[str, Any]] = []
        for smi, margins in per_smiles.items():
            worst = min(margins) if margins else 0.0
            if worst <= -margin_req:
                v, code, text = (
                    VERDICT_PASS,
                    "intended_preferred",
                    f"Intended target preferred by {abs(worst):.2f} kcal/mol.",
                )
            elif worst <= 0.0:
                v, code, text = (
                    VERDICT_BORDERLINE,
                    "small_margin",
                    f"Intended margin only {abs(worst):.2f} kcal/mol.",
                )
            else:
                v, code, text = (
                    VERDICT_FAIL,
                    "offtarget_competitive",
                    "High-risk off-target remains competitive.",
                )
            verdicts.append(
                _verdict(
                    gate=self.gate_name,
                    verdict=v,
                    reason_code=code,
                    reason_text=text,
                    metrics={"worst_intended_margin_kcal_mol": worst},
                    thresholds=self._thresholds,
                    candidate_id=smi,
                )
            )
        if not verdicts:
            verdicts.append(
                _verdict(
                    gate=self.gate_name,
                    verdict=VERDICT_BORDERLINE,
                    reason_code="no_tier2_data",
                    reason_text="Tier 2 refinement not run.",
                    metrics={},
                    thresholds=self._thresholds,
                )
            )
        return {"verdicts": verdicts}


class RetrosynthesisGateNode(_GateBase):
    workflow_type_id: ClassVar[str] = "dd_gate_retrosynthesis"
    gate_name: ClassVar[str] = "retrosynthesis"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Gate: retrosynthesis",
            description="Require a plausible route (doc 2.6).",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"route_summary": "dict"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        summary = inputs.get("route_summary") or {}
        rows = summary.get("rows") if isinstance(summary, dict) else []
        min_score = float(self._thresholds.get("min_route_score", 0.5))
        max_steps = int(self._thresholds.get("max_step_count", 8))
        verdicts: list[dict[str, Any]] = []
        for r in rows or []:
            if not isinstance(r, dict):
                continue
            smi = r.get("smiles")
            if not isinstance(smi, str):
                continue
            score = float(r.get("route_score") or 0.0)
            steps = int(r.get("step_count") or 0)
            purchasable = bool(r.get("purchasable_precursors"))
            if not r.get("best_route"):
                v, code, text = VERDICT_FAIL, "no_route", "No viable retrosynthesis route."
            elif score >= min_score and steps <= max_steps and purchasable:
                v, code, text = VERDICT_PASS, "route_ok", f"Route {score:.2f} in {steps} steps."
            elif score >= min_score * 0.6 and steps <= max_steps + 4:
                v, code, text = (
                    VERDICT_BORDERLINE,
                    "route_weak",
                    f"Route exists ({score:.2f}/{steps} steps) but marginal.",
                )
            else:
                v, code, text = (
                    VERDICT_FAIL,
                    "route_impractical",
                    f"Route impractical ({score:.2f}/{steps} steps).",
                )
            verdicts.append(
                _verdict(
                    gate=self.gate_name,
                    verdict=v,
                    reason_code=code,
                    reason_text=text,
                    metrics={
                        "route_score": score,
                        "step_count": steps,
                        "purchasable_precursors": purchasable,
                    },
                    thresholds=self._thresholds,
                    candidate_id=smi,
                )
            )
        return {"verdicts": verdicts}


# ---------------------------------------------------------------------------
# Decision log aggregator
# ---------------------------------------------------------------------------


class DecisionLogNode(WorkflowNode):
    """Aggregate all gate verdicts and compute a promotion tier per candidate."""

    workflow_type_id: ClassVar[str] = "dd_decision_log"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Decision log (promotion)",
            description="Merge gate verdicts → tier per candidate (doc 2.7).",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {
            "structure_readiness": "list",
            "docking_validation": "list",
            "boltz_hit_screen": "list",
            "consensus": "list",
            "early_admet": "list",
            "offtarget_tier1": "list",
            "offtarget_tier2": "list",
            "retrosynthesis": "list",
        }

    @property
    def output_schema(self) -> dict[str, str]:
        return {"decision_log": DECISION_LOG_TAG, "gate_verdicts": "list", "summary": "dict"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        all_verdicts: list[dict[str, Any]] = []
        for key in (
            "structure_readiness",
            "docking_validation",
            "boltz_hit_screen",
            "consensus",
            "early_admet",
            "offtarget_tier1",
            "offtarget_tier2",
            "retrosynthesis",
        ):
            entries = inputs.get(key) or []
            for e in entries:
                if isinstance(e, dict) and "gate" in e:
                    all_verdicts.append(e)

        global_verdicts = [v for v in all_verdicts if v.get("candidate_id") == _GLOBAL]
        per_candidate_verdicts: dict[str, list[dict[str, Any]]] = {}
        for v in all_verdicts:
            cid = v.get("candidate_id") or _GLOBAL
            if cid == _GLOBAL:
                continue
            per_candidate_verdicts.setdefault(cid, []).append(v)

        global_block = [v["verdict"] for v in global_verdicts]
        any_global_fail = VERDICT_FAIL in global_block

        log_entries: list[dict[str, Any]] = []
        tier_counts: dict[str, int] = {
            TIER_REJECT: 0,
            TIER_HIT: 0,
            TIER_OPTIMIZABLE_HIT: 0,
            TIER_LEAD: 0,
            TIER_DE_RISKED_LEAD: 0,
        }
        all_candidates = set(per_candidate_verdicts) or {_GLOBAL}
        for cid in sorted(all_candidates):
            verdicts = list(global_verdicts) + list(per_candidate_verdicts.get(cid, []))
            tier = self._compute_tier(verdicts, any_global_fail=any_global_fail)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            summary = self._summarize(tier, verdicts)
            log_entries.append(
                {
                    "candidate_id": cid,
                    "tier": tier,
                    "gates": verdicts,
                    "summary": summary,
                }
            )

        return {
            "decision_log": log_entries,
            "gate_verdicts": all_verdicts,
            "summary": {
                "candidate_count": len(all_candidates),
                "tier_counts": tier_counts,
                "promotions": [e for e in log_entries if e["tier"] != TIER_REJECT],
                "rejections": [e for e in log_entries if e["tier"] == TIER_REJECT],
            },
        }

    @staticmethod
    def _collect(gate: str, verdicts: list[dict[str, Any]]) -> str | None:
        for v in verdicts:
            if v.get("gate") == gate:
                return v.get("verdict")
        return None

    def _compute_tier(self, verdicts: list[dict[str, Any]], *, any_global_fail: bool) -> str:
        structure = self._collect("structure_readiness", verdicts)
        dock_val = self._collect("docking_validation", verdicts)
        boltz_hit = self._collect("boltz_hit_screen", verdicts)
        consensus = self._collect("consensus_structural_review", verdicts)
        admet = self._collect("early_admet", verdicts)
        ot1 = self._collect("off_target_tier1", verdicts)
        ot2 = self._collect("off_target_tier2", verdicts)
        retro = self._collect("retrosynthesis", verdicts)

        any_fail = any(v.get("verdict") == VERDICT_FAIL for v in verdicts)
        if any_fail or any_global_fail:
            return TIER_REJECT

        # Basic plausibility: a structural signal + no ADMET/off-target fail.
        structural_ok = any(
            v in (VERDICT_PASS, VERDICT_BORDERLINE)
            for v in (dock_val, boltz_hit, consensus)
            if v is not None
        )
        if not structural_ok:
            return TIER_REJECT

        # Hit: structural + admet ok + no off-target critical
        if admet in (VERDICT_PASS, VERDICT_BORDERLINE, None) and ot1 in (
            VERDICT_PASS,
            None,
        ):
            tier = TIER_HIT
        else:
            return TIER_REJECT

        # Optimizable hit: consensus at least borderline + route at least borderline
        if consensus in (VERDICT_PASS, VERDICT_BORDERLINE) and retro in (
            VERDICT_PASS,
            VERDICT_BORDERLINE,
        ):
            tier = TIER_OPTIMIZABLE_HIT

        # Lead: consensus pass + admet pass + ot1 pass + retro pass
        if (
            consensus == VERDICT_PASS
            and admet == VERDICT_PASS
            and ot1 == VERDICT_PASS
            and retro == VERDICT_PASS
        ):
            tier = TIER_LEAD

        # De-risked lead: additionally ot2 pass
        if tier == TIER_LEAD and ot2 == VERDICT_PASS:
            tier = TIER_DE_RISKED_LEAD

        # Don't lose structure info for weak-evidence cases.
        if structure == VERDICT_FAIL:
            return TIER_REJECT

        return tier

    @staticmethod
    def _summarize(tier: str, verdicts: list[dict[str, Any]]) -> str:
        fails = [v["gate"] for v in verdicts if v.get("verdict") == VERDICT_FAIL]
        borderlines = [v["gate"] for v in verdicts if v.get("verdict") == VERDICT_BORDERLINE]
        parts = [f"Tier={tier}"]
        if fails:
            parts.append("failed=" + ",".join(sorted(set(fails))))
        if borderlines:
            parts.append("borderline=" + ",".join(sorted(set(borderlines))))
        if not fails and not borderlines:
            parts.append("all gates pass")
        return "; ".join(parts)


GATE_NODES: list[type[WorkflowNode]] = [
    StructureReadinessGateNode,
    DockingValidationGateNode,
    BoltzHitScreenGateNode,
    ConsensusGateNode,
    EarlyAdmetGateNode,
    OffTargetTier1GateNode,
    OffTargetTier2GateNode,
    RetrosynthesisGateNode,
    DecisionLogNode,
]
