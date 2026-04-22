"""Unit + scenario + API tests for the drug-discovery workflow family.

Covers:
  • Gate-node pass/borderline/fail behaviour for every gate in doc 2.6.
  • DecisionLogNode tier promotion logic (doc 2.7).
  • Scenario graph builders (A/B/C/D) are DAGs with the expected aggregators.
  • JSON round-trip: each scenario serializes → deserializes identically.
  • /api/drug-discovery/scenarios returns the 4 scenarios + default thresholds.
  • /api/drug-discovery/run rejects unknown ids and missing required inputs.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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
from bioagents.workflows.drug_discovery.scenarios import (
    DECISION_LOG_NODE_ID,
    DOSSIER_NODE_ID,
    SCENARIO_FORM_FIELDS,
    SCENARIO_INFO,
    SCENARIO_INPUT_MAPPING,
    build_scenario_initial_inputs,
)
from bioagents.workflows.drug_discovery.schemas import (
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
from bioagents.workflows.preset_catalog import (
    PRESET_BY_ID,
    is_drug_discovery_preset,
)
from bioagents.workflows.serialization import graph_from_definition
from frontend.drug_discovery_routes import include_drug_discovery_routes

# ---------------------------------------------------------------------------
# Gate pass/borderline/fail behaviour (doc section 2.6)
# ---------------------------------------------------------------------------


class TestGateNodes:
    def test_structure_readiness_pass_on_holo(self) -> None:
        node = StructureReadinessGateNode()
        out = node.run({"structure_package": {"kind": "holo"}})
        (v,) = out["verdicts"]
        assert v["verdict"] == VERDICT_PASS

    def test_structure_readiness_borderline_on_predicted(self) -> None:
        node = StructureReadinessGateNode()
        out = node.run({"structure_package": {"kind": "predicted"}})
        (v,) = out["verdicts"]
        assert v["verdict"] == VERDICT_BORDERLINE

    def test_structure_readiness_fail_without_structure(self) -> None:
        node = StructureReadinessGateNode()
        out = node.run({"structure_package": {"kind": "none"}})
        (v,) = out["verdicts"]
        assert v["verdict"] == VERDICT_FAIL

    def test_docking_validation_pass(self) -> None:
        out = DockingValidationGateNode().run(
            {
                "validation": {
                    "has_native_ligand": True,
                    "rmsd": 1.5,
                    "key_contacts_recovered": True,
                }
            }
        )
        assert out["verdicts"][0]["verdict"] == VERDICT_PASS

    def test_docking_validation_fail_high_rmsd(self) -> None:
        out = DockingValidationGateNode().run(
            {
                "validation": {
                    "has_native_ligand": True,
                    "rmsd": 4.0,
                    "key_contacts_recovered": True,
                }
            }
        )
        assert out["verdicts"][0]["verdict"] == VERDICT_FAIL

    def test_docking_validation_borderline_without_native(self) -> None:
        out = DockingValidationGateNode().run({"validation": {"has_native_ligand": False}})
        assert out["verdicts"][0]["verdict"] == VERDICT_BORDERLINE
        assert out["verdicts"][0]["reason_code"] == "no_native_ligand"

    def test_boltz_hit_screen_thresholds(self) -> None:
        out = BoltzHitScreenGateNode().run(
            {
                "boltz_results": [
                    {"smiles": "CCO", "affinity_probability_binary": 0.85},
                    {"smiles": "CCN", "affinity_probability_binary": 0.55},
                    {"smiles": "CCC", "affinity_probability_binary": 0.20},
                ]
            }
        )
        by_smi = {v["candidate_id"]: v["verdict"] for v in out["verdicts"]}
        assert by_smi == {
            "CCO": VERDICT_PASS,
            "CCN": VERDICT_BORDERLINE,
            "CCC": VERDICT_FAIL,
        }

    def test_consensus_gate(self) -> None:
        out = ConsensusGateNode().run(
            {
                "consensus": [
                    {"smiles": "X1", "agreeing_methods": 2},
                    {"smiles": "X2", "agreeing_methods": 1},
                    {"smiles": "X3", "agreeing_methods": 0},
                ]
            }
        )
        by_smi = {v["candidate_id"]: v["verdict"] for v in out["verdicts"]}
        assert by_smi == {"X1": VERDICT_PASS, "X2": VERDICT_BORDERLINE, "X3": VERDICT_FAIL}

    def test_early_admet_gate(self) -> None:
        out = EarlyAdmetGateNode().run(
            {
                "triage": {
                    "rows": [
                        {"smiles": "A", "major_liabilities": 0},
                        {"smiles": "B", "major_liabilities": 1},
                        {"smiles": "C", "major_liabilities": 3},
                    ]
                }
            }
        )
        by_smi = {v["candidate_id"]: v["verdict"] for v in out["verdicts"]}
        assert by_smi == {"A": VERDICT_PASS, "B": VERDICT_BORDERLINE, "C": VERDICT_FAIL}

    def test_off_target_tier1_counts(self) -> None:
        panel = {"critical_flags": []}
        assert (
            OffTargetTier1GateNode().run({"offtarget_panel": panel})["verdicts"][0]["verdict"]
            == VERDICT_PASS
        )
        panel = {"critical_flags": [{"p": "X"}]}
        assert (
            OffTargetTier1GateNode().run({"offtarget_panel": panel})["verdicts"][0]["verdict"]
            == VERDICT_BORDERLINE
        )
        panel = {"critical_flags": [{"p": "X"}, {"p": "Y"}, {"p": "Z"}]}
        assert (
            OffTargetTier1GateNode().run({"offtarget_panel": panel})["verdicts"][0]["verdict"]
            == VERDICT_FAIL
        )

    def test_off_target_tier2_uses_worst_margin(self) -> None:
        out = OffTargetTier2GateNode().run(
            {
                "tier2_report": {
                    "rows": [
                        {"smiles": "S1", "intended_margin_kcal_mol": -0.5},
                        {"smiles": "S1", "intended_margin_kcal_mol": -0.2},
                        {"smiles": "S2", "intended_margin_kcal_mol": 0.4},
                    ]
                }
            }
        )
        by_smi = {v["candidate_id"]: v["verdict"] for v in out["verdicts"]}
        # S1 worst margin is -0.2, below -0.1 threshold → pass
        assert by_smi["S1"] == VERDICT_PASS
        # S2 is positive → offtarget competitive
        assert by_smi["S2"] == VERDICT_FAIL

    def test_off_target_tier2_borderline_when_no_data(self) -> None:
        out = OffTargetTier2GateNode().run({"tier2_report": {"rows": []}})
        assert len(out["verdicts"]) == 1
        assert out["verdicts"][0]["verdict"] == VERDICT_BORDERLINE

    def test_retrosynthesis_gate(self) -> None:
        out = RetrosynthesisGateNode().run(
            {
                "route_summary": {
                    "rows": [
                        {
                            "smiles": "ok",
                            "best_route": True,
                            "route_score": 0.8,
                            "step_count": 5,
                            "purchasable_precursors": True,
                        },
                        {
                            "smiles": "weak",
                            "best_route": True,
                            "route_score": 0.35,
                            "step_count": 10,
                            "purchasable_precursors": False,
                        },
                        {
                            "smiles": "none",
                            "best_route": None,
                            "route_score": 0.0,
                            "step_count": 0,
                            "purchasable_precursors": False,
                        },
                    ]
                }
            }
        )
        by_smi = {v["candidate_id"]: v["verdict"] for v in out["verdicts"]}
        assert by_smi["ok"] == VERDICT_PASS
        assert by_smi["weak"] in {VERDICT_BORDERLINE, VERDICT_FAIL}
        assert by_smi["none"] == VERDICT_FAIL


# ---------------------------------------------------------------------------
# Decision-log tier promotion (doc section 2.7)
# ---------------------------------------------------------------------------


def _v(gate: str, verdict: str, candidate_id: str = "__global__") -> dict:
    return {
        "gate": gate,
        "verdict": verdict,
        "reason_code": "t",
        "reason_text": "t",
        "metrics": {},
        "thresholds": {},
        "candidate_id": candidate_id,
    }


class TestDecisionLogNode:
    def test_all_pass_yields_lead_or_better(self) -> None:
        node = DecisionLogNode()
        out = node.run(
            {
                "structure_readiness": [_v("structure_readiness", VERDICT_PASS)],
                "docking_validation": [_v("docking_validation", VERDICT_PASS)],
                "boltz_hit_screen": [_v("boltz_hit_screen", VERDICT_PASS, "CCO")],
                "consensus": [_v("consensus_structural_review", VERDICT_PASS, "CCO")],
                "early_admet": [_v("early_admet", VERDICT_PASS, "CCO")],
                "offtarget_tier1": [_v("off_target_tier1", VERDICT_PASS)],
                "offtarget_tier2": [_v("off_target_tier2", VERDICT_PASS, "CCO")],
                "retrosynthesis": [_v("retrosynthesis", VERDICT_PASS, "CCO")],
            }
        )
        entries = out["decision_log"]
        assert len(entries) == 1
        assert entries[0]["candidate_id"] == "CCO"
        assert entries[0]["tier"] in {TIER_LEAD, TIER_DE_RISKED_LEAD}

    def test_fail_in_critical_gate_rejects(self) -> None:
        node = DecisionLogNode()
        out = node.run(
            {
                "structure_readiness": [_v("structure_readiness", VERDICT_PASS)],
                "docking_validation": [_v("docking_validation", VERDICT_PASS)],
                "boltz_hit_screen": [_v("boltz_hit_screen", VERDICT_PASS, "X")],
                "consensus": [],
                "early_admet": [_v("early_admet", VERDICT_FAIL, "X")],
                "offtarget_tier1": [_v("off_target_tier1", VERDICT_PASS)],
                "offtarget_tier2": [],
                "retrosynthesis": [_v("retrosynthesis", VERDICT_PASS, "X")],
            }
        )
        assert out["decision_log"][0]["tier"] == TIER_REJECT

    def test_produces_hit_when_only_hit_gates_pass(self) -> None:
        node = DecisionLogNode()
        out = node.run(
            {
                "structure_readiness": [_v("structure_readiness", VERDICT_PASS)],
                "docking_validation": [],
                "boltz_hit_screen": [_v("boltz_hit_screen", VERDICT_PASS, "H")],
                "consensus": [],
                "early_admet": [_v("early_admet", VERDICT_PASS, "H")],
                "offtarget_tier1": [_v("off_target_tier1", VERDICT_PASS)],
                "offtarget_tier2": [],
                "retrosynthesis": [],
            }
        )
        tier = out["decision_log"][0]["tier"]
        # No retrosynthesis + no consensus → not lead; passes early ADMET + tier1
        # + structure readiness + boltz → at least hit/optimizable_hit.
        assert tier in {TIER_HIT, TIER_OPTIMIZABLE_HIT}


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------


class TestScenarioGraphs:
    @pytest.mark.parametrize("scenario_id", sorted(SCENARIO_INFO.keys()))
    def test_each_scenario_builds_dag(self, scenario_id: str) -> None:
        info = SCENARIO_INFO[scenario_id]
        g = info["builder"]({"n_candidates": 2, "rounds": 1})
        # Must validate as a DAG and contain the terminal aggregator nodes.
        g.validate_is_dag()
        node_ids = set(g.node_ids())
        assert DECISION_LOG_NODE_ID in node_ids
        assert DOSSIER_NODE_ID in node_ids

    @pytest.mark.parametrize("scenario_id", sorted(SCENARIO_INFO.keys()))
    def test_scenario_json_round_trip(self, scenario_id: str) -> None:
        info = SCENARIO_INFO[scenario_id]
        g = info["builder"]({"n_candidates": 2, "rounds": 1})
        definition = g.to_definition_dict()
        g2 = graph_from_definition(definition)
        assert sorted(g2.node_ids()) == sorted(g.node_ids())
        assert set(g2.nx_graph.edges()) == set(g.nx_graph.edges())
        # Second round-trip should be byte-stable.
        assert g2.to_definition_dict() == definition

    def test_preset_catalog_registers_scenarios(self) -> None:
        for sid in SCENARIO_INFO:
            assert sid in PRESET_BY_ID, f"{sid} missing from preset catalog"
            assert is_drug_discovery_preset(sid)
        # Sanity: non-DD presets still classified separately.
        assert not is_drug_discovery_preset("protein_embedding")


# ---------------------------------------------------------------------------
# Input-mapping helper
# ---------------------------------------------------------------------------


class TestInitialInputMapping:
    def test_disease_first_maps_two_fields(self) -> None:
        out = build_scenario_initial_inputs(
            "dd_scenario_disease_first",
            {"disease_id": "EFO_0000311", "target_uniprot": "p04637"},
        )
        assert out["brief"]["primary_id"] == "EFO_0000311"
        # uniprot_id is uppercased by the mapper.
        assert out["uni_record"]["uniprot_id"] == "P04637"

    def test_target_first_duplicates_uniprot(self) -> None:
        out = build_scenario_initial_inputs("dd_scenario_target_first", {"uniprot_id": "p00533"})
        assert out["brief"]["primary_id"] == "P00533"
        assert out["uni_record"]["uniprot_id"] == "P00533"

    def test_molecule_first_splits_seed_smiles(self) -> None:
        out = build_scenario_initial_inputs(
            "dd_scenario_molecule_first",
            {"seed_smiles": "CCO, CCN;  CCC ", "target_uniprot": "p23219"},
        )
        # series node gets a list, brief/seed_std get the first entry.
        assert out["series"]["seed_smiles"] == ["CCO", "CCN", "CCC"]
        assert out["brief"]["primary_id"] == "CCO"
        assert out["seed_std"]["smiles"] == "CCO"
        assert out["uni_record"]["uniprot_id"] == "P23219"

    def test_missing_optional_input_is_dropped(self) -> None:
        out = build_scenario_initial_inputs(
            "dd_scenario_target_first", {"uniprot_id": "", "ignored": "x"}
        )
        assert out == {}

    def test_unknown_scenario_raises(self) -> None:
        with pytest.raises(ValueError):
            build_scenario_initial_inputs("dd_unknown", {"x": "y"})


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------


@pytest.fixture
def dd_client() -> TestClient:
    app = FastAPI()
    include_drug_discovery_routes(app)
    return TestClient(app)


class TestScenariosEndpoint:
    def test_lists_four_scenarios(self, dd_client: TestClient) -> None:
        r = dd_client.get("/api/drug-discovery/scenarios")
        assert r.status_code == 200
        data = r.json()
        ids = {s["id"] for s in data["scenarios"]}
        assert ids == set(SCENARIO_INFO.keys())
        # Each scenario must expose form_fields + options.
        for s in data["scenarios"]:
            assert isinstance(s["form_fields"], list)
            assert s["form_fields"] == SCENARIO_FORM_FIELDS[s["id"]]
            assert isinstance(s["options"], dict)
        # Default thresholds cover every gate in doc 2.6.
        assert set(data["default_thresholds"].keys()) == set(default_policy_thresholds().keys())


class TestRunEndpoint:
    def test_unknown_scenario_404(self, dd_client: TestClient) -> None:
        r = dd_client.post(
            "/api/drug-discovery/run",
            json={"scenario_id": "dd_does_not_exist", "inputs": {}},
        )
        assert r.status_code == 404

    def test_missing_required_inputs_400(self, dd_client: TestClient) -> None:
        r = dd_client.post(
            "/api/drug-discovery/run",
            json={"scenario_id": "dd_scenario_target_first", "inputs": {}},
        )
        assert r.status_code == 400
        detail = r.json()["detail"]
        assert "uniprot_id" in detail

    def test_option_clamping(self, dd_client: TestClient) -> None:
        # n_candidates must clamp to allowed range (1..32). We test this by
        # inspecting the GET metadata rather than running the graph (which
        # would hit the network for fetch nodes).
        r = dd_client.get("/api/drug-discovery/scenarios")
        opts = {s["id"]: s["options"] for s in r.json()["scenarios"]}
        target = opts["dd_scenario_target_first"]["n_candidates"]
        assert target["min"] == 1
        assert target["max"] == 32


class TestWorkflowPresetDefinitionEndpoint:
    """Endpoint used by the Workflow Builder to open a scenario as a template."""

    @pytest.fixture
    def wb_client(self) -> TestClient:
        from frontend.workflow_routes import include_workflow_routes

        app = FastAPI()
        include_workflow_routes(app)
        return TestClient(app)

    @pytest.mark.parametrize("sid", list(SCENARIO_INFO.keys()))
    def test_returns_graph_definition_and_layout(self, wb_client: TestClient, sid: str) -> None:
        r = wb_client.get(f"/api/workflows/presets/{sid}/definition")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["preset_id"] == sid
        assert data["category"] == "drug_discovery"
        nodes = data["definition"]["nodes"]
        edges = data["definition"]["edges"]
        assert isinstance(nodes, list) and nodes, "definition must have nodes"
        assert isinstance(edges, list), "definition must have edges array"
        # Every node must have a layout hint.
        layout = data["layout"]
        for n in nodes:
            assert n["id"] in layout, f"missing layout for {n['id']}"
            assert "x" in layout[n["id"]] and "y" in layout[n["id"]]
        # initial_inputs must reference real node ids.
        node_ids = {n["id"] for n in nodes}
        for nid in data["initial_inputs"]:
            assert nid in node_ids

    def test_unknown_preset_404(self, wb_client: TestClient) -> None:
        r = wb_client.get("/api/workflows/presets/does_not_exist/definition")
        assert r.status_code == 404

    def test_definition_is_graph_from_definition_compatible(self, wb_client: TestClient) -> None:
        """Round-trip through graph_from_definition (builder does the same server-side)."""
        from bioagents.workflows.serialization import graph_from_definition

        r = wb_client.get("/api/workflows/presets/dd_scenario_target_first/definition")
        data = r.json()
        g = graph_from_definition(data["definition"])
        assert len(list(g.node_ids())) == len(data["definition"]["nodes"])


class TestScenarioInputMappingMetadata:
    def test_each_mapping_has_matching_form_field(self) -> None:
        for sid, mapping in SCENARIO_INPUT_MAPPING.items():
            form_keys = {f["key"] for f in SCENARIO_FORM_FIELDS[sid]}
            for form_key, _node, _input, _transform in mapping:
                assert form_key in form_keys, (
                    f"Scenario {sid} maps non-existent form key: {form_key}"
                )
