"""HTTP tests for scientific workflow routes (lightweight app, no LangGraph import chain)."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bioagents.workflows.preset_catalog import PRESET_BY_ID, list_preset_api_payloads
from frontend.workflow_routes import (
    CUSTOM_WORKFLOW_MAX_EDGES,
    CUSTOM_WORKFLOW_MAX_NODES,
    include_workflow_routes,
)


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    include_workflow_routes(app)
    return TestClient(app)


def _mock_uniprot_fasta(mock_get: Mock, text: str = ">sp|P12345|X\nMASLKGFVP") -> None:
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = text
    mock_get.return_value = mock_response


class TestWorkflowPresetsEndpoint:
    def test_get_presets_ok(self, client: TestClient) -> None:
        r = client.get("/api/workflows/presets")
        assert r.status_code == 200
        data = r.json()
        assert "presets" in data and "total" in data
        assert data["total"] == len(PRESET_BY_ID)
        assert data["total"] == len(data["presets"])
        payloads = list_preset_api_payloads()
        assert len(data["presets"]) == len(payloads)
        first = data["presets"][0]
        assert "id" in first and "name" in first


class TestWorkflowNodeTypesEndpoint:
    def test_get_node_types_ok(self, client: TestClient) -> None:
        r = client.get("/api/workflows/node-types")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == len(data["node_types"])
        ids = [t["id"] for t in data["node_types"]]
        assert ids == sorted(ids)
        for t in data["node_types"]:
            assert set(t.keys()) >= {
                "id",
                "name",
                "description",
                "category",
                "version",
                "inputs",
                "outputs",
                "default_params",
            }


class TestRunCustomWorkflowEndpoint:
    def test_run_custom_success(self, client: TestClient) -> None:
        body = {
            "definition": {
                "nodes": [
                    {"id": "fetch", "type": "uniprot_fasta", "params": {}},
                    {"id": "prep", "type": "fasta_preprocess", "params": {}},
                ],
                "edges": [{"source": "fetch", "target": "prep"}],
            },
            "initial_inputs": {"fetch": {"protein_id": "P12345"}},
        }
        with patch("bioagents.tools.proteomics_tools.requests.get") as mock_get:
            _mock_uniprot_fasta(mock_get)
            r = client.post("/api/workflows/run-custom", json=body)
        assert r.status_code == 200
        out = r.json()
        assert out["success"] is True
        assert out["preset_id"] == "custom"
        assert "prep" in out["node_outputs"]
        assert out["node_outputs"]["prep"]["sequence"] == "MASLKGFVP"

    def test_run_custom_graph_error_cycle_returns_body_error(self, client: TestClient) -> None:
        body = {
            "definition": {
                "nodes": [
                    {"id": "a", "type": "reverse_sequence", "params": {}},
                    {"id": "b", "type": "reverse_sequence", "params": {}},
                ],
                "edges": [
                    {"source": "a", "target": "b", "port_map": {"reversed_sequence": "sequence"}},
                    {"source": "b", "target": "a", "port_map": {"reversed_sequence": "sequence"}},
                ],
            },
            "initial_inputs": {},
        }
        r = client.post("/api/workflows/run-custom", json=body)
        assert r.status_code == 200
        out = r.json()
        assert out["success"] is False
        assert out["preset_id"] == "custom"
        assert "acyclic" in (out.get("error") or "").lower()

    def test_run_custom_execution_error_returns_body_error(self, client: TestClient) -> None:
        body = {
            "definition": {
                "nodes": [{"id": "fetch", "type": "uniprot_fasta", "params": {}}],
                "edges": [],
            },
            "initial_inputs": {},
        }
        r = client.post("/api/workflows/run-custom", json=body)
        assert r.status_code == 200
        out = r.json()
        assert out["success"] is False
        assert "missing required input" in (out.get("error") or "").lower()

    def test_run_custom_definition_must_contain_node_edge_lists(self, client: TestClient) -> None:
        r = client.post("/api/workflows/run-custom", json={"definition": {}, "initial_inputs": {}})
        assert r.status_code == 400
        detail = r.json()["detail"]
        assert "nodes" in detail and "edges" in detail

    def test_run_custom_requires_nodes_and_edges_arrays(self, client: TestClient) -> None:
        r = client.post(
            "/api/workflows/run-custom",
            json={"definition": {"nodes": []}, "initial_inputs": {}},
        )
        assert r.status_code == 400
        detail = r.json()["detail"]
        assert "nodes" in detail and "edges" in detail

    def test_run_custom_duplicate_node_id(self, client: TestClient) -> None:
        body = {
            "definition": {
                "nodes": [
                    {"id": "x", "type": "dummy_embedder", "params": {"dim": 1}},
                    {"id": "x", "type": "dummy_embedder", "params": {"dim": 1}},
                ],
                "edges": [],
            },
            "initial_inputs": {},
        }
        r = client.post("/api/workflows/run-custom", json=body)
        assert r.status_code == 400
        assert "Duplicate" in r.json()["detail"]

    def test_run_custom_empty_node_id(self, client: TestClient) -> None:
        body = {
            "definition": {
                "nodes": [{"id": "  ", "type": "dummy_embedder", "params": {}}],
                "edges": [],
            },
            "initial_inputs": {},
        }
        r = client.post("/api/workflows/run-custom", json=body)
        assert r.status_code == 400
        assert "id" in r.json()["detail"]

    def test_run_custom_empty_node_type(self, client: TestClient) -> None:
        body = {
            "definition": {
                "nodes": [{"id": "a", "type": "", "params": {}}],
                "edges": [],
            },
            "initial_inputs": {},
        }
        r = client.post("/api/workflows/run-custom", json=body)
        assert r.status_code == 400
        assert "type" in r.json()["detail"]

    def test_run_custom_node_must_be_object(self, client: TestClient) -> None:
        body = {
            "definition": {"nodes": ["bad"], "edges": []},
            "initial_inputs": {},
        }
        r = client.post("/api/workflows/run-custom", json=body)
        assert r.status_code == 400
        assert "nodes[0]" in r.json()["detail"]

    def test_run_custom_params_must_be_object(self, client: TestClient) -> None:
        body = {
            "definition": {
                "nodes": [{"id": "a", "type": "dummy_embedder", "params": []}],
                "edges": [],
            },
            "initial_inputs": {},
        }
        r = client.post("/api/workflows/run-custom", json=body)
        assert r.status_code == 400
        assert "params" in r.json()["detail"]

    def test_run_custom_edge_must_be_object(self, client: TestClient) -> None:
        body = {
            "definition": {
                "nodes": [{"id": "a", "type": "dummy_embedder", "params": {"dim": 1}}],
                "edges": [1],
            },
            "initial_inputs": {"a": {"sequence": "M"}},
        }
        r = client.post("/api/workflows/run-custom", json=body)
        assert r.status_code == 400
        assert "edges[0]" in r.json()["detail"]

    def test_run_custom_unknown_edge_source(self, client: TestClient) -> None:
        body = {
            "definition": {
                "nodes": [{"id": "a", "type": "dummy_embedder", "params": {"dim": 1}}],
                "edges": [{"source": "missing", "target": "a"}],
            },
            "initial_inputs": {"a": {"sequence": "M"}},
        }
        r = client.post("/api/workflows/run-custom", json=body)
        assert r.status_code == 400
        assert "unknown source" in r.json()["detail"]

    def test_run_custom_unknown_edge_target(self, client: TestClient) -> None:
        body = {
            "definition": {
                "nodes": [{"id": "a", "type": "dummy_embedder", "params": {"dim": 1}}],
                "edges": [{"source": "a", "target": "missing"}],
            },
            "initial_inputs": {"a": {"sequence": "M"}},
        }
        r = client.post("/api/workflows/run-custom", json=body)
        assert r.status_code == 400
        assert "unknown target" in r.json()["detail"]

    def test_run_custom_empty_edge_endpoint(self, client: TestClient) -> None:
        body = {
            "definition": {
                "nodes": [{"id": "a", "type": "dummy_embedder", "params": {"dim": 1}}],
                "edges": [{"source": "", "target": "a"}],
            },
            "initial_inputs": {"a": {"sequence": "M"}},
        }
        r = client.post("/api/workflows/run-custom", json=body)
        assert r.status_code == 400

    def test_run_custom_port_map_must_be_object(self, client: TestClient) -> None:
        body = {
            "definition": {
                "nodes": [
                    {"id": "a", "type": "dummy_embedder", "params": {"dim": 1}},
                    {"id": "b", "type": "export_text_json", "params": {}},
                ],
                "edges": [{"source": "a", "target": "b", "port_map": "no"}],
            },
            "initial_inputs": {"a": {"sequence": "M"}},
        }
        r = client.post("/api/workflows/run-custom", json=body)
        assert r.status_code == 400
        assert "port_map" in r.json()["detail"]

    def test_run_custom_unknown_initial_inputs_key(self, client: TestClient) -> None:
        body = {
            "definition": {
                "nodes": [{"id": "a", "type": "dummy_embedder", "params": {"dim": 1}}],
                "edges": [],
            },
            "initial_inputs": {"ghost": {"sequence": "M"}},
        }
        r = client.post("/api/workflows/run-custom", json=body)
        assert r.status_code == 400
        assert "unknown node" in r.json()["detail"]

    def test_run_custom_too_many_nodes(self, client: TestClient) -> None:
        nodes = [
            {"id": f"n{i}", "type": "dummy_embedder", "params": {"dim": 1}}
            for i in range(CUSTOM_WORKFLOW_MAX_NODES + 1)
        ]
        r = client.post(
            "/api/workflows/run-custom",
            json={"definition": {"nodes": nodes, "edges": []}, "initial_inputs": {}},
        )
        assert r.status_code == 400
        assert "Too many nodes" in r.json()["detail"]

    def test_run_custom_too_many_edges(self, client: TestClient) -> None:
        nodes = [
            {"id": "a", "type": "dummy_embedder", "params": {"dim": 1}},
            {"id": "b", "type": "dummy_embedder", "params": {"dim": 1}},
        ]
        edges = [{"source": "a", "target": "b"} for _ in range(CUSTOM_WORKFLOW_MAX_EDGES + 1)]
        r = client.post(
            "/api/workflows/run-custom",
            json={"definition": {"nodes": nodes, "edges": edges}, "initial_inputs": {}},
        )
        assert r.status_code == 400
        assert "Too many edges" in r.json()["detail"]

    def test_run_custom_unknown_node_type_in_graph(self, client: TestClient) -> None:
        body = {
            "definition": {
                "nodes": [{"id": "z", "type": "not_a_real_type", "params": {}}],
                "edges": [],
            },
            "initial_inputs": {},
        }
        r = client.post("/api/workflows/run-custom", json=body)
        assert r.status_code == 200
        out = r.json()
        assert out["success"] is False
        assert "Unknown workflow node type" in (out.get("error") or "")


class TestRunPresetWorkflowEndpoint:
    def test_run_preset_unknown_returns_404(self, client: TestClient) -> None:
        r = client.post(
            "/api/workflows/run",
            json={
                "preset_id": "no_such_preset_ever",
                "protein_id": "P04637",
                "embedding_dim": 8,
                "esm2_model_name": "esm2_t6_8M_UR50D",
            },
        )
        assert r.status_code == 404
        assert "Unknown preset_id" in r.json()["detail"]

    def test_run_preset_bad_esm_model_returns_400(self, client: TestClient) -> None:
        r = client.post(
            "/api/workflows/run",
            json={
                "preset_id": "protein_embedding",
                "protein_id": "P04637",
                "embedding_dim": 8,
                "esm2_model_name": "not-a-model",
            },
        )
        assert r.status_code == 400
        assert "Unsupported esm2_model_name" in r.json()["detail"]

    @patch("bioagents.tools.proteomics_tools.requests.get")
    def test_run_preset_uniprot_molecular_weight_success(
        self, mock_get: Mock, client: TestClient
    ) -> None:
        _mock_uniprot_fasta(mock_get)
        r = client.post(
            "/api/workflows/run",
            json={
                "preset_id": "uniprot_molecular_weight",
                "protein_id": "P12345",
                "embedding_dim": 8,
                "esm2_model_name": "esm2_t6_8M_UR50D",
            },
        )
        assert r.status_code == 200
        out = r.json()
        assert out["success"] is True
        assert out["preset_id"] == "uniprot_molecular_weight"
        assert "ex" in out["sink_outputs"]

    @patch("bioagents.tools.proteomics_tools.requests.get")
    def test_run_preset_protein_embedding_dummy_success(
        self, mock_get: Mock, client: TestClient
    ) -> None:
        _mock_uniprot_fasta(mock_get)
        r = client.post(
            "/api/workflows/run",
            json={
                "preset_id": "protein_embedding_dummy",
                "protein_id": "P12345",
                "embedding_dim": 4,
                "esm2_model_name": "esm2_t6_8M_UR50D",
            },
        )
        assert r.status_code == 200
        out = r.json()
        assert out["success"] is True
        assert "json" in out["sink_outputs"].get("ex", {})
