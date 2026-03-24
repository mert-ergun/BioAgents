"""Tests for the scientific DAG workflow layer (``bioagents.workflows``)."""

import json
from typing import Any, ClassVar
from unittest.mock import Mock, patch

import pytest

from bioagents.workflows.executor import WorkflowExecutionError, WorkflowExecutor
from bioagents.workflows.graph import WorkflowGraph, WorkflowGraphError
from bioagents.workflows.node import WorkflowNode
from bioagents.workflows.nodes.dummy_embedder import DummyEmbedderNode
from bioagents.workflows.nodes.esm2_embedder import Esm2EmbedderNode
from bioagents.workflows.nodes.export_json import ExportJsonNode
from bioagents.workflows.nodes.fasta_preprocess import FastaPreprocessorNode
from bioagents.workflows.nodes.uniprot_fasta import UniprotFastaNode
from bioagents.workflows.preset_catalog import (
    WORKFLOW_PRESETS,
    build_graph_for_preset,
    list_preset_api_payloads,
)
from bioagents.workflows.presets import build_protein_embedding_pipeline_graph
from bioagents.workflows.schemas import TYPE_ANY, NodeMetadata, types_compatible
from bioagents.workflows.serialization import (
    NODE_REGISTRY,
    graph_from_definition,
    graph_from_json,
    graph_from_yaml,
    graph_to_json,
    graph_to_yaml,
    list_node_type_descriptors,
    register_node_type,
)


def _sample_fasta() -> str:
    return ">sp|P12345|TEST_HUMAN Test protein\nMASLKGFVP"


class PassthroughNode(WorkflowNode):
    """Minimal str->str node for structural graph tests (not in NODE_REGISTRY)."""

    workflow_type_id: ClassVar[str] = "test_passthrough"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="passthrough",
            description="test",
            version="0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"x": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"x": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"x": inputs["x"]}


class TestPresetCatalog:
    def test_preset_ids_unique_and_count(self) -> None:
        ids = [p.id for p in WORKFLOW_PRESETS]
        assert len(ids) == len(set(ids))
        assert len(ids) >= 28

    @patch("bioagents.tools.proteomics_tools.requests.get")
    def test_sample_preset_builds_and_runs(self, mock_get: Mock) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = _sample_fasta()
        mock_get.return_value = mock_response
        g = build_graph_for_preset(
            "uniprot_molecular_weight",
            embedding_dim=8,
            esm2_model_name="esm2_t6_8M_UR50D",
            options={},
        )
        ex = WorkflowExecutor(g)
        result = ex.run({"fetch": {"protein_id": "P12345"}})
        assert "ex" in result.sink_outputs
        assert "json" in result.sink_outputs["ex"]


class TestWorkflowGraphDag:
    def test_rejects_cycle(self) -> None:
        g = WorkflowGraph()
        n = PassthroughNode()
        g.add_node("a", n)
        g.add_node("b", n)
        g.add_node("c", n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", "a")
        with pytest.raises(WorkflowGraphError, match="acyclic"):
            g.validate_is_dag()

    def test_incompatible_edge_types(self) -> None:
        g = WorkflowGraph()
        g.add_node("src", DummyEmbedderNode())
        g.add_node("dst", FastaPreprocessorNode())
        with pytest.raises(WorkflowGraphError, match="Incompatible types"):
            g.add_edge("src", "dst", {"embedding": "fasta"})


class TestWorkflowExecutorLinear:
    @patch("bioagents.tools.proteomics_tools.requests.get")
    def test_protein_pipeline_mocked_http(self, mock_get: Mock) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = _sample_fasta()
        mock_get.return_value = mock_response

        g = WorkflowGraph()
        g.add_node("fetch", UniprotFastaNode())
        g.add_node("clean", FastaPreprocessorNode())
        g.add_node("embed", DummyEmbedderNode(dim=4))
        g.add_node("export", ExportJsonNode())
        g.add_edge("fetch", "clean")
        g.add_edge("clean", "embed")
        g.add_edge("clean", "export")
        g.add_edge("embed", "export")

        ex = WorkflowExecutor(g)
        result = ex.run({"fetch": {"protein_id": "P12345"}})

        assert "export" in result.sink_outputs
        out = result.sink_outputs["export"]["json"]
        assert "embedding" in out
        assert "residue_count" in out
        assert result.node_outputs["clean"]["residue_count"] == 9
        assert result.node_outputs["embed"]["embedding"] == [0.0, 0.0, 0.0, 0.0]

    @pytest.mark.slow
    @patch("bioagents.tools.proteomics_tools.requests.get")
    def test_protein_pipeline_esm2_mocked_http(self, mock_get: Mock) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("esm")
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = _sample_fasta()
        mock_get.return_value = mock_response

        g = build_protein_embedding_pipeline_graph(use_esm2=True)
        ex = WorkflowExecutor(g)
        result = ex.run({"fetch": {"protein_id": "P12345"}})

        emb = result.node_outputs["embed"]["embedding"]
        assert len(emb) == 320
        assert max(abs(x) for x in emb) > 1e-5

    def test_missing_source_input(self) -> None:
        g = WorkflowGraph()
        g.add_node("fetch", UniprotFastaNode())
        ex = WorkflowExecutor(g)
        with pytest.raises(WorkflowExecutionError, match="missing required input"):
            ex.run({})


class TestSerialization:
    def test_round_trip_json(self) -> None:
        g = WorkflowGraph()
        g.add_node("fetch", UniprotFastaNode(timeout=7))
        g.add_node("clean", FastaPreprocessorNode())
        g.add_edge("fetch", "clean")

        raw = graph_to_json(g)
        g2 = graph_from_definition(json.loads(raw))
        assert g2.node("fetch").params == {"timeout": 7}

        ex = WorkflowExecutor(g2)
        with patch("bioagents.tools.proteomics_tools.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = _sample_fasta()
            mock_get.return_value = mock_response
            result = ex.run({"fetch": {"protein_id": "P12345"}})
        assert result.node_outputs["clean"]["sequence"] == "MASLKGFVP"

    def test_yaml_round_trip(self) -> None:
        g = WorkflowGraph()
        g.add_node("x", DummyEmbedderNode(dim=3))
        yml = graph_to_yaml(g)
        g2 = graph_from_yaml(yml)
        assert g2.node("x").params == {"dim": 3}

    def test_yaml_round_trip_esm2(self) -> None:
        g = WorkflowGraph()
        g.add_node("e", Esm2EmbedderNode(model_name="esm2_t6_8M_UR50D"))
        yml = graph_to_yaml(g)
        g2 = graph_from_yaml(yml)
        assert g2.node("e").params["model_name"] == "esm2_t6_8M_UR50D"

    def test_unknown_node_type_raises(self) -> None:
        with pytest.raises(WorkflowGraphError, match="Unknown workflow node type"):
            graph_from_definition(
                {
                    "nodes": [{"id": "a", "type": "no_such_node", "params": {}}],
                    "edges": [],
                }
            )


class TestPresetApiPayloads:
    def test_list_preset_api_payloads_aligns_with_catalog(self) -> None:
        payloads = list_preset_api_payloads()
        assert len(payloads) == len(WORKFLOW_PRESETS)
        ids = {p["id"] for p in payloads}
        assert ids == {p.id for p in WORKFLOW_PRESETS}
        for row in payloads:
            assert row["id"]
            assert "name" in row and "description" in row


class TestTypesCompatible:
    def test_equal_types(self) -> None:
        assert types_compatible("str", "str") is True

    def test_mismatch(self) -> None:
        assert types_compatible("str", "int") is False

    def test_any_wildcards(self) -> None:
        assert types_compatible(TYPE_ANY, "int") is True
        assert types_compatible("str", TYPE_ANY) is True


class TestGraphFromDefinitionErrors:
    def test_nodes_must_be_list(self) -> None:
        with pytest.raises(WorkflowGraphError, match="nodes' and 'edges' lists"):
            graph_from_definition({"nodes": {}, "edges": []})

    def test_edges_must_be_list(self) -> None:
        with pytest.raises(WorkflowGraphError, match="nodes' and 'edges' lists"):
            graph_from_definition({"nodes": [], "edges": {}})

    def test_node_spec_must_be_dict(self) -> None:
        with pytest.raises(WorkflowGraphError, match="node spec must be a dict"):
            graph_from_definition({"nodes": [1], "edges": []})

    def test_node_requires_string_id_and_type(self) -> None:
        with pytest.raises(WorkflowGraphError, match="string 'id' and 'type'"):
            graph_from_definition(
                {
                    "nodes": [{"id": 1, "type": "dummy_embedder", "params": {}}],
                    "edges": [],
                }
            )

    def test_node_params_must_be_dict(self) -> None:
        with pytest.raises(WorkflowGraphError, match=r"params.*dict"):
            graph_from_definition(
                {
                    "nodes": [{"id": "a", "type": "dummy_embedder", "params": "nope"}],
                    "edges": [],
                }
            )

    def test_edge_spec_must_be_dict(self) -> None:
        with pytest.raises(WorkflowGraphError, match="edge spec must be a dict"):
            graph_from_definition(
                {
                    "nodes": [{"id": "a", "type": "dummy_embedder", "params": {"dim": 1}}],
                    "edges": [()],
                }
            )

    def test_edge_requires_string_endpoints(self) -> None:
        with pytest.raises(WorkflowGraphError, match="string 'source' and 'target'"):
            graph_from_definition(
                {
                    "nodes": [{"id": "a", "type": "dummy_embedder", "params": {"dim": 1}}],
                    "edges": [{"source": 1, "target": "a"}],
                }
            )

    def test_edge_port_map_must_be_dict_or_omitted(self) -> None:
        with pytest.raises(WorkflowGraphError, match="port_map"):
            graph_from_definition(
                {
                    "nodes": [
                        {"id": "a", "type": "dummy_embedder", "params": {"dim": 1}},
                        {"id": "b", "type": "dummy_embedder", "params": {"dim": 1}},
                    ],
                    "edges": [{"source": "a", "target": "b", "port_map": []}],
                }
            )

    def test_edge_unknown_endpoint(self) -> None:
        with pytest.raises(WorkflowGraphError, match="Both endpoints"):
            graph_from_definition(
                {
                    "nodes": [{"id": "a", "type": "dummy_embedder", "params": {"dim": 1}}],
                    "edges": [{"source": "a", "target": "missing"}],
                }
            )


class TestWorkflowGraphOperations:
    def test_duplicate_add_node(self) -> None:
        g = WorkflowGraph()
        n = DummyEmbedderNode()
        g.add_node("x", n)
        with pytest.raises(WorkflowGraphError, match="Duplicate node id"):
            g.add_node("x", n)

    def test_get_edge_port_map_missing(self) -> None:
        g = WorkflowGraph()
        g.add_node("a", UniprotFastaNode())
        g.add_node("b", FastaPreprocessorNode())
        g.add_edge("a", "b")
        with pytest.raises(WorkflowGraphError, match="No edge"):
            g.get_edge_port_map("a", "missing")

    def test_get_edge_port_map_identity_auto_wire(self) -> None:
        g = WorkflowGraph()
        g.add_node("a", UniprotFastaNode())
        g.add_node("b", FastaPreprocessorNode())
        g.add_edge("a", "b")
        assert g.get_edge_port_map("a", "b") == {"fasta": "fasta"}


class TestExecutorFanIn:
    def test_conflicting_values_same_target_port(self) -> None:
        g = WorkflowGraph()
        p = PassthroughNode()
        g.add_node("left", p)
        g.add_node("right", p)
        g.add_node("sink", p)
        g.add_edge("left", "sink", {"x": "x"})
        g.add_edge("right", "sink", {"x": "x"})
        ex = WorkflowExecutor(g)
        with pytest.raises(WorkflowExecutionError, match="Conflicting values"):
            ex.run({"left": {"x": "a"}, "right": {"x": "b"}})


class TestGraphFromJsonYaml:
    def test_graph_from_json_invalid(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            graph_from_json("not json")

    def test_graph_from_yaml_non_mapping_root(self) -> None:
        with pytest.raises(WorkflowGraphError, match="YAML root must be a mapping"):
            graph_from_yaml("- item")


class TestNodeTypeCatalog:
    def test_list_node_type_descriptors_contains_uniprot(self) -> None:
        rows = list_node_type_descriptors()
        ids = {r["id"] for r in rows}
        assert "uniprot_fasta" in ids
        assert "fasta_preprocess" in ids
        u = next(r for r in rows if r["id"] == "uniprot_fasta")
        assert u["inputs"] == {"protein_id": "str"}
        assert u["outputs"] == {"fasta": "str"}

    def test_list_sorted_and_matches_registry(self) -> None:
        rows = list_node_type_descriptors()
        assert len(rows) == len(NODE_REGISTRY)
        ids = [r["id"] for r in rows]
        assert ids == sorted(NODE_REGISTRY.keys())
        for r in rows:
            assert r["default_params"] is not None
            assert isinstance(r["inputs"], dict)
            assert isinstance(r["outputs"], dict)


class TestRegisterNodeType:
    def test_custom_type_in_graph_from_definition(self) -> None:
        register_node_type("test_passthrough", PassthroughNode)
        try:
            g = graph_from_definition(
                {
                    "nodes": [{"id": "a", "type": "test_passthrough", "params": {}}],
                    "edges": [],
                }
            )
            ex = WorkflowExecutor(g)
            result = ex.run({"a": {"x": "hello"}})
            assert result.sink_outputs["a"]["x"] == "hello"
        finally:
            NODE_REGISTRY.pop("test_passthrough", None)


class TestCustomDefinitionRun:
    @patch("bioagents.tools.proteomics_tools.requests.get")
    def test_json_definition_matches_hand_built_graph(self, mock_get: Mock) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = _sample_fasta()
        mock_get.return_value = mock_response

        definition = {
            "nodes": [
                {"id": "fetch", "type": "uniprot_fasta", "params": {}},
                {"id": "prep", "type": "fasta_preprocess", "params": {}},
                {"id": "mw", "type": "molecular_weight", "params": {}},
            ],
            "edges": [
                {"source": "fetch", "target": "prep"},
                {"source": "prep", "target": "mw"},
            ],
        }
        g = graph_from_definition(definition)
        ex = WorkflowExecutor(g)
        result = ex.run({"fetch": {"protein_id": "P12345"}})
        assert "mw" in result.sink_outputs
        assert result.sink_outputs["mw"]["mw_daltons"] > 0
