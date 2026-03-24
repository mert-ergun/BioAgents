"""Load and save workflow graphs as JSON or YAML via a small type registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from bioagents.workflows.graph import WorkflowGraph, WorkflowGraphError

if TYPE_CHECKING:
    from bioagents.workflows.node import WorkflowNode
from bioagents.workflows.nodes.dummy_embedder import DummyEmbedderNode
from bioagents.workflows.nodes.esm2_embedder import Esm2EmbedderNode
from bioagents.workflows.nodes.export_extra import (
    ExportDictJsonNode,
    ExportFastaJsonNode,
    ExportSequenceJsonNode,
    ExportTextJsonNode,
)
from bioagents.workflows.nodes.export_json import ExportJsonNode
from bioagents.workflows.nodes.fasta_preprocess import FastaPreprocessorNode
from bioagents.workflows.nodes.merge_text import MergeThreeTextNode, MergeTwoTextNode
from bioagents.workflows.nodes.protein_analysis_nodes import (
    AliphaticIndexNode,
    AminoAcidCompositionNode,
    AromaticFractionNode,
    ChargedFractionNode,
    CompactBiochemRecordNode,
    GravyScoreNode,
    HydrophobicFractionNode,
    InstabilityIndexNode,
    IsoelectricPointNode,
    KmerProfileNode,
    MolecularWeightNode,
    ReverseSequenceNode,
    SequenceDigestNode,
    SequenceLengthNode,
    SingleResidueCountNode,
    TerminalResidueNode,
)
from bioagents.workflows.nodes.rcsb_summary import RcsbEntrySummaryNode
from bioagents.workflows.nodes.uniprot_fasta import UniprotFastaNode

_NODE_TYPES: list[type[WorkflowNode]] = [
    UniprotFastaNode,
    FastaPreprocessorNode,
    DummyEmbedderNode,
    Esm2EmbedderNode,
    ExportJsonNode,
    ExportFastaJsonNode,
    ExportSequenceJsonNode,
    ExportTextJsonNode,
    ExportDictJsonNode,
    MergeTwoTextNode,
    MergeThreeTextNode,
    RcsbEntrySummaryNode,
    MolecularWeightNode,
    AminoAcidCompositionNode,
    IsoelectricPointNode,
    ReverseSequenceNode,
    KmerProfileNode,
    HydrophobicFractionNode,
    ChargedFractionNode,
    AromaticFractionNode,
    SequenceDigestNode,
    GravyScoreNode,
    AliphaticIndexNode,
    InstabilityIndexNode,
    SingleResidueCountNode,
    TerminalResidueNode,
    SequenceLengthNode,
    CompactBiochemRecordNode,
]

NODE_REGISTRY: dict[str, type[WorkflowNode]] = {cls.workflow_type_id: cls for cls in _NODE_TYPES}


def list_node_type_descriptors() -> list[dict[str, Any]]:
    """
    Metadata for each registered node type (for API / workflow builder UIs).

    Instantiates each class with default constructor arguments to read schemas
    and default ``params``.
    """
    out: list[dict[str, Any]] = []
    for type_id in sorted(NODE_REGISTRY):
        cls = NODE_REGISTRY[type_id]
        sample = cls()
        md = sample.metadata
        out.append(
            {
                "id": type_id,
                "name": md.name,
                "description": md.description,
                "category": md.category,
                "version": md.version,
                "inputs": dict(sample.input_schema),
                "outputs": dict(sample.output_schema),
                "default_params": dict(sample.params),
            }
        )
    return out


def register_node_type(type_id: str, cls: type[WorkflowNode]) -> None:
    """Allow plugins to add node classes before ``graph_from_definition``."""
    NODE_REGISTRY[type_id] = cls


def graph_from_definition(data: dict[str, Any]) -> WorkflowGraph:
    """Build a :class:`WorkflowGraph` from a structure produced by ``to_definition_dict``."""
    g = WorkflowGraph()
    nodes = data.get("nodes")
    edges = data.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise WorkflowGraphError("Definition must contain 'nodes' and 'edges' lists")

    for spec in nodes:
        if not isinstance(spec, dict):
            raise WorkflowGraphError("Each node spec must be a dict")
        nid = spec.get("id")
        type_id = spec.get("type")
        params = spec.get("params") or {}
        if not isinstance(nid, str) or not isinstance(type_id, str):
            raise WorkflowGraphError("Each node needs string 'id' and 'type'")
        if not isinstance(params, dict):
            raise WorkflowGraphError("Node 'params' must be a dict")
        cls = NODE_REGISTRY.get(type_id)
        if cls is None:
            raise WorkflowGraphError(f"Unknown workflow node type: {type_id!r}")
        node = cls(**params)
        g.add_node(nid, node)

    for spec in edges:
        if not isinstance(spec, dict):
            raise WorkflowGraphError("Each edge spec must be a dict")
        src = spec.get("source")
        tgt = spec.get("target")
        if not isinstance(src, str) or not isinstance(tgt, str):
            raise WorkflowGraphError("Each edge needs string 'source' and 'target'")
        port_map = spec.get("port_map")
        pm: dict[str, str] | None
        if port_map is None:
            pm = None
        elif isinstance(port_map, dict):
            pm = {str(k): str(v) for k, v in port_map.items()}
        else:
            raise WorkflowGraphError("Edge 'port_map' must be a dict or omitted")
        g.add_edge(src, tgt, pm)

    return g


def graph_to_json(graph: WorkflowGraph, *, indent: int | None = 2) -> str:
    return json.dumps(graph.to_definition_dict(), indent=indent)


def graph_from_json(raw: str) -> WorkflowGraph:
    return graph_from_definition(json.loads(raw))


def graph_to_yaml(graph: WorkflowGraph) -> str:
    return yaml.safe_dump(graph.to_definition_dict(), sort_keys=False)


def graph_from_yaml(raw: str) -> WorkflowGraph:
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise WorkflowGraphError("YAML root must be a mapping")
    return graph_from_definition(data)


def save_workflow_yaml(graph: WorkflowGraph, path: str | Path) -> None:
    Path(path).write_text(graph_to_yaml(graph), encoding="utf-8")


def load_workflow_yaml(path: str | Path) -> WorkflowGraph:
    return graph_from_yaml(Path(path).read_text(encoding="utf-8"))
