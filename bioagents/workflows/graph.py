"""Directed workflow graph with DAG validation and edge schema checks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import networkx as nx

from bioagents.workflows.schemas import types_compatible

if TYPE_CHECKING:
    from bioagents.workflows.node import WorkflowNode


class WorkflowGraphError(ValueError):
    """Invalid workflow graph structure or wiring."""


class WorkflowGraph:
    """
    A DAG of :class:`WorkflowNode` instances keyed by string ids.

    Edges carry an optional ``port_map`` mapping ``source_output_key -> target_input_key``.
    When omitted, keys present in both the source output schema and target input schema
    are connected with identity mapping.
    """

    def __init__(self) -> None:
        self._nx: nx.DiGraph = nx.DiGraph()
        self._nodes: dict[str, WorkflowNode] = {}

    def add_node(self, node_id: str, node: WorkflowNode) -> None:
        if node_id in self._nodes:
            raise WorkflowGraphError(f"Duplicate node id: {node_id!r}")
        self._nodes[node_id] = node
        self._nx.add_node(node_id, node=node)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        port_map: dict[str, str] | None = None,
    ) -> None:
        if source_id not in self._nodes or target_id not in self._nodes:
            raise WorkflowGraphError("Both endpoints must be added before adding an edge.")
        src = self._nodes[source_id]
        tgt = self._nodes[target_id]
        resolved = self._resolve_port_map(src, tgt, port_map)
        self._validate_port_map_types(src, tgt, resolved)
        self._nx.add_edge(source_id, target_id, port_map=resolved)

    def _resolve_port_map(
        self,
        source: WorkflowNode,
        target: WorkflowNode,
        port_map: dict[str, str] | None,
    ) -> dict[str, str]:
        if port_map is not None:
            return dict(port_map)
        out_keys = set(source.output_schema)
        in_keys = set(target.input_schema)
        shared = sorted(out_keys & in_keys)
        return {k: k for k in shared}

    def _validate_port_map_types(
        self,
        source: WorkflowNode,
        target: WorkflowNode,
        port_map: dict[str, str],
    ) -> None:
        outs = source.output_schema
        ins = target.input_schema
        for src_key, tgt_key in port_map.items():
            if src_key not in outs:
                raise WorkflowGraphError(
                    f"Port map references unknown source output {src_key!r} "
                    f"on node {source.metadata.name!r}"
                )
            if tgt_key not in ins:
                raise WorkflowGraphError(
                    f"Port map references unknown target input {tgt_key!r} "
                    f"on node {target.metadata.name!r}"
                )
            if not types_compatible(outs[src_key], ins[tgt_key]):
                raise WorkflowGraphError(
                    f"Incompatible types on edge: {src_key!r} ({outs[src_key]}) -> "
                    f"{tgt_key!r} ({ins[tgt_key]})"
                )

    def validate_is_dag(self) -> None:
        if not nx.is_directed_acyclic_graph(self._nx):
            cycles = list(nx.simple_cycles(self._nx))
            raise WorkflowGraphError(f"Workflow graph must be acyclic; found cycles: {cycles}")

    def get_edge_port_map(self, source_id: str, target_id: str) -> dict[str, str]:
        data = self._nx.get_edge_data(source_id, target_id)
        if data is None:
            raise WorkflowGraphError(f"No edge {source_id!r} -> {target_id!r}")
        return cast("dict[str, str]", data["port_map"])

    @property
    def nx_graph(self) -> nx.DiGraph:
        return self._nx

    def node(self, node_id: str) -> WorkflowNode:
        return self._nodes[node_id]

    def node_ids(self) -> list[str]:
        return list(self._nodes.keys())

    def source_node_ids(self) -> list[str]:
        return [n for n in self._nx.nodes() if self._nx.in_degree(n) == 0]

    def sink_node_ids(self) -> list[str]:
        return [n for n in self._nx.nodes() if self._nx.out_degree(n) == 0]

    def predecessors(self, node_id: str) -> list[str]:
        return list(self._nx.predecessors(node_id))

    def topological_order(self) -> list[str]:
        self.validate_is_dag()
        return list(nx.lexicographical_topological_sort(self._nx))

    def to_definition_dict(self) -> dict[str, Any]:
        """Serialize structure only; node instances must be rebuilt via a registry."""
        nodes_spec: list[dict[str, Any]] = []
        for nid in sorted(self._nodes):
            n = self._nodes[nid]
            type_id = getattr(type(n), "workflow_type_id", None)
            if not type_id:
                raise WorkflowGraphError(
                    f"Cannot serialize node {nid!r}: class {type(n).__name__} "
                    "has no workflow_type_id"
                )
            nodes_spec.append({"id": nid, "type": type_id, "params": dict(n.params)})
        edges_spec: list[dict[str, Any]] = []
        for u, v, data in sorted(self._nx.edges(data=True), key=lambda t: (t[0], t[1])):
            edges_spec.append(
                {
                    "source": u,
                    "target": v,
                    "port_map": dict(data["port_map"]),
                }
            )
        return {"nodes": nodes_spec, "edges": edges_spec}
