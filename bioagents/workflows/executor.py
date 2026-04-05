"""Topological execution of a :class:`WorkflowGraph`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bioagents.workflows.graph import WorkflowGraph


class WorkflowExecutionError(RuntimeError):
    """Raised when inputs are missing or predecessor outputs conflict."""


@dataclass(frozen=True)
class WorkflowExecutionResult:
    """Outputs for every node and a convenience view of sink nodes only."""

    node_outputs: dict[str, dict[str, Any]]
    sink_outputs: dict[str, dict[str, Any]]


class WorkflowExecutor:
    """
    Runs a validated DAG: source nodes take ``initial_inputs[node_id]``;
    other nodes merge mapped outputs from predecessors (sorted by predecessor id
    for deterministic fan-in). Key collisions across predecessors raise.
    """

    def __init__(self, graph: WorkflowGraph) -> None:
        self._graph = graph

    def run(self, initial_inputs: dict[str, dict[str, Any]]) -> WorkflowExecutionResult:
        self._graph.validate_is_dag()
        order = self._graph.topological_order()
        node_outputs: dict[str, dict[str, Any]] = {}

        for nid in order:
            node = self._graph.node(nid)
            preds = self._graph.predecessors(nid)

            if not preds:
                inputs = dict(initial_inputs.get(nid, {}))
            else:
                inputs = {}
                for pred_id in sorted(preds):
                    port_map = self._graph.get_edge_port_map(pred_id, nid)
                    pred_out = node_outputs[pred_id]
                    for src_key, tgt_key in port_map.items():
                        if src_key not in pred_out:
                            raise WorkflowExecutionError(
                                f"Missing output {src_key!r} from node {pred_id!r}"
                            )
                        val = pred_out[src_key]
                        if tgt_key in inputs and inputs[tgt_key] != val:
                            raise WorkflowExecutionError(
                                f"Conflicting values for input port {tgt_key!r} into node {nid!r}"
                            )
                        inputs[tgt_key] = val

            for req in node.input_schema:
                if req not in inputs:
                    raise WorkflowExecutionError(
                        f"Node {nid!r} missing required input {req!r} (have keys: {sorted(inputs)})"
                    )

            out = node.run(inputs)
            for ok in node.output_schema:
                if ok not in out:
                    raise WorkflowExecutionError(
                        f"Node {nid!r} did not produce declared output {ok!r}"
                    )
            node_outputs[nid] = out

        sinks = self._graph.sink_node_ids()
        sink_outputs = {sid: node_outputs[sid] for sid in sinks}
        return WorkflowExecutionResult(node_outputs=node_outputs, sink_outputs=sink_outputs)
