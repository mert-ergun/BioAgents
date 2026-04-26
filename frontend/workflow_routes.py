"""Scientific workflow REST routes (split from ``server`` so tests avoid heavy LangGraph imports)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field

from bioagents.workflows.esm_models import ALLOWED_ESM2_MODEL_NAMES, DEFAULT_ESM2_MODEL_NAME
from bioagents.workflows.executor import WorkflowExecutionError, WorkflowExecutor
from bioagents.workflows.graph import WorkflowGraphError
from bioagents.workflows.preset_catalog import (
    PRESET_BY_ID,
    build_graph_for_preset,
    initial_inputs_for_preset,
    list_preset_api_payloads,
)
from bioagents.workflows.serialization import graph_from_definition, list_node_type_descriptors

logger = logging.getLogger(__name__)

router = APIRouter()


class WorkflowRunRequest(BaseModel):
    """Run a preset scientific DAG (see GET /api/workflows/presets)."""

    preset_id: str = "protein_embedding"
    protein_id: str
    embedding_dim: int = 8
    esm2_model_name: str = DEFAULT_ESM2_MODEL_NAME
    options: dict[str, Any] = Field(default_factory=dict)


class WorkflowRunResponse(BaseModel):
    success: bool
    preset_id: str
    error: str | None = None
    sink_outputs: dict[str, dict[str, Any]] = {}
    node_outputs: dict[str, dict[str, Any]] = {}


CUSTOM_WORKFLOW_MAX_NODES = 48
CUSTOM_WORKFLOW_MAX_EDGES = 160


class CustomWorkflowRunRequest(BaseModel):
    """Run a user-defined DAG (see GET /api/workflows/node-types)."""

    definition: dict[str, Any]
    initial_inputs: dict[str, dict[str, Any]] = Field(default_factory=dict)


@router.get("/api/workflows/presets")
async def list_workflow_presets():
    """Catalog of runnable preset graphs for the UI."""
    return {"presets": list_preset_api_payloads(), "total": len(PRESET_BY_ID)}


@router.get("/api/workflows/node-types")
async def list_workflow_node_types():
    """Registered workflow node types with port schemas (for the workflow builder UI)."""
    types_ = list_node_type_descriptors()
    return {"node_types": types_, "total": len(types_)}


@router.get("/api/workflows/presets/{preset_id}/definition")
async def get_preset_definition(preset_id: str):
    """Serialized graph definition for a preset so the builder can load it.

    Returns ``{definition: {nodes, edges}, initial_inputs, layout, preset_id}``
    where ``definition`` matches the shape accepted by ``graph_from_definition``
    and the ``/api/workflows/run-custom`` endpoint, and ``layout`` supplies
    auto-computed x/y hints for the builder canvas.
    """
    entry = PRESET_BY_ID.get(preset_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown preset_id: {preset_id!r}")
    try:
        graph = build_graph_for_preset(
            preset_id,
            embedding_dim=8,
            esm2_model_name=DEFAULT_ESM2_MODEL_NAME,
            options=None,
        )
    except Exception as exc:
        logger.exception("Failed to build preset graph %s", preset_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    definition = graph.to_definition_dict()
    layout = _compute_preset_layout(graph)

    # Default initial_inputs derived from the preset's declared form fields.
    initial_inputs: dict[str, dict[str, Any]] = {}
    if entry.source_spec:
        # Multi-source scenario: populate every declared input with a placeholder
        # that the UI can edit.
        for spec in entry.source_spec:
            nid = spec["node_id"]
            key = spec["input_key"]
            initial_inputs.setdefault(nid, {})[key] = ""
    else:
        initial_inputs[entry.source_node_id] = {entry.source_input_key: ""}

    return {
        "preset_id": preset_id,
        "name": entry.name,
        "description": entry.description,
        "category": entry.category,
        "definition": definition,
        "layout": layout,
        "initial_inputs": initial_inputs,
    }


def _compute_preset_layout(graph: Any) -> dict[str, dict[str, int]]:
    """Assign (x, y) positions to every node using a longest-path layered layout.

    Column = longest path from any source → node (so all predecessors sit left).
    Within a column, rows are stable-sorted by node id. Spacing matches the
    builder's node footprint (260 x 200).
    """
    import networkx as nx  # local import keeps module import cheap

    nxg = graph.nx_graph
    # Longest-path distance from any root to each node == topological depth.
    depth: dict[str, int] = {}
    for nid in nx.lexicographical_topological_sort(nxg):
        preds = list(nxg.predecessors(nid))
        depth[nid] = 0 if not preds else 1 + max(depth[p] for p in preds)

    by_col: dict[int, list[str]] = {}
    for nid, d in depth.items():
        by_col.setdefault(d, []).append(nid)

    col_w = 320
    row_h = 140
    layout: dict[str, dict[str, int]] = {}
    for col, ids in by_col.items():
        ids.sort()
        for row, nid in enumerate(ids):
            layout[nid] = {"x": 80 + col * col_w, "y": 80 + row * row_h}
    return layout


def validate_custom_workflow_request(request: CustomWorkflowRunRequest) -> None:
    d = request.definition
    if not isinstance(d, dict):
        raise HTTPException(status_code=400, detail="definition must be a JSON object")
    nodes = d.get("nodes")
    edges = d.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise HTTPException(
            status_code=400,
            detail="definition must contain 'nodes' and 'edges' arrays",
        )
    if len(nodes) > CUSTOM_WORKFLOW_MAX_NODES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many nodes (max {CUSTOM_WORKFLOW_MAX_NODES})",
        )
    if len(edges) > CUSTOM_WORKFLOW_MAX_EDGES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many edges (max {CUSTOM_WORKFLOW_MAX_EDGES})",
        )
    seen_ids: set[str] = set()
    for i, spec in enumerate(nodes):
        if not isinstance(spec, dict):
            raise HTTPException(status_code=400, detail=f"nodes[{i}] must be an object")
        nid = spec.get("id")
        if not isinstance(nid, str) or not nid.strip():
            raise HTTPException(
                status_code=400,
                detail=f"nodes[{i}] needs a non-empty string 'id'",
            )
        if nid in seen_ids:
            raise HTTPException(status_code=400, detail=f"Duplicate node id: {nid!r}")
        seen_ids.add(nid)
        if not isinstance(spec.get("type"), str) or not spec["type"].strip():
            raise HTTPException(
                status_code=400,
                detail=f"nodes[{i}] needs a non-empty string 'type'",
            )
        params = spec.get("params")
        if params is not None and not isinstance(params, dict):
            raise HTTPException(status_code=400, detail=f"nodes[{i}].params must be an object")
    for i, spec in enumerate(edges):
        if not isinstance(spec, dict):
            raise HTTPException(status_code=400, detail=f"edges[{i}] must be an object")
        for key in ("source", "target"):
            v = spec.get(key)
            if not isinstance(v, str) or not v.strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"edges[{i}] needs non-empty string '{key}'",
                )
        pm = spec.get("port_map")
        if pm is not None and not isinstance(pm, dict):
            raise HTTPException(status_code=400, detail=f"edges[{i}].port_map must be an object")
        src, tgt = spec["source"].strip(), spec["target"].strip()
        if src not in seen_ids:
            raise HTTPException(status_code=400, detail=f"edges[{i}] unknown source node: {src!r}")
        if tgt not in seen_ids:
            raise HTTPException(status_code=400, detail=f"edges[{i}] unknown target node: {tgt!r}")
    for src_nid in request.initial_inputs:
        if src_nid not in seen_ids:
            raise HTTPException(
                status_code=400,
                detail=f"initial_inputs references unknown node id: {src_nid!r}",
            )


@router.post("/api/workflows/run-custom", response_model=WorkflowRunResponse)
async def run_custom_workflow(request: CustomWorkflowRunRequest):
    """Execute a JSON graph definition built in the workflow builder (or any client)."""
    validate_custom_workflow_request(request)
    try:
        graph = graph_from_definition(request.definition)
        executor = WorkflowExecutor(graph)
        result = executor.run(request.initial_inputs)
    except WorkflowGraphError as exc:
        return WorkflowRunResponse(success=False, preset_id="custom", error=str(exc))
    except WorkflowExecutionError as exc:
        return WorkflowRunResponse(success=False, preset_id="custom", error=str(exc))
    except Exception as exc:
        logger.exception("Custom workflow run failed")
        return WorkflowRunResponse(success=False, preset_id="custom", error=str(exc))

    return WorkflowRunResponse(
        success=True,
        preset_id="custom",
        sink_outputs=result.sink_outputs,
        node_outputs=result.node_outputs,
    )


@router.post("/api/workflows/run", response_model=WorkflowRunResponse)
async def run_scientific_workflow(request: WorkflowRunRequest):
    """Execute a preset workflow synchronously and return per-node and sink outputs."""
    if request.preset_id not in PRESET_BY_ID:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown preset_id: {request.preset_id!r}",
        )
    dim = max(1, min(1280, request.embedding_dim))
    model_name = request.esm2_model_name.strip()
    if model_name not in ALLOWED_ESM2_MODEL_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported esm2_model_name; allowed: {sorted(ALLOWED_ESM2_MODEL_NAMES)}",
        )
    try:
        graph = build_graph_for_preset(
            request.preset_id,
            embedding_dim=dim,
            esm2_model_name=model_name,
            options=request.options,
        )
        executor = WorkflowExecutor(graph)
        initial = initial_inputs_for_preset(request.preset_id, request.protein_id.strip())
        result = executor.run(initial)
    except WorkflowExecutionError as exc:
        return WorkflowRunResponse(
            success=False,
            preset_id=request.preset_id,
            error=str(exc),
        )
    except Exception as exc:
        logger.exception("Workflow run failed")
        return WorkflowRunResponse(
            success=False,
            preset_id=request.preset_id,
            error=str(exc),
        )

    return WorkflowRunResponse(
        success=True,
        preset_id=request.preset_id,
        sink_outputs=result.sink_outputs,
        node_outputs=result.node_outputs,
    )


def include_workflow_routes(app: FastAPI) -> None:
    """Register workflow endpoints on the main FastAPI application."""
    app.include_router(router)
