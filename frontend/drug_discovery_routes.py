"""Drug-discovery REST routes exposing scenario presets + a scenario runner.

These endpoints are a thin layer over the existing workflow engine. The scenario
graph builders live in :mod:`bioagents.workflows.drug_discovery.scenarios` and
the preset entries are registered in
:mod:`bioagents.workflows.preset_catalog`. This router just:

* surfaces the four drug-discovery scenarios as one-form launchers, and
* translates form-level inputs into per-node ``initial_inputs`` for the
  executor, then returns the decision log plus any other sink outputs.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field

from bioagents.workflows.drug_discovery.scenarios import (
    DECISION_LOG_NODE_ID,
    DOSSIER_NODE_ID,
    SCENARIO_FORM_FIELDS,
    SCENARIO_INFO,
    SCENARIO_OPTIONS_HELP,
    build_scenario_initial_inputs,
)
from bioagents.workflows.drug_discovery.schemas import default_policy_thresholds
from bioagents.workflows.executor import WorkflowExecutionError, WorkflowExecutor
from bioagents.workflows.graph import WorkflowGraphError
from bioagents.workflows.preset_catalog import PRESET_BY_ID, is_drug_discovery_preset

logger = logging.getLogger(__name__)

router = APIRouter()


class DrugDiscoveryRunRequest(BaseModel):
    """Launch a drug-discovery scenario as a single-form run."""

    scenario_id: str = Field(
        ..., description="One of the dd_scenario_* ids from GET /api/drug-discovery/scenarios"
    )
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description=("User-facing form values keyed by SCENARIO_FORM_FIELDS[scenario_id][*].key"),
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Scenario option overrides (objective, n_candidates, rounds). See "
            "SCENARIO_OPTIONS_HELP on GET /api/drug-discovery/scenarios."
        ),
    )
    thresholds: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Per-gate threshold overrides (see default_policy_thresholds()). "
            "Any gate omitted here uses its documented default."
        ),
    )


class DrugDiscoveryRunResponse(BaseModel):
    success: bool
    scenario_id: str
    error: str | None = None
    # Full per-node outputs keyed by node id (useful for the workflow builder
    # users). The scenario runner UI only needs a subset of these:
    node_outputs: dict[str, dict[str, Any]] = {}
    sink_outputs: dict[str, dict[str, Any]] = {}
    # Convenience top-level fields extracted from the decision-log aggregator
    # so the UI can render them without digging through node_outputs.
    decision_log: list[dict[str, Any]] = []
    verdicts: dict[str, list[dict[str, Any]]] = {}
    target_dossier: dict[str, Any] | None = None


def _scenario_metadata_payload() -> list[dict[str, Any]]:
    """Return the list of scenario descriptors served at the GET endpoint."""
    payload: list[dict[str, Any]] = []
    for sid, info in SCENARIO_INFO.items():
        preset = PRESET_BY_ID.get(sid)
        payload.append(
            {
                "id": sid,
                "name": info["name"],
                "description": info["description"],
                "category": getattr(preset, "category", "drug_discovery"),
                "form_fields": SCENARIO_FORM_FIELDS.get(sid, []),
                "options": SCENARIO_OPTIONS_HELP.get(sid, {}),
            }
        )
    return payload


@router.get("/api/drug-discovery/scenarios")
async def list_drug_discovery_scenarios() -> dict[str, Any]:
    """List the four drug-discovery scenarios with their form fields + option help."""
    return {
        "scenarios": _scenario_metadata_payload(),
        "default_thresholds": default_policy_thresholds(),
    }


def _coerce_options(scenario_id: str, options: dict[str, Any]) -> dict[str, Any]:
    """Validate + clamp scenario options using ``SCENARIO_OPTIONS_HELP``."""
    spec = SCENARIO_OPTIONS_HELP.get(scenario_id, {})
    out: dict[str, Any] = {}
    for key, meta in spec.items():
        if key not in options:
            continue
        raw = options[key]
        if meta.get("type") == "str" or key == "objective":
            out[key] = str(raw).strip()
        else:
            try:
                val = int(raw)
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Option {key!r} must be an integer",
                ) from exc
            lo = meta.get("min")
            hi = meta.get("max")
            if lo is not None:
                val = max(lo, val)
            if hi is not None:
                val = min(hi, val)
            out[key] = val
    return out


def _required_fields_missing(scenario_id: str, inputs: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for fld in SCENARIO_FORM_FIELDS.get(scenario_id, []):
        if not fld.get("required"):
            continue
        key = fld["key"]
        val = inputs.get(key)
        if val is None or (isinstance(val, str) and not val.strip()):
            missing.append(key)
    return missing


@router.post("/api/drug-discovery/run", response_model=DrugDiscoveryRunResponse)
async def run_drug_discovery_scenario(
    request: DrugDiscoveryRunRequest,
) -> DrugDiscoveryRunResponse:
    """Execute a drug-discovery scenario synchronously."""
    sid = request.scenario_id
    if sid not in SCENARIO_INFO or not is_drug_discovery_preset(sid):
        raise HTTPException(
            status_code=404,
            detail=f"Unknown drug-discovery scenario_id: {sid!r}",
        )

    missing = _required_fields_missing(sid, request.inputs)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required inputs: {sorted(missing)}",
        )

    options = _coerce_options(sid, request.options)
    if request.thresholds:
        options = {**options, "thresholds": request.thresholds}

    try:
        initial_inputs = build_scenario_initial_inputs(sid, request.inputs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        graph = SCENARIO_INFO[sid]["builder"](options)
        executor = WorkflowExecutor(graph)
        result = executor.run(initial_inputs)
    except WorkflowGraphError as exc:
        return DrugDiscoveryRunResponse(success=False, scenario_id=sid, error=str(exc))
    except WorkflowExecutionError as exc:
        return DrugDiscoveryRunResponse(success=False, scenario_id=sid, error=str(exc))
    except Exception as exc:
        logger.exception("Drug-discovery scenario %s failed", sid)
        return DrugDiscoveryRunResponse(success=False, scenario_id=sid, error=str(exc))

    decision_out = result.node_outputs.get(DECISION_LOG_NODE_ID, {})
    dossier_out = result.node_outputs.get(DOSSIER_NODE_ID, {})

    # Build a {gate_name: [verdict, ...]} map by regrouping the flat list of
    # verdicts emitted by :class:`DecisionLogNode`. This keeps the response
    # self-contained so the UI doesn't have to reach into node_outputs to
    # render the per-gate verdict panel.
    verdicts_by_gate: dict[str, list[dict[str, Any]]] = {}
    for v in decision_out.get("gate_verdicts", []) or []:
        if not isinstance(v, dict):
            continue
        gate = v.get("gate")
        if isinstance(gate, str) and gate:
            verdicts_by_gate.setdefault(gate, []).append(v)

    return DrugDiscoveryRunResponse(
        success=True,
        scenario_id=sid,
        node_outputs=result.node_outputs,
        sink_outputs=result.sink_outputs,
        decision_log=list(decision_out.get("decision_log", []) or []),
        verdicts=verdicts_by_gate,
        target_dossier=(dossier_out.get("target_dossier") if dossier_out else None),
    )


def include_drug_discovery_routes(app: FastAPI) -> None:
    """Register drug-discovery endpoints on the main FastAPI application."""
    app.include_router(router)
