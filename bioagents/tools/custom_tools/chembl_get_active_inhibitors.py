"""chembl_get_active_inhibitors — sample N active inhibitors of a target from ChEMBL.

Uses the retrying ``ChemblClient`` (timeouts, exponential backoff on 429/5xx/connection
errors, batched molecule lookups) so a transient EBI outage does not fail the run, and
writes outputs under the run workspace so the docking step can find them.
"""

from __future__ import annotations

import csv
import json
import random
from typing import Any

from bioagents.sandbox.workspace import to_portable_path
from bioagents.tools.chembl_client import (
    ChemblClient,
    ChemblError,
    canonical_smiles,
    normalise_activity_types,
)
from bioagents.tools.workspace_paths import resolve_output_dir

_CSV_FIELDS = ["chembl_id", "smiles", "activity_type", "standard_value_nM", "assay_chembl_id"]


def _best_per_molecule(activities: list[dict[str, Any]], max_nM: float) -> list[dict[str, Any]]:
    """Keep the most potent qualifying record per molecule, most potent first."""
    best: dict[str, dict[str, Any]] = {}
    for act in activities:
        mid = act.get("molecule_chembl_id")
        try:
            value = float(act.get("standard_value"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if not mid or value > max_nM:
            continue
        if mid not in best or value < best[mid]["standard_value_nM"]:
            best[mid] = {
                "chembl_id": mid,
                "activity_type": act.get("standard_type"),
                "standard_value_nM": value,
                "assay_chembl_id": act.get("assay_chembl_id") or "",
            }
    return sorted(best.values(), key=lambda r: r["standard_value_nM"])


def _write_outputs(out_dir, stem: str, payload: dict[str, Any]) -> tuple[str, str]:
    # Never clobber: a rerun with different filters produced a different ligand set and
    # replaced the one the rest of the deliverable had already been built from.
    from bioagents.tools.workspace_paths import prepare_output_file

    json_path, _ = prepare_output_file(out_dir / f"{stem}.json")
    csv_path, _ = prepare_output_file(out_dir / f"{stem}.csv")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for rec in payload["molecules"]:
            writer.writerow({k: rec.get(k, "") for k in _CSV_FIELDS})
    return to_portable_path(json_path), to_portable_path(csv_path)


def chembl_get_active_inhibitors(
    target_chembl_id: str = "",
    target_name: str = "",
    n_molecules: int = 10,
    max_activity_nM: float = 1000.0,
    activity_types: str = "IC50,Ki,EC50,Kd",
    random_seed: int = 42,
    output_dir: str = "",
    max_activity_records: int = 2000,
    time_budget_s: float = 180.0,
) -> dict[str, Any]:
    """Retrieve N random active inhibitors of a target from ChEMBL.

    Args:
        target_chembl_id: ChEMBL target ID, e.g. CHEMBL4282 for human AKT1. Required
            unless target_name is given.
        target_name: Target name to search for, e.g. "AKT1". Required unless
            target_chembl_id is given.
        n_molecules: Number of random actives to return.
        max_activity_nM: Potency threshold; records must be '=', '<' or '<=' this value.
        activity_types: Comma-separated activity types (case-insensitive).
        random_seed: Seed for reproducible sampling.
        output_dir: Output directory; relative paths are anchored at the run workspace.
        max_activity_records: Upper bound on activity records fetched (bounds latency;
            EBI serves ~1000 records per 10-40 s).
        time_budget_s: Per-request retry budget in seconds.

    Returns:
        Dict with status, target, molecules (chembl_id, smiles, activity_type,
        standard_value_nM) and the JSON/CSV paths; or status "error" with a message
        stating whether the failure was transient (EBI outage) or a query problem.
    """
    # No default target: this used to fall back to AKT1/CHEMBL4282, so a call that
    # omitted the target returned AKT1 ligands for whatever protein was being studied,
    # under the filename chembl_akt1_ligands.csv.
    if not (target_chembl_id or "").strip() and not (target_name or "").strip():
        return {
            "status": "error",
            "transient": False,
            "message": (
                "No target given. Pass target_chembl_id (e.g. 'CHEMBL4282') or "
                "target_name (e.g. 'AKT1') for the protein you are actually studying."
            ),
        }

    client = ChemblClient(time_budget=time_budget_s)
    types = normalise_activity_types(activity_types)
    try:
        target = client.resolve_target(target_chembl_id, target_name)
        tid = target["target_chembl_id"]
        activities = client.get_activities(
            tid,
            activity_types=types,
            max_value_nM=max_activity_nM,
            max_records=max_activity_records,
        )
        candidates = _best_per_molecule(activities, float(max_activity_nM))
        if not candidates:
            return {
                "status": "error",
                "transient": False,
                "message": f"No activities for {tid} with {types} <= {max_activity_nM} nM.",
            }

        pool = list(candidates)
        random.Random(random_seed).shuffle(pool)  # nosec B311 - reproducible sampling
        selected: list[dict[str, Any]] = []
        # Fetch SMILES in batches until enough molecules have a structure.
        for start in range(0, len(pool), 50):
            if len(selected) >= n_molecules:
                break
            chunk = pool[start : start + 50]
            molecules = client.get_molecules([r["chembl_id"] for r in chunk])
            for rec in chunk:
                smiles = canonical_smiles(molecules.get(rec["chembl_id"], {}))
                if smiles and len(selected) < n_molecules:
                    selected.append({**rec, "smiles": smiles})
    except ChemblError as exc:
        return {"status": "error", "transient": exc.transient, "message": str(exc)}

    if len(selected) < n_molecules:
        return {
            "status": "error",
            "transient": False,
            "message": f"Only {len(selected)} of {n_molecules} molecules have a SMILES "
            f"({len(candidates)} candidates).",
        }

    payload = {
        "target_chembl_id": tid,
        "target_pref_name": target.get("pref_name"),
        "organism": target.get("organism"),
        "target_name": target_name,
        "n_candidates": len(candidates),
        "n_molecules": len(selected),
        "max_activity_nM": max_activity_nM,
        "activity_types": types,
        "random_seed": random_seed,
        "molecules": selected,
    }
    out_dir = resolve_output_dir(output_dir or None)
    stem = f"chembl_{str(target_name or tid).lower()}_ligands"
    json_file, csv_file = _write_outputs(out_dir, stem, payload)
    return {
        "status": "success",
        "target_chembl_id": tid,
        "target_pref_name": target.get("pref_name"),
        "n_candidates": len(candidates),
        "n_molecules": len(selected),
        "molecules": [
            {k: m[k] for k in ("chembl_id", "smiles", "activity_type", "standard_value_nM")}
            for m in selected
        ],
        "json_file": json_file,
        "csv_file": csv_file,
    }
