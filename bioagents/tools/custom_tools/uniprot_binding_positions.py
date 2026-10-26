"""
uniprot_binding_positions - Extract ligand-binding residue positions from a
UniProt features JSON file produced by uniprot_get_features.

Category: utilities

Reads the compact features JSON (either a bare list or {"features": [...]})
and returns the sorted unique positions of all "Binding site" features, the
ligand(s) bound at each position, and a ready-to-use positions list for
saturation-mutagenesis tools (e.g. esm1b_fitness_scores). Can additionally
check the existence of other artifact paths so later pipeline steps are not
repeated unnecessarily. Read-only: never modifies artifacts.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def uniprot_binding_positions(
    features_json: str,
    check_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Extract binding-site positions from a UniProt features JSON file.

    Args:
        features_json: Path to the compact JSON saved by uniprot_get_features
            (must contain "Binding site" features, e.g. the output of
            feature_types="binding").
        check_paths: Optional list of file paths to check for existence
            (e.g. to detect whether a downstream step already ran).

    Returns:
        dict with status, source_file, n_binding_features, n_positions,
        positions (sorted list of ints), positions_csv, position_ligands
        (mapping position -> list of ligand names), path_checks.

    Raises:
        ValueError: if features_json is missing/empty or contains no
            binding-site features.
        FileNotFoundError: if the JSON file does not exist.

    Example:
        uniprot_binding_positions(
            "generated_artifacts/experiments/.../cdk2_P24941_binding_sites.json")
    """
    if not features_json or not isinstance(features_json, str):
        raise ValueError("features_json must be a non-empty file path string")
    if not Path(features_json).exists():
        raise FileNotFoundError(f"Features JSON not found: {features_json}")

    with Path(features_json).open() as fh:
        data = json.load(fh)

    feats: list[dict[str, Any]] = []
    if isinstance(data, list):
        feats = data
    elif isinstance(data, dict):
        feats = data.get("features") or []

    pos_ligands: dict[int, list[str]] = {}
    n_binding = 0
    for f in feats:
        if not isinstance(f, dict) or f.get("type") != "Binding site":
            continue
        n_binding += 1
        start = None
        loc = f.get("location") or {}
        if isinstance(loc, dict):
            s = loc.get("start") or {}
            if isinstance(s, dict):
                start = s.get("value")
        if start is None:
            start = f.get("start")
        if start is None:
            logger.warning("Binding site feature without start position skipped: %r", f)
            continue
        pos = int(start)
        ligand = f.get("ligand") or f.get("description") or "unknown"
        ligands = pos_ligands.setdefault(pos, [])
        if ligand not in ligands:
            ligands.append(ligand)

    if not pos_ligands:
        raise ValueError(f"No 'Binding site' features found in {features_json}")

    positions = sorted(pos_ligands)
    result: dict[str, Any] = {
        "status": "success",
        "source_file": features_json,
        "n_binding_features": n_binding,
        "n_positions": len(positions),
        "positions": positions,
        "positions_csv": ",".join(str(p) for p in positions),
        "position_ligands": {str(p): pos_ligands[p] for p in positions},
    }
    if check_paths:
        result["path_checks"] = {p: bool(Path(p).exists()) for p in check_paths}
    return result
