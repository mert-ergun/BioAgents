"""
analyze_viral_conservation - Rank query proteins by conservation across viruses.

Category: genomics

Aggregates NCBI blastp result JSON files (as produced by ncbi_blastp), filters
hits by identity/e-value and organism exclusions, computes per-target
conservation metrics (number of distinct viral species with qualifying best
hits, mean/median/max/min best-hit percent identity), ranks targets and
selects the most conserved one. Uses only the Python standard library.
"""

import json
import statistics
from pathlib import Path
from typing import Any


def analyze_viral_conservation(
    blast_result_files: str,
    min_identity: float | None = None,
    max_evalue: float | None = None,
    exclude_organism_substrings: str | None = None,
    literature_targets_json: str | None = None,
    output_file: str | None = None,
) -> dict[str, Any]:
    """Aggregate blastp result JSON files and rank query proteins by
    conservation across other viruses.

    Args:
        blast_result_files: JSON array of paths to ncbi_blastp result files.
        min_identity: minimum best-hit percent identity for a species to count.
        max_evalue: maximum e-value for a qualifying hit.
        exclude_organism_substrings: JSON array of case-insensitive organism
            name substrings to exclude (e.g. the query species itself).
        literature_targets_json: optional JSON string of literature-derived
            target candidates embedded as provenance in the summary.
        output_file: optional path to save the conservation summary JSON.

    Returns:
        dict with status, n_targets, per_target metrics, ranking,
        most_conserved_target, filters and output_file when saved.

    Raises:
        ValueError: malformed inputs or unrecognized result layout.
    """
    try:
        files = json.loads(blast_result_files)
        if not isinstance(files, list) or not files:
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError(
            "blast_result_files must be a non-empty JSON array of file paths"
        ) from None

    exclude: list[str] = []
    if exclude_organism_substrings:
        try:
            exclude = [str(s).lower() for s in json.loads(exclude_organism_substrings)]
        except (ValueError, TypeError):
            raise ValueError(
                "exclude_organism_substrings must be a JSON array of strings"
            ) from None

    per_target: list[dict[str, Any]] = []
    for path in files:
        with Path(path).open(encoding="utf-8") as fh:
            data = json.load(fh)
        org_map: dict[str, dict[str, Any]] = {}
        if isinstance(data, dict) and isinstance(data.get("organisms"), dict):
            org_map = data["organisms"]
        elif isinstance(data, dict) and isinstance(data.get("hits"), list):
            for h in data["hits"]:
                org = h.get("organism") or "unknown"
                cur = org_map.get(org)
                if cur is None or h.get("bit_score", 0) > cur.get("bit_score", 0):
                    org_map[org] = h
        elif isinstance(data, list):
            for h in data:
                org = h.get("organism") or "unknown"
                cur = org_map.get(org)
                if cur is None or h.get("bit_score", 0) > cur.get("bit_score", 0):
                    org_map[org] = h
        else:
            raise ValueError(f"Unrecognized blastp result layout in {path}")

        target_name = data.get("target_name") if isinstance(data, dict) else None
        if not target_name:
            target_name = Path(path).stem

        qualifying: dict[str, dict[str, Any]] = {}
        for org, h in org_map.items():
            if any(s in org.lower() for s in exclude):
                continue
            ident = h.get("identity_pct", 0)
            ev = h.get("evalue", None)
            if min_identity is not None and ident < min_identity:
                continue
            if max_evalue is not None and ev is not None and ev > max_evalue:
                continue
            qualifying[org] = h

        idents = [h.get("identity_pct", 0) for h in qualifying.values()]
        top = sorted(qualifying.items(), key=lambda kv: kv[1].get("bit_score", 0), reverse=True)[
            :10
        ]
        per_target.append(
            {
                "target": target_name,
                "file": path,
                "n_viral_species": len(qualifying),
                "mean_identity": round(statistics.mean(idents), 2) if idents else None,
                "median_identity": round(statistics.median(idents), 2) if idents else None,
                "max_identity": round(max(idents), 2) if idents else None,
                "min_identity": round(min(idents), 2) if idents else None,
                "top_species": [
                    {
                        "organism": org,
                        "identity_pct": h.get("identity_pct"),
                        "bit_score": h.get("bit_score"),
                        "evalue": h.get("evalue"),
                        "hit_accession": h.get("hit_accession"),
                    }
                    for org, h in top
                ],
            }
        )

    ranked = sorted(
        per_target,
        key=lambda m: (
            m["mean_identity"] if m["mean_identity"] is not None else -1.0,
            m["n_viral_species"],
        ),
        reverse=True,
    )
    most_conserved = ranked[0]["target"] if ranked else None

    summary = {
        "status": "success",
        "n_targets": len(per_target),
        "per_target": per_target,
        "ranking": [m["target"] for m in ranked],
        "most_conserved_target": most_conserved,
        "filters": {
            "min_identity": min_identity,
            "max_evalue": max_evalue,
            "exclude_organism_substrings": exclude,
        },
        "literature_targets": json.loads(literature_targets_json)
        if literature_targets_json
        else None,
    }

    if output_file:
        parent = str(Path(str(Path(output_file).absolute())).parent)
        if parent:
            Path(parent).mkdir(parents=True, exist_ok=True)
        with Path(output_file).open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        summary["output_file"] = output_file

    return summary
