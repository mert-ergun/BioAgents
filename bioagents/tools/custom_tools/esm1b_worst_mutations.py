"""
esm1b_worst_mutations - Extract the worst-scoring mutants from ESM-1b fitness
artifacts written by esm1b_fitness_scores.

Category: ml

Reads esm1b_fitness_summary.json and/or esm1b_fitness_scores.csv from an
output directory and returns the N mutants with the lowest log-likelihood
ratios (most fitness-reducing), in a compact normalized form suitable for
downstream literature-comparison tools. Read-only: never modifies artifacts.
Column/key names are matched defensively so schema variations are tolerated.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_WT_KEYS = ["wild_type", "wildtype", "wt", "wt_aa", "wild_type_aa", "original"]
_MUT_KEYS = ["mutant", "mut", "mut_aa", "mutant_aa", "alt", "alternative", "substitution"]
_POS_KEYS = ["position", "pos", "residue", "site", "location"]
_LLR_KEYS = ["llr", "log_likelihood_ratio", "loglik_change", "delta_loglik", "log_ratio"]
_RATIO_KEYS = ["fitness_ratio", "ratio", "exp_llr", "fitness"]


def _pick(row: dict[str, Any], candidates: list[str], default: Any = None) -> Any:
    lowered = {str(k).lower(): v for k, v in row.items()}
    for key in candidates:
        if key in lowered:
            return lowered[key]
    for key in candidates:
        for k, v in lowered.items():
            if key in k:
                return v
    return default


def _to_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def esm1b_worst_mutations(output_dir: str, top_n: int = 5) -> dict[str, Any]:
    """Return the worst-N mutants by ESM-1b LLR from the scoring artifacts.

    Args:
        output_dir: Directory containing esm1b_fitness_summary.json and/or
            esm1b_fitness_scores.csv produced by esm1b_fitness_scores.
        top_n: Number of worst mutants to return (default 5). Lower LLR =
            worse fitness change; ties broken by CSV order.

    Returns:
        dict with status, output_dir, n_mutants_total, model_backend,
        worst_mutations: list of {rank, mutation, position, wild_type, mutant,
        llr, fitness_ratio} sorted from worst (lowest LLR) upward, and
        artifacts paths found.

    Raises:
        ValueError: if output_dir is invalid or artifacts contain no usable
            score rows.
        FileNotFoundError: if neither artifact file exists.

    Example:
        esm1b_worst_mutations("results/esm1b", top_n=5)
    """
    if not output_dir or not isinstance(output_dir, str):
        raise ValueError("output_dir must be a non-empty directory path string")
    if not Path(output_dir).is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    summary_path = str(Path(output_dir).joinpath("esm1b_fitness_summary.json"))
    scores_path = str(Path(output_dir).joinpath("esm1b_fitness_scores.csv"))

    summary: dict[str, Any] = {}
    if Path(summary_path).exists():
        with Path(summary_path).open() as fh:
            summary = json.load(fh)

    all_rows: list[dict[str, Any]] = []
    if Path(scores_path).exists():
        with Path(scores_path).open(newline="") as fh:
            all_rows = list(csv.DictReader(fh))
        if not all_rows:
            raise ValueError(f"esm1b_fitness_scores.csv in {output_dir} is empty")
    elif not summary:
        raise FileNotFoundError(
            f"No esm1b_fitness_scores.csv or esm1b_fitness_summary.json found in {output_dir}"
        )

    if not all_rows and summary:
        for key in ("scores", "all_scores", "results", "rows", "mutations"):
            v = summary.get(key)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                all_rows = v
                break

    worst_raw: list[Any] = []
    if summary:
        for key in ("worst_mutants", "worst_5", "worst5", "top_worst", "worst"):
            v = summary.get(key)
            if isinstance(v, list) and v:
                worst_raw = v
                break

    normalized: list[dict[str, Any]] = []
    if worst_raw:
        for i, item in enumerate(worst_raw[:top_n], 1):
            if isinstance(item, str):
                normalized.append({"rank": i, "mutation": item})
                continue
            if not isinstance(item, dict):
                continue
            pos = _to_float(_pick(item, _POS_KEYS, 0))
            pos = int(pos) if pos is not None else 0
            wt = str(_pick(item, _WT_KEYS, "")).upper()[:1]
            mut = str(_pick(item, _MUT_KEYS, "")).upper()[:1]
            mutstr = item.get("mutation") or f"{wt}{pos:d}{mut}"
            normalized.append(
                {
                    "rank": item.get("rank", i),
                    "mutation": mutstr,
                    "position": pos,
                    "wild_type": wt,
                    "mutant": mut,
                    "llr": _to_float(_pick(item, _LLR_KEYS)),
                    "fitness_ratio": _to_float(_pick(item, _RATIO_KEYS)),
                }
            )
    else:
        if not all_rows:
            raise ValueError(f"No usable score rows found in artifacts in {output_dir}")

        def llr_of(row: dict[str, Any]) -> float:
            v = _to_float(_pick(row, _LLR_KEYS))
            if v is None:
                raise ValueError("No LLR-like column found in score rows")
            return v

        srt = sorted(all_rows, key=llr_of)
        for i, row in enumerate(srt[:top_n], 1):
            pos = _to_float(_pick(row, _POS_KEYS, 0))
            pos = int(pos) if pos is not None else 0
            wt = str(_pick(row, _WT_KEYS, "")).upper()[:1]
            mut = str(_pick(row, _MUT_KEYS, "")).upper()[:1]
            normalized.append(
                {
                    "rank": i,
                    "mutation": f"{wt}{pos:d}{mut}",
                    "position": pos,
                    "wild_type": wt,
                    "mutant": mut,
                    "llr": llr_of(row),
                    "fitness_ratio": _to_float(_pick(row, _RATIO_KEYS)),
                }
            )
        worst_raw = srt

    result: dict[str, Any] = {
        "status": "success",
        "output_dir": output_dir,
        "n_mutants_total": summary.get("n_mutants", summary.get("n_scores", len(all_rows))),
        "model_backend": summary.get("model_backend", summary.get("backend")),
        "worst_mutations": normalized,
        "artifacts": {
            "summary_json": summary_path if Path(summary_path).exists() else None,
            "scores_csv": scores_path if Path(scores_path).exists() else None,
        },
    }
    return result
