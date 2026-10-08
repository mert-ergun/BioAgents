"""Descriptors, similarity, filtering, fingerprints and dataset statistics."""

from __future__ import annotations

import logging
import statistics
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from bioagents.tools.rdkit_native._common import (
    invalid_smiles_result,
    parse_smiles,
    require_rdkit,
    split_smiles_list,
)

logger = logging.getLogger(__name__)

#: Descriptor name -> callable. Names match what the tool docstrings advertise, so an
#: agent asking for "MW" or "logP" gets what it asked for.
DESCRIPTOR_FIELDS = (
    "MW",
    "logP",
    "TPSA",
    "HBA",
    "HBD",
    "RotBonds",
    "Rings",
    "AromaticRings",
    "HeavyAtoms",
    "FractionCSP3",
    "QED",
    "SA_Score",
    "Formula",
)


@lru_cache(maxsize=1)
def _sascorer():
    """Load RDKit's contributed synthetic-accessibility scorer.

    SA_Score ships inside the RDKit distribution but lives in Contrib, which is not on
    the import path by default. Agents that could not find it have previously faked the
    score with a random number, so it is wired up properly here.
    """
    from rdkit.Chem import RDConfig

    contrib = str(Path(RDConfig.RDContribDir) / "SA_Score")
    if contrib not in sys.path:
        sys.path.append(contrib)
    import sascorer

    return sascorer


def _descriptor_values(mol) -> dict[str, Any]:
    from rdkit.Chem import QED as qed_module
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    values: dict[str, Any] = {
        "MW": round(Descriptors.MolWt(mol), 4),
        "logP": round(Crippen.MolLogP(mol), 4),
        "TPSA": round(Descriptors.TPSA(mol), 4),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "Rings": rdMolDescriptors.CalcNumRings(mol),
        "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "HeavyAtoms": mol.GetNumHeavyAtoms(),
        "FractionCSP3": round(rdMolDescriptors.CalcFractionCSP3(mol), 4),
        "Formula": rdMolDescriptors.CalcMolFormula(mol),
    }
    try:
        values["QED"] = round(qed_module.qed(mol), 4)
    except Exception as exc:
        logger.debug("QED failed: %s", exc)
        values["QED"] = None
    try:
        values["SA_Score"] = round(_sascorer().calculateScore(mol), 4)
    except Exception as exc:
        # Report the gap rather than substituting a plausible number.
        logger.debug("SA_Score unavailable: %s", exc)
        values["SA_Score"] = None
        values["SA_Score_error"] = str(exc)
    return values


def compute_descriptors(
    smiles_list: list[str] | str,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    """Compute molecular descriptors for one or more molecules.

    Returns:
        {"molecules": [{"smiles": str, "MW": float, "logP": float, ...}], ...}
    """
    require_rdkit()
    requested = [f.strip() for f in fields if f.strip()] if fields else None
    if requested:
        unknown = [f for f in requested if f not in DESCRIPTOR_FIELDS]
        if unknown:
            return {
                "error": (
                    f"Unknown descriptor field(s): {', '.join(unknown)}. "
                    f"Available: {', '.join(DESCRIPTOR_FIELDS)}."
                ),
                "molecules": [],
            }

    molecules: list[dict[str, Any]] = []
    for smiles in split_smiles_list(smiles_list):
        mol = parse_smiles(smiles)
        if mol is None:
            molecules.append(invalid_smiles_result(smiles))
            continue
        values = _descriptor_values(mol)
        if requested:
            values = {k: v for k, v in values.items() if k in requested}
        molecules.append({"smiles": smiles, "valid": True, **values})

    return {
        "molecules": molecules,
        "count": len(molecules),
        "fields": requested or list(DESCRIPTOR_FIELDS),
    }


def _morgan_generator(radius: int, nbits: int):
    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)


def _fingerprint(mol, fp_type: str = "Morgan", radius: int = 2, nbits: int = 2048):
    from rdkit.Chem import rdFingerprintGenerator

    if fp_type.strip().lower() in ("morgan", "ecfp", "circular"):
        return _morgan_generator(radius, nbits).GetFingerprint(mol)
    generator = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=nbits)
    return generator.GetFingerprint(mol)


def compute_fingerprint(
    smiles: str,
    fp_type: str = "Morgan",
    radius: int = 2,
    nbits: int = 2048,
) -> dict[str, Any]:
    """Compute a Morgan (circular) or topological RDKit fingerprint."""
    require_rdkit()
    mol = parse_smiles(smiles)
    if mol is None:
        return invalid_smiles_result(smiles)

    fingerprint = _fingerprint(mol, fp_type, radius, nbits)
    on_bits = list(fingerprint.GetOnBits())
    return {
        "smiles": smiles,
        "type": fp_type,
        "radius": radius if fp_type.strip().lower() != "topological" else None,
        "nbits": nbits,
        "num_on_bits": len(on_bits),
        "on_bits": on_bits,
        "fingerprint": fingerprint.ToBitString(),
    }


def search_similar_molecules(
    query: str,
    targets: list[str],
    threshold: float = 0.5,
    top: int = 5,
) -> dict[str, Any]:
    """Rank target molecules by Tanimoto similarity to a query molecule."""
    from rdkit import DataStructs

    require_rdkit()
    query_mol = parse_smiles(query)
    if query_mol is None:
        return {"query": query, "error": "Query SMILES could not be parsed.", "hits": []}

    query_fp = _fingerprint(query_mol)
    target_list = split_smiles_list(targets)

    hits: list[dict[str, Any]] = []
    unparseable: list[str] = []
    for smiles in target_list:
        mol = parse_smiles(smiles)
        if mol is None:
            unparseable.append(smiles)
            continue
        similarity = DataStructs.TanimotoSimilarity(query_fp, _fingerprint(mol))
        if similarity >= threshold:
            hits.append({"smiles": smiles, "similarity": round(float(similarity), 4)})

    hits.sort(key=lambda h: h["similarity"], reverse=True)
    return {
        "query": query,
        "targets_count": len(target_list),
        "threshold": threshold,
        "metric": "Tanimoto over Morgan(radius=2, 2048 bits)",
        "hits": hits[:top],
        "hit_count": len(hits),
        "unparseable_targets": unparseable,
    }


# Lipinski's Rule of Five, as the tool docstring advertises it.
LIPINSKI_LIMITS = {"mw_max": 500.0, "logp_max": 5.0, "hba_max": 10, "hbd_max": 5}


def filter_molecules(
    smiles_list: list[str],
    mw_min: float | None = None,
    mw_max: float | None = None,
    logp_min: float | None = None,
    logp_max: float | None = None,
    tpsa_min: float | None = None,
    tpsa_max: float | None = None,
    hbd_max: int | None = None,
    hba_max: int | None = None,
    lipinski: bool = False,
) -> dict[str, Any]:
    """Filter molecules by descriptor ranges, or by Lipinski's Rule of Five."""
    require_rdkit()
    if lipinski:
        mw_max = LIPINSKI_LIMITS["mw_max"]
        logp_max = LIPINSKI_LIMITS["logp_max"]
        hba_max = int(LIPINSKI_LIMITS["hba_max"])
        hbd_max = int(LIPINSKI_LIMITS["hbd_max"])

    constraints = {
        "mw_min": mw_min,
        "mw_max": mw_max,
        "logp_min": logp_min,
        "logp_max": logp_max,
        "tpsa_min": tpsa_min,
        "tpsa_max": tpsa_max,
        "hbd_max": hbd_max,
        "hba_max": hba_max,
    }

    candidates = split_smiles_list(smiles_list)
    passed: list[str] = []
    rejected: list[dict[str, Any]] = []
    unparseable: list[str] = []

    for smiles in candidates:
        mol = parse_smiles(smiles)
        if mol is None:
            unparseable.append(smiles)
            continue
        values = _descriptor_values(mol)
        failures = _constraint_failures(values, constraints)
        if failures:
            rejected.append({"smiles": smiles, "failed": failures})
        else:
            passed.append(smiles)

    return {
        "input_count": len(candidates),
        "filtered_count": len(passed),
        "filtered_smiles": passed,
        "rejected": rejected,
        "unparseable": unparseable,
        "applied_constraints": (
            "Lipinski Rule of Five"
            if lipinski
            else {k: v for k, v in constraints.items() if v is not None}
        ),
    }


def _constraint_failures(values: dict[str, Any], constraints: dict[str, Any]) -> list[str]:
    """Return a readable list of the constraints a molecule violated."""
    checks = (
        ("mw_min", "MW", lambda v, limit: v < limit, ">="),
        ("mw_max", "MW", lambda v, limit: v > limit, "<="),
        ("logp_min", "logP", lambda v, limit: v < limit, ">="),
        ("logp_max", "logP", lambda v, limit: v > limit, "<="),
        ("tpsa_min", "TPSA", lambda v, limit: v < limit, ">="),
        ("tpsa_max", "TPSA", lambda v, limit: v > limit, "<="),
        ("hbd_max", "HBD", lambda v, limit: v > limit, "<="),
        ("hba_max", "HBA", lambda v, limit: v > limit, "<="),
    )
    failures = []
    for key, field, violated, operator in checks:
        limit = constraints.get(key)
        if limit is None:
            continue
        value = values.get(field)
        if value is not None and violated(value, limit):
            failures.append(f"{field}={value} violates {field} {operator} {limit}")
    return failures


def dataset_statistics(smiles_list: list[str]) -> dict[str, Any]:
    """Summarise descriptor distributions across a set of molecules."""
    require_rdkit()
    candidates = split_smiles_list(smiles_list)
    collected: dict[str, list[float]] = {}
    unparseable: list[str] = []

    for smiles in candidates:
        mol = parse_smiles(smiles)
        if mol is None:
            unparseable.append(smiles)
            continue
        for field, value in _descriptor_values(mol).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                collected.setdefault(field, []).append(float(value))

    stats: dict[str, dict[str, float]] = {}
    for field, values in collected.items():
        stats[field] = {
            "mean": round(statistics.fmean(values), 4),
            "median": round(statistics.median(values), 4),
            "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "n": len(values),
        }

    return {
        "molecule_count": len(candidates) - len(unparseable),
        "input_count": len(candidates),
        "unparseable": unparseable,
        "descriptors_stats": stats,
    }
