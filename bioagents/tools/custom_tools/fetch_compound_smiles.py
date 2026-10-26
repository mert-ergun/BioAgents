"""fetch_compound_smiles.py

Resolve a compound name (e.g. 'capivasertib') to its SMILES and basic identity
using the PubChem PUG REST API, with an automatic ChEMBL web service fallback.

Requires only the 'requests' package.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import requests

__all__ = ["fetch_compound_smiles"]

logger = logging.getLogger(__name__)

PUBCHEM_PROPERTY_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/"
    "CanonicalSMILES,IsomericSMILES,MolecularFormula,MolecularWeight/JSON"
)

CHEMBL_SEARCH_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json"

REQUEST_TIMEOUT = 30  # seconds
USER_AGENT = "fetch_compound_smiles-wrapper/1.0 (bioagents tool; python-requests)"


def _resolve_path(path: str) -> str:
    return str(Path(os.path.expandvars(path)).expanduser().absolute())


def _versioned_path(path: str) -> str:
    """Return a non-clobbering versioned path (appends _1, _2, ... if needed)."""
    if not Path(path).exists():
        return path
    stem, ext = os.path.splitext(path)  # noqa: PTH122 -- preserve legacy suffix handling
    i = 1
    while True:
        candidate = f"{stem}_{i}{ext}"
        if not Path(candidate).exists():
            return candidate
        i += 1


def fetch_compound_smiles(name: str, output_file: str | None = None) -> dict:
    """Resolve a compound *name* to SMILES/identity via PubChem (ChEMBL fallback).

    Queries PubChem PUG REST first; if no record is found there, falls back to
    the ChEMBL web service name search. Never fabricates a SMILES: if both
    sources fail, raises RuntimeError.

    Args:
        name: Compound name to resolve, e.g. "capivasertib".
        output_file: Optional JSON path to save the resolved record.

    Returns:
        Dict with name, cid/chembl_id, isomeric/canonical SMILES, molecular
        formula/weight, and 'sources_queried'/'sources_succeeded'.

    Raises:
        RuntimeError: If both PubChem and ChEMBL fail to return a usable record.
        ValueError: If *name* is empty.
    """
    if not name or not str(name).strip():
        raise ValueError("Compound name must be a non-empty string")
    name = str(name).strip()
    headers = {"User-Agent": USER_AGENT}
    sources_queried = []
    sources_succeeded = []
    record: dict[str, Any] | None = None

    # ---- Primary: PubChem PUG REST ---------------------------------------
    pubchem_url = PUBCHEM_PROPERTY_URL.format(name=name)
    try:
        resp = requests.get(pubchem_url, headers=headers, timeout=REQUEST_TIMEOUT)
        sources_queried.append("pubchem")
        if resp.status_code == 200:
            data = resp.json()
            table = data.get("PropertyTable", {}).get("Properties", [])
            if table:
                props = table[0]
                smi = props.get("IsomericSMILES") or props.get("CanonicalSMILES")
                if smi:
                    sources_succeeded.append("pubchem")
                    record = {
                        "name": name,
                        "source": "pubchem",
                        "cid": props.get("CID"),
                        "canonical_smiles": props.get("CanonicalSMILES"),
                        "isomeric_smiles": props.get("IsomericSMILES"),
                        "molecular_formula": props.get("MolecularFormula"),
                        "molecular_weight": props.get("MolecularWeight"),
                    }
        else:
            logger.info("PubChem returned HTTP %s for %s", resp.status_code, name)
    except Exception as exc:
        logger.info("PubChem lookup failed for %s: %s", name, exc)

    # ---- Fallback: ChEMBL web service ------------------------------------
    if record is None:
        try:
            resp = requests.get(
                CHEMBL_SEARCH_URL,
                params={"q": name, "limit": "10"},
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
            sources_queried.append("chembl")
            if resp.status_code == 200:
                molecules = resp.json().get("molecules", [])
                for mol in molecules:
                    struct = mol.get("molecule_structures") or {}
                    smi = struct.get("canonical_smiles")
                    pref = (mol.get("pref_name") or "").lower()
                    if not smi:
                        continue
                    if name.lower() not in pref and name.lower() not in smi.lower():
                        continue
                    sources_succeeded.append("chembl")
                    record = {
                        "name": name,
                        "source": "chembl",
                        "chembl_id": mol.get("molecule_chembl_id"),
                        "canonical_smiles": smi,
                        "isomeric_smiles": smi,
                        "molecular_formula": mol.get("molecule_properties", {}).get(
                            "full_molformula"
                        ),
                        "molecular_weight": mol.get("molecule_properties", {}).get("full_mw"),
                    }
                    break
        except Exception as exc:
            logger.info("ChEMBL lookup failed for %s: %s", name, exc)

    if record is None:
        raise RuntimeError(
            f"Could not resolve compound '{name}' from PubChem or ChEMBL online. "
            "Check network connectivity or provide a SMILES directly."
        )

    record["sources_queried"] = sources_queried
    record["sources_succeeded"] = sources_succeeded

    if output_file:
        out_path = _versioned_path(_resolve_path(output_file))
        Path(str(Path(out_path).parent) or ".").mkdir(parents=True, exist_ok=True)
        with Path(out_path).open("w") as fh:
            json.dump(record, fh, indent=2)
        record["output_file"] = out_path

    return record
