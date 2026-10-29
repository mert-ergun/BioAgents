"""pdb_fetch_structure.py — fetch PDB from RCSB (PDBe fallback) and parse metadata."""

import json
import math
import re
from typing import Any

import requests

from bioagents.sandbox.workspace import to_portable_path
from bioagents.tools.workspace_paths import prepare_output_file, resolve_output_dir

SKIP_HETATM = {
    "HOH",
    "DOD",
    "WAT",
    "H2O",
    "TIP3",
    "SPC",
    "SO4",
    "PO4",
    "GOL",
    "EDO",
    "PEG",
    "PGE",
    "ACT",
    "ACY",
    "EPE",
    "MES",
    "TRS",
    "BME",
    "DTT",
    "MPD",
    "IPA",
    "DMS",
    "FMT",
    "NO3",
    "CL",
    "NA",
    "MG",
    "CA",
    "ZN",
    "FE",
    "MN",
    "CU",
    "NI",
    "CO",
    "CD",
    "HG",
    "K",
    "BR",
    "IOD",
    "SCN",
    "OXY",
    "O2",
    "NCO",
    "NH4",
    "BA",
    "SR",
}


def _has_protein_atoms(text):
    for line in text.splitlines():
        if not line.startswith("ATOM  ") or len(line) < 54:
            continue
        try:
            if all(math.isfinite(float(line[a:b])) for a, b in ((30, 38), (38, 46), (46, 54))):
                return True
        except ValueError:
            continue
    return False


def pdb_fetch_structure(pdb_id, output_dir=""):
    """Download a PDB structure and extract metadata for docking box definition.

    Args:
        pdb_id: 4-character PDB ID (e.g., 4GV1).
        output_dir: directory for output files.

    Returns:
        dict with pdb_id, pdb_file, source_url, chains, n_protein_atoms,
        n_hetatm_atoms, co_crystal_ligands [{resname, n_atoms, centroid}], info_file.

    Raises:
        ValueError: invalid PDB ID.
        RuntimeError: download failed from all mirrors.
    """
    pdb_id = str(pdb_id).strip().upper()
    if not re.fullmatch(r"[0-9][A-Z0-9]{3}", pdb_id):
        raise ValueError(f"Invalid PDB ID: {pdb_id}")
    out_dir = resolve_output_dir(output_dir or None)
    pdb_path, _ = prepare_output_file(out_dir / f"{pdb_id}.pdb")
    sources = [
        f"https://files.rcsb.org/download/{pdb_id}.pdb",
        f"https://www.ebi.ac.uk/pdbe/entry-files/download/pdb{pdb_id.lower()}.ent",
    ]
    text, used = None, None
    for url in sources:
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            t = r.text
            if _has_protein_atoms(t):
                text, used = t, url
                break
        except requests.RequestException:
            continue
    if text is None:
        raise RuntimeError(f"Could not download {pdb_id} from RCSB or PDBe")
    with pdb_path.open("w") as f:
        f.write(text)

    chains: set[str] = set()
    n_protein, n_hetatm = 0, 0
    het: dict[tuple[str, str, str], dict[str, Any]] = {}
    for line in text.splitlines():
        rec = line[:6].strip()
        if rec == "ATOM":
            n_protein += 1
            if len(line) > 21:
                chains.add(line[21])
        elif rec == "HETATM":
            n_hetatm += 1
            resn = line[17:20].strip()
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            key = (resn, line[21:22], line[22:27].strip())
            d = het.setdefault(
                key,
                {
                    "resname": resn,
                    "chain": key[1],
                    "residue_id": key[2],
                    "n_atoms": 0,
                    "coords": [],
                },
            )
            d["n_atoms"] += 1
            d["coords"].append((x, y, z))
    co_crystal = []
    for (resn, _, _), d in sorted(het.items(), key=lambda kv: -kv[1]["n_atoms"]):
        if resn in SKIP_HETATM or d["n_atoms"] < 5:
            continue
        c = d["coords"]
        centroid = [round(sum(p[i] for p in c) / len(c), 3) for i in range(3)]
        co_crystal.append(
            {
                "resname": resn,
                "chain": d["chain"],
                "residue_id": d["residue_id"],
                "n_atoms": d["n_atoms"],
                "centroid": centroid,
            }
        )
    info = {
        "pdb_id": pdb_id,
        "source_url": used,
        "chains": sorted(chains),
        "n_protein_atoms": n_protein,
        "n_hetatm_atoms": n_hetatm,
        "co_crystal_ligands": co_crystal,
    }
    info_path, _ = prepare_output_file(out_dir / f"{pdb_id}_info.json")
    with info_path.open("w") as f:
        json.dump(info, f, indent=2)
    return {
        "status": "success",
        "pdb_id": pdb_id,
        "pdb_file": to_portable_path(pdb_path),
        "source_url": used,
        "chains": sorted(chains),
        "n_protein_atoms": n_protein,
        "n_hetatm_atoms": n_hetatm,
        "co_crystal_ligands": co_crystal,
        "info_file": to_portable_path(info_path),
    }
