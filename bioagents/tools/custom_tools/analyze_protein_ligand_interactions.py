"""Heuristic protein-ligand contact estimates; not a validated PLIP calculation."""

import csv
import importlib
import json
import math
from pathlib import Path
from typing import Any

from bioagents.sandbox.workspace import to_portable_path
from bioagents.tools.workspace_paths import (
    prepare_output_file,
    resolve_input_file,
    resolve_output_dir,
)

COVALENT = {
    "H": 0.37,
    "C": 0.77,
    "N": 0.75,
    "O": 0.73,
    "S": 1.02,
    "P": 1.06,
    "F": 0.71,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}
DONOR_N = {
    "ARG": {"NE", "NH1", "NH2"},
    "LYS": {"NZ"},
    "TRP": {"NE1"},
    "HIS": {"ND1", "NE2"},
    "ASN": {"ND2"},
    "GLN": {"NE2"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
    "CYS": {"SG"},
}
ACCEPT = {
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
    "ASN": {"OD1"},
    "GLN": {"OE1"},
    "HIS": {"ND1", "NE2"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
    "CYS": {"SG"},
    "MET": {"SD"},
}
AROM_RES = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "HIS": {"CG", "ND1", "CD2", "CE1", "NE2"},
}
POSITIVE_RES = {"LYS": {"NZ"}, "ARG": {"NE", "NH1", "NH2"}, "HIS": {"ND1", "NE2"}}
NEGATIVE_RES = {"ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}}
PDBQT_TYPE2ELEM = {
    "A": "C",
    "C": "C",
    "N": "N",
    "NA": "N",
    "NS": "N",
    "O": "O",
    "OA": "O",
    "OS": "O",
    "S": "S",
    "SA": "S",
    "HD": "H",
    "H": "H",
    "F": "F",
    "CL": "Cl",
    "BR": "Br",
    "I": "I",
    "P": "P",
    "SI": "Si",
    "B": "B",
}


def _ensure(mod, pip_name=None):
    try:
        return importlib.import_module(mod)
    except ModuleNotFoundError as exc:
        if exc.name != mod and not mod.startswith((exc.name or "") + "."):
            raise
        raise ModuleNotFoundError(
            f"Missing dependency {mod!r}; install {pip_name or mod!r} in the sandbox image",
            name=exc.name,
        ) from exc


def _dist(a, b):
    return math.sqrt(sum((a[k] - b[k]) ** 2 for k in range(3)))


def _angle(p1, p2, p3):
    v1 = [p1[k] - p2[k] for k in range(3)]
    v2 = [p3[k] - p2[k] for k in range(3)]
    c = sum(v1[k] * v2[k] for k in range(3))
    n1 = math.sqrt(sum(v**2 for v in v1))
    n2 = math.sqrt(sum(v**2 for v in v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    cosv = max(-1.0, min(1.0, c / (n1 * n2)))
    return math.degrees(math.acos(cosv))


def _ring_normal(pts):
    p0, p1, p2 = pts[0], pts[1], pts[2]
    v1 = [p1[k] - p0[k] for k in range(3)]
    v2 = [p2[k] - p0[k] for k in range(3)]
    n = [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ]
    nn = math.sqrt(sum(x * x for x in n))
    if nn == 0:
        return None
    return [x / nn for x in n]


def _load_receptor(receptor_pdb):
    _ensure("rdkit", "rdkit")
    from rdkit import Chem

    kept = []
    for line in Path(receptor_pdb).read_text().splitlines():
        if line[:6].strip() == "ATOM" and line[16] in (" ", "A"):
            kept.append(line.rstrip())
    try:
        mol = Chem.MolFromPDBBlock(
            "\n".join(kept) + "\nEND\n", removeHs=False, proximityBonding=True
        )
        if mol is None:
            raise RuntimeError("RDKit parse failed")
        mol = Chem.AddHs(mol, addCoords=True)
        conf = mol.GetConformer()
        atoms: list[dict[str, Any]] = []
        bonds: list[tuple[int, int]] = []
        for atom in mol.GetAtoms():
            ri = atom.GetPDBResidueInfo()
            pos = conf.GetAtomPosition(atom.GetIdx())
            atoms.append(
                {
                    "element": atom.GetSymbol(),
                    "name": (ri.GetName().strip() if ri else atom.GetSymbol()),
                    "resname": (ri.GetResidueName() if ri else "UNK"),
                    "chain": (ri.GetChainId() if ri else " "),
                    "resnum": (ri.GetResidueNumber() if ri else 0),
                    "coord": [pos.x, pos.y, pos.z],
                    "charge": float(atom.GetFormalCharge()),
                }
            )
        for b in mol.GetBonds():
            bonds.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        return atoms, bonds
    except Exception:
        atoms = []
        for line in kept:
            elem = line[76:78].strip().upper()
            if not elem:
                nm = line[12:16].strip()
                elem = nm[0] if not nm[0].isdigit() else nm[1]
            elem = elem[0].upper() + elem[1:].lower() if len(elem) == 2 else elem.upper()
            atoms.append(
                {
                    "element": elem,
                    "name": line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "chain": line[21],
                    "resnum": int(line[22:26] or 0),
                    "coord": [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    "charge": 0.0,
                }
            )
        bonds = []
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                ei, ej = atoms[i]["element"], atoms[j]["element"]
                if ei == "H" and ej == "H":
                    continue
                if (
                    _dist(atoms[i]["coord"], atoms[j]["coord"])
                    <= (COVALENT.get(ei, 0.8) + COVALENT.get(ej, 0.8)) * 1.25
                ):
                    bonds.append((i, j))
        return atoms, bonds


def _classify_receptor(atoms, bonds):
    nb: dict[int, set[int]] = {i: set() for i in range(len(atoms))}
    for i, j in bonds:
        nb[i].add(j)
        nb[j].add(i)
    for a in atoms:
        rn, nm, e = a["resname"], a["name"], a["element"]
        a["is_donor"] = a["is_acceptor"] = a["is_cation"] = a["is_anion"] = a["is_aromatic"] = False
        if e == "N":
            if (nm == "N" and rn != "PRO") or nm in DONOR_N.get(rn, set()):
                a["is_donor"] = True
            if nm in ACCEPT.get(rn, set()):
                a["is_acceptor"] = True
            if nm in POSITIVE_RES.get(rn, set()):
                a["is_cation"] = True
        elif e == "O":
            if nm in ("O", "OXT") or nm in ACCEPT.get(rn, set()):
                a["is_acceptor"] = True
            if nm in NEGATIVE_RES.get(rn, set()):
                a["is_anion"] = True
        elif e == "S":
            if nm in ACCEPT.get(rn, set()):
                a["is_acceptor"] = True
        if e == "C" and nm in AROM_RES.get(rn, set()):
            a["is_aromatic"] = True
    rings = []
    groups: dict[tuple[str, int, str], list[int]] = {}
    for i, a in enumerate(atoms):
        if a["is_aromatic"]:
            groups.setdefault((a["chain"], a["resnum"], a["resname"]), []).append(i)
    for key, idxs in groups.items():
        if len(idxs) >= 4:
            pts = [atoms[i]["coord"] for i in idxs]
            centroid = [sum(p[k] for p in pts) / len(pts) for k in range(3)]
            normal = _ring_normal(pts)
            rings.append(
                {
                    "centroid": centroid,
                    "normal": normal,
                    "n_atoms": len(idxs),
                    "residue": f"{key[2]}:{key[0]}{key[1]}",
                }
            )
    return nb, rings


def _load_ligand(pose_sdf, pose_pdbqt):
    _ensure("rdkit", "rdkit")
    from rdkit import Chem

    if pose_sdf and Path(pose_sdf).is_file():
        mol = Chem.MolFromMolFile(pose_sdf, removeHs=False)
        if mol is not None:
            conf = mol.GetConformer()
            atoms = [
                {
                    "element": a.GetSymbol(),
                    "coord": [
                        conf.GetAtomPosition(i).x,
                        conf.GetAtomPosition(i).y,
                        conf.GetAtomPosition(i).z,
                    ],
                    "charge": float(a.GetFormalCharge()),
                    "aromatic": a.GetIsAromatic(),
                    "idx": i,
                }
                for i, a in enumerate(mol.GetAtoms())
            ]
            bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
            rings = []
            for ring in mol.GetRingInfo().AtomRings():
                ratoms = [mol.GetAtomWithIdx(i) for i in ring]
                if sum(1 for a in ratoms if a.GetIsAromatic()) >= 4:
                    pts = [list(conf.GetAtomPosition(i)) for i in ring]
                    centroid = [sum(p[k] for p in pts) / len(pts) for k in range(3)]
                    rings.append(
                        {"centroid": centroid, "normal": _ring_normal(pts), "n_atoms": len(ring)}
                    )
            return atoms, bonds, rings, "sdf"
    if pose_pdbqt and Path(pose_pdbqt).is_file():
        atoms = []
        for line in Path(pose_pdbqt).read_text().splitlines():
            if line.startswith("ENDMDL"):
                break  # Vina stores multiple poses: analyse only the best, first model.
            if line[:6].strip() not in ("ATOM", "HETATM"):
                continue
            parts = line.split()
            atype = parts[-1] if parts else "C"
            elem = PDBQT_TYPE2ELEM.get(atype.upper(), PDBQT_TYPE2ELEM.get(atype, "C"))
            atoms.append(
                {
                    "element": elem,
                    "coord": [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    "charge": 0.0,
                    "aromatic": atype.upper() == "A",
                    "idx": len(atoms),
                }
            )
        bonds = []
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                ei, ej = atoms[i]["element"], atoms[j]["element"]
                if ei == "H" and ej == "H":
                    continue
                if (
                    _dist(atoms[i]["coord"], atoms[j]["coord"])
                    <= (COVALENT.get(ei, 0.8) + COVALENT.get(ej, 0.8)) * 1.3
                ):
                    bonds.append((i, j))
        return atoms, bonds, [], "pdbqt"
    raise RuntimeError("No readable pose file (SDF or PDBQT)")


def _analyze_pair(rec_atoms, rec_nb, rec_rings, lig_atoms, lig_bonds, lig_rings, p):
    lig_nb: dict[int, set[int]] = {i: set() for i in range(len(lig_atoms))}
    for i, j in lig_bonds:
        lig_nb[i].add(j)
        lig_nb[j].add(i)
    interactions = []

    def rec_label(i):
        a = rec_atoms[i]
        return f"{a['resname']}:{a['chain']}{a['resnum']}:{a['name']}"

    def lig_label(i):
        a = lig_atoms[i]
        return f"{a['element']}{i}({a['element']})"

    rec_donors = [i for i, a in enumerate(rec_atoms) if a["is_donor"]]
    rec_acceptors = [i for i, a in enumerate(rec_atoms) if a["is_acceptor"]]
    rec_cations = [i for i, a in enumerate(rec_atoms) if a["is_cation"]]
    rec_anions = [i for i, a in enumerate(rec_atoms) if a["is_anion"]]
    rec_carbons = [i for i, a in enumerate(rec_atoms) if a["element"] == "C"]
    lig_polar = [i for i, a in enumerate(lig_atoms) if a["element"] in ("N", "O", "S")]
    lig_carbons = [i for i, a in enumerate(lig_atoms) if a["element"] == "C"]
    lig_anions = [i for i, a in enumerate(lig_atoms) if a["element"] == "O" and a["charge"] < 0]
    lig_cations = [i for i, a in enumerate(lig_atoms) if a["element"] == "N" and a["charge"] > 0]
    lig_halogens = [i for i, a in enumerate(lig_atoms) if a["element"] in ("F", "Cl", "Br", "I")]

    def hs_on(atoms, nb, i):
        return [j for j in nb[i] if atoms[j]["element"] == "H"]

    # Hydrogen bonds: receptor donor -> ligand acceptor
    for di in rec_donors:
        for ai in lig_polar:
            if lig_atoms[ai]["element"] == "S" and rec_atoms[di]["element"] == "S":
                continue
            d_da = _dist(rec_atoms[di]["coord"], lig_atoms[ai]["coord"])
            if d_da > p["hbond_max_dist"]:
                continue
            found = False
            for h in hs_on(rec_atoms, rec_nb, di):
                d_ha = _dist(rec_atoms[h]["coord"], lig_atoms[ai]["coord"])
                ang = _angle(rec_atoms[di]["coord"], rec_atoms[h]["coord"], lig_atoms[ai]["coord"])
                if d_ha <= p["hbond_max_h_dist"] and ang >= p["hbond_min_angle"]:
                    interactions.append(
                        {
                            "type": "hbond",
                            "ligand_atom": lig_label(ai),
                            "protein": rec_label(di),
                            "distance_A": round(d_da, 2),
                            "angle_deg": round(ang, 1),
                        }
                    )
                    found = True
                    break
            if not found and d_da <= 3.0:
                interactions.append(
                    {
                        "type": "hbond",
                        "ligand_atom": lig_label(ai),
                        "protein": rec_label(di),
                        "distance_A": round(d_da, 2),
                        "angle_deg": None,
                    }
                )
    # Hydrogen bonds: ligand donor -> receptor acceptor
    for di in lig_polar:
        for h in hs_on(lig_atoms, lig_nb, di):
            for ai in rec_acceptors:
                d_ha = _dist(lig_atoms[h]["coord"], rec_atoms[ai]["coord"])
                if d_ha > p["hbond_max_h_dist"]:
                    continue
                d_da = _dist(lig_atoms[di]["coord"], rec_atoms[ai]["coord"])
                ang = _angle(lig_atoms[di]["coord"], lig_atoms[h]["coord"], rec_atoms[ai]["coord"])
                if d_da <= p["hbond_max_dist"] and ang >= p["hbond_min_angle"]:
                    interactions.append(
                        {
                            "type": "hbond",
                            "ligand_atom": lig_label(di),
                            "protein": rec_label(ai),
                            "distance_A": round(d_da, 2),
                            "angle_deg": round(ang, 1),
                        }
                    )
    # Hydrophobic contacts
    for ci in rec_carbons:
        for cj in lig_carbons:
            d = _dist(rec_atoms[ci]["coord"], lig_atoms[cj]["coord"])
            if d <= p["hydrophobic_max_dist"]:
                interactions.append(
                    {
                        "type": "hydrophobic",
                        "ligand_atom": lig_label(cj),
                        "protein": rec_label(ci),
                        "distance_A": round(d, 2),
                    }
                )
    # Salt bridges
    for ci in rec_cations:
        for aj in lig_anions:
            d = _dist(rec_atoms[ci]["coord"], lig_atoms[aj]["coord"])
            if d <= p["salt_bridge_max_dist"]:
                interactions.append(
                    {
                        "type": "salt_bridge",
                        "ligand_atom": lig_label(aj),
                        "protein": rec_label(ci),
                        "distance_A": round(d, 2),
                    }
                )
    for ai in rec_anions:
        for cj in lig_cations:
            d = _dist(rec_atoms[ai]["coord"], lig_atoms[cj]["coord"])
            if d <= p["salt_bridge_max_dist"]:
                interactions.append(
                    {
                        "type": "salt_bridge",
                        "ligand_atom": lig_label(cj),
                        "protein": rec_label(ai),
                        "distance_A": round(d, 2),
                    }
                )
    # Pi-pi stacking
    for rr in rec_rings:
        for lr in lig_rings:
            d = _dist(rr["centroid"], lr["centroid"])
            if d <= p["pi_stack_max_dist"] and rr["normal"] and lr["normal"]:
                dot = abs(sum(rr["normal"][k] * lr["normal"][k] for k in range(3)))
                ang = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
                if ang < 40 or 50 <= ang <= 90:
                    interactions.append(
                        {
                            "type": "pi_stack",
                            "ligand_atom": "ring",
                            "protein": rr["residue"],
                            "distance_A": round(d, 2),
                            "angle_deg": round(ang, 1),
                        }
                    )
    # Pi-cation
    for ci in rec_cations:
        for lr in lig_rings:
            d = _dist(rec_atoms[ci]["coord"], lr["centroid"])
            if d <= p["pi_cation_max_dist"]:
                interactions.append(
                    {
                        "type": "pi_cation",
                        "ligand_atom": "ring",
                        "protein": rec_label(ci),
                        "distance_A": round(d, 2),
                    }
                )
    for cj in lig_cations:
        for rr in rec_rings:
            d = _dist(lig_atoms[cj]["coord"], rr["centroid"])
            if d <= p["pi_cation_max_dist"]:
                interactions.append(
                    {
                        "type": "pi_cation",
                        "ligand_atom": lig_label(cj),
                        "protein": rr["residue"],
                        "distance_A": round(d, 2),
                    }
                )
    # Halogen bonds
    for hi in lig_halogens:
        for ai in rec_acceptors:
            d = _dist(lig_atoms[hi]["coord"], rec_atoms[ai]["coord"])
            if d <= 3.5:
                interactions.append(
                    {
                        "type": "halogen",
                        "ligand_atom": lig_label(hi),
                        "protein": rec_label(ai),
                        "distance_A": round(d, 2),
                    }
                )
    return interactions


def analyze_protein_ligand_interactions(
    receptor_pdb,
    docking_results_json,
    output_dir="",
    hbond_max_dist=3.5,
    hbond_max_h_dist=2.5,
    hbond_min_angle=110.0,
    hydrophobic_max_dist=4.0,
    salt_bridge_max_dist=4.0,
    pi_stack_max_dist=5.5,
    pi_cation_max_dist=6.0,
    dry_run=False,
):
    """Detect protein-ligand interactions and rank ligands.

    Args:
        receptor_pdb: path to receptor PDB.
        docking_results_json: path to docking_results.json from vina_dock_ligands.
        output_dir: output directory.
        hbond_max_dist: max heavy-atom D-A distance for H-bonds (A).
        hbond_max_h_dist: max H...A distance for H-bonds (A).
        hbond_min_angle: min D-H...A angle (degrees).
        hydrophobic_max_dist: max C...C contact distance (A).
        salt_bridge_max_dist: max distance for salt bridges (A).
        pi_stack_max_dist: max ring centroid distance for pi-stacking (A).
        pi_cation_max_dist: max distance for pi-cation (A).
        dry_run: parse inputs and report counts without full analysis.

    Returns:
        dict with per-ligand interaction breakdowns and final ranked table
        (rank, chembl_id, affinity_kcal_mol, rank_affinity, total_interactions,
        rank_interactions, combined_score, per-type counts, key residues);
        saves interaction_analysis.json and final_ranked_table.csv.

    Raises:
        FileNotFoundError: input files missing.
        RuntimeError: no docked ligands found.
    """
    receptor_pdb = str(resolve_input_file(receptor_pdb, "Receptor PDB", (".pdb", ".ent")))
    docking_results_json = str(
        resolve_input_file(docking_results_json, "Docking results", (".json",))
    )
    with Path(docking_results_json).open() as f:
        dock = json.load(f)
    docked = [r for r in dock.get("results", []) if "affinity_kcal_mol" in r]
    if not docked:
        raise RuntimeError(f"No successfully docked ligands in {docking_results_json}")
    p = {
        "hbond_max_dist": float(hbond_max_dist),
        "hbond_max_h_dist": float(hbond_max_h_dist),
        "hbond_min_angle": float(hbond_min_angle),
        "hydrophobic_max_dist": float(hydrophobic_max_dist),
        "salt_bridge_max_dist": float(salt_bridge_max_dist),
        "pi_stack_max_dist": float(pi_stack_max_dist),
        "pi_cation_max_dist": float(pi_cation_max_dist),
    }
    if dry_run:
        return {
            "status": "dry_run",
            "n_docked_ligands": len(docked),
            "receptor_pdb": receptor_pdb,
            "criteria": p,
        }
    rec_atoms, rec_bonds = _load_receptor(receptor_pdb)
    rec_nb, rec_rings = _classify_receptor(rec_atoms, rec_bonds)
    per_ligand = []
    for r in docked:
        try:

            def pose_path(value):
                return str(resolve_input_file(value, "Docked pose")) if value else None

            lig_atoms, lig_bonds, lig_rings, src = _load_ligand(
                pose_path(r.get("pose_sdf")), pose_path(r.get("pose_pdbqt"))
            )
            inter = _analyze_pair(rec_atoms, rec_nb, rec_rings, lig_atoms, lig_bonds, lig_rings, p)
        except Exception as e:
            per_ligand.append({"chembl_id": r["chembl_id"], "error": str(e)})
            continue
        counts: dict[str, int] = {}
        residues: dict[str, set[str]] = {}
        for it in inter:
            counts[it["type"]] = counts.get(it["type"], 0) + 1
            residues.setdefault(it["type"], set()).add(it["protein"])
        all_res: dict[str, int] = {}
        for it in inter:
            all_res[it["protein"]] = all_res.get(it["protein"], 0) + 1
        key_res = sorted(all_res.items(), key=lambda kv: -kv[1])[:10]
        per_ligand.append(
            {
                "chembl_id": r["chembl_id"],
                "affinity_kcal_mol": r["affinity_kcal_mol"],
                "smiles": r.get("smiles", ""),
                "pose_source": src,
                "total_interactions": len(inter),
                "counts": counts,
                "unique_residues_by_type": {k: sorted(v) for k, v in residues.items()},
                "key_residues": [{"residue": k, "n_contacts": v} for k, v in key_res],
                "interactions": inter,
            }
        )
    ok = [pl for pl in per_ligand if "error" not in pl]
    if not ok:
        raise RuntimeError("Interaction analysis failed for all ligands")
    rows = [
        {
            "chembl_id": pl["chembl_id"],
            "affinity_kcal_mol": pl["affinity_kcal_mol"],
            "total_interactions": pl["total_interactions"],
            "hbonds": pl["counts"].get("hbond", 0),
            "hydrophobic": pl["counts"].get("hydrophobic", 0),
            "salt_bridges": pl["counts"].get("salt_bridge", 0),
            "pi_stacking": pl["counts"].get("pi_stack", 0),
            "pi_cation": pl["counts"].get("pi_cation", 0),
            "halogen": pl["counts"].get("halogen", 0),
            "key_residues": "; ".join(
                f"{k['residue']}({k['n_contacts']})" for k in pl["key_residues"]
            ),
        }
        for pl in ok
    ]
    by_aff = sorted(rows, key=lambda r: r["affinity_kcal_mol"])
    for k, r in enumerate(by_aff):
        r["rank_affinity"] = k + 1
    by_int = sorted(rows, key=lambda r: -r["total_interactions"])
    for k, r in enumerate(by_int):
        r["rank_interactions"] = k + 1
    for r in rows:
        r["combined_score"] = round((r["rank_affinity"] + r["rank_interactions"]) / 2.0, 2)
    rows.sort(key=lambda r: (r["combined_score"], r["rank_affinity"]))
    for k, r in enumerate(rows):
        r["rank"] = k + 1
    out_dir = resolve_output_dir(output_dir or None)
    json_path, _ = prepare_output_file(out_dir / "interaction_analysis.json")
    method = "distance and residue heuristics (not a validated PLIP calculation)"
    limitations = "Contact counts are heuristic; PDBQT does not provide full bond orders or aromatic rings. Only the first docking pose is analysed."
    with json_path.open("w") as f:
        json.dump(
            {
                "method": method,
                "limitations": limitations,
                "criteria": p,
                "per_ligand": per_ligand,
                "ranked_table": rows,
            },
            f,
            indent=2,
        )
    csv_path, _ = prepare_output_file(out_dir / "final_ranked_table.csv")
    cols = [
        "rank",
        "chembl_id",
        "affinity_kcal_mol",
        "rank_affinity",
        "total_interactions",
        "rank_interactions",
        "combined_score",
        "hbonds",
        "hydrophobic",
        "salt_bridges",
        "pi_stacking",
        "pi_cation",
        "halogen",
        "key_residues",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    return {
        "status": "success" if len(ok) == len(per_ligand) else "partial",
        "method": method,
        "limitations": limitations,
        "n_ligands_analyzed": len(ok),
        "n_failed": len(per_ligand) - len(ok),
        "ranked_table": rows,
        "json_file": to_portable_path(json_path),
        "csv_file": to_portable_path(csv_path),
    }
