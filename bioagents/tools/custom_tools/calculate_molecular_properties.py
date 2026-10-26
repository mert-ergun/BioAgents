import csv
import json
import os
import statistics
import time
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Crippen, Descriptors
from rdkit.Contrib.SA_Score import sascorer

RDLogger.DisableLog("rdApp.*")


def _rp(p):
    return str(Path(p).expanduser().absolute())


def _vp(p):
    if not Path(p).exists():
        return p
    s, e = os.path.splitext(p)  # noqa: PTH122 -- preserve legacy suffix handling
    i = 1
    while Path(f"{s}_{i}{e}").exists():
        i += 1
    return f"{s}_{i}{e}"


def calculate_molecular_properties(molecules_file, output_file, property_list=None):
    """Compute QED, SA score, Crippen logP and exact MW per molecule.

    Args:
        molecules_file: JSON file with a list of {id, smiles} or {"molecules": [...]}.
        output_file: CSV path for the per-molecule property table.
        property_list: subset of ['qed', 'sa_score', 'logp', 'mw'] (default all).

    Returns:
        dict with status ('success'/'partial'), n_molecules, n_failed,
        summary_statistics (min/max/mean/median per property from real values),
        elapsed_seconds and output_file.

    Raises:
        ValueError if the input has no molecules.
    """
    started = time.time()
    with Path(_rp(molecules_file)).open() as fh:
        data = json.load(fh)
    mols = data if isinstance(data, list) else data.get("molecules", [])
    if not mols:
        raise ValueError("no molecules found in input")
    if property_list is None:
        property_list = ["qed", "sa_score", "logp", "mw"]
    rows = []
    pools: dict[str, list[float]] = {p: [] for p in property_list}
    n_failed = 0
    for ent in mols:
        row = {"id": ent.get("id", ""), "smiles": ent.get("smiles", ""), "error": ""}
        mol = Chem.MolFromSmiles(row["smiles"]) if row["smiles"] else None
        if mol is None:
            row["error"] = "unparseable SMILES"
            for p in property_list:
                row[p] = ""
            rows.append(row)
            n_failed += 1
            continue
        try:
            if "qed" in property_list:
                v = QED.qed(mol)
                row["qed"] = round(v, 4)
                pools["qed"].append(v)
            if "sa_score" in property_list:
                v = sascorer.calculateScore(mol)
                row["sa_score"] = round(v, 3)
                pools["sa_score"].append(v)
            if "logp" in property_list:
                v = Crippen.MolLogP(mol)
                row["logp"] = round(v, 3)
                pools["logp"].append(v)
            if "mw" in property_list:
                v = Descriptors.ExactMolWt(mol)
                row["mw"] = round(v, 3)
                pools["mw"].append(v)
            if "tanimoto_to_seed" in ent:
                row["tanimoto_to_seed"] = ent["tanimoto_to_seed"]
        except Exception as exc:
            row["error"] = f"property computation failed: {exc}"
            for p in property_list:
                row.setdefault(p, "")
            n_failed += 1
        rows.append(row)
    out = _vp(_rp(output_file))
    Path(str(Path(out).parent) or ".").mkdir(parents=True, exist_ok=True)
    fields = ["id", "smiles", "tanimoto_to_seed", *list(property_list), "error"]
    with Path(out).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for p, v in pools.items():
        if v:
            summary[p] = {
                "n": len(v),
                "min": round(min(v), 4),
                "max": round(max(v), 4),
                "mean": round(statistics.fmean(v), 4),
                "median": round(statistics.median(v), 4),
            }
    return {
        "status": "success" if n_failed == 0 else "partial",
        "n_molecules": len(mols),
        "n_failed": n_failed,
        "properties": property_list,
        "summary_statistics": summary,
        "elapsed_seconds": round(time.time() - started, 1),
        "output_file": out,
    }
