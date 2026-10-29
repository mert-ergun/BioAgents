import json
import os
import statistics
import time
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

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


def generate_ecfp4_fingerprints(molecules_file, output_file, radius=2, n_bits=2048):
    """Generate Morgan fingerprint bitstrings (radius=2 -> ECFP4-equivalent).

    Args:
        molecules_file: JSON file with a list of {id, smiles} or {"molecules": [...]}.
        output_file: JSON path for {"fingerprints": [{id, smiles, fingerprint_bitstring, n_on_bits}]}.
        radius: Morgan radius (default 2).
        n_bits: fingerprint size in bits (default 2048).

    Returns:
        dict with status, n_fingerprints, n_failed, n_bits, radius,
        fingerprint_type, mean_on_bits, elapsed_seconds, output_file.

    Raises:
        ValueError if the input has no molecules;
        RuntimeError if every SMILES fails to parse.
    """
    started = time.time()
    with Path(_rp(molecules_file)).open() as fh:
        data = json.load(fh)
    mols = data if isinstance(data, list) else data.get("molecules", [])
    if not mols:
        raise ValueError("no molecules found in input")
    rows = []
    n_failed = 0
    for ent in mols:
        mid = ent.get("id", "")
        smi = ent.get("smiles", "")
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            rows.append({"id": mid, "smiles": smi, "error": "unparseable SMILES"})
            n_failed += 1
            continue
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
            bits = DataStructs.BitVectToText(fp)
            rows.append(
                {
                    "id": mid,
                    "smiles": smi,
                    "fingerprint_bitstring": bits,
                    "n_on_bits": sum(1 for c in bits if c == "1"),
                }
            )
        except Exception as exc:
            rows.append({"id": mid, "smiles": smi, "error": f"fingerprint failed: {exc}"})
            n_failed += 1
    onbits = [r["n_on_bits"] for r in rows if "n_on_bits" in r]
    if not onbits:
        raise RuntimeError("all molecules failed fingerprint generation")
    out = _vp(_rp(output_file))
    Path(str(Path(out).parent) or ".").mkdir(parents=True, exist_ok=True)
    payload = {
        "fingerprint_type": f"Morgan radius={radius} (ECFP4-equivalent when radius=2)",
        "radius": radius,
        "n_bits": n_bits,
        "n_fingerprints": len(onbits),
        "fingerprints": rows,
    }
    with Path(out).open("w") as fh:
        json.dump(payload, fh, indent=2)
    return {
        "status": "success" if n_failed == 0 else "partial",
        "n_fingerprints": len(onbits),
        "n_failed": n_failed,
        "n_bits": n_bits,
        "radius": radius,
        "fingerprint_type": payload["fingerprint_type"],
        "mean_on_bits": round(statistics.fmean(onbits), 2),
        "elapsed_seconds": round(time.time() - started, 1),
        "output_file": out,
    }
