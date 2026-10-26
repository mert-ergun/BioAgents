import json
import os
import random
import time
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import BRICS, AllChem, rdMolDescriptors

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


def ligand_based_brics_generator(
    seed_smiles,
    n_molecules=100,
    similarity_threshold=0.35,
    min_fragment_size=4,
    random_seed=42,
    max_attempts=60000,
    mw_min=150.0,
    mw_max=700.0,
    output_file=None,
):
    """Ligand-based de novo generation: BRICS decomposition of the seed plus BRICSBuild recombination.

    Compatible with older RDKit versions: BRICSBuild is called without the
    minFragmentSize kwarg (not supported in this environment); fragment-size
    control comes from the decomposition and the MW window filters products.

    Args:
        seed_smiles: SMILES of the seed ligand.
        n_molecules: number of unique analogs to generate (default 100).
        similarity_threshold: min ECFP4 Tanimoto similarity to the seed (default 0.35).
        min_fragment_size: min BRICS fragment size in heavy atoms during decomposition (default 4).
        random_seed: RNG seed for fragment shuffling (default 42).
        max_attempts: cap on BRICSBuild iterations (default 60000).
        mw_min, mw_max: exact-MW window for accepted analogs.
        output_file: optional JSON path to save the generated molecules.

    Returns:
        dict with status ('success' if n_generated==n_requested else 'partial'),
        seed_canonical_smiles, n_fragments, n_requested, n_generated, method,
        elapsed_seconds, molecules [{id, smiles, tanimoto_to_seed, exact_mw}]
        and output_file when written.

    Raises:
        RuntimeError on unparseable seed or empty fragment pool;
        ValueError if n_molecules < 1 or the MW window is inverted.
    """
    if n_molecules < 1:
        raise ValueError("n_molecules must be >= 1")
    if mw_max <= mw_min:
        raise ValueError("mw_max must be greater than mw_min")
    started = time.time()
    seed = Chem.MolFromSmiles(seed_smiles)
    if seed is None:
        raise RuntimeError(f"cannot parse seed SMILES: {seed_smiles}")
    seed_canon = Chem.MolToSmiles(seed)
    seed_fp = AllChem.GetMorganFingerprintAsBitVect(seed, 2, nBits=2048)
    seed_mw = rdMolDescriptors.CalcExactMolWt(seed)
    frags = list(BRICS.BRICSDecompose(seed, minFragmentSize=min_fragment_size, returnMols=True))
    stock = list(
        BRICS.BRICSDecompose(seed, minFragmentSize=1, keepNonLeafNodes=True, returnMols=True)
    )
    pool = []
    seenf = set()
    for f in frags + stock:
        cs = Chem.MolToSmiles(f)
        if cs not in seenf:
            seenf.add(cs)
            pool.append(f)
    # Reproducible scientific sampling, never a credential or security token.
    random.Random(random_seed).shuffle(pool)  # nosec B311
    if not pool:
        raise RuntimeError("BRICS decomposition produced no fragments")
    collected: list[dict[str, Any]] = []
    seen = {seed_canon}
    attempts = 0

    def build(threshold):
        nonlocal attempts
        for prod in BRICS.BRICSBuild(pool, maxDepth=3):
            attempts += 1
            if attempts > max_attempts or len(collected) >= n_molecules:
                break
            try:
                if any(a.GetAtomicNum() == 0 for a in prod.GetAtoms()):
                    continue
                Chem.SanitizeMol(prod)
                smi = Chem.MolToSmiles(prod)
                mw = rdMolDescriptors.CalcExactMolWt(prod)
            except (ValueError, RuntimeError):
                continue
            if smi in seen or mw < mw_min or mw > mw_max:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(prod, 2, nBits=2048)
            sim = DataStructs.TanimotoSimilarity(seed_fp, fp)
            if sim >= threshold:
                seen.add(smi)
                collected.append(
                    {"smiles": smi, "tanimoto_to_seed": round(sim, 4), "exact_mw": round(mw, 3)}
                )

    build(similarity_threshold)
    first_pass = len(collected)
    if first_pass < n_molecules and attempts < max_attempts:
        build(max(0.20, similarity_threshold - 0.15))
    collected.sort(key=lambda d: d["tanimoto_to_seed"], reverse=True)
    collected = collected[:n_molecules]
    mols = []
    for i, rec in enumerate(collected, 1):
        mols.append(
            {
                "id": f"gen_{i:03d}",
                "smiles": rec["smiles"],
                "tanimoto_to_seed": rec["tanimoto_to_seed"],
                "exact_mw": rec["exact_mw"],
            }
        )
    result = {
        "status": "success" if len(mols) == n_molecules else "partial",
        "seed_smiles": seed_smiles,
        "seed_canonical_smiles": seed_canon,
        "seed_exact_mw": round(seed_mw, 3),
        "n_fragments": len(pool),
        "n_requested": n_molecules,
        "n_generated": len(mols),
        "first_pass_count": first_pass,
        "method": "RDKit BRICS decomposition + BRICSBuild recombination, ECFP4 Tanimoto filter to seed, MW window, deduplication",
        "elapsed_seconds": round(time.time() - started, 1),
        "molecules": mols,
    }
    if output_file:
        out = _vp(_rp(output_file))
        Path(str(Path(out).parent) or ".").mkdir(parents=True, exist_ok=True)
        with Path(out).open("w") as fh:
            json.dump(
                {
                    "seed_canonical_smiles": seed_canon,
                    "n_generated": len(mols),
                    "method": result["method"],
                    "molecules": mols,
                },
                fh,
                indent=2,
            )
        result["output_file"] = out
    return result
