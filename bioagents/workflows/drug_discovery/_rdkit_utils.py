"""Lazy RDKit helpers for drug-discovery nodes.

We avoid importing RDKit at module import time so the rest of the workflow
package remains usable on systems where RDKit failed to install.
"""

from __future__ import annotations

import hashlib
from typing import Any


def rdkit_available() -> bool:
    try:
        import rdkit  # noqa: F401
    except Exception:
        return False
    return True


def stable_seed(*parts: Any) -> int:
    """Deterministic 32-bit seed from arbitrary string-y parts."""
    joined = "||".join("" if p is None else str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(joined, digest_size=4).digest(), "big")


def safe_mol_from_smiles(smiles: str):  # -> Optional[Chem.Mol]
    if not rdkit_available() or not smiles:
        return None
    from rdkit import Chem

    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def standardize_smiles(smiles: str) -> str | None:
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        from rdkit import Chem

        smi: str = Chem.MolToSmiles(mol, canonical=True)
        return smi
    except Exception:
        return None


def compute_descriptors(smiles: str) -> dict[str, float]:
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        return {}
    try:
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

        return {
            "molecular_weight": float(Descriptors.MolWt(mol)),
            "logp": float(Crippen.MolLogP(mol)),
            "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
            "hbd": int(Lipinski.NumHDonors(mol)),
            "hba": int(Lipinski.NumHAcceptors(mol)),
            "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
            "heavy_atoms": int(mol.GetNumHeavyAtoms()),
            "rings": int(rdMolDescriptors.CalcNumRings(mol)),
            "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        }
    except Exception:
        return {}


def pains_alert_count(smiles: str) -> int:
    """Count PAINS substructure hits via RDKit's FilterCatalog."""
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        return 0
    try:
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog(params)
        return len(catalog.GetMatches(mol))
    except Exception:
        return 0


def enumerate_methyl_analogues(smiles: str, max_out: int) -> list[str]:
    """Deterministic methyl-walk style enumeration used by stub design nodes."""
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        return []
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        canonical = Chem.MolToSmiles(mol, canonical=True)
        outs: list[str] = [canonical]
        rxn = AllChem.ReactionFromSmarts("[cH:1]>>[c:1]C")
        try:
            products = rxn.RunReactants((mol,))
            seen = {canonical}
            for prods in products:
                if not prods:
                    continue
                p_smi = Chem.MolToSmiles(prods[0], canonical=True)
                if p_smi and p_smi not in seen:
                    seen.add(p_smi)
                    outs.append(p_smi)
                    if len(outs) >= max_out:
                        break
        except Exception:  # nosec B110 - best-effort RDKit reaction; fall back to canonical below
            pass
        while len(outs) < max_out:
            outs.append(canonical)
        return outs[:max_out]
    except Exception:
        return [smiles] * max_out


def synthetic_complexity_score(smiles: str) -> float:
    """Heuristic 0-1 score where 1.0 is "easy to synthesize"."""
    desc = compute_descriptors(smiles)
    if not desc:
        return 0.0
    # Weight: fewer heavy atoms, fewer rings, fewer rotatable bonds => easier.
    hv = desc["heavy_atoms"]
    rings = desc["rings"]
    rot = desc["rotatable_bonds"]
    raw = 1.0 - (hv / 80.0) * 0.4 - (rings / 10.0) * 0.35 - (rot / 15.0) * 0.25
    return float(max(0.0, min(1.0, raw)))
