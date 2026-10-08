"""Validation, repair, conversion and editing, backed by the RDKit Python library."""

from __future__ import annotations

import logging
import re
from typing import Any

from bioagents.tools.rdkit_native._common import (
    canonical_smiles,
    invalid_smiles_result,
    parse_smarts,
    parse_smiles,
    require_rdkit,
)

logger = logging.getLogger(__name__)

# Repair strategies, tried in order of how much they alter the input. Each returns a
# candidate SMILES string or None, and carries the confidence we have in a molecule
# recovered that way.
_REPAIR_CONFIDENCE = {
    "as_written": 1.0,
    "sanitize_relaxed": 0.8,
    "strip_whitespace": 0.95,
    "uppercase_halogens": 0.7,
    "close_unbalanced_parens": 0.5,
    "drop_atom_maps": 0.85,
}


def check_smiles(smiles: str) -> dict[str, Any]:
    """Validate a SMILES string for chemical correctness.

    Returns:
        {"overall_pass", "valid", "canonical_smiles", "summary", "confidence",
         "functional_groups"}
    """
    result = repair_smiles(smiles)
    best = result.get("best_candidate") or {}
    is_valid = bool(best.get("valid"))
    return {
        "overall_pass": is_valid,
        "valid": is_valid,
        "canonical_smiles": best.get("canonical_smiles", smiles),
        "summary": "Valid SMILES" if is_valid else result.get("message", "Invalid SMILES"),
        "confidence": result.get("confidence", 0.0),
        "repair_applied": result.get("strategy") not in (None, "as_written"),
        "strategy": result.get("strategy"),
    }


def check_smirks(smirks: str) -> dict[str, Any]:
    """Validate a SMIRKS/reaction SMARTS string."""
    from rdkit.Chem import AllChem

    require_rdkit()
    text = (smirks or "").strip()
    if not text:
        return {"smirks": smirks, "valid": False, "summary": "Empty SMIRKS string."}

    try:
        rxn = AllChem.ReactionFromSmarts(text)
    except Exception as exc:
        return {"smirks": text, "valid": False, "summary": f"Could not parse SMIRKS: {exc}"}

    if rxn is None:
        return {"smirks": text, "valid": False, "summary": "RDKit could not parse this SMIRKS."}

    try:
        rxn.Initialize()
        init_error = ""
    except Exception as exc:
        init_error = str(exc)

    return {
        "smirks": text,
        "valid": not init_error,
        "reactant_templates": rxn.GetNumReactantTemplates(),
        "product_templates": rxn.GetNumProductTemplates(),
        "summary": init_error or "Valid SMIRKS.",
    }


def check_reaction(reactants: list[str], products: list[str]) -> dict[str, Any]:
    """Validate a reaction by checking that atoms balance between both sides."""
    require_rdkit()

    def _tally(side: list[str]) -> tuple[dict[str, int], list[str]]:
        counts: dict[str, int] = {}
        bad: list[str] = []
        for smi in side:
            mol = parse_smiles(smi)
            if mol is None:
                bad.append(smi)
                continue
            # Explicit hydrogens matter for balance and are implicit after parsing.
            for atom in mol.GetAtoms():
                counts[atom.GetSymbol()] = counts.get(atom.GetSymbol(), 0) + 1
                hydrogens = atom.GetTotalNumHs()
                if hydrogens:
                    counts["H"] = counts.get("H", 0) + hydrogens
        return counts, bad

    left, bad_left = _tally(list(reactants))
    right, bad_right = _tally(list(products))

    if bad_left or bad_right:
        return {
            "valid": False,
            "balanced": False,
            "summary": "Some structures could not be parsed; balance was not checked.",
            "unparseable_reactants": bad_left,
            "unparseable_products": bad_right,
        }

    differences = {
        element: left.get(element, 0) - right.get(element, 0)
        for element in set(left) | set(right)
        if left.get(element, 0) != right.get(element, 0)
    }

    return {
        "valid": True,
        "balanced": not differences,
        "reactant_atoms": left,
        "product_atoms": right,
        "atom_differences": differences,
        "summary": (
            "Reaction is atom balanced."
            if not differences
            else f"Reaction is NOT balanced. Surplus on reactant side (negative = product side): {differences}"
        ),
    }


def _repair_candidates(raw: str):
    """Yield (strategy, candidate_smiles) pairs from least to most invasive."""
    yield "as_written", raw
    stripped = "".join(raw.split())
    if stripped != raw:
        yield "strip_whitespace", stripped
    if "[" in raw and ":" in raw:
        yield "drop_atom_maps", re.sub(r":\d+(?=\])", "", raw)
    # Lowercase halogens are a common hand-typing slip: 'cl' means aromatic-C + L.
    # 'cl'/'br' typed in lowercase parse as aromatic-c + L, which is never valid. A
    # following lowercase letter is a legitimate aromatic atom, so it must not block
    # the fix (e.g. 'clc1ccccc1' -> 'Clc1ccccc1').
    fixed_halogens = re.sub(r"(?<![A-Za-z\[])(cl|br)", lambda m: m.group(0).title(), raw)
    if fixed_halogens != raw:
        yield "uppercase_halogens", fixed_halogens
    unbalanced = raw.count("(") - raw.count(")")
    if unbalanced > 0:
        yield "close_unbalanced_parens", raw + ")" * unbalanced


def repair_smiles(input_str: str) -> dict[str, Any]:
    """Repair a malformed SMILES string, reporting what had to be changed.

    Returns:
        {"success", "canonical_smiles", "strategy", "confidence", "attempts",
         "best_candidate", "message"}
    """
    Chem = require_rdkit()
    raw = (input_str or "").strip()
    if not raw:
        return {
            "success": False,
            "canonical_smiles": "",
            "strategy": None,
            "confidence": 0.0,
            "attempts": 0,
            "best_candidate": {"valid": False, "canonical_smiles": ""},
            "message": "Empty input string.",
        }

    attempts = 0
    for strategy, candidate in _repair_candidates(raw):
        attempts += 1
        mol = Chem.MolFromSmiles(candidate)
        if mol is None:
            continue
        canonical = canonical_smiles(mol)
        return {
            "success": True,
            "canonical_smiles": canonical,
            "strategy": strategy,
            "confidence": _REPAIR_CONFIDENCE.get(strategy, 0.5),
            "attempts": attempts,
            "input_smiles": raw,
            "best_candidate": {
                "valid": True,
                "canonical_smiles": canonical,
                "formula": _formula(mol),
                "heavy_atoms": mol.GetNumHeavyAtoms(),
            },
            "message": (
                "Parsed as written."
                if strategy == "as_written"
                else f"Repaired using strategy '{strategy}'."
            ),
        }

    # Last resort: parse without sanitisation so we can say *why* it fails.
    reason = "RDKit could not parse this SMILES with any repair strategy."
    unsanitized = Chem.MolFromSmiles(raw, sanitize=False)
    if unsanitized is not None:
        try:
            Chem.SanitizeMol(unsanitized)
        except Exception as exc:
            reason = f"SMILES parses structurally but fails sanitisation: {exc}"

    return {
        "success": False,
        "canonical_smiles": raw,
        "strategy": None,
        "confidence": 0.0,
        "attempts": attempts,
        "input_smiles": raw,
        "best_candidate": {"valid": False, "canonical_smiles": raw},
        "message": reason,
    }


def _formula(mol) -> str:
    from rdkit.Chem import rdMolDescriptors

    return str(rdMolDescriptors.CalcMolFormula(mol))


def convert_notation(input_str: str, from_format: str, to_format: str) -> dict[str, Any]:
    """Convert between smiles, inchi, inchikey, mol and sdf representations."""
    Chem = require_rdkit()
    src = (from_format or "").strip().lower()
    dst = (to_format or "").strip().lower()
    supported = {"smiles", "inchi", "inchikey", "mol", "sdf"}

    if src not in supported or dst not in supported:
        return {
            "success": False,
            "error": (
                f"Unsupported conversion {from_format!r} -> {to_format!r}. "
                f"Supported formats: {', '.join(sorted(supported))}."
            ),
        }
    if src == "inchikey":
        return {
            "success": False,
            "error": "InChIKey is a hash and cannot be converted back to a structure.",
        }

    if src == "smiles":
        mol = Chem.MolFromSmiles(input_str.strip())
    elif src == "inchi":
        mol = Chem.MolFromInchi(input_str.strip())
    else:  # mol / sdf share the molblock parser
        mol = Chem.MolFromMolBlock(input_str)

    if mol is None:
        return {
            "success": False,
            "input": input_str,
            "from": src,
            "error": f"Could not parse the input as {src}.",
        }

    try:
        if dst == "smiles":
            output = Chem.MolToSmiles(mol)
        elif dst == "inchi":
            output = Chem.MolToInchi(mol)
        elif dst == "inchikey":
            output = Chem.MolToInchiKey(mol)
        else:
            output = Chem.MolToMolBlock(mol)
    except Exception as exc:
        return {"success": False, "from": src, "to": dst, "error": f"Conversion failed: {exc}"}

    return {"success": True, "from": src, "to": dst, "input": input_str, "output": output}


_EDIT_OPERATIONS = ("neutralize", "sanitize", "add-h", "remove-h", "strip-maps")


def edit_molecule(smiles: str, operation: str) -> dict[str, Any]:
    """Apply a structural edit: neutralize, sanitize, add-h, remove-h or strip-maps."""
    Chem = require_rdkit()
    op = (operation or "").strip().lower()
    if op not in _EDIT_OPERATIONS:
        return {
            "success": False,
            "error": f"Unknown operation {operation!r}. Supported: {', '.join(_EDIT_OPERATIONS)}.",
        }

    mol = parse_smiles(smiles, sanitize=(op != "sanitize"))
    if mol is None:
        return {"success": False, **invalid_smiles_result(smiles)}

    try:
        if op == "neutralize":
            mol = _neutralize(mol)
        elif op == "sanitize":
            Chem.SanitizeMol(mol)
        elif op == "add-h":
            mol = Chem.AddHs(mol)
        elif op == "remove-h":
            mol = Chem.RemoveHs(mol)
        elif op == "strip-maps":
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
    except Exception as exc:
        return {"success": False, "input_smiles": smiles, "operation": op, "error": str(exc)}

    return {
        "success": True,
        "input_smiles": smiles,
        "output_smiles": Chem.MolToSmiles(mol),
        "operation": op,
    }


def _neutralize(mol):
    """Neutralise charged atoms by adjusting hydrogen counts where chemically possible."""
    Chem = require_rdkit()
    # Matches a charged atom that carries, or can accept, a hydrogen.
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    editable = Chem.RWMol(mol)
    for (idx,) in editable.GetSubstructMatches(pattern):
        atom = editable.GetAtomWithIdx(idx)
        charge = atom.GetFormalCharge()
        hydrogens = atom.GetTotalNumHs()
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(hydrogens - charge)
        atom.UpdatePropertyCache()
    Chem.SanitizeMol(editable)
    return editable.GetMol()


def substructure_search(smiles: str, smarts_pattern: str) -> dict[str, Any]:
    """Find SMARTS substructure matches inside a molecule."""
    mol = parse_smiles(smiles)
    if mol is None:
        return invalid_smiles_result(smiles)

    pattern = parse_smarts(smarts_pattern)
    if pattern is None:
        return {
            "smiles": smiles,
            "smarts": smarts_pattern,
            "matched": False,
            "error": "Could not parse the SMARTS pattern.",
        }

    matches = mol.GetSubstructMatches(pattern)
    return {
        "smiles": smiles,
        "smarts": smarts_pattern,
        "matched": bool(matches),
        "match_count": len(matches),
        "match_indices": [list(m) for m in matches],
    }
