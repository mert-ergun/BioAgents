"""Ring systems, stereochemistry, functional groups, atom mapping, reactions, drawing."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

from bioagents.tools.rdkit_native._common import (
    invalid_smiles_result,
    parse_smarts,
    parse_smiles,
    require_rdkit,
)

logger = logging.getLogger(__name__)

#: Functional-group SMARTS catalogue. Ordered from more to less specific so that, for
#: example, a carboxylic acid is not merely reported as a generic carbonyl.
FUNCTIONAL_GROUP_SMARTS: tuple[tuple[str, str], ...] = (
    ("carboxylic_acid", "[CX3](=O)[OX2H1]"),
    ("ester", "[CX3](=O)[OX2H0][#6]"),
    ("amide", "[NX3][CX3](=[OX1])"),
    ("anhydride", "[CX3](=[OX1])[OX2][CX3](=[OX1])"),
    ("aldehyde", "[CX3H1](=O)[#6]"),
    ("ketone", "[#6][CX3](=O)[#6]"),
    ("carbamate", "[NX3][CX3](=[OX1])[OX2]"),
    ("urea", "[NX3][CX3](=[OX1])[NX3]"),
    ("sulfonamide", "[SX4](=[OX1])(=[OX1])[NX3]"),
    ("sulfone", "[SX4](=[OX1])(=[OX1])([#6])[#6]"),
    ("sulfoxide", "[SX3](=[OX1])([#6])[#6]"),
    ("nitro", "[NX3](=O)=O"),
    ("nitrile", "[NX1]#[CX2]"),
    ("primary_amine", "[NX3;H2;!$(N[#6]=[!#6])][#6]"),
    ("secondary_amine", "[NX3;H1;!$(N[#6]=[!#6])]([#6])[#6]"),
    ("tertiary_amine", "[NX3;H0;!$(N[#6]=[!#6]);!$(N a)]([#6])([#6])[#6]"),
    ("aromatic_amine", "[NX3][c]"),
    ("phenol", "[OX2H][c]"),
    ("alcohol", "[OX2H][CX4]"),
    ("ether", "[OD2]([#6])[#6]"),
    ("thiol", "[SX2H]"),
    ("thioether", "[SD2]([#6])[#6]"),
    ("halide", "[F,Cl,Br,I;$([*][#6])]"),
    ("alkene", "[CX3]=[CX3]"),
    ("alkyne", "[CX2]#[CX2]"),
    ("aromatic_ring", "c1ccccc1"),
    ("heteroaromatic", "[a;!c]"),
    ("phosphate", "[PX4](=[OX1])([OX2])([OX2])[OX2]"),
    ("guanidine", "[NX3][CX3](=[NX2])[NX3]"),
    ("imine", "[CX3]=[NX2]"),
)


def detect_functional_groups(smiles: str) -> dict[str, Any]:
    """Detect functional groups present in a molecule via a SMARTS catalogue."""
    require_rdkit()
    mol = parse_smiles(smiles)
    if mol is None:
        return invalid_smiles_result(smiles)

    found: list[str] = []
    details: list[dict[str, Any]] = []
    for name, smarts in FUNCTIONAL_GROUP_SMARTS:
        pattern = parse_smarts(smarts)
        if pattern is None:
            continue
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            found.append(name)
            details.append({"group": name, "count": len(matches), "smarts": smarts})

    return {
        "smiles": smiles,
        "functional_groups": found,
        "group_count": len(found),
        "details": details,
    }


def analyze_rings(smiles: str) -> dict[str, Any]:
    """Describe the ring systems in a molecule."""
    require_rdkit()
    from rdkit.Chem import rdMolDescriptors

    mol = parse_smiles(smiles)
    if mol is None:
        return invalid_smiles_result(smiles)

    ring_info = mol.GetRingInfo()
    rings: list[dict[str, Any]] = []
    aromatic = 0
    aliphatic = 0

    for atom_indices in ring_info.AtomRings():
        is_aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in atom_indices)
        if is_aromatic:
            aromatic += 1
        else:
            aliphatic += 1
        rings.append(
            {
                "size": len(atom_indices),
                "aromatic": is_aromatic,
                "atom_indices": list(atom_indices),
                "elements": [mol.GetAtomWithIdx(i).GetSymbol() for i in atom_indices],
                "is_heterocycle": any(
                    mol.GetAtomWithIdx(i).GetSymbol() != "C" for i in atom_indices
                ),
            }
        )

    return {
        "smiles": smiles,
        "ring_count": ring_info.NumRings(),
        "aromatic_rings": aromatic,
        "aliphatic_rings": aliphatic,
        "spiro_atoms": rdMolDescriptors.CalcNumSpiroAtoms(mol),
        "bridgehead_atoms": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        "rings": rings,
    }


def analyze_stereo(smiles: str) -> dict[str, Any]:
    """Report tetrahedral and double-bond stereochemistry, including unspecified centres."""
    Chem = require_rdkit()
    mol = parse_smiles(smiles)
    if mol is None:
        return invalid_smiles_result(smiles)

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)

    centers: list[dict[str, Any]] = []
    specified = 0
    for atom_index, chirality in Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False
    ):
        is_specified = chirality != "?"
        specified += int(is_specified)
        centers.append(
            {
                "atom_index": atom_index,
                "element": mol.GetAtomWithIdx(atom_index).GetSymbol(),
                "cip_code": chirality if is_specified else None,
                "specified": is_specified,
                "type": "tetrahedral",
            }
        )

    for bond in mol.GetBonds():
        if bond.GetStereo() == Chem.BondStereo.STEREONONE:
            continue
        centers.append(
            {
                "bond_index": bond.GetIdx(),
                "begin_atom": bond.GetBeginAtomIdx(),
                "end_atom": bond.GetEndAtomIdx(),
                "cip_code": str(bond.GetStereo()).replace("STEREO", ""),
                "specified": True,
                "type": "double_bond",
            }
        )
        specified += 1

    return {
        "smiles": smiles,
        "stereo_centers": centers,
        "stereo_center_count": len(centers),
        "specified_count": specified,
        "has_unspecified_stereo": any(not c["specified"] for c in centers),
    }


def atom_map_list(smiles: str) -> dict[str, Any]:
    """List the atom-map numbers present in a SMILES string."""
    require_rdkit()
    mol = parse_smiles(smiles)
    if mol is None:
        return invalid_smiles_result(smiles)

    maps = [
        {"atom_index": a.GetIdx(), "element": a.GetSymbol(), "map_number": a.GetAtomMapNum()}
        for a in mol.GetAtoms()
        if a.GetAtomMapNum()
    ]
    return {
        "smiles": smiles,
        "atom_maps": maps,
        "mapped_atom_count": len(maps),
        "total_atoms": mol.GetNumAtoms(),
    }


def atom_map_add(smiles: str) -> dict[str, Any]:
    """Assign sequential atom-map numbers to every heavy atom."""
    Chem = require_rdkit()
    mol = parse_smiles(smiles)
    if mol is None:
        return invalid_smiles_result(smiles)

    for index, atom in enumerate(mol.GetAtoms(), start=1):
        atom.SetAtomMapNum(index)
    mapped = Chem.MolToSmiles(mol)
    return {
        "input_smiles": smiles,
        "output_smiles": mapped,
        # Kept under its historical name too; callers and tests use both spellings.
        "mapped_smiles": mapped,
        "mapped_atom_count": mol.GetNumAtoms(),
    }


def atom_map_remove(smiles: str) -> dict[str, Any]:
    """Strip all atom-map numbers from a SMILES string."""
    Chem = require_rdkit()
    mol = parse_smiles(smiles)
    if mol is None:
        return invalid_smiles_result(smiles)

    removed = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum():
            atom.SetAtomMapNum(0)
            removed += 1
    return {
        "input_smiles": smiles,
        "output_smiles": Chem.MolToSmiles(mol),
        "removed_count": removed,
    }


def atom_map_check(smirks: str) -> dict[str, Any]:
    """Check that atom maps balance between the reactant and product sides of a SMIRKS."""
    from rdkit.Chem import AllChem

    require_rdkit()
    try:
        rxn = AllChem.ReactionFromSmarts((smirks or "").strip())
    except Exception as exc:
        return {"smirks": smirks, "valid": False, "error": f"Could not parse SMIRKS: {exc}"}
    if rxn is None:
        return {"smirks": smirks, "valid": False, "error": "RDKit could not parse this SMIRKS."}

    def _maps(templates) -> set[int]:
        numbers: set[int] = set()
        for template in templates:
            numbers.update(a.GetAtomMapNum() for a in template.GetAtoms() if a.GetAtomMapNum())
        return numbers

    reactant_maps = _maps(rxn.GetReactants())
    product_maps = _maps(rxn.GetProducts())
    unmapped_products = sorted(product_maps - reactant_maps)
    unused_reactants = sorted(reactant_maps - product_maps)

    return {
        "smirks": smirks,
        "valid": not unmapped_products,
        "balanced": not unmapped_products and not unused_reactants,
        "reactant_map_numbers": sorted(reactant_maps),
        "product_map_numbers": sorted(product_maps),
        "product_maps_without_reactant": unmapped_products,
        "reactant_maps_not_in_products": unused_reactants,
        "summary": (
            "Atom mapping is balanced."
            if not unmapped_products and not unused_reactants
            else "Atom mapping is incomplete; see the unmatched map numbers."
        ),
    }


def apply_reaction(smirks: str, reactants: list[str]) -> dict[str, Any]:
    """Apply a reaction SMIRKS to reactant SMILES and return the product sets.

    The previous CLI backend could not do this at all (the WASM build has no
    RunReactants); the Python RDKit library can, so this now runs for real.
    """
    Chem = require_rdkit()
    from rdkit.Chem import AllChem

    try:
        rxn = AllChem.ReactionFromSmarts((smirks or "").strip())
    except Exception as exc:
        return {"reaction": smirks, "error": f"Could not parse SMIRKS: {exc}", "products": []}
    if rxn is None:
        return {"reaction": smirks, "error": "RDKit could not parse this SMIRKS.", "products": []}

    reactant_list = [r.strip() for r in reactants if str(r).strip()]
    mols = []
    for smiles in reactant_list:
        mol = parse_smiles(smiles)
        if mol is None:
            return {
                "reaction": smirks,
                "error": f"Reactant SMILES could not be parsed: {smiles}",
                "products": [],
            }
        mols.append(mol)

    expected = rxn.GetNumReactantTemplates()
    if len(mols) != expected:
        return {
            "reaction": smirks,
            "error": (f"This reaction needs {expected} reactant(s) but {len(mols)} were supplied."),
            "products": [],
        }

    try:
        product_sets = rxn.RunReactants(tuple(mols))
    except Exception as exc:
        return {"reaction": smirks, "error": f"Reaction failed: {exc}", "products": []}

    products: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for product_set in product_sets:
        smiles_set = []
        for product in product_set:
            try:
                Chem.SanitizeMol(product)
            except Exception:
                # An unsanitisable product is still worth reporting to the caller.
                logger.debug("Product failed sanitisation; reporting unsanitised SMILES.")
            smiles_set.append(Chem.MolToSmiles(product))
        key = tuple(smiles_set)
        if key not in seen:
            seen.add(key)
            products.append(smiles_set)

    return {
        "reaction": smirks,
        "reactant_count": len(mols),
        "reactants": reactant_list,
        "product_set_count": len(products),
        "products": products,
    }


def draw_molecule(
    smiles: str,
    output_file: str | None = None,
    output_format: str = "svg",
    width: int = 300,
    height: int = 300,
    highlight_atoms: dict[str, str] | None = None,
    highlight_bonds: dict[str, str] | None = None,
    highlight_radius: float = 0.3,
) -> dict[str, Any]:
    """Render a molecule to SVG or PNG, optionally highlighting atoms and bonds."""
    Chem = require_rdkit()
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = parse_smiles(smiles)
    if mol is None:
        return invalid_smiles_result(smiles)

    fmt = (output_format or "svg").strip().lower()
    if fmt not in ("svg", "png"):
        return {"smiles": smiles, "error": f"Unsupported format {output_format!r}; use svg or png."}

    rdMolDraw2D.PrepareMolForDrawing(mol)
    Draw.rdDepictor.Compute2DCoords(mol)

    atom_ids, atom_colors = _parse_highlights(highlight_atoms)
    bond_ids, bond_colors = _parse_highlights(highlight_bonds)

    drawer = (
        rdMolDraw2D.MolDraw2DSVG(width, height)
        if fmt == "svg"
        else rdMolDraw2D.MolDraw2DCairo(width, height)
    )
    if atom_ids:
        drawer.drawOptions().highlightRadius = highlight_radius
    drawer.DrawMolecule(
        mol,
        highlightAtoms=atom_ids or None,
        highlightAtomColors=atom_colors or None,
        highlightBonds=bond_ids or None,
        highlightBondColors=bond_colors or None,
    )
    drawer.FinishDrawing()
    payload = drawer.GetDrawingText()

    result: dict[str, Any] = {
        "smiles": smiles,
        "format": fmt,
        "width": width,
        "height": height,
        "canonical_smiles": Chem.MolToSmiles(mol),
    }

    if output_file:
        path = Path(output_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "svg":
            path.write_text(payload)
        else:
            path.write_bytes(payload)
        from bioagents.sandbox.workspace import to_portable_path

        result["output_file"] = to_portable_path(path)
        result["bytes_written"] = path.stat().st_size
        return result

    if fmt == "svg":
        result["svg"] = payload
    else:
        result["png_base64"] = base64.b64encode(payload).decode("ascii")
    return result


def _parse_highlights(spec: dict[str, str] | None):
    """Turn a {"index": "#rrggbb"} mapping into RDKit's id list + colour dict."""
    if not spec:
        return [], {}
    ids: list[int] = []
    colors: dict[int, tuple[float, float, float]] = {}
    for key, value in spec.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        ids.append(index)
        rgb = _hex_to_rgb(value)
        if rgb:
            colors[index] = rgb
    return ids, colors


def _hex_to_rgb(value: str) -> tuple[float, float, float] | None:
    """Convert '#rrggbb' to the 0-1 RGB triple RDKit expects."""
    text = (value or "").strip().lstrip("#")
    if len(text) != 6:
        return None
    try:
        return tuple(int(text[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return None
