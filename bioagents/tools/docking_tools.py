"""Molecular docking tools for protein-ligand docking using AutoDock Vina.

Provides tools for:
- Preparing receptor structures (PDB → PDBQT)
- Preparing ligands from SMILES (SMILES → 3D PDBQT)
- Running molecular docking with AutoDock Vina
- Parsing and summarizing docking results
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from langchain_core.tools import BaseTool, tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Workspace base directory for file operations (prevents path traversal)
_WORKSPACE_BASE = Path(os.environ.get("BIOAGENTS_SANDBOX_DIR", str(Path.cwd()))).resolve()

# Distance cutoffs for interaction analysis (Angstroms)
HBOND_DISTANCE_CUTOFF = 3.5
HYDROPHOBIC_DISTANCE_CUTOFF = 4.0

# Max SMILES length to prevent resource exhaustion
MAX_SMILES_LENGTH = 1000

# Water residue names to strip during receptor preparation
_WATER_RESNAMES = {"HOH", "WAT", "DOD", "TIP3", "H2O"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_path(path_str: str) -> Path:
    """Validate a file path stays within the allowed workspace.

    Args:
        path_str: Raw file path string to validate.

    Returns:
        Resolved Path object.

    Raises:
        ValueError: If path traversal is detected or path is outside workspace.
    """
    resolved = Path(path_str).resolve()
    # Block obvious traversal attempts
    if ".." in Path(path_str).parts:
        raise ValueError(f"Path traversal rejected: {path_str}")
    # Allow absolute paths under workspace, and all relative paths
    if (
        resolved.is_absolute()
        and not str(resolved).startswith(str(_WORKSPACE_BASE))
        and not str(resolved).startswith(tempfile.gettempdir())
    ):
        raise ValueError(
            f"Path outside allowed workspace: {path_str}. Workspace: {_WORKSPACE_BASE}"
        )
    return resolved


def _parse_pdb_structure(path: Path, name: str = "structure"):
    """Parse a PDB file and validate it contains usable structure data.

    Args:
        path: Validated, resolved path to the PDB file.
        name: Name for the structure in error messages.

    Returns:
        BioPython Structure object with at least one model.

    Raises:
        ValueError: If the file is empty, corrupted, or contains no models.
    """
    from Bio.PDB import PDBParser

    # Check file content is not empty or obviously non-PDB
    content = path.read_text(errors="replace").strip()
    if not content:
        raise ValueError(f"PDB file is empty: {path}")
    if content.startswith("<?xml") or content.startswith("<Error"):
        raise ValueError(
            f"PDB file appears corrupted (contains XML error response instead of PDB data): {path}. "
            "The file may need to be re-downloaded."
        )
    if not any(line.startswith(("ATOM", "HETATM", "MODEL")) for line in content.splitlines()):
        raise ValueError(f"PDB file contains no ATOM/HETATM records (possibly corrupted): {path}")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(name, str(path))

    if len(structure) == 0:
        raise ValueError(f"PDB file contains no structural models (possibly corrupted): {path}")

    atom_count = sum(1 for _ in structure.get_atoms())
    if atom_count == 0:
        raise ValueError(f"PDB file contains no atoms (possibly corrupted or empty): {path}")

    return structure


def _assign_vina_atom_type(element: str, atom_name: str) -> str:
    """Assign AutoDock Vina atom type from element and atom name.

    Args:
        element: Chemical element symbol (e.g. 'C', 'N', 'O').
        atom_name: Full atom name from PDB (e.g. 'NZ', 'OH').

    Returns:
        Vina atom type string.
    """
    if element == "C":
        return "C"
    elif element == "N":
        # Backbone N is typically HD (donor), sidechain N can be NA (acceptor)
        if atom_name == "N":
            return "HD"
        return "NA"
    elif element == "O":
        return "OA"
    elif element == "S":
        return "S"
    elif element == "H":
        return "HD"
    elif element == "P":
        return "P"
    elif element in ("F", "CL", "BR", "I"):
        return element
    else:
        return element


def _pdb_to_pdbqt(pdb_path: str, output_path: str | None = None) -> dict:
    """Convert a PDB file to PDBQT format for AutoDock Vina.

    Strips waters and non-protein ligands, keeps ATOM records only,
    and assigns Vina-compatible atom types.

    Args:
        pdb_path: Path to input PDB file.
        output_path: Optional output path for PDBQT file.

    Returns:
        Dict with status and file_path or error message.
    """
    try:
        from Bio.PDB import PDBParser  # noqa: F401
    except ImportError:
        return {
            "status": "error",
            "message": "BioPython is required. Install with: pip install biopython",
        }

    try:
        path = _validate_path(pdb_path)
        if not path.exists():
            return {"status": "error", "message": f"PDB file not found: {pdb_path}"}

        out = _validate_path(output_path) if output_path is not None else path.with_suffix(".pdbqt")

        structure = _parse_pdb_structure(path, name="receptor")

        pdbqt_lines: list[str] = []
        atom_serial = 0

        for atom in structure.get_atoms():
            res = atom.get_parent()
            chain = res.get_parent()

            # Skip non-standard residues and waters
            hetflag = res.id[0]
            if hetflag != " ":
                continue
            resname = res.get_resname().strip()
            if resname in _WATER_RESNAMES:
                continue

            atom_serial += 1
            coord = atom.get_coord()
            element = atom.element.strip().upper()
            atype = _assign_vina_atom_type(element, atom.get_name().strip())

            pdbqt_lines.append(
                f"ATOM  {atom_serial:5d} {atom.get_name():4s} "
                f"{resname:3s} {chain.id}{res.id[1]:4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  "
                f"0.00  0.00    +0.000 {atype:>2s}"
            )

        with out.open("w") as f:
            f.write("\n".join(pdbqt_lines) + "\nTER\nEND\n")

        return {
            "status": "success",
            "file_path": str(out.absolute()),
            "atom_count": atom_serial,
        }

    except ValueError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.error("Error preparing receptor: %s", e)
        return {"status": "error", "message": str(e)}


def _smiles_to_pdbqt(smiles: str, output_path: str | None = None) -> dict:
    """Convert a SMILES string to a 3D PDBQT ligand file.

    Uses RDKit for 3D conformer generation and MMFF94 energy minimization,
    then meeko for PDBQT conversion with proper torsion tree assignment.

    Args:
        smiles: SMILES string of the ligand.
        output_path: Optional output path for PDBQT file.

    Returns:
        Dict with status, file_path, and molecular properties.
    """
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors
    except ImportError as e:
        return {
            "status": "error",
            "message": f"Required package not installed: {e}. "
            "Install with: pip install rdkit meeko",
        }

    try:
        if len(smiles) > MAX_SMILES_LENGTH:
            return {
                "status": "error",
                "message": f"SMILES string too long ({len(smiles)} chars, max {MAX_SMILES_LENGTH})",
            }

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"status": "error", "message": f"Invalid SMILES: {smiles}"}

        mol = Chem.AddHs(mol)

        # Generate 3D coordinates
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result == -1:
            result = AllChem.EmbedMolecule(
                mol, randomSeed=42, useRandomCoords=True, maxAttempts=100
            )
        if result == -1:
            return {
                "status": "error",
                "message": f"Could not generate 3D coordinates for: {smiles}",
            }

        # Energy minimization
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

        # Compute properties
        mw = round(Descriptors.ExactMolWt(mol), 2)
        logp = round(Descriptors.MolLogP(mol), 2)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)

        # Convert to PDBQT
        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)
        pdbqt_str, _, _ = PDBQTWriterLegacy.write_string(mol_setups[0])

        if output_path is None:
            tmp = tempfile.mkdtemp(prefix="bioagents_dock_")
            output_path = str(Path(tmp) / "ligand.pdbqt")

        validated_out = _validate_path(output_path)
        with validated_out.open("w") as f:
            f.write(pdbqt_str)

        return {
            "status": "success",
            "file_path": str(validated_out.absolute()),
            "smiles": smiles,
            "properties": {
                "molecular_weight": mw,
                "logp": logp,
                "hbd": hbd,
                "hba": hba,
                "rotatable_bonds": rotatable_bonds,
                "lipinski_rule_of_5": mw < 500 and logp < 5 and hbd <= 5 and hba <= 10,
            },
        }

    except ValueError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.error("Error preparing ligand: %s", e)
        return {"status": "error", "message": str(e)}


def _parse_docking_output(pdbqt_path: str) -> list[dict]:
    """Parse AutoDock Vina output PDBQT file for pose energies and metadata.

    Args:
        pdbqt_path: Path to Vina output PDBQT file.

    Returns:
        List of pose dicts with affinity, rmsd_lb, rmsd_ub, and model index.
    """
    poses: list[dict] = []
    current_model = 0

    with Path(pdbqt_path).open() as f:
        for line in f:
            if line.startswith("MODEL"):
                current_model = int(line.split()[1])
            elif line.startswith("REMARK VINA RESULT:"):
                parts = line.split()
                if len(parts) >= 6:
                    affinity = float(parts[3])
                    rmsd_lb = float(parts[4])
                    rmsd_ub = float(parts[5])
                    poses.append(
                        {
                            "model": current_model,
                            "affinity_kcal_mol": affinity,
                            "rmsd_lb": rmsd_lb,
                            "rmsd_ub": rmsd_ub,
                        }
                    )

    return poses


# ---------------------------------------------------------------------------
# LangChain Tool Definitions
# ---------------------------------------------------------------------------


@tool
def prepare_receptor(
    pdb_path: str,
    output_dir: str | None = None,
) -> str:
    """Prepare a receptor protein structure for molecular docking.

    Converts a PDB file to PDBQT format suitable for AutoDock Vina.
    Removes water molecules and non-protein ligands. The output file
    is saved alongside the input with a .pdbqt extension unless
    output_dir is specified.

    Args:
        pdb_path: Path to the receptor PDB file.
        output_dir: Optional directory for the output PDBQT file.

    Returns:
        JSON string with preparation status and output file path.
    """
    try:
        if output_dir:
            validated_dir = _validate_path(output_dir)
            name = Path(pdb_path).stem + ".pdbqt"
            out = str(validated_dir / name)
        else:
            out = None

        result = _pdb_to_pdbqt(pdb_path, out)
        return json.dumps(result, indent=2)
    except ValueError as e:
        return json.dumps({"status": "error", "message": str(e)})


@tool
def prepare_ligand(
    smiles: str,
    ligand_name: str = "ligand",
    output_dir: str | None = None,
) -> str:
    """Prepare a ligand molecule from SMILES for molecular docking.

    Generates a 3D conformation, minimizes energy using MMFF94 force field,
    and converts to PDBQT format with proper torsion tree assignment.
    Also computes molecular properties (MW, LogP, HBD, HBA, Lipinski Ro5).

    Args:
        smiles: SMILES string of the ligand molecule.
        ligand_name: Name identifier for the ligand (used in filename).
        output_dir: Optional directory for the output PDBQT file.

    Returns:
        JSON string with preparation status, file path, and molecular properties.
    """
    try:
        if output_dir:
            validated_dir = _validate_path(output_dir)
            validated_dir.mkdir(parents=True, exist_ok=True)
            out = str(validated_dir / f"{ligand_name}.pdbqt")
        else:
            out = None

        result = _smiles_to_pdbqt(smiles, out)
        return json.dumps(result, indent=2)
    except ValueError as e:
        return json.dumps({"status": "error", "message": str(e)})


@tool
def run_docking(
    receptor_pdbqt_path: str,
    ligand_pdbqt_path: str,
    center_x: float,
    center_y: float,
    center_z: float,
    box_size_x: float = 20.0,
    box_size_y: float = 20.0,
    box_size_z: float = 20.0,
    exhaustiveness: int = 8,
    num_poses: int = 5,
    output_dir: str | None = None,
) -> str:
    """Run molecular docking using AutoDock Vina.

    Performs protein-ligand docking with the prepared receptor and ligand
    PDBQT files. Returns binding affinities and RMSD values for all poses.
    The best-docked poses are saved as a PDBQT file.

    Args:
        receptor_pdbqt_path: Path to prepared receptor PDBQT file.
        ligand_pdbqt_path: Path to prepared ligand PDBQT file.
        center_x: Grid box center X coordinate in Angstroms.
        center_y: Grid box center Y coordinate in Angstroms.
        center_z: Grid box center Z coordinate in Angstroms.
        box_size_x: Grid box size X in Angstroms (default: 20).
        box_size_y: Grid box size Y in Angstroms (default: 20).
        box_size_z: Grid box size Z in Angstroms (default: 20).
        exhaustiveness: Sampling thoroughness, higher = more accurate but slower (default: 8).
        num_poses: Maximum number of poses to generate (default: 5).
        output_dir: Optional directory for docking output files.

    Returns:
        JSON string with docking results including binding affinities,
        RMSD values, and paths to output pose files.
    """
    try:
        from vina import Vina
    except ImportError:
        return json.dumps(
            {
                "status": "error",
                "message": "AutoDock Vina is required. Install with: pip install vina",
            }
        )

    try:
        # Validate input file paths
        rec_path = _validate_path(receptor_pdbqt_path)
        lig_path = _validate_path(ligand_pdbqt_path)

        for path_obj, label in [(rec_path, "Receptor"), (lig_path, "Ligand")]:
            if not path_obj.exists():
                return json.dumps(
                    {"status": "error", "message": f"{label} file not found: {path_obj}"}
                )

        # Resolve output directory
        if output_dir:
            out_dir = _validate_path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = Path(tempfile.mkdtemp(prefix="bioagents_dock_"))

        # Initialize Vina
        v = Vina(sf_name="vina")
        v.set_receptor(str(rec_path))
        v.set_ligand_from_file(str(lig_path))

        # Compute affinity maps
        center = [center_x, center_y, center_z]
        box_size = [box_size_x, box_size_y, box_size_z]
        v.compute_vina_maps(center=center, box_size=box_size)

        # Run docking
        v.dock(exhaustiveness=exhaustiveness, n_poses=num_poses)

        # Extract energies with defensive length check
        energies = v.energies()
        poses: list[dict] = []
        for i, energy_row in enumerate(energies):
            if len(energy_row) < 5:
                logger.warning("Unexpected energy row format: %s", energy_row)
                continue
            poses.append(
                {
                    "pose_number": i + 1,
                    "affinity_kcal_mol": round(float(energy_row[0]), 3),
                    "inter_energy": round(float(energy_row[1]), 3),
                    "intra_energy": round(float(energy_row[2]), 3),
                    "torsional_energy": round(float(energy_row[3]), 3),
                    "unbound_energy": round(float(energy_row[4]), 3),
                }
            )

        if not poses:
            return json.dumps({"status": "error", "message": "Docking produced no valid poses."})

        # Save poses
        ligand_stem = lig_path.stem
        poses_path = out_dir / f"{ligand_stem}_docked.pdbqt"
        v.write_poses(str(poses_path), n_poses=num_poses)

        # Parse the output for RMSD information
        parsed_poses = _parse_docking_output(str(poses_path))
        for i, pose in enumerate(poses):
            if i < len(parsed_poses):
                pose["rmsd_lb"] = parsed_poses[i].get("rmsd_lb")
                pose["rmsd_ub"] = parsed_poses[i].get("rmsd_ub")

        best_affinity = poses[0]["affinity_kcal_mol"]

        return json.dumps(
            {
                "status": "success",
                "best_affinity_kcal_mol": best_affinity,
                "grid_center": center,
                "grid_size": box_size,
                "exhaustiveness": exhaustiveness,
                "num_poses": len(poses),
                "poses": poses,
                "docked_poses_file": str(poses_path.absolute()),
            },
            indent=2,
        )

    except ValueError as e:
        return json.dumps({"status": "error", "message": str(e)})
    except Exception as e:
        logger.error("Error running docking: %s", e)
        return json.dumps({"status": "error", "message": str(e)})


@tool
def identify_binding_site(
    pdb_path: str,
    chain_id: str | None = None,
    residue_numbers: list[int] | None = None,
    padding: float = 5.0,
) -> str:
    """Identify a binding site and compute grid box coordinates for docking.

    Calculates the center and optimal box size for a binding site defined
    by either specific residue numbers or the full protein. Uses the
    geometric center of selected residues with padding.

    Args:
        pdb_path: Path to the protein PDB file.
        chain_id: Optional chain ID to focus on. If None, uses all chains.
        residue_numbers: Optional list of residue numbers defining the binding site.
            If None, computes for the entire structure.
        padding: Padding around the site in Angstroms (default: 5.0).

    Returns:
        JSON string with grid box center coordinates, box dimensions,
        and binding site residue information.
    """
    try:
        import numpy as np
        from Bio.PDB import PDBParser  # noqa: F401
    except ImportError:
        return json.dumps(
            {
                "status": "error",
                "message": "BioPython and numpy are required. "
                "Install with: pip install biopython numpy",
            }
        )

    try:
        path = _validate_path(pdb_path)
        if not path.exists():
            return json.dumps({"status": "error", "message": f"PDB file not found: {pdb_path}"})

        structure = _parse_pdb_structure(path, name="protein")
        model = structure[0]

        coords: list[list[float]] = []
        site_residues: list[dict] = []

        for chain in model:
            if chain_id and chain.id != chain_id:
                continue
            for residue in chain:
                if residue.id[0] != " ":
                    continue  # Skip HETATM
                if residue_numbers and residue.id[1] not in residue_numbers:
                    continue

                res_info = {
                    "chain": chain.id,
                    "residue": residue.get_resname(),
                    "position": residue.id[1],
                }
                site_residues.append(res_info)

                for atom in residue:
                    if atom.element.strip() != "H":
                        coords.append(atom.get_coord().tolist())

        if not coords:
            return json.dumps(
                {
                    "status": "error",
                    "message": "No atoms found for the specified selection. "
                    "Check chain_id and residue_numbers.",
                }
            )

        coords_array = np.array(coords)
        center = coords_array.mean(axis=0)
        mins = coords_array.min(axis=0)
        maxs = coords_array.max(axis=0)
        size = maxs - mins + 2 * padding

        return json.dumps(
            {
                "status": "success",
                "grid_center": {
                    "x": round(float(center[0]), 3),
                    "y": round(float(center[1]), 3),
                    "z": round(float(center[2]), 3),
                },
                "grid_size": {
                    "x": round(float(size[0]), 3),
                    "y": round(float(size[1]), 3),
                    "z": round(float(size[2]), 3),
                },
                "padding_angstrom": padding,
                "site_residue_count": len(site_residues),
                "site_residues": site_residues[:50],
            },
            indent=2,
        )

    except ValueError as e:
        return json.dumps({"status": "error", "message": str(e)})
    except Exception as e:
        logger.error("Error identifying binding site: %s", e)
        return json.dumps({"status": "error", "message": str(e)})


@tool
def analyze_docking_results(
    docked_poses_path: str,
    receptor_pdb_path: str,
    distance_cutoff: float = 3.5,
) -> str:
    """Analyze docking results to identify protein-ligand interactions.

    Parses the docked poses file for binding affinities, then identifies
    potential hydrogen bonds and hydrophobic contacts between the best-docked
    ligand pose and the receptor based on distance criteria.

    The distance_cutoff controls the maximum distance for the initial contact
    detection. Hydrogen bonds are detected at <= 3.5 Å and hydrophobic
    contacts at <= 4.0 Å, regardless of the distance_cutoff value.

    Args:
        docked_poses_path: Path to the Vina output PDBQT file with docked poses.
        receptor_pdb_path: Path to the receptor PDB file.
        distance_cutoff: Maximum distance in Angstroms for initial contact
            detection (default: 3.5). Must be >= 3.5 to detect all interaction types.

    Returns:
        JSON string with interaction analysis including hydrogen bonds,
        hydrophobic contacts, and per-residue contact summary.
    """
    try:
        import numpy as np
        from Bio.PDB import PDBParser  # noqa: F401
    except ImportError:
        return json.dumps(
            {
                "status": "error",
                "message": "BioPython and numpy are required. "
                "Install with: pip install biopython numpy",
            }
        )

    try:
        poses_path = _validate_path(docked_poses_path)
        if not poses_path.exists():
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Docked poses file not found: {docked_poses_path}",
                }
            )

        rec_path = _validate_path(receptor_pdb_path)
        if not rec_path.exists():
            return json.dumps(
                {"status": "error", "message": f"Receptor file not found: {receptor_pdb_path}"}
            )

        parsed_poses = _parse_docking_output(str(poses_path))

        rec_structure = _parse_pdb_structure(rec_path, name="receptor")
        lig_structure = _parse_pdb_structure(poses_path, name="ligand")

        # Analyze best pose (first model)
        best_pose = parsed_poses[0] if parsed_poses else {}
        lig_model = lig_structure[0]

        # Collect receptor atoms
        rec_atoms: list[dict] = []
        for atom in rec_structure[0].get_atoms():
            res = atom.get_parent()
            chain = res.get_parent()
            if res.id[0] != " ":
                continue
            rec_atoms.append(
                {
                    "coord": atom.get_coord(),
                    "element": atom.element.strip().upper(),
                    "name": atom.get_name().strip(),
                    "residue": f"{res.get_resname()}{chain.id}{res.id[1]}",
                }
            )

        # Collect ligand atoms (first model only)
        lig_atoms: list[dict] = []
        for atom in lig_model.get_atoms():
            element = atom.element.strip().upper()
            if not element or len(element) > 2:
                name = atom.get_name().strip()
                element = name[0] if name else "C"
            lig_atoms.append(
                {
                    "coord": atom.get_coord(),
                    "element": element,
                    "name": atom.get_name().strip(),
                }
            )

        # Use the wider of distance_cutoff and the interaction-specific cutoffs
        max_detection_dist = max(distance_cutoff, HYDROPHOBIC_DISTANCE_CUTOFF)

        # Interaction element sets
        hbond_elements = {"N", "O"}
        hydrophobic_elements = {"C"}

        interactions: list[dict] = []
        # Use consistent keys matching interaction_type values
        residue_contacts: dict[str, dict] = {}

        for lig_atom in lig_atoms:
            for rec_atom in rec_atoms:
                dist = float(np.linalg.norm(lig_atom["coord"] - rec_atom["coord"]))
                if dist > max_detection_dist:
                    continue

                interaction_type = None

                # Check hydrogen bonds (both directions)
                if (
                    lig_atom["element"] in hbond_elements
                    and rec_atom["element"] in hbond_elements
                    and dist <= HBOND_DISTANCE_CUTOFF
                ):
                    interaction_type = "potential_hbond"
                elif (
                    lig_atom["element"] in hydrophobic_elements
                    and rec_atom["element"] in hydrophobic_elements
                    and dist <= HYDROPHOBIC_DISTANCE_CUTOFF
                ):
                    interaction_type = "hydrophobic"
                elif dist <= distance_cutoff:
                    interaction_type = "close_contact"

                if interaction_type:
                    residue = rec_atom["residue"]
                    if residue not in residue_contacts:
                        residue_contacts[residue] = {
                            "potential_hbond": 0,
                            "hydrophobic": 0,
                            "close_contact": 0,
                        }
                    residue_contacts[residue][interaction_type] += 1

                    interactions.append(
                        {
                            "type": interaction_type,
                            "ligand_atom": lig_atom["name"],
                            "receptor_residue": residue,
                            "receptor_atom": rec_atom["name"],
                            "distance_angstrom": round(dist, 2),
                        }
                    )

        # Summarize unique residues
        contact_summary = []
        for residue, counts in sorted(residue_contacts.items()):
            contact_summary.append({"residue": residue, **counts})

        return json.dumps(
            {
                "status": "success",
                "best_pose": best_pose,
                "total_interactions": len(interactions),
                "interaction_types": {
                    "potential_hbonds": sum(
                        1 for i in interactions if i["type"] == "potential_hbond"
                    ),
                    "hydrophobic_contacts": sum(
                        1 for i in interactions if i["type"] == "hydrophobic"
                    ),
                    "close_contacts": sum(1 for i in interactions if i["type"] == "close_contact"),
                },
                "contacting_residues": contact_summary,
                "interactions": interactions[:100],
            },
            indent=2,
        )

    except ValueError as e:
        return json.dumps({"status": "error", "message": str(e)})
    except Exception as e:
        logger.error("Error analyzing docking results: %s", e)
        return json.dumps({"status": "error", "message": str(e)})


def get_docking_tools() -> list[BaseTool]:
    """Return list of all molecular docking tools."""
    return [
        prepare_receptor,
        prepare_ligand,
        run_docking,
        identify_binding_site,
        analyze_docking_results,
    ]
