import re
from pathlib import Path
from typing import Any

# NumPy for distance calculation
import numpy as np

# Biopython imports
from Bio.PDB import PDBList, PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionException


def _parse_residue_spec(spec: str) -> tuple[str, int]:
    """
    Parses a residue specification string (e.g., 'ASP72') into its name and number.

    Args:
        spec (str): The residue specification string.

    Returns:
        Tuple[str, int]: A tuple containing the 3-letter residue name and its number.

    Raises:
        ValueError: If the specification format is invalid.
    """
    match = re.match(r"([A-Z]{3})(\d+)", spec)
    if not match:
        raise ValueError(
            f"Invalid residue specification format: '{spec}'. Expected format like 'ASP72'."
        )
    res_name = match.group(1).upper()
    res_num = int(match.group(2))
    return res_name, res_num


def analyze_pdb_chain(
    pdb_id: str, chain_id: str, residue1_spec: str | None = None, residue2_spec: str | None = None
) -> dict[str, Any]:
    """
    Parses a PDB file, extracts a specified chain, counts VAL and TRP residues (alpha carbons),
    and calculates the distance between alpha carbons of two specified residues.

    Args:
        pdb_id (str): The PDB ID (e.g., '1PC2').
        chain_id (str): The chain ID to analyze (e.g., 'A').
        residue1_spec (Optional[str]): Specification for the first residue for distance calculation
                                        (e.g., 'ASP72'). Format: RESIDUE_NAME_3_LETTER_CODE + RESIDUE_NUMBER.
                                        Required if residue2_spec is provided.
        residue2_spec (Optional[str]): Specification for the second residue for distance calculation
                                        (e.g., 'GLY144'). Format: RESIDUE_NAME_3_LETTER_CODE + RESIDUE_NUMBER.

    Returns:
        Dict[str, Any]: A dictionary containing the counts of VAL and TRP residues, and the calculated distance
                        if residue specifications are provided.

    Raises:
        ValueError: If input parameters are invalid or required residues are not found.
        RuntimeError: If PDB file cannot be downloaded or parsed, or if Biopython is not installed.

    Example:
        >>> # To count VAL and TRP residues in chain A of 1PC2
        >>> analyze_pdb_chain(pdb_id='1PC2', chain_id='A')
        {'val_count': 10, 'trp_count': 2, 'distance_angstroms': None}

        >>> # To count VAL and TRP and calculate distance between ASP72 and GLY144 in chain A of 1PC2
        >>> analyze_pdb_chain(pdb_id='1PC2', chain_id='A', residue1_spec='ASP72', residue2_spec='GLY144')
        {'val_count': 10, 'trp_count': 2, 'distance_angstroms': 15.78}
    """
    if (residue1_spec and not residue2_spec) or (not residue1_spec and residue2_spec):
        raise ValueError(
            "Both 'residue1_spec' and 'residue2_spec' must be provided for distance calculation."
        )

    # Initialize PDBList to download PDB files
    pdbl = PDBList()
    try:
        # Attempt to download PDB file. By default, it downloads to current directory.
        # We'll try to find it in the current directory or a 'pdb' subdirectory.
        pdb_file_path = pdbl.retrieve_pdb_file(pdb_id, pdir=".", file_format="pdb")
        if not Path(pdb_file_path).exists():
            # Try .cif format if .pdb fails
            pdb_file_path = pdbl.retrieve_pdb_file(pdb_id, pdir=".", file_format="cif")
            if not Path(pdb_file_path).exists():
                raise RuntimeError(f"Could not download PDB or CIF file for {pdb_id}.")

        # Determine parser based on file extension
        parser: PDBParser | MMCIFParser
        if pdb_file_path.endswith(".pdb"):
            parser = PDBParser(QUIET=True)
        elif pdb_file_path.endswith(".cif"):
            parser = MMCIFParser(QUIET=True)
        else:
            raise RuntimeError(f"Unsupported file format for {pdb_id}: {pdb_file_path}")

        structure = parser.get_structure(pdb_id, pdb_file_path)
    except PDBConstructionException as e:
        raise RuntimeError(f"Error parsing PDB file {pdb_id}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve or parse PDB file {pdb_id}: {e}") from e

    val_count = 0
    trp_count = 0
    ca_atom_res1 = None
    ca_atom_res2 = None

    target_res1_name, target_res1_num = (None, None)
    target_res2_name, target_res2_num = (None, None)

    if residue1_spec and residue2_spec:
        target_res1_name, target_res1_num = _parse_residue_spec(residue1_spec)
        target_res2_name, target_res2_num = _parse_residue_spec(residue2_spec)

    found_chain = False
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                found_chain = True
                for residue in chain:
                    # Biopython residue ID is a tuple: (hetero_flag, residue_number, insertion_code)
                    res_id = residue.get_id()
                    res_name = residue.get_resname()
                    res_num = res_id[1]

                    # Count VAL and TRP alpha carbons
                    if res_name == "VAL" and "CA" in residue:
                        val_count += 1
                    elif res_name == "TRP" and "CA" in residue:
                        trp_count += 1

                    # Find alpha carbons for specified residues for distance calculation
                    if (
                        target_res1_name
                        and target_res1_num
                        and res_name == target_res1_name
                        and res_num == target_res1_num
                    ):
                        if "CA" in residue:
                            ca_atom_res1 = residue["CA"]
                        else:
                            raise ValueError(
                                f"Alpha carbon not found for {residue1_spec} in chain {chain_id}."
                            )
                    if (
                        target_res2_name
                        and target_res2_num
                        and res_name == target_res2_name
                        and res_num == target_res2_num
                    ):
                        if "CA" in residue:
                            ca_atom_res2 = residue["CA"]
                        else:
                            raise ValueError(
                                f"Alpha carbon not found for {residue2_spec} in chain {chain_id}."
                            )
                break  # Chain found, no need to iterate further in this model
        if found_chain:
            break  # Chain found, no need to iterate further in other models

    if not found_chain:
        raise ValueError(f"Chain '{chain_id}' not found in PDB ID '{pdb_id}'.")

    distance_angstroms = None
    if ca_atom_res1 is not None and ca_atom_res2 is not None:
        distance_angstroms = np.linalg.norm(ca_atom_res1.get_coord() - ca_atom_res2.get_coord())
    elif (residue1_spec and residue2_spec) and (ca_atom_res1 is None or ca_atom_res2 is None):
        raise ValueError(
            f"Could not find both specified residues ({residue1_spec}, {residue2_spec}) in chain {chain_id}."
        )

    # Clean up downloaded PDB file
    if pdb_file_path and Path(pdb_file_path).exists():
        Path(pdb_file_path).unlink()
        # Also remove the directory if it's empty and was created by PDBList
        pdb_dir = Path(pdb_file_path).parent
        if str(pdb_dir) != "." and not any(pdb_dir.iterdir()):
            pdb_dir.rmdir()

    return {
        "val_count": val_count,
        "trp_count": trp_count,
        "distance_angstroms": distance_angstroms,
    }
