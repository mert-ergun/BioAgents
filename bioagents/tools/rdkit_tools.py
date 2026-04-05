"""LangChain @tool decorators for rdkit-agent commands.

This module wraps rdkit-agent wrapper functions as LangChain tools,
providing structured input validation and LLM-friendly output formatting.

Each tool returns a JSON string suitable for LLM interpretation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

from bioagents.tools.rdkit_wrapper import (
    RdkitAgentValidationError,
    RdkitAgentWASMError,
    analyze_rings,
    analyze_stereo,
    apply_reaction,
    atom_map_add,
    atom_map_check,
    atom_map_list,
    atom_map_remove,
    check_reaction,
    check_smiles,
    check_smirks,
    compute_descriptors,
    compute_fingerprint,
    convert_notation,
    dataset_statistics,
    detect_functional_groups,
    draw_molecule,
    edit_molecule,
    filter_molecules,
    repair_smiles,
    search_similar_molecules,
    substructure_search,
)

logger = logging.getLogger(__name__)


def _format_response(data: dict[str, Any], success: bool = True) -> str:
    """Format wrapper response as pretty JSON string for LLM."""
    response = {
        "success": success,
        "data": data,
    }
    return json.dumps(response, indent=2, default=str)


def _format_error(error: Exception) -> str:
    """Format error as JSON string for LLM."""
    if isinstance(error, RdkitAgentWASMError):
        severity = "WASM_LIMITATION"
        suggestion = error.message or "Use Python RDKit library as alternative"
    elif isinstance(error, RdkitAgentValidationError):
        severity = "VALIDATION_ERROR"
        suggestion = "Check chemical notation format and try repairs"
    else:
        severity = "ERROR"
        suggestion = str(error)

    return json.dumps(
        {
            "success": False,
            "error": {
                "type": type(error).__name__,
                "severity": severity,
                "message": str(error),
                "suggestion": suggestion,
            },
        },
        indent=2,
    )


# ============================================================================
# VALIDATION TOOLS
# ============================================================================


@tool
def validate_smiles(smiles: str) -> str:
    """
    Validate a SMILES string for chemical correctness.

    Returns validation status, issues found, and repair suggestions.

    Args:
        smiles: A SMILES string to validate

    Returns:
        JSON string with validation result including:
        - overall_pass: Whether validation passed
        - fix_suggestions: Suggested repairs if invalid
        - corrected_values: Corrected SMILES if available
    """
    try:
        result = check_smiles(smiles)
        return _format_response(result)
    except Exception as e:
        logger.error(f"Error validating SMILES '{smiles}': {e}")
        return _format_error(e)


@tool
def validate_smirks(smirks: str) -> str:
    """
    Validate a SMIRKS string (SMILES Arbitrary Target Specification).

    SMIRKS are used for reaction definitions and substructure searching.

    Args:
        smirks: A SMIRKS string to validate

    Returns:
        JSON string with validation result
    """
    try:
        result = check_smirks(smirks)
        return _format_response(result)
    except Exception as e:
        logger.error(f"Error validating SMIRKS '{smirks}': {e}")
        return _format_error(e)


@tool
def validate_reaction(reactants: list[str], products: list[str]) -> str:
    """
    Validate a chemical reaction by checking atom balance.

    Args:
        reactants: List of SMILES strings for reactants
        products: List of SMILES strings for products

    Returns:
        JSON string with atom balance check results
    """
    try:
        result = check_reaction(reactants, products)
        return _format_response(result)
    except Exception as e:
        logger.error(f"Error validating reaction: {e}")
        return _format_error(e)


# ============================================================================
# REPAIR TOOLS
# ============================================================================


@tool
def repair_invalid_smiles(input_str: str) -> str:
    """
    Repair or reconstruct malformed SMILES strings.

    Can fix ring closure errors, formula normalization, and other issues.
    Returns canonical SMILES with repair strategy and confidence score.

    Args:
        input_str: A potentially invalid SMILES string

    Returns:
        JSON string with:
        - success: Whether repair succeeded
        - canonical_smiles: Repaired SMILES
        - strategy: Repair method used
        - confidence: Confidence score (0-1)
    """
    try:
        result = repair_smiles(input_str)
        return _format_response(result)
    except Exception as e:
        logger.error(f"Error repairing SMILES '{input_str}': {e}")
        return _format_error(e)


# ============================================================================
# CONVERSION TOOLS
# ============================================================================


@tool
def convert_chemical_notation(
    input_str: str,
    from_format: str,
    to_format: str,
) -> str:
    """
    Convert between chemical notation formats.

    Supported formats: smiles, inchi, inchikey, mol, sdf

    Args:
        input_str: Chemical notation to convert
        from_format: Source format (smiles, inchi, mol, sdf)
        to_format: Target format (smiles, inchi, inchikey, mol, sdf)

    Returns:
        JSON string with converted notation

    Examples:
        - SMILES -> InChI: convert_chemical_notation("CCO", "smiles", "inchi")
        - InChI -> SMILES: convert_chemical_notation("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3", "inchi", "smiles")
    """
    try:
        result = convert_notation(input_str, from_format, to_format)
        return _format_response(result)
    except Exception as e:
        logger.error(f"Error converting {from_format} to {to_format}: {e}")
        return _format_error(e)


# ============================================================================
# DESCRIPTOR TOOLS
# ============================================================================


@tool
def compute_molecular_descriptors(
    smiles_list: str,
    fields: str | None = None,
) -> str:
    """
    Compute molecular descriptors (MW, logP, TPSA, HBA, HBD, rotatable bonds, rings).

    Args:
        smiles_list: Single SMILES string or comma-separated SMILES
        fields: Optional comma-separated descriptor names to compute
                (default: all - MW, logP, TPSA, HBA, HBD, RotBonds, Rings)

    Returns:
        JSON string with computed descriptors for each molecule:
        {
            "success": true,
            "data": {
                "molecules": [
                    {"smiles": "CCO", "MW": 46.04, "logP": -0.31, "TPSA": 20.23, ...}
                ]
            }
        }

    Examples:
        - Single molecule: compute_molecular_descriptors("CCO")
        - Multiple molecules: compute_molecular_descriptors("CCO,c1ccccc1,C1CCCCC1")
        - Specific fields: compute_molecular_descriptors("CCO", fields="MW,logP")
    """
    try:
        # Parse SMILES list (comma-separated)
        if isinstance(smiles_list, str):
            smiles_array = [s.strip() for s in smiles_list.split(",") if s.strip()]
        else:
            smiles_array = smiles_list

        # Parse fields
        fields_list = None
        if fields:
            fields_list = [f.strip() for f in fields.split(",") if f.strip()]

        result = compute_descriptors(smiles_array, fields_list)
        return _format_response(result)
    except Exception as e:
        logger.error(f"Error computing descriptors: {e}")
        return _format_error(e)


# ============================================================================
# SIMILARITY & FILTERING TOOLS
# ============================================================================


@tool
def search_similar_molecules_tool(
    query: str,
    targets: str,
    threshold: float = 0.5,
    top: int = 5,
) -> str:
    """
    Find molecules similar to a query molecule (Tanimoto similarity).

    Args:
        query: Query SMILES string
        targets: Comma-separated SMILES strings to search
        threshold: Similarity threshold (0.0-1.0, default 0.5)
        top: Maximum number of hits to return (default 5)

    Returns:
        JSON string with similar molecules ranked by similarity score

    Example:
        query="c1ccccc1", targets="Cc1ccccc1,CCO,c1ccc2ccccc2c1", threshold=0.5
    """
    try:
        targets_list = [s.strip() for s in targets.split(",") if s.strip()]
        result = search_similar_molecules(query, targets_list, threshold, top)
        return _format_response(result)
    except Exception as e:
        logger.error(f"Error searching similar molecules: {e}")
        return _format_error(e)


@tool
def filter_by_descriptor_constraints(
    smiles_list: str,
    mw_min: float | None = None,
    mw_max: float | None = None,
    logp_min: float | None = None,
    logp_max: float | None = None,
    tpsa_min: float | None = None,
    tpsa_max: float | None = None,
    hbd_max: int | None = None,
    hba_max: int | None = None,
    apply_lipinski: bool = False,
) -> str:
    """
    Filter molecules by molecular descriptor ranges.

    Useful for drug discovery (e.g., Lipinski's Rule of Five).

    Args:
        smiles_list: Comma-separated SMILES strings to filter
        mw_min/mw_max: Molecular weight range (Da)
        logp_min/logp_max: LogP (lipophilicity) range
        tpsa_min/tpsa_max: Topological polar surface area range (Ų)
        hbd_max: Maximum hydrogen bond donors
        hba_max: Maximum hydrogen bond acceptors
        apply_lipinski: If True, apply Lipinski's Rule of Five (MW<500, LogP<5, HBA<10, HBD<5)

    Returns:
        JSON string with filtered molecule list

    Examples:
        - Lipinski filter: filter_by_descriptor_constraints(smiles_list="...", apply_lipinski=True)
        - Custom range: filter_by_descriptor_constraints(smiles_list="...", mw_max=400, logp_max=3)
    """
    try:
        smiles_array = [s.strip() for s in smiles_list.split(",") if s.strip()]
        result = filter_molecules(
            smiles_array,
            mw_min=mw_min,
            mw_max=mw_max,
            logp_min=logp_min,
            logp_max=logp_max,
            tpsa_min=tpsa_min,
            tpsa_max=tpsa_max,
            hbd_max=hbd_max,
            hba_max=hba_max,
            lipinski=apply_lipinski,
        )
        return _format_response(result)
    except Exception as e:
        logger.error(f"Error filtering molecules: {e}")
        return _format_error(e)


# ============================================================================
# FUNCTIONAL GROUP & SUBSTRUCTURE TOOLS
# ============================================================================


@tool
def detect_functional_groups_tool(smiles: str) -> str:
    """
    Detect functional groups in a molecule using a tiered SMARTS catalog.

    Uses a curated set of SMARTS patterns with overlap management.

    Args:
        smiles: SMILES string to analyze

    Returns:
        JSON string listing detected functional groups

    Example:
        detect_functional_groups_tool("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
        # Should find: ester, carboxylic acid, aromatic ring
    """
    try:
        result = detect_functional_groups(smiles)
        return _format_response(result)
    except Exception as e:
        logger.error(f"Error detecting functional groups: {e}")
        return _format_error(e)


@tool
def search_substructure(smiles: str, smarts_pattern: str) -> str:
    """
    Search for a SMARTS substructure pattern in a molecule.

    Args:
        smiles: SMILES string to search in
        smarts_pattern: SMARTS pattern to search for

    Returns:
        JSON string with match results and atom indices

    Example:
        search_substructure("c1ccccc1CC(=O)O", "[cR1][cR1]")  # aromatic ring
    """
    try:
        result = substructure_search(smiles, smarts_pattern)
        return _format_response(result)
    except Exception as e:
        logger.error(f"Error searching substructure: {e}")
        return _format_error(e)


# ============================================================================
# RING & STEREOCHEMISTRY TOOLS
# ============================================================================


@tool
def analyze_ring_systems(smiles: str) -> str:
    """
    Analyze ring systems in a molecule (count, aromaticity, spiro atoms).

    Args:
        smiles: SMILES string to analyze

    Returns:
        JSON string with ring information

    Example:
        analyze_ring_systems("c1ccc2c(c1)cccn2")  # indole
    """
    try:
        result = analyze_rings(smiles)
        return _format_response(result)
    except Exception as e:
        logger.error(f"Error analyzing rings: {e}")
        return _format_error(e)


@tool
def analyze_stereochemistry(smiles: str) -> str:
    """
    Analyze stereocenters in a molecule.

    Identifies tetrahedral and E/Z stereocenters with:
    - Specified vs unspecified status
    - CIP codes (when available)

    Args:
        smiles: SMILES string with or without stereochemistry

    Returns:
        JSON string with stereocenter details

    Example:
        analyze_stereochemistry("CC(O)C(N)C")  # has unspecified stereocenters
    """
    try:
        result = analyze_stereo(smiles)
        return _format_response(result)
    except Exception as e:
        logger.error(f"Error analyzing stereochemistry: {e}")
        return _format_error(e)


# ============================================================================
# ATOM MAPPING TOOLS
# ============================================================================


@tool
def list_atom_maps(smiles: str) -> str:
    """
    List atom mapping numbers in a SMILES string.

    Args:
        smiles: SMILES with atom map numbers (e.g., "[CH3:1][OH:2]")

    Returns:
        JSON with atom_index -> map_number mapping
    """
    try:
        result = atom_map_list(smiles)
        return _format_response(result)
    except Exception as e:
        logger.error(f"Error listing atom maps: {e}")
        return _format_error(e)


@tool
def add_atom_maps(smiles: str) -> str:
    """
    Add sequential atom map numbers to all heavy atoms in a SMILES.

    Args:
        smiles: SMILES string without maps

    Returns:
        SMILES with sequential atom numbers added

    Example:
        add_atom_maps("CCO") -> "[CH3:1][CH2:2][OH:3]"
    """
    try:
        result = atom_map_add(smiles)
        return _format_response(result)
    except Exception as e:
        logger.error(f"Error adding atom maps: {e}")
        return _format_error(e)


@tool
def remove_atom_maps(smiles: str) -> str:
    """
    Remove all atom map numbers from a SMILES string.

    Args:
        smiles: SMILES with atom maps

    Returns:
        Canonical SMILES with maps removed
    """
    try:
        result = atom_map_remove(smiles)
        return _format_response(result)
    except Exception as e:
        logger.error(f"Error removing atom maps: {e}")
        return _format_error(e)


@tool
def validate_atom_mapping(smirks: str) -> str:
    """
    Validate atom mapping balance in a SMIRKS string.

    Checks if reactant atoms are correctly mapped to product atoms.

    Args:
        smirks: SMIRKS reaction string with atom maps

    Returns:
        JSON with validation results and balance info

    Example:
        validate_atom_mapping("[C:1][OH:2]>>[C:1]Br")
    """
    try:
        result = atom_map_check(smirks)
        return _format_response(result)
    except Exception as e:
        logger.error(f"Error validating atom mapping: {e}")
        return _format_error(e)


# ============================================================================
# FINGERPRINTING TOOLS
# ============================================================================


@tool
def compute_molecular_fingerprint(
    smiles: str,
    fp_type: str = "Morgan",
    radius: int = 2,
    nbits: int = 2048,
) -> str:
    """
    Compute molecular fingerprint (Morgan or topological).

    Used for similarity searching and molecular clustering.

    Args:
        smiles: SMILES string
        fp_type: "Morgan" (default) or "topological"
        radius: Radius for Morgan fingerprints (default 2)
        nbits: Number of bits in fingerprint (default 2048)

    Returns:
        JSON with fingerprint and metadata
    """
    try:
        result = compute_fingerprint(smiles, fp_type, radius, nbits)
        return _format_response(result)
    except Exception as e:
        logger.error(f"Error computing fingerprint: {e}")
        return _format_error(e)


# ============================================================================
# REACTION TOOLS (with WASM handling)
# ============================================================================


@tool
def apply_chemical_reaction(smirks: str, reactants: str) -> str:
    """
    Apply a reaction SMIRKS to reactant SMILES.

    Returns product SMILES from reaction application.

    WARNING: May not be supported in WASM build. If unavailable,
    use Python RDKit AllChem.RunReactants() instead.

    Args:
        smirks: SMIRKS reaction definition (e.g., "[C:1][OH]>>[C:1]Br")
        reactants: Comma-separated SMILES reactants

    Returns:
        JSON with product SMILES for each reactant

    Raises:
        RdkitAgentWASMError: If WASM build doesn't support reactions
    """
    try:
        reactants_list = [s.strip() for s in reactants.split(",") if s.strip()]
        result = apply_reaction(smirks, reactants_list)
        return _format_response(result)
    except Exception as e:
        logger.error(f"Error applying reaction: {e}")
        return _format_error(e)


# ============================================================================
# MOLECULAR EDITING TOOLS
# ============================================================================


@tool
def edit_molecule_structure(smiles: str, operation: str) -> str:
    """
    Apply molecular transformations (neutralize, sanitize, add-h, remove-h, strip-maps).

    Args:
        smiles: SMILES string to modify
        operation: One of:
            - "neutralize": Remove formal charges
            - "sanitize": RDKit sanitization
            - "add-h": Add explicit hydrogens
            - "remove-h": Remove explicit hydrogens
            - "strip-maps": Remove atom mapping numbers

    Returns:
        JSON with modified SMILES

    Examples:
        - Neutralize charges: edit_molecule_structure("[NH4+].[OH-]", "neutralize")
        - Add hydrogens: edit_molecule_structure("CCO", "add-h")
    """
    try:
        result = edit_molecule(smiles, operation)
        return _format_response(result)
    except Exception as e:
        logger.error(f"Error editing molecule: {e}")
        return _format_error(e)


# ============================================================================
# VISUALIZATION TOOLS
# ============================================================================


@tool
def draw_molecule_svg(
    smiles: str,
    width: int = 300,
    height: int = 300,
    highlight_atoms: str | None = None,
    highlight_bonds: str | None = None,
) -> str:
    """
    Render a molecule as SVG image.

    Can highlight specific atoms or bonds with colors.

    Args:
        smiles: SMILES string to draw
        width: Image width in pixels (default 300)
        height: Image height in pixels (default 300)
        highlight_atoms: JSON string mapping atom indices to CSS colors
                        e.g., '{"0":"#ff0000","1":"#ff0000"}'
        highlight_bonds: JSON string mapping bond indices to CSS colors
                        e.g., '{"0":"#00ff00"}'

    Returns:
        JSON with base64-encoded SVG image

    Examples:
        - Simple: draw_molecule_svg("c1ccccc1")
        - With highlighting: draw_molecule_svg(
            "c1ccccc1",
            highlight_atoms='{"0":"#ff0000","1":"#00ff00"}'
          )
    """
    try:
        highlight_atoms_dict = None
        highlight_bonds_dict = None

        if highlight_atoms:
            highlight_atoms_dict = json.loads(highlight_atoms)
        if highlight_bonds:
            highlight_bonds_dict = json.loads(highlight_bonds)

        result = draw_molecule(
            smiles,
            output_format="svg",
            width=width,
            height=height,
            highlight_atoms=highlight_atoms_dict,
            highlight_bonds=highlight_bonds_dict,
        )
        return _format_response(result)
    except Exception as e:
        logger.error(f"Error drawing molecule: {e}")
        return _format_error(e)


# ============================================================================
# BATCH ANALYSIS TOOLS
# ============================================================================


@tool
def compute_dataset_statistics(smiles_list: str) -> str:
    """
    Compute statistics across a molecule dataset.

    Returns mean, median, std, min, max for all molecular descriptors.

    Args:
        smiles_list: Comma-separated SMILES strings

    Returns:
        JSON with statistical summary of the dataset
    """
    try:
        smiles_array = [s.strip() for s in smiles_list.split(",") if s.strip()]
        result = dataset_statistics(smiles_array)
        return _format_response(result)
    except Exception as e:
        logger.error(f"Error computing dataset statistics: {e}")
        return _format_error(e)


# ============================================================================
# TOOL COLLECTION FUNCTIONS
# ============================================================================


def get_rdkit_validation_tools() -> list:
    """Get all rdkit validation tools."""
    return [
        validate_smiles,
        validate_smirks,
        validate_reaction,
        repair_invalid_smiles,
    ]


def get_rdkit_analysis_tools() -> list:
    """Get all rdkit analysis and descriptor tools."""
    return [
        compute_molecular_descriptors,
        analyze_ring_systems,
        analyze_stereochemistry,
        detect_functional_groups_tool,
        compute_molecular_fingerprint,
    ]


def get_rdkit_query_tools() -> list:
    """Get all rdkit search and filter tools."""
    return [
        search_similar_molecules_tool,
        filter_by_descriptor_constraints,
        search_substructure,
    ]


def get_rdkit_conversion_tools() -> list:
    """Get all rdkit conversion and editing tools."""
    return [
        convert_chemical_notation,
        edit_molecule_structure,
        repair_invalid_smiles,
    ]


def get_rdkit_reaction_tools() -> list:
    """Get all rdkit reaction and mapping tools."""
    return [
        apply_chemical_reaction,
        validate_atom_mapping,
        list_atom_maps,
        add_atom_maps,
        remove_atom_maps,
    ]


def get_all_rdkit_tools() -> list:
    """Get all available rdkit tools."""
    return (
        get_rdkit_validation_tools()
        + get_rdkit_analysis_tools()
        + get_rdkit_query_tools()
        + get_rdkit_conversion_tools()
        + get_rdkit_reaction_tools()
        + [draw_molecule_svg, compute_dataset_statistics]
    )
