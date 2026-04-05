"""BioAgents tools module.

Provides access to various bioinformatics and cheminformatics tools.
"""

# Import rdkit-agent tools
from bioagents.tools.rdkit_tools import (
    add_atom_maps,
    analyze_ring_systems,
    analyze_stereochemistry,
    apply_chemical_reaction,
    compute_dataset_statistics,
    compute_molecular_descriptors,
    compute_molecular_fingerprint,
    convert_chemical_notation,
    detect_functional_groups_tool,
    draw_molecule_svg,
    edit_molecule_structure,
    filter_by_descriptor_constraints,
    get_all_rdkit_tools,
    get_rdkit_analysis_tools,
    get_rdkit_conversion_tools,
    get_rdkit_query_tools,
    get_rdkit_reaction_tools,
    get_rdkit_validation_tools,
    list_atom_maps,
    remove_atom_maps,
    repair_invalid_smiles,
    search_similar_molecules_tool,
    search_substructure,
    validate_reaction,
    validate_smiles,
    validate_smirks,
)

# Import existing tools
from bioagents.tools.analysis_tools import (
    analyze_amino_acid_composition,
    calculate_isoelectric_point,
    calculate_molecular_weight,
)

__all__ = [
    # RDKit cheminformatics tools
    "validate_smiles",
    "validate_smirks",
    "validate_reaction",
    "repair_invalid_smiles",
    "compute_molecular_descriptors",
    "search_similar_molecules_tool",
    "filter_by_descriptor_constraints",
    "detect_functional_groups_tool",
    "search_substructure",
    "analyze_ring_systems",
    "analyze_stereochemistry",
    "list_atom_maps",
    "add_atom_maps",
    "remove_atom_maps",
    "convert_chemical_notation",
    "edit_molecule_structure",
    "compute_molecular_fingerprint",
    "apply_chemical_reaction",
    "draw_molecule_svg",
    "compute_dataset_statistics",
    # Tool collection functions
    "get_all_rdkit_tools",
    "get_rdkit_validation_tools",
    "get_rdkit_analysis_tools",
    "get_rdkit_query_tools",
    "get_rdkit_conversion_tools",
    "get_rdkit_reaction_tools",
    # Existing BioAgents tools
    "calculate_molecular_weight",
    "analyze_amino_acid_composition",
    "calculate_isoelectric_point",
]
