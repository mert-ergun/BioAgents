"""BioAgents tools module.

Provides access to various bioinformatics and cheminformatics tools.
"""

# Import rdkit-agent tools
# Import existing tools
from bioagents.tools.analysis_tools import (
    analyze_amino_acid_composition,
    calculate_isoelectric_point,
    calculate_molecular_weight,
)
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

__all__ = [
    "add_atom_maps",
    "analyze_amino_acid_composition",
    "analyze_ring_systems",
    "analyze_stereochemistry",
    "apply_chemical_reaction",
    "calculate_isoelectric_point",
    "calculate_molecular_weight",
    "compute_dataset_statistics",
    "compute_molecular_descriptors",
    "compute_molecular_fingerprint",
    "convert_chemical_notation",
    "detect_functional_groups_tool",
    "draw_molecule_svg",
    "edit_molecule_structure",
    "filter_by_descriptor_constraints",
    "get_all_rdkit_tools",
    "get_rdkit_analysis_tools",
    "get_rdkit_conversion_tools",
    "get_rdkit_query_tools",
    "get_rdkit_reaction_tools",
    "get_rdkit_validation_tools",
    "list_atom_maps",
    "remove_atom_maps",
    "repair_invalid_smiles",
    "search_similar_molecules_tool",
    "search_substructure",
    "validate_reaction",
    "validate_smiles",
    "validate_smirks",
]
