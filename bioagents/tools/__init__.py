"""Lazy public tool exports; importing standalone submodules needs no LangChain."""

from importlib import import_module

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


_ANALYSIS_EXPORTS = {
    "analyze_amino_acid_composition",
    "calculate_isoelectric_point",
    "calculate_molecular_weight",
}


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name = "analysis_tools" if name in _ANALYSIS_EXPORTS else "rdkit_tools"
    try:
        module = import_module(f"{__name__}.{module_name}")
    except ModuleNotFoundError as exc:
        if exc.name == "langchain_core" or (exc.name or "").startswith("langchain_core."):
            raise ModuleNotFoundError(
                f"Tool {name!r} requires langchain-core; install the project dependencies",
                name=exc.name,
            ) from exc
        raise
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
