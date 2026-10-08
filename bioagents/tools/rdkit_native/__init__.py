"""Native RDKit backend for the chemistry tools.

These functions replace a Node.js `rdkit-agent` CLI that was not installed in this
deployment, which left 22 agent-facing tools unusable — and, worse, made some of them
report a missing CLI as a chemically invalid molecule. The RDKit Python library was
already a transitive dependency and does everything the CLI did, plus reactions, which
the CLI's WASM build could not run at all.

Each function returns the same dict shape the previous wrapper documented, so the
`@tool` layer in ``rdkit_tools.py`` is unchanged.
"""

from bioagents.tools.rdkit_native._common import RdkitNotAvailableError
from bioagents.tools.rdkit_native.descriptors import (
    DESCRIPTOR_FIELDS,
    LIPINSKI_LIMITS,
    compute_descriptors,
    compute_fingerprint,
    dataset_statistics,
    filter_molecules,
    search_similar_molecules,
)
from bioagents.tools.rdkit_native.molecules import (
    check_reaction,
    check_smiles,
    check_smirks,
    convert_notation,
    edit_molecule,
    repair_smiles,
    substructure_search,
)
from bioagents.tools.rdkit_native.structure import (
    FUNCTIONAL_GROUP_SMARTS,
    analyze_rings,
    analyze_stereo,
    apply_reaction,
    atom_map_add,
    atom_map_check,
    atom_map_list,
    atom_map_remove,
    detect_functional_groups,
    draw_molecule,
)

__all__ = [
    "DESCRIPTOR_FIELDS",
    "FUNCTIONAL_GROUP_SMARTS",
    "LIPINSKI_LIMITS",
    "RdkitNotAvailableError",
    "analyze_rings",
    "analyze_stereo",
    "apply_reaction",
    "atom_map_add",
    "atom_map_check",
    "atom_map_list",
    "atom_map_remove",
    "check_reaction",
    "check_smiles",
    "check_smirks",
    "compute_descriptors",
    "compute_fingerprint",
    "convert_notation",
    "dataset_statistics",
    "detect_functional_groups",
    "draw_molecule",
    "edit_molecule",
    "filter_molecules",
    "repair_smiles",
    "search_similar_molecules",
    "substructure_search",
]
