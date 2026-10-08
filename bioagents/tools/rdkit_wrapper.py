"""Chemistry backend for the RDKit tools.

Historically this shelled out to a Node.js ``rdkit-agent`` CLI. That CLI was not
installed in this deployment, so all 22 agent-facing chemistry tools were unusable —
and several of them reported the missing CLI as a *chemical* verdict, e.g.
``validate_smiles`` returning ``{"success": true, "valid": false}`` for a perfectly
good molecule. Its WASM build also could not run reactions at all.

The work now runs through the RDKit Python library (see ``rdkit_native``), which was
already installed. This module stays as the import surface so ``rdkit_tools.py`` and
its exception handling are unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

from bioagents.tools.rdkit_native import (
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


class RdkitAgentError(Exception):
    """Base exception for chemistry backend failures (infrastructure, not chemistry)."""


class RdkitAgentWASMError(RdkitAgentError):
    """Kept for compatibility.

    The WASM build imposed limits — notably no ``RunReactants`` — that the native RDKit
    backend does not have. Nothing raises this any more; it remains so that existing
    ``except`` clauses keep working.
    """

    def __init__(self, feature: str, message: str = ""):
        self.feature = feature
        self.message = message
        super().__init__(f"Unsupported feature: {feature}. {message}")


class RdkitAgentValidationError(RdkitAgentError):
    """Exception for validation failures."""


def get_version() -> dict[str, Any]:
    """Report the chemistry backend in use and its version."""
    try:
        import rdkit

        return {
            "backend": "rdkit-python",
            "version": rdkit.__version__,
            "available": True,
            "note": "Native RDKit library; the Node rdkit-agent CLI is no longer used.",
        }
    except ImportError as exc:
        return {
            "backend": "rdkit-python",
            "available": False,
            "error": f"RDKit is not installed: {exc}. Install with `uv pip install rdkit`.",
        }


def get_command_schema(command: str) -> dict[str, Any]:
    """Describe the arguments a backend command accepts."""
    schemas: dict[str, dict[str, Any]] = {
        "check": {"args": ["smiles | smirks | reactants+products"]},
        "repair-smiles": {"args": ["input"]},
        "convert": {"args": ["input", "from", "to"]},
        "descriptors": {"args": ["smiles_list", "fields?"]},
        "similarity": {"args": ["query", "targets", "threshold?", "top?"]},
        "filter": {"args": ["smiles_list", "descriptor bounds?", "lipinski?"]},
        "fg": {"args": ["smiles"]},
        "substructure": {"args": ["smiles", "smarts_pattern"]},
        "rings": {"args": ["smiles"]},
        "stereo": {"args": ["smiles"]},
        "atom-map": {"args": ["smiles | smirks"]},
        "fingerprint": {"args": ["smiles", "fp_type?", "radius?", "nbits?"]},
        "react": {"args": ["smirks", "reactants"]},
        "edit": {"args": ["smiles", "operation"]},
        "draw": {"args": ["smiles", "output_file?", "output_format?", "width?", "height?"]},
        "stats": {"args": ["smiles_list"]},
    }
    if command not in schemas:
        return {
            "command": command,
            "error": f"Unknown command. Known commands: {', '.join(sorted(schemas))}.",
        }
    return {"command": command, **schemas[command]}


__all__ = [
    "RdkitAgentError",
    "RdkitAgentValidationError",
    "RdkitAgentWASMError",
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
    "get_command_schema",
    "get_version",
    "repair_smiles",
    "search_similar_molecules",
    "substructure_search",
]
