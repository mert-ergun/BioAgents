"""Shared parsing helpers for the native RDKit backend.

These functions draw a hard line between two kinds of failure, because conflating
them is what made the previous CLI-backed implementation dangerous:

* A **chemical** result ("this SMILES is invalid") is data. It is returned as a dict.
* An **infrastructure** failure ("RDKit is not installed") is an error. It is raised,
  so the caller reports a failure instead of recording the molecule as invalid.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rdkit.Chem import Mol

logger = logging.getLogger(__name__)


class RdkitNotAvailableError(RuntimeError):
    """Raised when RDKit itself cannot be imported — an infrastructure failure."""


def require_rdkit():
    """Import and return the RDKit Chem module, or raise a clear infrastructure error."""
    try:
        from rdkit import Chem, RDLogger
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RdkitNotAvailableError(
            "RDKit is not installed in this environment. Install it with "
            "`uv pip install rdkit`. No chemistry was computed."
        ) from exc

    # RDKit logs parse failures to stderr by default, which floods agent logs with
    # noise for input we already handle and report structurally.
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    return Chem


def parse_smiles(smiles: str, *, sanitize: bool = True) -> Mol | None:
    """Parse a SMILES string, returning None when it is chemically invalid."""
    Chem = require_rdkit()
    if not smiles or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles.strip(), sanitize=sanitize)


def parse_smarts(smarts: str):
    """Parse a SMARTS pattern, returning None when it is malformed."""
    Chem = require_rdkit()
    if not smarts or not smarts.strip():
        return None
    return Chem.MolFromSmarts(smarts.strip())


def canonical_smiles(mol: Mol) -> str:
    """Return the canonical SMILES for a parsed molecule."""
    Chem = require_rdkit()
    return str(Chem.MolToSmiles(mol))


def invalid_smiles_result(smiles: str, reason: str = "") -> dict[str, Any]:
    """Build the standard payload for a SMILES RDKit could not parse.

    This is a *chemical* verdict, not a tool failure, so it carries ``valid: False``
    with an explanation rather than raising.
    """
    message = reason or "RDKit could not parse this SMILES string."
    return {
        "smiles": smiles,
        "valid": False,
        "overall_pass": False,
        "summary": message,
        "error": message,
    }


def split_smiles_list(smiles_list: list[str] | str) -> list[str]:
    """Normalise a SMILES collection supplied as a list or a comma-separated string."""
    if isinstance(smiles_list, str):
        return [s.strip() for s in smiles_list.split(",") if s.strip()]
    return [str(s).strip() for s in smiles_list if str(s).strip()]
