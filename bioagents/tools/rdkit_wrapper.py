"""Python wrapper for rdkit-agent CLI using subprocess.

This module provides a clean interface to rdkit-agent commands via subprocess,
handling JSON serialization, error parsing, and WASM limitation graceful degradation.

Installation: npm install -g rdkit-agent (requires Node.js >= 16)
Verify: rdkit-agent version
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess  # nosec B404
import sys
from typing import Any, cast

logger = logging.getLogger(__name__)


class RdkitAgentError(Exception):
    """Base exception for rdkit-agent errors."""

    pass


class RdkitAgentWASMError(RdkitAgentError):
    """Exception for WASM-unsupported features."""

    def __init__(self, feature: str, message: str = ""):
        self.feature = feature
        self.message = message
        super().__init__(f"WASM Limitation: {feature} not supported in browser build. {message}")


class RdkitAgentValidationError(RdkitAgentError):
    """Exception for validation failures."""

    pass


def _get_rdkit_agent_path() -> str:
    """Find rdkit-agent executable, checking npm global bin."""
    # Try to find in PATH
    rdkit_path = shutil.which("rdkit-agent")
    if rdkit_path:
        return rdkit_path

    # Try npm global bin directory
    try:
        npm_exe = shutil.which("npm")
        if not npm_exe:
            raise FileNotFoundError("npm not on PATH")
        npm_prefix = subprocess.check_output(  # nosec B603
            [npm_exe, "config", "get", "prefix"],
            text=True,
            timeout=5,
        ).strip()

        # Construct path to rdkit-agent
        bin_dir = (
            f"{npm_prefix}\\node_modules\\.bin" if sys.platform == "win32" else f"{npm_prefix}/bin"
        )
        rdkit_path = shutil.which("rdkit-agent", path=bin_dir)
        if rdkit_path:
            return rdkit_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # If nothing found, return the command and let subprocess.run handle the error
    return "rdkit-agent"


def _run_rdkit_command(
    command: str,
    args: list[str] | None = None,
    json_input: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """
    Execute an rdkit-agent CLI command and return JSON output.

    Args:
        command: rdkit-agent command (e.g., "check", "descriptors", "convert")
        args: List of CLI arguments/flags (e.g., ["--smiles", "CCO"])
        json_input: Dict to pass via --json stdin
        timeout: Command timeout in seconds

    Returns:
        Parsed JSON response from rdkit-agent

    Raises:
        RdkitAgentError: If rdkit-agent not installed or command fails
        RdkitAgentWASMError: If WASM-unsupported feature requested
        RdkitAgentValidationError: If validation fails (exit code 1)
    """
    rdkit_agent = _get_rdkit_agent_path()
    cmd = [rdkit_agent, command]

    if args:
        cmd.extend(args)

    # Force JSON output
    cmd.append("--output")
    cmd.append("json")

    try:
        if json_input:
            # Pass JSON via stdin
            result = subprocess.run(  # nosec B603
                [*cmd, "--json", "-"],
                input=json.dumps(json_input),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        else:
            result = subprocess.run(  # nosec B603
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )

        # Handle exit codes
        if result.returncode == 0:
            # Success
            try:
                return cast("dict[str, Any]", json.loads(result.stdout))
            except json.JSONDecodeError as e:
                raise RdkitAgentError(
                    f"Invalid JSON from rdkit-agent: {result.stdout}. Error: {e}"
                ) from e

        elif result.returncode == 1:
            # Validation failure (expected for invalid SMILES, etc.)
            try:
                return cast("dict[str, Any]", json.loads(result.stdout))
            except json.JSONDecodeError as e:
                raise RdkitAgentValidationError(
                    f"Validation failed: {result.stderr or result.stdout}"
                ) from e

        elif result.returncode == 2:
            # Usage error
            raise RdkitAgentError(f"Usage error: {result.stderr}")

        elif result.returncode == 3:
            # RDKit error (including WASM limitations)
            stderr = result.stderr or result.stdout

            # Check for WASM-specific errors
            if "NOT_SUPPORTED_IN_WASM" in stderr:
                if "react" in stderr or "RunReactants" in stderr:
                    raise RdkitAgentWASMError(
                        "Reaction application (react)",
                        "Use Python RDKit installation instead: AllChem.RunReactants",
                    )
                elif "stereo" in stderr and "enumerate" in stderr:
                    raise RdkitAgentWASMError(
                        "Stereo enumeration (stereo --enumerate)",
                        "Use Python RDKit: EnumerateStereoisomers",
                    )
                raise RdkitAgentWASMError("Unknown", stderr)

            raise RdkitAgentError(f"RDKit error: {stderr}")

        else:
            raise RdkitAgentError(f"Unknown error (exit code {result.returncode}): {result.stderr}")

    except subprocess.TimeoutExpired as e:
        raise RdkitAgentError(f"rdkit-agent command timed out after {timeout}s") from e
    except FileNotFoundError as e:
        raise RdkitAgentError(
            "rdkit-agent CLI not found. Install with: npm install -g rdkit-agent"
        ) from e


# ============================================================================
# VALIDATION COMMANDS
# ============================================================================


def _detect_and_close_unclosed_rings(smiles: str) -> tuple[str, list[str]]:
    """
    Detect unclosed ring markers (1-9) and close them.
    
    Returns: (fixed_smiles, repairs_applied)
    """
    import re
    repairs = []
    fixed = smiles
    
    # Find all ring closure markers
    markers = {}
    for match in re.finditer(r'([1-9])', fixed):
        marker = match.group(1)
        if marker not in markers:
            markers[marker] = []
        markers[marker].append(match.start())
    
    # Find unclosed markers (appear odd number of times)
    for marker, positions in markers.items():
        if len(positions) % 2 == 1:  # Unclosed
            fixed += marker
            repairs.append(f"close_ring_{marker}")
    
    return fixed, repairs


def check_smiles(smiles: str) -> dict[str, Any]:
    """
    Validate a SMILES string for chemical correctness.

    Returns:
        {
            "overall_pass": bool,
            "valid": bool,
            "canonical_smiles": str,
            "summary": str
        }
    """
    # First, detect and close unclosed rings
    fixed_smiles, ring_repairs = _detect_and_close_unclosed_rings(smiles)
    smiles_to_validate = fixed_smiles
    
    # Use repair-smiles for validation since it validates SMILES syntax
    try:
        result = repair_smiles(smiles_to_validate)
        if result.get("success", False):
            best_candidate = result.get("best_candidate", {})
            is_valid = best_candidate.get("valid", False)
            canonical = best_candidate.get("canonical_smiles", smiles_to_validate)
            
            return {
                "overall_pass": is_valid,
                "valid": is_valid,
                "canonical_smiles": canonical,
                "summary": "Valid SMILES" if is_valid else "Invalid SMILES",
                "confidence": result.get("confidence", 0),
                "functional_groups": best_candidate.get("functional_groups", []),
                "repairs_applied": ring_repairs,
            }
        else:
            return {
                "overall_pass": False,
                "valid": False,
                "canonical_smiles": smiles,
                "summary": result.get("message", "Failed to validate SMILES"),
                "repairs_applied": ring_repairs,
            }
    except Exception as e:
        return {
            "overall_pass": False,
            "valid": False,
            "canonical_smiles": smiles,
            "summary": str(e),
            "repairs_applied": ring_repairs,
        }


def check_smirks(smirks: str) -> dict[str, Any]:
    """Validate a SMIRKS string."""
    return _run_rdkit_command("check", ["--smirks", smirks])


def check_reaction(reactants: list[str], products: list[str]) -> dict[str, Any]:
    """Validate a chemical reaction (atom balance check)."""
    args = ["--reactants", ",".join(reactants), "--products", ",".join(products)]
    return _run_rdkit_command("check", args)


# ============================================================================
# REPAIR COMMANDS
# ============================================================================


def repair_smiles(input_str: str) -> dict[str, Any]:
    """
    Repair/reconstruct malformed SMILES.
    
    Detects and fixes unclosed ring markers (1-9) before attempting repair.

    Returns:
        {
            "success": bool,
            "canonical_smiles": str,
            "strategy": str,
            "confidence": float,
            "intent": str,
            "attempts": int
        }
    """
    # First, detect and close unclosed rings
    fixed_smiles, ring_repairs = _detect_and_close_unclosed_rings(input_str)
    
    # If rings were closed, use the fixed version
    if ring_repairs:
        input_str = fixed_smiles
    
    result = _run_rdkit_command("repair-smiles", ["--input", input_str])
    
    # Add repair trace
    if ring_repairs:
        if "best_candidate" in result:
            result["best_candidate"]["ring_closures_applied"] = ring_repairs
        result["ring_closures_applied"] = ring_repairs
    
    return result


# ============================================================================
# CONVERSION COMMANDS
# ============================================================================


def convert_notation(
    input_str: str,
    from_format: str,
    to_format: str,
) -> dict[str, Any]:
    """
    Convert between chemical notations (smiles, inchi, inchikey, mol, sdf).

    Args:
        input_str: The chemical notation to convert
        from_format: Source format (smiles, inchi, mol, sdf)
        to_format: Target format (smiles, inchi, inchikey, mol, sdf)
    """
    args = ["--from", from_format, "--to", to_format, "--input", input_str]
    return _run_rdkit_command("convert", args)


# ============================================================================
# DESCRIPTOR COMMANDS
# ============================================================================


def compute_descriptors(
    smiles_list: list[str] | str,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute molecular descriptors (MW, logP, TPSA, HBA, HBD, rotatable bonds, rings).

    Args:
        smiles_list: Single SMILES or list of SMILES strings (comma-separated string or list)
        fields: Optional list of descriptors to compute (e.g., ["MW", "logP", "TPSA"])

    Returns:
        {
            "molecules": [
                {"smiles": str, "MW": float, "logP": float, "TPSA": float, ...}
            ]
        }
    """
    # Parse input
    if isinstance(smiles_list, str):
        if "," in smiles_list:
            smiles_array = [s.strip() for s in smiles_list.split(",") if s.strip()]
        else:
            smiles_array = [smiles_list]
    else:
        smiles_array = list(smiles_list)

    molecules = []
    for smiles in smiles_array:
        try:
            args = ["--smiles", smiles]
            if fields:
                args.extend(["--fields", ",".join(fields)])

            result = _run_rdkit_command("descriptors", args)
            if result:
                # Wrap single result in expected format
                molecules.append(result)
        except Exception as e:
            logger.warning(f"Failed to compute descriptors for {smiles}: {e}")
            molecules.append({"smiles": smiles, "error": str(e)})

    return {"molecules": molecules}


# ============================================================================
# SIMILARITY & FILTERING COMMANDS
# ============================================================================


def search_similar_molecules(
    query: str,
    targets: list[str],
    threshold: float = 0.5,
    top: int = 5,
) -> dict[str, Any]:
    """
    Find molecules similar to query (Tanimoto similarity).

    Returns:
        {
            "query": str,
            "targets_count": int,
            "threshold": float,
            "hits": [
                {"smiles": str, "similarity": float}
            ]
        }
    """
    args = [
        "--query",
        query,
        "--targets",
        ",".join(targets),
        "--threshold",
        str(threshold),
        "--top",
        str(top),
    ]
    return _run_rdkit_command("similarity", args)


def filter_molecules(
    smiles_list: list[str],
    mw_min: float | None = None,
    mw_max: float | None = None,
    logp_min: float | None = None,
    logp_max: float | None = None,
    tpsa_min: float | None = None,
    tpsa_max: float | None = None,
    hbd_max: int | None = None,
    hba_max: int | None = None,
    lipinski: bool = False,
) -> dict[str, Any]:
    """
    Filter molecules by descriptor ranges.

    Args:
        smiles_list: List of SMILES to filter
        Various descriptor constraints (see rdkit-agent filter command)
        lipinski: If True, apply Lipinski's Rule of Five

    Returns:
        {
            "input_count": int,
            "filtered_count": int,
            "filtered_smiles": [str]
        }
    """
    json_input = {"molecules": smiles_list}
    args = []

    if lipinski:
        args.append("--lipinski")
    else:
        if mw_min is not None:
            args.extend(["--mw-min", str(mw_min)])
        if mw_max is not None:
            args.extend(["--mw-max", str(mw_max)])
        if logp_min is not None:
            args.extend(["--logp-min", str(logp_min)])
        if logp_max is not None:
            args.extend(["--logp-max", str(logp_max)])
        if tpsa_min is not None:
            args.extend(["--tpsa-min", str(tpsa_min)])
        if tpsa_max is not None:
            args.extend(["--tpsa-max", str(tpsa_max)])
        if hbd_max is not None:
            args.extend(["--hbd-max", str(hbd_max)])
        if hba_max is not None:
            args.extend(["--hba-max", str(hba_max)])

    return _run_rdkit_command("filter", args, json_input=json_input)


# ============================================================================
# FUNCTIONAL GROUP & SUBSTRUCTURE COMMANDS
# ============================================================================


def detect_functional_groups(smiles: str) -> dict[str, Any]:
    """
    Detect functional groups in a molecule (tiered SMARTS catalog).

    Returns:
        {
            "smiles": str,
            "functional_groups": [str]
        }
    """
    return _run_rdkit_command("fg", ["--smiles", smiles])


def substructure_search(smiles: str, smarts_pattern: str) -> dict[str, Any]:
    """
    Search for SMARTS substructure in molecule.

    Returns:
        {
            "smiles": str,
            "smarts": str,
            "matched": bool,
            "match_count": int,
            "match_indices": [...] (if matched)
        }
    """
    args = ["--smiles", smiles, "--smarts", smarts_pattern]
    return _run_rdkit_command("subsearch", args)


# ============================================================================
# RING & STEREOCHEMISTRY COMMANDS
# ============================================================================


def analyze_rings(smiles: str) -> dict[str, Any]:
    """
    Analyze ring systems in a molecule.

    Returns:
        {
            "smiles": str,
            "ring_count": int,
            "aromatic_rings": int,
            "aliphatic_rings": int,
            "spiro_atoms": int,
            "rings": [...]
        }
    """
    return _run_rdkit_command("rings", ["--smiles", smiles])


def analyze_stereo(smiles: str) -> dict[str, Any]:
    """
    Analyze stereocenters (tetrahedral + E/Z with CIP codes).

    Returns:
        {
            "smiles": str,
            "stereo_centers": [...],
            "stereo_center_count": int,
            "specified_count": int,
            "has_unspecified_stereo": bool
        }
    """
    return _run_rdkit_command("stereo", ["--smiles", smiles])


# ============================================================================
# ATOM MAPPING COMMANDS
# ============================================================================


def atom_map_list(smiles: str) -> dict[str, Any]:
    """List atom mapping numbers in SMILES."""
    return _run_rdkit_command("atom-map", ["list", "--smiles", smiles])


def atom_map_add(smiles: str) -> dict[str, Any]:
    """Add sequential atom map numbers to all heavy atoms."""
    return _run_rdkit_command("atom-map", ["add", "--smiles", smiles])


def atom_map_remove(smiles: str) -> dict[str, Any]:
    """Remove all atom map numbers from SMILES."""
    return _run_rdkit_command("atom-map", ["remove", "--smiles", smiles])


def atom_map_check(smirks: str) -> dict[str, Any]:
    """Validate atom mapping balance in SMIRKS."""
    return _run_rdkit_command("atom-map", ["check", "--smirks", smirks])


# ============================================================================
# FINGERPRINTING COMMANDS
# ============================================================================


def compute_fingerprint(
    smiles: str,
    fp_type: str = "Morgan",
    radius: int = 2,
    nbits: int = 2048,
) -> dict[str, Any]:
    """
    Compute molecular fingerprint (Morgan or topological).

    Args:
        smiles: SMILES string
        fp_type: "Morgan" or "topological"
        radius: Radius for Morgan fingerprints (default 2)
        nbits: Number of bits (default 2048)

    Returns:
        {
            "smiles": str,
            "fingerprint": str (hex or binary),
            "type": str,
            ...
        }
    """
    args = ["--smiles", smiles, "--type", fp_type]
    if fp_type == "Morgan":
        args.extend(["--radius", str(radius)])
    args.extend(["--nbits", str(nbits)])

    return _run_rdkit_command("fingerprint", args)


# ============================================================================
# REACTION COMMANDS (with WASM limitation handling)
# ============================================================================


def apply_reaction(smirks: str, reactants: list[str]) -> dict[str, Any]:
    """
    Apply a reaction SMIRKS to reactant SMILES.

    Raises:
        RdkitAgentWASMError: If WASM build doesn't support reactions

    Returns:
        {
            "reaction": str,
            "reactant_count": int,
            "products": [[str]] (list of product list per reactant)
        }
    """
    json_input = {"smirks": smirks, "reactants": reactants}
    return _run_rdkit_command("react", [], json_input=json_input)


# ============================================================================
# MOLECULAR EDITING COMMANDS
# ============================================================================


def edit_molecule(smiles: str, operation: str) -> dict[str, Any]:
    """
    Modify a molecule (neutralize, sanitize, add-h, remove-h, strip-maps).

    Args:
        smiles: SMILES string
        operation: One of "neutralize", "sanitize", "add-h", "remove-h", "strip-maps"

    Returns:
        {
            "input_smiles": str,
            "output_smiles": str,
            "operation": str
        }
    """
    args = ["--smiles", smiles, "--operation", operation]
    return _run_rdkit_command("edit", args)


# ============================================================================
# VISUALIZATION COMMANDS
# ============================================================================


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
    """
    Draw molecule to SVG or PNG.

    Args:
        smiles: SMILES string
        output_file: Path to save output (optional, returns base64 if not specified)
        output_format: "svg" or "png"
        width: Image width in pixels
        height: Image height in pixels
        highlight_atoms: Dict mapping atom index -> CSS hex colour
        highlight_bonds: Dict mapping bond index -> CSS hex colour
        highlight_radius: Highlight circle radius (default 0.3)

    Returns:
        {
            "smiles": str,
            "format": str,
            "output": str (base64 or file path)
        }
    """
    args = [
        "--smiles",
        smiles,
        "--format",
        output_format,
        "--width",
        str(width),
        "--height",
        str(height),
        "--highlight-radius",
        str(highlight_radius),
    ]

    if output_file:
        args.extend(["--output", output_file])

    if highlight_atoms:
        args.extend(["--highlight-atoms", json.dumps(highlight_atoms)])

    if highlight_bonds:
        args.extend(["--highlight-bonds", json.dumps(highlight_bonds)])

    return _run_rdkit_command("draw", args)


# ============================================================================
# BATCH ANALYSIS COMMANDS
# ============================================================================


def dataset_statistics(smiles_list: list[str]) -> dict[str, Any]:
    """
    Compute statistics across a molecule dataset.

    Returns:
        {
            "molecule_count": int,
            "descriptors_stats": {...}  (Mean, median, std, min, max for each descriptor)
        }
    """
    json_input = {"molecules": smiles_list}
    return _run_rdkit_command("stats", [], json_input=json_input)


# ============================================================================
# VERSION & SCHEMA COMMANDS
# ============================================================================


def get_version() -> dict[str, Any]:
    """Get rdkit-agent version."""
    return _run_rdkit_command("version", [])


def get_command_schema(command: str) -> dict[str, Any]:
    """Get JSON schema for a specific command."""
    return _run_rdkit_command("schema", ["--command", command])
