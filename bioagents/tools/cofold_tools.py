"""Protein-ligand co-folding with a local Boltz-2 backend.

Before this module the only co-folding paths were hosted ones (ToolUniverse's
``NvidiaNIM_boltz2`` / ``NvidiaNIM_openfold3``, which need ``NVIDIA_API_KEY``) and a
``run_boltz`` stub that always said "not installed". With no key the agents silently
fell back to docking, even though Boltz runs fine locally — in task1 the coder had to
``pip install boltz`` inside its sandbox by hand (which also downgraded numpy).

Backends, in order of preference:

1. ``BIOAGENTS_BOLTZ_CMD`` — an explicit command (e.g. a dedicated venv's ``boltz``).
2. The prebuilt sandbox's dedicated Boltz venv.
3. ``boltz`` on ``PATH``.
4. The ``boltz`` Python module importable in this interpreter.
5. ``uvx`` — runs Boltz in an isolated, cached environment so its pins (numpy<2,
   its own torch) never touch the BioAgents environment. First use downloads Boltz.
   Disable with ``BIOAGENTS_COFOLD_UVX=0``.

``cofold_backend_status()`` reports exactly which backend would run (or why none can),
so the tool never claims to need an API key when a local backend is available.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shlex
import shutil

# Runs the Boltz CLI with an argument list, without a shell.
import subprocess  # nosec B404
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bioagents.sandbox.python_environments import environment_python

logger = logging.getLogger(__name__)

DEFAULT_UVX_SPEC = "boltz==2.2.1"
DEFAULT_TIMEOUT_S = 3600


@dataclass(frozen=True)
class CofoldBackend:
    """Which co-folding backend is usable here, and how to invoke it."""

    name: str | None
    command: tuple[str, ...] = ()
    detail: str = ""
    hosted_alternatives: tuple[str, ...] = field(default_factory=tuple)

    @property
    def available(self) -> bool:
        return self.name is not None


def _truthy(value: str | None, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _module_importable(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def detect_cofold_backend() -> CofoldBackend:
    """Pick the first usable local Boltz backend; never requires an API key."""
    hosted: tuple[str, ...] = ()
    if os.getenv("NVIDIA_API_KEY"):
        hosted = ("ToolUniverse NvidiaNIM_boltz2 / NvidiaNIM_openfold3 (NVIDIA_API_KEY set)",)

    explicit = os.getenv("BIOAGENTS_BOLTZ_CMD", "").strip()
    if explicit:
        return CofoldBackend(
            "boltz-explicit",
            tuple(shlex.split(explicit)),
            f"BIOAGENTS_BOLTZ_CMD={explicit}",
            hosted,
        )
    try:
        python = environment_python("boltz")
    except FileNotFoundError:
        pass
    else:
        cli = python.parent / "boltz"
        if cli.is_file():
            return CofoldBackend(
                "boltz-venv", (str(cli),), f"isolated Boltz environment at {python}", hosted
            )
    on_path = shutil.which("boltz")
    if on_path:
        return CofoldBackend("boltz-cli", (on_path,), f"boltz executable at {on_path}", hosted)
    if _module_importable("boltz"):
        return CofoldBackend(
            "boltz-module",
            (sys.executable, "-c", "from boltz.main import cli; cli()"),
            f"boltz module in {sys.executable}",
            hosted,
        )
    uvx = shutil.which("uvx")
    if uvx and _truthy(os.getenv("BIOAGENTS_COFOLD_UVX"), default=True):
        spec = os.getenv("BIOAGENTS_BOLTZ_UVX_SPEC", DEFAULT_UVX_SPEC)
        return CofoldBackend(
            "boltz-uvx",
            (uvx, "--python", "3.12", "--from", spec, "boltz"),
            f"isolated uvx environment ({spec}); first run downloads Boltz and its weights",
            hosted,
        )
    return CofoldBackend(None, (), "no local Boltz backend found", hosted)


def cofold_backend_status() -> dict[str, Any]:
    """Machine-readable availability report for the co-folding capability."""
    backend = detect_cofold_backend()
    status: dict[str, Any] = {
        "capability": "protein-ligand co-folding (Boltz-2)",
        "available": backend.available,
        "backend": backend.name,
        "detail": backend.detail,
        "requires_api_key": False,
    }
    if backend.hosted_alternatives:
        status["hosted_alternatives"] = list(backend.hosted_alternatives)
    if not backend.available:
        status["how_to_enable"] = (
            "Install Boltz in an isolated environment (`uv tool install boltz` or "
            "`pipx install boltz`) so `boltz` is on PATH, set BIOAGENTS_BOLTZ_CMD, or "
            "install `uv` so it can be run on demand via uvx. A CUDA GPU is strongly "
            "recommended (CPU works but takes ~10 min per complex). Hosted alternative: "
            "set NVIDIA_API_KEY for ToolUniverse's NvidiaNIM_boltz2."
        )
    return status


def cofold_available() -> bool:
    return detect_cofold_backend().available


def write_boltz_yaml(
    path: Path, protein_sequence: str, ligand_smiles: str | None, predict_affinity: bool
) -> None:
    """Write a Boltz-2 input file: chain A protein, chain L ligand."""
    lines = ["version: 1", "sequences:", "  - protein:", "      id: [A]"]
    lines.append(f"      sequence: {protein_sequence}")
    if ligand_smiles:
        lines += ["  - ligand:", "      id: [L]", f"      smiles: '{ligand_smiles}'"]
        if predict_affinity:
            lines += ["properties:", "  - affinity:", "      binder: L"]
    path.write_text("\n".join(lines) + "\n")


def _clean_sequence(sequence: str) -> str:
    lines = [ln.strip() for ln in sequence.strip().splitlines() if not ln.startswith(">")]
    return "".join(lines).replace(" ", "").upper()


def _collect_predictions(out_dir: Path, name: str) -> dict[str, Any]:
    pred_dir = out_dir / f"boltz_results_{name}" / "predictions" / name
    structures = sorted(
        str(p) for p in pred_dir.glob(f"{name}_model_*") if p.suffix in (".pdb", ".cif")
    )
    confidence: dict[str, Any] = {}
    conf_file = pred_dir / f"confidence_{name}_model_0.json"
    if conf_file.is_file():
        try:
            confidence = json.loads(conf_file.read_text())
        except (OSError, ValueError):
            confidence = {}
    affinity: dict[str, Any] = {}
    aff_file = pred_dir / f"affinity_{name}.json"
    if aff_file.is_file():
        try:
            affinity = json.loads(aff_file.read_text())
        except (OSError, ValueError):
            affinity = {}
    return {
        "prediction_dir": str(pred_dir),
        "structures": structures,
        "confidence": confidence,
        "affinity": affinity,
    }


def run_boltz_cofold(
    protein_sequence: str,
    ligand_smiles: str | None,
    output_dir: str | Path,
    *,
    use_msa_server: bool = True,
    predict_affinity: bool = False,
    timeout_s: int | None = None,
) -> str:
    """Co-fold a protein (and optional ligand) with the best local Boltz backend.

    Returns a JSON string: status='success' with the predicted complex path and Boltz
    confidence scores, or a capability_unavailable / execution error payload.
    """
    from bioagents.tools.capability_reporting import (
        capability_unavailable,
        cuda_is_usable,
        subprocess_env_for_torch,
    )

    seq = _clean_sequence(protein_sequence or "")
    if not seq or not seq.isalpha():
        return capability_unavailable(
            "Boltz-2 co-folding",
            reason="A protein sequence (single-letter amino acids) is required.",
        )
    backend = detect_cofold_backend()
    status = cofold_backend_status()
    if not backend.available:
        return capability_unavailable(
            "Boltz-2 co-folding",
            reason="No local Boltz backend is installed (no API key is needed for local use).",
            how_to_enable=status.get("how_to_enable"),
            use_instead=(
                "; ".join(backend.hosted_alternatives)
                if backend.hosted_alternatives
                else "the docking tools (prepare_ligand / run_docking) — note docking is "
                "not co-folding; report it as docking"
            ),
            backend_status=status,
        )

    out_dir = Path(output_dir or "cofold_output").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "cofold_input"
    yaml_path = out_dir / f"{name}.yaml"
    write_boltz_yaml(yaml_path, seq, (ligand_smiles or "").strip() or None, predict_affinity)

    gpu = cuda_is_usable()
    cmd = [
        *backend.command,
        "predict",
        str(yaml_path),
        "--out_dir",
        str(out_dir),
        "--output_format",
        "pdb",
        "--accelerator",
        "gpu" if gpu else "cpu",
    ]
    if use_msa_server:
        cmd.append("--use_msa_server")
    timeout = timeout_s or int(os.getenv("BIOAGENTS_BOLTZ_TIMEOUT", DEFAULT_TIMEOUT_S))
    log_path = out_dir / "boltz_run.log"
    started = time.monotonic()
    try:
        with log_path.open("w") as log:
            # Fixed argv; paths and options are not interpreted by a shell.
            proc = subprocess.run(  # nosec B603
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                env=subprocess_env_for_torch(),
                check=False,
            )
    except subprocess.TimeoutExpired:
        return capability_unavailable(
            "Boltz-2 co-folding",
            reason=f"Boltz did not finish within {timeout}s.",
            backend=backend.name,
            log_file=str(log_path),
        )
    except OSError as exc:
        return capability_unavailable(
            "Boltz-2 co-folding",
            reason=f"Could not start Boltz ({backend.detail}): {exc}",
            backend=backend.name,
        )
    elapsed = round(time.monotonic() - started, 1)
    preds = _collect_predictions(out_dir, name)
    if proc.returncode != 0 or not preds["structures"]:
        tail = ""
        with contextlib.suppress(OSError):
            tail = log_path.read_text()[-2000:]
        return capability_unavailable(
            "Boltz-2 co-folding",
            reason=f"Boltz exited with code {proc.returncode} and produced no structure.",
            backend=backend.name,
            log_file=str(log_path),
            log_tail=tail,
        )
    return json.dumps(
        {
            "status": "success",
            "performed_any_computation": True,
            "method": "Boltz-2 co-folding",
            "backend": backend.name,
            "device": "gpu" if gpu else "cpu",
            "runtime_s": elapsed,
            "input_yaml": str(yaml_path),
            "predicted_complex": preds["structures"][0],
            "all_models": preds["structures"],
            "chains": {"protein": "A", "ligand": "L" if ligand_smiles else None},
            "confidence": preds["confidence"],
            "affinity": preds["affinity"] or None,
            "log_file": str(log_path),
            "note": (
                "Predicted coordinates are not in the crystal frame: superpose on protein "
                "CA atoms before computing a ligand RMSD."
            ),
        },
        indent=2,
    )
