"""vina_dock_ligands — dock a list of SMILES ligands into one receptor with AutoDock Vina.

Paths are resolved through ``bioagents.tools.workspace_paths``: relative paths are
tried against the run workspace (where agents save files), the project root and the
cwd; ``/app/...`` container paths and absolute paths are accepted; the file name is
matched case-insensitively (``4gv1.pdb`` finds ``4GV1.pdb``). A missing file raises
``FileNotFoundError`` listing the candidate files that do exist.

Previously the tool did ``os.path.isfile(receptor_pdb)`` against the host process cwd
(the project root), so the workspace-relative ``001_task2_molecule_ranking/4gv1.pdb``
was never found and Vina was never run.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import subprocess  # nosec B404
import tempfile
import time
from pathlib import Path
from typing import Any

from bioagents.sandbox.workspace import to_portable_path
from bioagents.tools.workspace_paths import (
    prepare_output_file,
    resolve_input_file,
    resolve_output_dir,
)

DEFAULT_BOX_SIZE = [22.5, 22.5, 22.5]
_RECEPTOR_SUFFIXES = (".pdb", ".ent", ".pdbqt")
# HETATM groups that are not a bound ligand: solvent, additives, ions, modified residues.
_NON_LIGAND_HET = {
    "HOH", "WAT", "DOD", "H2O", "TIP3", "GOL", "EDO", "PEG", "PG4", "SO4", "PO4", "ACT",
    "DMS", "MPD", "CL", "NA", "K", "MG", "CA", "ZN", "MN", "FE", "NI", "CU", "CD", "IOD",
    "TPO", "SEP", "PTR", "MSE", "CSO", "HYP",
}  # fmt: skip


def _load_ligands(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    mols = data.get("molecules", []) if isinstance(data, dict) else data
    ligands = []
    if not isinstance(mols, list):
        raise ValueError("Ligands must be a list of molecule objects")
    seen = set()
    for i, m in enumerate(mols or [], start=1):
        if not isinstance(m, dict):
            raise ValueError("Each ligand must be a molecule object")
        smiles = (m or {}).get("smiles")
        if smiles:
            cid = str(m.get("chembl_id") or f"lig_{i}")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", cid):
                raise ValueError(f"Unsafe ligand identifier: {cid!r}")
            if cid in seen:
                raise ValueError(f"Duplicate ligand identifier: {cid!r}")
            if not isinstance(smiles, str):
                raise ValueError("Ligand SMILES must be a string")
            seen.add(cid)
            ligands.append({"chembl_id": cid, "smiles": smiles})
    if not ligands:
        raise ValueError(f"No molecules with a 'smiles' field in {path}")
    return ligands


def _het_ligand_centroid(pdb_path: Path) -> list[float] | None:
    """Centroid of the largest co-crystallised ligand (by heavy-atom count)."""
    groups: dict[tuple[str, str, str], list[tuple[float, float, float]]] = {}
    for line in pdb_path.read_text(errors="replace").splitlines():
        if not line.startswith("HETATM"):
            continue
        resname = line[17:20].strip()
        if resname in _NON_LIGAND_HET or line[76:78].strip() == "H":
            continue
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        groups.setdefault((resname, line[21], line[22:26]), []).append(xyz)
    if not groups:
        return None
    atoms = max(groups.values(), key=len)
    return [round(sum(a[i] for a in atoms) / len(atoms), 3) for i in range(3)]


def _protein_centroid(pdb_path: Path) -> list[float]:
    xs: list[tuple[float, float, float]] = []
    for line in pdb_path.read_text(errors="replace").splitlines():
        if line.startswith("ATOM"):
            try:
                xs.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except ValueError:
                continue
    if not xs:
        raise ValueError(f"No ATOM records in {pdb_path}")
    return [round(sum(a[i] for a in xs) / len(xs), 3) for i in range(3)]


def _sidecar_center(receptor: Path) -> list[float] | None:
    """Use a box center saved next to the receptor (binding_site.json / *_info.json)."""
    site = receptor.parent / "binding_site.json"
    candidates = [site] if site.is_file() else []
    candidates += sorted(receptor.parent.glob("*_info.json"))
    for path in candidates:
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(info.get("center"), list) and len(info["center"]) == 3:
            return [float(c) for c in info["center"]]
        ligs = info.get("co_crystal_ligands") or []
        if ligs and isinstance(ligs[0].get("centroid"), list):
            return [float(c) for c in ligs[0]["centroid"]]
    return None


def _choose_center(receptor: Path, center: list[float] | None) -> tuple[list[float], str]:
    if center is not None:
        return [float(c) for c in center], "user"
    found = _sidecar_center(receptor)
    if found is not None:
        return found, "sidecar"
    if receptor.suffix.lower() != ".pdbqt":
        het = _het_ligand_centroid(receptor)
        if het is not None:
            return het, "co-crystal ligand"
    return _protein_centroid(receptor), "protein centroid"


def _ensure(mod, pip_name=None):
    """Require preinstalled dependencies without changing a running interpreter."""
    try:
        return importlib.import_module(mod)
    except ModuleNotFoundError as exc:
        if exc.name != mod and not mod.startswith((exc.name or "") + "."):
            raise
        raise ModuleNotFoundError(
            f"Missing dependency {mod!r}; install {pip_name or mod!r} in the sandbox image before execution",
            name=exc.name,
        ) from exc


def _convert_with_obabel(source, destination, *, options=(), timeout=60):
    """Publish only a fresh nonempty CLI output; native crashes stay in the child."""
    executable = shutil.which("obabel")
    if not executable:
        raise RuntimeError("obabel CLI is required; install openbabel-wheel in the sandbox image")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(suffix=target.suffix, dir=target.parent)
    os.close(descriptor)
    try:
        # Resolved executable and separate argv; inputs are never interpreted by a shell.
        result = subprocess.run(  # nosec B603
            [executable, str(source), "-O", temporary, *options],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0 or Path(temporary).stat().st_size == 0:
            raise RuntimeError(
                f"obabel conversion failed (exit {result.returncode}): {result.stderr[-2000:]}"
            )
        Path(temporary).replace(target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _receptor_to_pdbqt(source, destination, ph=7.4):
    _convert_with_obabel(source, destination, options=("-xr", "-p", str(ph)), timeout=300)
    return "obabel-cli"


def _prepare_receptor(receptor: Path, out_dir: Path, ph=7.4) -> tuple[Path, str]:
    if receptor.suffix.lower() == ".pdbqt":
        return receptor, "provided"
    target, _ = prepare_output_file(out_dir / "receptor.pdbqt")
    with tempfile.TemporaryDirectory(prefix="receptor-", dir=out_dir) as temporary:
        clean = Path(temporary) / "receptor_clean.pdb"
        clean.write_text(
            "\n".join(
                ln
                for ln in receptor.read_text(errors="replace").splitlines()
                if ln.startswith(("ATOM", "TER", "END"))
            )
            + "\n"
        )
        _receptor_to_pdbqt(clean, target, ph)
    return target, "obabel-cli"


def _dock_one(vina: Any, lig: dict[str, Any], out_dir: Path, n_poses: int, exh: int) -> dict:
    from bioagents.tools.docking_tools import _smiles_to_pdbqt

    cid = lig["chembl_id"]
    ligand_path, _ = prepare_output_file(out_dir / f"{cid}.pdbqt")
    prep = _smiles_to_pdbqt(lig["smiles"], str(ligand_path))
    if prep.get("status") != "success":
        return {**lig, "status": "error", "error": prep.get("message")}
    vina.set_ligand_from_file(str(ligand_path))
    vina.dock(exhaustiveness=int(exh), n_poses=int(n_poses))
    energies = vina.energies()
    pose, _ = prepare_output_file(out_dir / f"{cid}_pose.pdbqt")
    vina.write_poses(str(pose), n_poses=int(n_poses))
    return {
        **lig,
        "status": "success",
        "affinity_kcal_mol": round(float(energies[0][0]), 3),
        "pose_affinities_kcal_mol": [round(float(e[0]), 3) for e in energies],
        "ligand_pdbqt": to_portable_path(ligand_path),
        "pose_pdbqt": to_portable_path(pose),
    }


def _write_results(out_dir: Path, summary: dict[str, Any]) -> tuple[str, str]:
    # A rerun writes a new version rather than replacing a completed docking run.
    json_path, _ = prepare_output_file(out_dir / "docking_results.json")
    csv_path, _ = prepare_output_file(out_dir / "docking_results.csv")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fields = ["chembl_id", "smiles", "status", "affinity_kcal_mol", "pose_pdbqt", "error"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["results"])
    return to_portable_path(json_path), to_portable_path(csv_path)


def _resume_results(out_dir: Path, fingerprint: str, ligands: list[dict]) -> list[dict]:
    """Resume the latest matching run, including versioned result files."""
    candidates = [
        p
        for p in out_dir.glob("docking_results*.json")
        if re.fullmatch(r"docking_results(?:_v\d+)?\.json", p.name)
    ]
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime_ns, reverse=True):
        try:
            previous = json.loads(path.read_text())
            if previous.get("input_fingerprint") != fingerprint:
                continue
            requested = {lig["chembl_id"]: lig["smiles"] for lig in ligands}
            results = []
            for record in previous.get("results", []):
                if record.get("status") == "success" and requested.get(
                    record.get("chembl_id")
                ) == record.get("smiles"):
                    try:
                        resolve_input_file(record["pose_pdbqt"], "Docked pose", (".pdbqt",))
                    except (KeyError, ValueError, FileNotFoundError):
                        continue
                    results.append(record)
            return results
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return []


def vina_dock_ligands(
    receptor_pdb: str,
    ligands_json: str,
    output_dir: str = "",
    center: list[float] | None = None,
    box_size: list[float] | None = None,
    exhaustiveness: int = 8,
    n_poses: int = 5,
    seed: int = 42,
    dry_run: bool = False,
    resume: bool = False,
    time_budget_sec: float = 900,
    ph: float = 7.4,
) -> dict[str, Any]:
    """Dock ligands into a receptor with AutoDock Vina.

    Args:
        receptor_pdb: Receptor PDB/PDBQT. Workspace-relative, project-relative, ``/app``
            or absolute; file name matched case-insensitively.
        ligands_json: JSON with a ``molecules`` list of ``{chembl_id, smiles}`` (or a list).
        output_dir: Output directory; relative paths are anchored at the run workspace
            (default ``<workspace>/docking``).
        center: Box center [x, y, z]; default binding_site.json / *_info.json next to the
            receptor, else the largest co-crystal ligand, else the protein centroid.
        box_size: Box size in Angstrom (default 22.5 per axis).
        exhaustiveness: Vina exhaustiveness.
        n_poses: Poses per ligand.
        seed: Vina random seed.
        dry_run: Resolve inputs and report the plan without docking.
        resume: Skip ligands already present in ``docking_results.json``.
        time_budget_sec: Stop docking new ligands after this many seconds.
        ph: Receptor hydrogenation pH used by the OpenBabel CLI.

    Returns:
        Dict with status, n_docked, n_failed, center, per-ligand results (affinity in
        kcal/mol, pose paths) and the docking_results.json/csv paths.

    Raises:
        FileNotFoundError: Receptor or ligands file not found (lists candidate files).
        ValueError: Invalid inputs.
    """
    t0 = time.monotonic()
    box = [float(b) for b in (box_size if box_size is not None else DEFAULT_BOX_SIZE)]
    if len(box) != 3 or any(not math.isfinite(b) or b <= 0 for b in box):
        raise ValueError("box_size must contain three finite positive values")
    if not math.isfinite(time_budget_sec) or time_budget_sec <= 0:
        raise ValueError("time_budget_sec must be finite and positive")
    if exhaustiveness < 1 or n_poses < 1 or not math.isfinite(ph) or not 0 <= ph <= 14:
        raise ValueError("exhaustiveness/n_poses must be positive and pH must be between 0 and 14")
    receptor = resolve_input_file(receptor_pdb, "Receptor PDB", _RECEPTOR_SUFFIXES)
    ligands_path = resolve_input_file(ligands_json, "Ligands JSON", (".json",))
    ligands = _load_ligands(ligands_path)
    out_dir = resolve_output_dir(output_dir or None, default_subdir="docking")
    box_center, center_source = _choose_center(receptor, center)
    if len(box_center) != 3 or any(not math.isfinite(c) for c in box_center):
        raise ValueError("center must contain three finite values")

    plan: dict[str, Any] = {
        "receptor_pdb": to_portable_path(receptor),
        "ligands_json": to_portable_path(ligands_path),
        "output_dir": to_portable_path(out_dir),
        "n_ligands": len(ligands),
        "center": box_center,
        "center_source": center_source,
        "box_size": box,
        "input_fingerprint": hashlib.sha256(
            receptor.read_bytes()
            + json.dumps(
                {
                    "ligands": ligands,
                    "center": box_center,
                    "box": box,
                    "seed": seed,
                    "exhaustiveness": exhaustiveness,
                    "n_poses": n_poses,
                    "ph": ph,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    }
    if dry_run:
        return {"status": "dry_run", **plan}

    try:
        from vina import Vina
    except ImportError as exc:
        raise RuntimeError("AutoDock Vina is not installed (pip install vina)") from exc

    receptor_pdbqt, prep_method = _prepare_receptor(receptor, out_dir, ph)
    vina = Vina(sf_name="vina", seed=int(seed), verbosity=0)
    vina.set_receptor(str(receptor_pdbqt))
    vina.compute_vina_maps(center=box_center, box_size=box)

    previous = _resume_results(out_dir, plan["input_fingerprint"], ligands) if resume else []
    done = {r["chembl_id"] for r in previous if r.get("status") == "success"}
    results = [r for r in previous if r["chembl_id"] in done]
    for lig in ligands:
        if lig["chembl_id"] in done:
            continue
        if time.monotonic() - t0 > time_budget_sec:
            results.append({**lig, "status": "skipped", "error": "time budget exhausted"})
            continue
        try:
            results.append(_dock_one(vina, lig, out_dir, n_poses, exhaustiveness))
        except Exception as exc:  # one bad ligand must not abort the batch
            results.append({**lig, "status": "error", "error": str(exc)})

    n_docked = sum(r.get("status") == "success" for r in results)
    summary = {
        **plan,
        "status": ("success" if n_docked == len(ligands) else "partial" if n_docked else "error"),
        "receptor_pdbqt": to_portable_path(receptor_pdbqt),
        "prep_method": prep_method,
        "exhaustiveness": exhaustiveness,
        "n_poses": n_poses,
        "n_docked": n_docked,
        "n_failed": len(results) - n_docked,
        "results": results,
        "elapsed_sec": round(time.monotonic() - t0, 1),
    }
    summary["results_json"], summary["results_csv"] = _write_results(out_dir, summary)
    summary["json_file"], summary["csv_file"] = summary["results_json"], summary["results_csv"]
    return summary
