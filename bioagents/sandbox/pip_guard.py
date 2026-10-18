"""Keep code agents from breaking the sandbox's core scientific stack with pip.

Two failure modes observed in real runs:

* ``pip install PyTDC`` (or ``boltz``) silently downgraded numpy 2.x to 1.26 and rdkit
  inside the running kernel. Already-imported C extensions then crashed with
  ``numpy.dtype size changed`` and every later ``import pandas`` failed.
* smolagents' ``send_tools`` derives "requirements" from each tool's imports. The
  BioAgents tool wrappers import ``bioagents``, so every ``agent.run()`` issued
  ``pip install bioagents smolagents`` inside the container. Because the project is
  mounted at ``/app`` that resolved the full project dependency set and re-downloaded
  a 550 MB torch wheel.

The guard pins the core packages to whatever version is already installed through a
pip constraints file (``PIP_CONSTRAINT`` / ``UV_CONSTRAINT``). An innocuous
``pip install numpy`` becomes a no-op, ``pip install -U torch`` cannot replace torch,
and a package that needs a different numpy fails to resolve instead of corrupting the
environment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

#: Distributions that must never be upgraded, downgraded or reinstalled by agent code.
CORE_PINNED_PACKAGES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "scipy",
    "torch",
    "torchvision",
    "torchaudio",
    "rdkit",
    "scikit-learn",
    "matplotlib",
    "biopython",
    "pyarrow",
    "numba",
    "llvmlite",
    "tensorflow",
    "jax",
    "jaxlib",
    "smolagents",
    "ipykernel",
    "jupyter_client",
    "jupyter-kernel-gateway",
    "tornado",
    "pyzmq",
)

#: Tool "requirements" smolagents infers that must never be pip-installed: the project
#: itself is importable from the mounted workspace, and smolagents is already present.
NEVER_INSTALL: frozenset[str] = frozenset({"bioagents", "smolagents"})

CONSTRAINTS_FILENAME = "bioagents_pip_constraints.txt"


def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(".", "-")


def build_constraints(installed: Mapping[str, str]) -> str:
    """Render a pip constraints file pinning each installed core package."""
    lines = [f"{name}=={version}" for name, version in sorted(installed.items()) if version]
    return "\n".join(lines) + ("\n" if lines else "")


def installed_core_versions(packages: Iterable[str] = CORE_PINNED_PACKAGES) -> dict[str, str]:
    """Return ``{distribution: version}`` for the core packages installed here."""
    from importlib import metadata

    found: dict[str, str] = {}
    for name in packages:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return found


def filter_install_requests(packages: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split requested installs into (allowed, skipped).

    Skips the project / framework packages and any core package that is already
    installed, so an install request never reaches pip just to reshuffle the stack.
    """
    installed = {_normalize(n) for n in installed_core_versions()}
    allowed: list[str] = []
    skipped: list[str] = []
    for raw in packages:
        name = _normalize(raw.split("[")[0].split("=")[0].split("<")[0].split(">")[0])
        if not name:
            continue
        if name in NEVER_INSTALL or name in installed:
            skipped.append(raw)
        else:
            allowed.append(raw)
    return allowed, skipped


def apply_pip_guard(directory: str | Path | None = None) -> Path | None:
    """Pin core packages for every pip/uv run spawned from this process.

    Writes the constraints file and exports ``PIP_CONSTRAINT`` and ``UV_CONSTRAINT``
    so subprocesses (``subprocess.run([sys.executable, "-m", "pip", ...])``,
    ``!pip install``) inherit it. Returns the constraints path, or None on failure.
    """
    import tempfile

    try:
        target_dir = Path(directory) if directory else Path(tempfile.gettempdir())
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / CONSTRAINTS_FILENAME
        path.write_text(build_constraints(installed_core_versions()))
    except OSError:
        return None
    os.environ["PIP_CONSTRAINT"] = str(path)
    os.environ["UV_CONSTRAINT"] = str(path)
    return path


def kernel_guard_code() -> str:
    """Python source that installs the guard inside a (remote) Jupyter kernel.

    Self-contained on purpose: it runs in the sandbox interpreter, which cannot import
    this module before ``/app`` is on its path, and it must run again after every
    kernel restart.
    """
    packages = ", ".join(repr(p) for p in CORE_PINNED_PACKAGES)
    return f"""
def __bioagents_install_pip_guard():
    import os, tempfile
    from importlib import metadata
    lines = []
    for _name in ({packages},):
        try:
            lines.append(f"{{_name}}=={{metadata.version(_name)}}")
        except metadata.PackageNotFoundError:
            pass
    _path = os.path.join(tempfile.gettempdir(), {CONSTRAINTS_FILENAME!r})
    with open(_path, "w") as _fh:
        _fh.write("\\n".join(lines) + "\\n")
    os.environ["PIP_CONSTRAINT"] = _path
    os.environ["UV_CONSTRAINT"] = _path

__bioagents_install_pip_guard()
del __bioagents_install_pip_guard
"""
