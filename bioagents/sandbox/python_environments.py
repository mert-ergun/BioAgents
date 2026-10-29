#!/usr/bin/env python
"""Run code with an isolated Coder, Boltz or PyTDC interpreter in the sandbox."""

from __future__ import annotations

import argparse
import math
import os
import subprocess  # nosec B404
from pathlib import Path

ENVIRONMENTS = ("coder", "boltz", "pytdc")


def environment_python(name: str) -> Path:
    """Locate an explicitly named environment without falling back to another one."""
    if name not in ENVIRONMENTS:
        raise ValueError(f"Unknown Python environment {name!r}; choose from {ENVIRONMENTS}")
    root = Path(os.environ.get("BIOAGENTS_VENV_ROOT", "/opt/venvs"))
    python = root / name / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(
            f"Sandbox Python environment {name!r} is missing at {python}. "
            "Build docker/coder.Dockerfile and select that sandbox image."
        )
    return python


def _environment(python: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["VIRTUAL_ENV"] = str(python.parent.parent)
    environment["PATH"] = str(python.parent) + os.pathsep + environment.get("PATH", "")
    # Coder's pip guard pins NumPy/RDKit versions that deliberately differ here.
    for key in ("PIP_CONSTRAINT", "UV_CONSTRAINT", "PYTHONHOME"):
        environment.pop(key, None)
    return environment


def _audit_invocation(code: str, environment: dict[str, str]) -> tuple[str, dict[str, str]]:
    if os.getenv("BIOAGENTS_RUN_ID"):
        root = Path(__file__).resolve().parents[2]
        if not (root / "bioagents/__init__.py").is_file():
            root = Path("/app")
        if (root / "bioagents/__init__.py").is_file():
            import sys

            sys.path.insert(0, str(root))
    try:
        from bioagents.sandbox import dependency_events
    except ModuleNotFoundError as exc:
        if exc.name != "bioagents":
            raise
    else:
        if dependency_events.current_audit() is not None:
            environment.update(dependency_events.dependency_environment())
            package_root = str(Path(dependency_events.__file__).resolve().parents[2])
            environment["PYTHONPATH"] = (
                package_root + os.pathsep + environment.get("PYTHONPATH", "")
            )
            code = (
                "from bioagents.sandbox.dependency_events import enable_dependency_tracking; "
                "enable_dependency_tracking(); "
                f"exec(compile({code!r}, '<bioagents-environment>', 'exec'))"
            )
    return code, environment


def run_in_environment(
    name: str, code: str, *, timeout: float = 300
) -> subprocess.CompletedProcess[str]:
    """Execute Python in a separate process, sharing cwd, files and GPU visibility.

    Exchange results through stdout or files; Jupyter variables are process-local.
    Child failures and timeouts raise rather than appearing to be successful runs.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    python = environment_python(name)
    environment = _environment(python)
    code, environment = _audit_invocation(code, environment)
    return subprocess.run(  # nosec B603
        [str(python), "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", choices=ENVIRONMENTS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Arguments passed to Python")
    arguments = parser.parse_args()
    python = environment_python(arguments.environment)
    rest = arguments.arguments
    if rest[:1] == ["--"]:
        rest = rest[1:]
    environment = _environment(python)
    if rest[:1] == ["-c"] and len(rest) >= 2:
        code, environment = _audit_invocation(rest[1], environment)
        rest = ["-c", code, *rest[2:]]
    os.execve(str(python), [str(python), *rest], environment)  # nosec B606


if __name__ == "__main__":
    main()
