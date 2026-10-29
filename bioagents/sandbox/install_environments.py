"""Build the three independent Python environments inside a sandbox image."""

from __future__ import annotations

import os
import subprocess  # nosec B404
import tempfile
import venv
from contextlib import nullcontext
from pathlib import Path

ENVIRONMENTS = ("coder", "boltz", "pytdc")


def install_environments(names: tuple[str, ...] = ENVIRONMENTS) -> None:
    root = Path(os.environ.get("BIOAGENTS_VENV_ROOT", "/opt/venvs"))
    requirements = Path(__file__).with_name("requirements")
    torch_version = os.environ.get("TORCH_VERSION", "2.14.1")
    torch_index = os.environ.get("TORCH_INDEX_URL", "https://download.pytorch.org/whl/cpu")
    persistent_cache = os.environ.get("BIOAGENTS_INSTALL_CACHE")
    cache_context = (
        nullcontext(persistent_cache)
        if persistent_cache
        else tempfile.TemporaryDirectory(prefix="bioagents-wheel-cache-")
    )
    with cache_context as cache:
        environment = os.environ.copy()
        environment["PIP_CACHE_DIR"] = cache
        for key in ("PIP_CONSTRAINT", "UV_CONSTRAINT", "PYTHONHOME", "PYTHONPATH"):
            environment.pop(key, None)
        for name in names:
            directory = root / name
            if name == "pytdc":
                python311 = subprocess.check_output(  # nosec B603 B607
                    ["uv", "python", "find", "--no-project", "--managed-python", "3.11"],
                    env=environment,
                    text=True,
                ).strip()
                subprocess.run(  # nosec B603
                    [python311, "-m", "venv", str(directory)], env=environment, check=True
                )
            else:
                venv.EnvBuilder(with_pip=True).create(directory)
            python = str(directory / "bin" / "python")
            pip = [python, "-m", "pip", "--disable-pip-version-check"]
            subprocess.run(  # nosec B603
                [*pip, "install", "--upgrade", "pip>=25.3,<27", "setuptools==80.9.0", "wheel"],
                env=environment,
                check=True,
            )
            subprocess.run(  # nosec B603
                [
                    *pip,
                    "download",
                    f"torch=={torch_version}",
                    "--index-url",
                    torch_index,
                    "--dest",
                    str(Path(cache) / "torch-wheels"),
                ],
                env=environment,
                check=True,
            )
            subprocess.run(  # nosec B603
                [
                    *pip,
                    "install",
                    f"torch=={torch_version}",
                    "--no-index",
                    "--find-links",
                    str(Path(cache) / "torch-wheels"),
                ],
                env=environment,
                check=True,
            )
            subprocess.run(  # nosec B603
                [*pip, "install", "-r", str(requirements / f"{name}.lock.txt")],
                env=environment,
                check=True,
            )
            subprocess.run([*pip, "check"], env=environment, check=True)  # nosec B603


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", nargs="?", choices=ENVIRONMENTS)
    selected = parser.parse_args().environment
    install_environments((selected,) if selected else ENVIRONMENTS)
