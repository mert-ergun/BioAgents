"""Docker executor setup for coder agent."""

import contextlib
import importlib.util
import json
import logging
import os
import re
import shutil
import socket
import subprocess  # nosec B404
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Patched below: upstream stores __enter__()'s return value and calls __exit__ on it,
# so a context manager returning None (torch.no_grad()) crashes.
import smolagents.local_python_executor as _lpe
from rich.console import Console
from smolagents import AgentLogger, LocalPythonExecutor, LogLevel

# Patched below to be idempotent: smolagents re-applies it on every run(), and applying
# it twice makes forward and _forward call each other forever.
from smolagents.remote_executors import RemotePythonExecutor as _RPE

module_logger = logging.getLogger(__name__)

_original_patch = _RPE._patch_final_answer_with_exception


def _idempotent_patch(self, final_answer_tool):
    if final_answer_tool.__class__.__name__ == "_FinalAnswerTool":
        return  # already patched — re-patching would create infinite recursion
    _original_patch(self, final_answer_tool)


_RPE._patch_final_answer_with_exception = _idempotent_patch


def _patched_evaluate_with(with_node, state, static_tools, custom_tools, authorized_imports):
    contexts = []
    for item in with_node.items:
        ctx = _lpe.evaluate_ast(
            item.context_expr, state, static_tools, custom_tools, authorized_imports
        )
        enter_result = ctx.__enter__()
        if item.optional_vars:
            state[item.optional_vars.id] = enter_result
        contexts.append(ctx)

    try:
        for stmt in with_node.body:
            _lpe.evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
    except Exception as e:
        for context in reversed(contexts):
            context.__exit__(type(e), e, e.__traceback__)
        raise
    else:
        for context in reversed(contexts):
            context.__exit__(None, None, None)


_lpe.evaluate_with = _patched_evaluate_with


def _site_packages_roots() -> list[str]:
    import site

    roots: list[str] = []
    roots.extend(site.getsitepackages())
    us = site.getusersitepackages()
    if us:
        roots.append(us)
    return [p for p in roots if p and Path(p).is_dir()]


def _prepend_nvidia_pip_libs_to_ld_path() -> None:
    """If nvidia-* wheels are installed, expose their lib/ dirs to the dynamic linker."""
    try:
        lib_dirs: list[str] = []
        for sp in _site_packages_roots():
            lib_dirs.extend(str(p) for p in Path(sp).glob("nvidia/*/lib"))
        lib_dirs = [p for p in lib_dirs if Path(p).is_dir()]
        if not lib_dirs:
            return
        extra = os.pathsep.join(lib_dirs)
        prev = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = extra + (os.pathsep + prev if prev else "")
    except Exception:
        return


def _patch_json_numpy_serialization() -> None:
    """Allow ``json.dumps`` to handle numpy scalar types (int64, float64, etc.).

    Coder-generated code frequently calls ``json.dumps()`` on dicts containing
    pandas/numpy scalars, which crashes with ``TypeError: Object of type int64
    is not JSON serializable``.  This one-time patch makes the default encoder
    fall back to native Python types.
    """

    _original_default = json.JSONEncoder.default

    def _numpy_safe_default(self, obj):
        try:
            import numpy as np

            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
        except ImportError:
            pass
        return _original_default(self, obj)

    json.JSONEncoder.default = _numpy_safe_default  # type: ignore[method-assign]


def apply_local_executor_runtime_env() -> None:
    """
    Tune process env before LocalPythonExecutor runs user code (same interpreter).

    Partial CUDA installs (driver present, cuDNN missing) often break ``import torch``.
    - Prepends pip ``nvidia/*/lib`` paths so optional ``pip install nvidia-cudnn-cu12`` libs load.
    - Exposes the GPU only when a real kernel launch succeeds, so a broken or
      mismatched CUDA stack degrades to CPU instead of crashing mid-computation.
    - Patches ``json.JSONEncoder`` to handle numpy scalar types.

    ``BIOAGENTS_TORCH_CUDA`` overrides the probe in either direction: set it truthy to
    force the GPU through, or falsy to mask it even when it works.
    """
    _patch_json_numpy_serialization()
    _prepend_nvidia_pip_libs_to_ld_path()
    # Agent code runs in this interpreter: pin the core stack so `pip install X`
    # cannot downgrade numpy/torch/rdkit underneath the running process.
    from bioagents.sandbox.pip_guard import apply_pip_guard

    apply_pip_guard()
    from bioagents.sandbox.dependency_events import enable_dependency_tracking

    enable_dependency_tracking()

    override = os.getenv("BIOAGENTS_TORCH_CUDA", "").strip().lower()
    if override in ("1", "true", "yes"):
        return
    if override in ("0", "false", "no"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return

    # torch.cuda.is_available() is True for a GPU whose compute capability the installed
    # torch lacks kernels for; that only fails at the first launch, so probe instead.
    from bioagents.tools.capability_reporting import cuda_is_usable

    if not cuda_is_usable():
        module_logger.info("No usable CUDA device for the code sandbox; running on CPU.")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _extra_builtins() -> dict[str, object]:
    """Builtins that smolagents' LocalPythonExecutor needs but doesn't include by default."""
    return {"super": super}


def is_module_installed(module_name: str) -> bool:
    """Check if a Python module is installed."""
    base_module = module_name.split(".")[0].replace("*", "")
    if not base_module:
        return True
    return importlib.util.find_spec(base_module) is not None


def find_available_port(start_port: int = 8888, max_retries: int = 100) -> int:
    """
    Find an available port starting from start_port.

    Args:
        start_port: Starting port number to check
        max_retries: Maximum number of ports to check

    Returns:
        An available port number

    Raises:
        RuntimeError: If no available port is found in the range
    """
    for port in range(start_port, start_port + max_retries):
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(
        f"Could not find an available port in range {start_port}-{start_port + max_retries}"
    )


SANDBOX_IMAGE = os.getenv("BIOAGENTS_SANDBOX_IMAGE", "").strip() or "bioagents-sandbox"

# Without an explicit GPU the CUDA torch build is 5+ GB of kernels that cannot run.
_SANDBOX_GPU = os.getenv("BIOAGENTS_SANDBOX_GPU", "").strip().lower() in ("1", "true", "yes")

_TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

_PACKAGE_ALIASES = {
    "Bio": "biopython",
    "sklearn": "scikit-learn",
    "pkg_resources": "setuptools",
}

_STDLIB_MODULES = frozenset(
    {
        "typing",
        "json",
        "os",
        "os.path",
        "sys",
        "pathlib",
        "subprocess",
        "io",
        "base64",
        "binascii",
        "importlib",
        "posixpath",
        "ntpath",
        "collections",
        "itertools",
        "math",
        "random",
        "re",
        "time",
        "datetime",
        "statistics",
        "queue",
    }
)


def _pip_packages(imports: list[str]) -> list[str]:
    """Map an authorized-import list to the pip packages the image needs."""
    packages: list[str] = []
    for imp in imports:
        base = imp[:-2] if imp.endswith(".*") else imp
        if base in _STDLIB_MODULES or base.startswith("bioagents"):
            continue
        name = _PACKAGE_ALIASES.get(base, base)
        if name not in packages:
            packages.append(name)
    return packages


def sandbox_packages() -> list[str]:
    """Packages for the shared image: the union of what coder, ML and DL may import."""
    from bioagents.sandbox.coder_helpers import (
        DEFAULT_CODER_IMPORTS,
        DEFAULT_DL_IMPORTS,
        DEFAULT_ML_IMPORTS,
    )

    union: list[str] = []
    for imports in (DEFAULT_CODER_IMPORTS, DEFAULT_ML_IMPORTS, DEFAULT_DL_IMPORTS):
        for pkg in _pip_packages(imports):
            if pkg not in union:
                union.append(pkg)
    from importlib.resources import files

    requirements = files("bioagents.sandbox").joinpath("requirements/coder.txt").read_text()
    for line in requirements.splitlines():
        if not line or line.startswith("#"):
            continue
        pkg = re.split(r"[<>=!~;\[]", line, maxsplit=1)[0].strip()
        if pkg not in union:
            union.append(pkg)
    return union


def build_dockerfile(packages: list[str] | None = None) -> str:
    """Self-contained image recipe using the same three venv inputs as the Dockerfile."""
    import base64
    import shlex
    from importlib.resources import files

    packages = sandbox_packages() if packages is None else packages
    resources = files("bioagents.sandbox")
    requirements = resources.joinpath("requirements")
    payload = {
        f"/opt/bioagents/requirements/{entry.name}": entry.read_text()
        for entry in requirements.iterdir()
        if entry.name.endswith(".txt")
    }
    if packages:
        # Optional extra authorized imports augment Coder only, never the other venvs.
        source = payload["/opt/bioagents/requirements/coder.txt"]
        existing = {
            re.split(r"[<>=!~;\[]", line, maxsplit=1)[0].strip().lower().replace("_", "-")
            for line in source.splitlines()
            if line and not line.startswith("#")
        }
        extra = [
            p for p in packages if p != "torch" and p.lower().replace("_", "-") not in existing
        ]
        if extra:
            payload["/opt/bioagents/requirements/coder.lock.txt"] += "\n".join(extra) + "\n"
    payload["/opt/bioagents/install_environments.py"] = resources.joinpath(
        "install_environments.py"
    ).read_text()
    payload["/usr/local/bin/bioagents-python"] = resources.joinpath(
        "python_environments.py"
    ).read_text()
    encoded = base64.b64encode(json.dumps(payload, sort_keys=True).encode()).decode()
    unpack = (
        "import base64,json; from pathlib import Path; "
        f"files=json.loads(base64.b64decode({encoded!r})); "
        "[(Path(p).parent.mkdir(parents=True,exist_ok=True),Path(p).write_text(s)) "
        "for p,s in files.items()]"
    )
    torch_index = "https://download.pytorch.org/whl/cu130" if _SANDBOX_GPU else _TORCH_CPU_INDEX
    return f"""
        FROM python:3.12-slim
        WORKDIR /app
        # smolagents runs in Coder; Boltz and PyTDC have isolated dependencies.
        ENV PYTHONPATH=/app \\
            BIOAGENTS_VENV_ROOT=/opt/venvs \\
            UV_PYTHON_INSTALL_DIR=/opt/python \\
            PATH=/opt/venvs/coder/bin:$PATH \\
            VIRTUAL_ENV=/opt/venvs/coder \\
            HOME=/tmp/sandbox-home \\
            JUPYTER_RUNTIME_DIR=/tmp/sandbox-home/jupyter/runtime \\
            JUPYTER_DATA_DIR=/tmp/sandbox-home/jupyter/data \\
            XDG_CACHE_HOME=/tmp/sandbox-home/cache \\
            MPLCONFIGDIR=/tmp/sandbox-home/matplotlib
        RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \\
            && rm -rf /var/lib/apt/lists/* \\
            && mkdir -p /tmp/sandbox-home/jupyter/runtime /tmp/sandbox-home/jupyter/data \\
                /tmp/sandbox-home/cache /tmp/sandbox-home/matplotlib \\
            && chmod -R 777 /tmp/sandbox-home
        RUN /usr/local/bin/python -m pip install --no-cache-dir uv==0.12.5 \\
            && uv python install --no-cache --no-bin 3.11
        RUN /usr/local/bin/python -c {shlex.quote(unpack)}
        RUN TORCH_VERSION=2.14.1 TORCH_INDEX_URL={torch_index} \\
            /usr/local/bin/python /opt/bioagents/install_environments.py \\
            && chmod +x /usr/local/bin/bioagents-python
        EXPOSE 8888
        CMD ["jupyter", "kernelgateway", "--KernelGatewayApp.ip=0.0.0.0", "--KernelGatewayApp.port=8888", "--KernelGatewayApp.allow_origin=*"]
        """


def sandbox_image_tag(dockerfile: str) -> str:
    """Tag the image with a digest of its recipe.

    A fixed tag meant an edited Dockerfile kept resolving to the already-built image,
    so recipe changes appeared to do nothing until the image was removed by hand.
    """
    import hashlib

    digest = hashlib.sha256(dockerfile.encode()).hexdigest()[:12]
    return f"{SANDBOX_IMAGE}:{digest}"


def _sandbox_user() -> str | None:
    """The uid:gid the sandbox container should run as, or None to leave it to Docker."""
    override = os.getenv("BIOAGENTS_SANDBOX_UID", "").strip()
    if override:
        return override if ":" in override else f"{override}:{override}"
    if not hasattr(os, "getuid"):  # pragma: no cover - non-POSIX
        return None
    return f"{os.getuid()}:{os.getgid()}"


def _image_exists(image_name: str) -> bool:
    return _existing_image_id(image_name) is not None


def _existing_image_id(image_name: str) -> str | None:
    docker_path = shutil.which("docker")
    if not docker_path:
        return None
    try:
        result = subprocess.run(  # nosec B603
            [docker_path, "image", "inspect", "--format", "{{.Id}}", image_name],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.CalledProcessError:
        return None
    return result.stdout.strip() or None


_IDLE_ATTRS: dict[str, Callable[[], Any]] = {"fatal_error": lambda: None, "state": dict}


class LazyExecutor:
    """Defers executor creation until the agent actually runs code.

    Building the graph constructs the coder, ML and DL agents; starting a container for
    each would cost three for a run that typically uses one.
    """

    def __init__(self, factory):
        self._factory = factory
        self._executor = None

    @property
    def started(self) -> bool:
        return self._executor is not None

    def _ensure(self):
        if self._executor is None:
            self._executor = self._factory()
        return self._executor

    def __call__(self, *args, **kwargs):
        return self._ensure()(*args, **kwargs)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        # Status reads must not start a container.
        if self._executor is None and name in _IDLE_ATTRS:
            return _IDLE_ATTRS[name]()
        return getattr(self._ensure(), name)

    def cleanup(self) -> None:
        """Release the executor, without starting one that was never needed."""
        if self._executor is not None and hasattr(self._executor, "cleanup"):
            self._executor.cleanup()


def _local_executor(additional_imports: list) -> LocalPythonExecutor:
    apply_local_executor_runtime_env()
    from bioagents.sandbox.coder_helpers import PermissiveList

    return LocalPythonExecutor(
        additional_authorized_imports=PermissiveList(additional_imports),
        additional_functions=_extra_builtins(),
    )


def create_executor(agent_type: str, additional_imports: list) -> Any:
    """Create the code executor for ``agent_type``.

    Coder, ML and DL share one image built from the union of their allowed imports;
    the container stays per-agent so their variables cannot leak, and starts on first use.

    Set USE_LOCAL_EXECUTOR=true to run agent code in this process instead (not sandboxed).
    """
    if os.getenv("USE_LOCAL_EXECUTOR", "false").lower() in ("true", "1", "yes"):
        print(f"Using LocalPythonExecutor for {agent_type} (USE_LOCAL_EXECUTOR=true)")
        return _local_executor(additional_imports)

    def factory():
        try:
            terminal_console = Console(file=sys.stdout, width=None, force_terminal=True)
            logger = AgentLogger(level=LogLevel.INFO, console=terminal_console)
            from bioagents.sandbox.time_budget import sandbox_time_budget_env

            # Run as the host user: files the container writes into the shared
            # workspace were root-owned, so a host-side agent editing the same artifact
            # got PermissionError and treated the finished work as a stale relic to
            # archive and redo. BIOAGENTS_SANDBOX_UID=0 restores running as root.
            container_run_kwargs: dict[str, Any] = {
                "volumes": {
                    os.getenv("BIOAGENTS_DOCKER_PROJECT_ROOT") or str(Path.cwd()): {
                        "bind": "/app",
                        "mode": "rw",
                    }
                },
                # In-cell tools cap their budgets below the cell deadline, so they
                # need the host's deadline and budget overrides (issue #67).
                "environment": {"BIOAGENTS_BLAST_IN_CELL": "1", **sandbox_time_budget_env()},
                "command": [
                    "jupyter",
                    "kernelgateway",
                    "--KernelGatewayApp.ip=0.0.0.0",
                    "--KernelGatewayApp.port=8888",
                    "--KernelGatewayApp.allow_origin=*",
                ],
            }
            run_id = os.getenv("BIOAGENTS_RUN_ID", "")
            from bioagents.sandbox.dependency_events import current_audit, dependency_environment

            audit = current_audit()
            if audit:
                run_id = audit.run_id
                audit.prepare()
                container_run_kwargs["environment"].update(dependency_environment())
                if audit.directory:
                    audit.directory.mkdir(parents=True, exist_ok=True)
                    from bioagents.sandbox.workspace import to_portable_path

                    portable = to_portable_path(audit.directory)
                    if not Path(portable).is_absolute():
                        container_run_kwargs["environment"]["BIOAGENTS_DEPENDENCY_EVENTS_DIR"] = (
                            "/app/" + portable
                        )
                    else:
                        container_run_kwargs["volumes"][str(audit.directory.resolve())] = {
                            "bind": "/bioagents-audit",
                            "mode": "rw",
                        }
                        container_run_kwargs["environment"]["BIOAGENTS_DEPENDENCY_EVENTS_DIR"] = (
                            "/bioagents-audit"
                        )
            if run_id:
                if not re.fullmatch(r"[a-f0-9]{32}", run_id):
                    raise ValueError("BIOAGENTS_RUN_ID must be a 32-character hexadecimal ID")
                container_run_kwargs["labels"] = {
                    "bioagents.run_id": run_id,
                    "bioagents.agent": agent_type,
                }
            if os.getenv("BIOAGENTS_DOCKER_INHERIT_VOLUMES", "false").lower() in (
                "1",
                "true",
                "yes",
            ):
                parent = os.getenv("HOSTNAME", "").strip()
                if not parent:
                    raise RuntimeError("HOSTNAME is required to inherit parent container volumes")
                container_run_kwargs["volumes_from"] = [parent]
            user = _sandbox_user()
            if user:
                container_run_kwargs["user"] = user
            if _SANDBOX_GPU:
                from docker.types import DeviceRequest

                driver = os.getenv("BIOAGENTS_SANDBOX_GPU_DRIVER", "auto").strip().lower()
                if driver == "auto":
                    request = DeviceRequest(count=-1, capabilities=[["gpu"]])
                elif driver == "cdi":
                    request = DeviceRequest(driver="cdi", device_ids=["nvidia.com/gpu=all"])
                elif driver == "nvidia":
                    request = DeviceRequest(driver="nvidia", count=-1, capabilities=[["gpu"]])
                else:
                    raise ValueError("BIOAGENTS_SANDBOX_GPU_DRIVER must be auto, cdi or nvidia")
                container_run_kwargs["device_requests"] = [request]

            from bioagents.sandbox.resilient_executor import ResilientDockerExecutor

            dockerfile = build_dockerfile()
            explicit_image = os.getenv("BIOAGENTS_SANDBOX_IMAGE", "").strip()
            image = explicit_image or sandbox_image_tag(dockerfile)
            image_id = _existing_image_id(image)
            # Keep installations with prebuilt per-agent images working during migration.
            if not image_id and not explicit_image:
                image_id = _existing_image_id(f"bioagents-{agent_type}")
            allow_build = os.getenv("BIOAGENTS_SANDBOX_ALLOW_BUILD", "false").lower() in (
                "1",
                "true",
                "yes",
            )
            if not image_id and not allow_build:
                raise RuntimeError(
                    f"No existing sandbox image found for {image!r}. Build it explicitly or "
                    "set BIOAGENTS_SANDBOX_IMAGE to an existing image. "
                    "BIOAGENTS_SANDBOX_ALLOW_BUILD=true explicitly enables automatic builds."
                )
            return ResilientDockerExecutor(
                image_name=image_id or image,
                dockerfile_content=dockerfile,
                additional_imports=[],
                logger=logger,
                port=find_available_port(),
                build_new_image=not bool(image_id),
                container_run_kwargs=container_run_kwargs,
            )
        except Exception as e:
            raise RuntimeError(
                f"Docker sandbox could not initialize for {agent_type}: {e}. "
                "Set USE_LOCAL_EXECUTOR=true only if in-process execution is intended."
            ) from e

    return LazyExecutor(factory)


def cleanup_run_containers(run_id: str) -> list[str]:
    """Release only sandbox containers labelled with this specific CLI run ID."""
    if not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise ValueError("A complete run ID is required for scoped container cleanup")
    import docker

    errors = []
    client = None
    try:
        client = docker.from_env(timeout=10)
        for container in client.containers.list(
            all=True, filters={"label": f"bioagents.run_id={run_id}"}
        ):
            if container.labels.get("bioagents.run_id") != run_id:
                continue
            try:
                container.remove(force=True)
            except docker.errors.NotFound:
                pass
            except docker.errors.DockerException as exc:
                errors.append(str(exc))
    except docker.errors.DockerException as exc:
        errors.append(str(exc))
    finally:
        if client is not None:
            client.close()
    return errors
