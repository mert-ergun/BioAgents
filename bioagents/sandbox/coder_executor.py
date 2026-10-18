"""Docker executor setup for coder agent."""

import contextlib
import importlib.util
import json
import logging
import os
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


SANDBOX_IMAGE = os.getenv("BIOAGENTS_SANDBOX_IMAGE", "bioagents-sandbox")

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
    return union


def build_dockerfile(packages: list[str] | None = None) -> str:
    """Dockerfile for the shared sandbox image."""
    packages = sandbox_packages() if packages is None else packages
    others = [p for p in packages if p != "torch"]
    kernel = "jupyter_kernel_gateway jupyter_client ipykernel"

    if "torch" in packages and not _SANDBOX_GPU:
        torch_line = (
            f"RUN pip install --no-cache-dir --index-url {_TORCH_CPU_INDEX} torch\n        "
        )
    elif "torch" in packages:
        torch_line = "RUN pip install --no-cache-dir torch\n        "
    else:
        torch_line = ""

    return f"""
        FROM python:3.12-slim

        WORKDIR /app

        # The container runs as the host uid, which has no passwd entry and so no home.
        # Jupyter, matplotlib and pip all need a writable one or the kernel never starts.
        ENV PYTHONPATH=/app:$PYTHONPATH \\
            PIP_NO_CACHE_DIR=1 \\
            PIP_DISABLE_PIP_VERSION_CHECK=1 \\
            HOME=/tmp/sandbox-home \\
            JUPYTER_RUNTIME_DIR=/tmp/sandbox-home/jupyter/runtime \\
            JUPYTER_DATA_DIR=/tmp/sandbox-home/jupyter/data \\
            XDG_CACHE_HOME=/tmp/sandbox-home/cache \\
            MPLCONFIGDIR=/tmp/sandbox-home/matplotlib

        RUN mkdir -p /tmp/sandbox-home/jupyter/runtime /tmp/sandbox-home/jupyter/data \\
            /tmp/sandbox-home/cache /tmp/sandbox-home/matplotlib && chmod -R 777 /tmp/sandbox-home

        {torch_line}RUN pip install --no-cache-dir {kernel} {" ".join(others)}

        EXPOSE 8888
        CMD ["jupyter", "kernelgateway", "--KernelGatewayApp.ip='0.0.0.0'", "--KernelGatewayApp.port=8888", "--KernelGatewayApp.allow_origin='*'"]
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
    docker_path = shutil.which("docker")
    if not docker_path:
        return False
    try:
        subprocess.run(  # nosec B603
            [docker_path, "image", "inspect", image_name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return False
    return True


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

            # Run as the host user: files the container writes into the shared
            # workspace were root-owned, so a host-side agent editing the same artifact
            # got PermissionError and treated the finished work as a stale relic to
            # archive and redo. BIOAGENTS_SANDBOX_UID=0 restores running as root.
            container_run_kwargs: dict[str, Any] = {
                "volumes": {str(Path.cwd()): {"bind": "/app", "mode": "rw"}},
                "environment": {"BIOAGENTS_BLAST_IN_CELL": "1"},
            }
            user = _sandbox_user()
            if user:
                container_run_kwargs["user"] = user
            if _SANDBOX_GPU:
                from docker.types import DeviceRequest

                container_run_kwargs["device_requests"] = [
                    DeviceRequest(count=-1, capabilities=[["gpu"]])
                ]

            from bioagents.sandbox.resilient_executor import ResilientDockerExecutor

            dockerfile = build_dockerfile()
            image = sandbox_image_tag(dockerfile)
            return ResilientDockerExecutor(
                image_name=image,
                dockerfile_content=dockerfile,
                additional_imports=[],
                logger=logger,
                port=find_available_port(),
                build_new_image=not _image_exists(image),
                container_run_kwargs=container_run_kwargs,
            )
        except Exception as e:
            print(f"Warning: Could not initialize DockerExecutor for {agent_type}: {e}")
            print("Falling back to LocalExecutor (NOT SANDBOXED) for development purposes.")
            return _local_executor(additional_imports)

    return LazyExecutor(factory)
