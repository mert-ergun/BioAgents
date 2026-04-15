"""Docker executor setup for coder agent."""

import contextlib
import importlib.util
import os
import shutil
import socket
import subprocess  # nosec B404
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Monkey-patch smolagents evaluate_with — upstream stores __enter__() return
# in the contexts list, then calls __exit__ on it.  Context managers whose
# __enter__ returns None (e.g. torch.no_grad()) crash with
# "NoneType has no attribute __exit__".  Fix: store the context manager itself.
# ---------------------------------------------------------------------------
import smolagents.local_python_executor as _lpe
from rich.console import Console
from smolagents import AgentLogger, DockerExecutor, LocalPythonExecutor, LogLevel

# ---------------------------------------------------------------------------
# Monkey-patch RemotePythonExecutor._patch_final_answer_with_exception to be
# idempotent.  smolagents calls this on every agent.run() invocation via
# send_tools(), but the patch is not safe to apply twice: the second call
# grabs the already-patched `forward` (which calls self._forward) as the
# "original forward", stores it as `_forward`, and from that point both
# `forward` and `_forward` call `self._forward` → infinite recursion.
#
# The fix: if the tool's class is already named `_FinalAnswerTool` (i.e. has
# already been patched), skip the re-patch entirely.
# ---------------------------------------------------------------------------
from smolagents.remote_executors import RemotePythonExecutor as _RPE

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
    import json

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
    - Masks GPUs unless BIOAGENTS_TORCH_CUDA=1 so many stacks fall back to CPU.
    - Patches ``json.JSONEncoder`` to handle numpy scalar types.
    """
    _patch_json_numpy_serialization()
    if os.getenv("BIOAGENTS_TORCH_CUDA", "").lower() in ("1", "true", "yes"):
        _prepend_nvidia_pip_libs_to_ld_path()
        return
    _prepend_nvidia_pip_libs_to_ld_path()
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


def create_executor(
    agent_type: str, additional_imports: list
) -> DockerExecutor | LocalPythonExecutor:
    """
    Create a Docker executor for code execution, falling back to LocalExecutor if Docker fails.

    Can be forced to use LocalPythonExecutor by setting USE_LOCAL_EXECUTOR=true environment variable.

    Args:
        agent_type: Type of agent (e.g., 'coder', 'ml', 'dl')
        additional_imports: List of Python packages to install in the Docker image

    Returns:
        DockerExecutor or LocalPythonExecutor instance
    """
    # Check if local executor should be used (via environment variable)
    use_local = os.getenv("USE_LOCAL_EXECUTOR", "false").lower() in ("true", "1", "yes")

    if use_local:
        apply_local_executor_runtime_env()
        print(f"Using LocalPythonExecutor for {agent_type} (USE_LOCAL_EXECUTOR=true)")
        from bioagents.sandbox.coder_helpers import PermissiveList

        return LocalPythonExecutor(
            additional_authorized_imports=PermissiveList(additional_imports),
            additional_functions=_extra_builtins(),
        )

    image_name = f"bioagents-{agent_type}"
    try:
        terminal_console = Console(
            file=sys.stdout,
            width=None,
            force_terminal=True,
        )
        logger = AgentLogger(level=LogLevel.INFO, console=terminal_console)
        port = find_available_port()

        # Filter imports for pip install (remove wildcards and stdlib modules)
        pip_packages = []
        package_mapping = {
            "Bio": "biopython",
            "sklearn": "scikit-learn",
            "pkg_resources": "setuptools",
        }
        # Common standard library modules to exclude from pip install
        stdlib_modules = {
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
        for imp in additional_imports:
            base_pkg = imp[:-2] if imp.endswith(".*") else imp

            if base_pkg in stdlib_modules:
                continue
            if base_pkg.startswith("bioagents"):
                continue

            pkg_to_install = package_mapping.get(base_pkg, base_pkg)
            if pkg_to_install not in pip_packages:
                pip_packages.append(pkg_to_install)

        dockerfile_content = f"""
        FROM python:3.12-bullseye

        WORKDIR /app

        ENV PYTHONPATH=/app:$PYTHONPATH

        RUN pip install jupyter_kernel_gateway jupyter_client ipykernel {" ".join(pip_packages)}

        EXPOSE 8888
        CMD ["jupyter", "kernelgateway", "--KernelGatewayApp.ip='0.0.0.0'", "--KernelGatewayApp.port=8888", "--KernelGatewayApp.allow_origin='*'"]
        """

        docker_path = shutil.which("docker")
        if docker_path:
            try:
                subprocess.run(  # nosec B603
                    [docker_path, "image", "inspect", image_name],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                should_build = False
            except subprocess.CalledProcessError:
                should_build = True
        else:
            should_build = True

        cwd = str(Path.cwd())
        container_run_kwargs = {"volumes": {cwd: {"bind": "/app", "mode": "rw"}}}

        return DockerExecutor(
            image_name=image_name,
            dockerfile_content=dockerfile_content,
            additional_imports=[],
            logger=logger,
            port=port,
            build_new_image=should_build,
            container_run_kwargs=container_run_kwargs,
        )
    except Exception as e:
        print(f"Warning: Could not initialize DockerExecutor for {agent_type}: {e}")
        print("Falling back to LocalExecutor (NOT SANDBOXED) for development purposes.")

        apply_local_executor_runtime_env()
        from bioagents.sandbox.coder_helpers import PermissiveList

        return LocalPythonExecutor(
            additional_authorized_imports=PermissiveList(additional_imports),
            additional_functions=_extra_builtins(),
        )
