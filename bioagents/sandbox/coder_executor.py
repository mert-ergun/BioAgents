"""Docker executor setup for coder agent."""

import contextlib
import os
import socket
import subprocess
import sys
from typing import Union

from rich.console import Console
from smolagents import AgentLogger, DockerExecutor, LocalPythonExecutor, LogLevel


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
    raise RuntimeError(f"Could not find an available port in range {start_port}-{start_port + max_retries}")


def create_executor(additional_imports: list) -> Union[DockerExecutor, LocalPythonExecutor]:
    """
    Create a Docker executor for code execution, falling back to LocalExecutor if Docker fails.

    Args:
        additional_imports: List of Python packages to install in the Docker image

    Returns:
        DockerExecutor or LocalPythonExecutor instance
    """
    try:
        log_file_path = None
        try:
            import logging
            root_logger = logging.getLogger()
            for handler in root_logger.handlers:
                if hasattr(handler, 'log_file_path'):
                    log_file_path = handler.log_file_path
                    break
                elif hasattr(handler, 'baseFilename'):
                    log_file_path = handler.baseFilename
                    break
                elif hasattr(handler, 'stream') and hasattr(handler.stream, 'name'):
                    log_file_path = handler.stream.name
                    break
        except Exception:
            pass
        
        terminal_console = Console(
            file=sys.stdout,
            width=None,
            force_terminal=True,
        )
        logger = AgentLogger(level=LogLevel.INFO, console=terminal_console)
        port = find_available_port()
        
        # Filter imports for pip install (remove wildcards and stdlib modules)
        pip_packages = []
        for imp in additional_imports:
            if imp.endswith(".*"):
                base_pkg = imp[:-2]
                if base_pkg not in pip_packages:
                    pip_packages.append(base_pkg)
            elif imp not in ["typing", "json", "os", "os.path", "sys", "pathlib"]:
                if not imp.startswith("bioagents"):
                    pip_packages.append(imp)
        
        dockerfile_content = f"""
        FROM python:3.12-bullseye
        
        WORKDIR /app
        
        ENV PYTHONPATH=/app:$PYTHONPATH
        
        RUN pip install jupyter_kernel_gateway jupyter_client ipykernel {' '.join(pip_packages)}
        
        EXPOSE 8888
        CMD ["jupyter", "kernelgateway", "--KernelGatewayApp.ip='0.0.0.0'", "--KernelGatewayApp.port=8888", "--KernelGatewayApp.allow_origin='*'"]
        """
        
        try:
            subprocess.run(["docker", "image", "inspect", "bioagents-coder"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            should_build = False
        except subprocess.CalledProcessError:
            should_build = True

        cwd = os.getcwd()
        container_run_kwargs = {
            "volumes": {cwd: {"bind": "/app", "mode": "rw"}}
        }
        
        return DockerExecutor(
            image_name="bioagents-coder",
            dockerfile_content=dockerfile_content,
            additional_imports=[],
            logger=logger,
            port=port,
            build_new_image=should_build,
            container_run_kwargs=container_run_kwargs
        )
    except Exception as e:
        print(f"Warning: Could not initialize DockerExecutor: {e}")
        print("Falling back to LocalExecutor (NOT SANDBOXED) for development purposes.")
        return LocalPythonExecutor(additional_authorized_imports=additional_imports)

