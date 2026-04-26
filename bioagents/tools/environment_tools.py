"""Environment tools for managing virtual environments, packages, and system info."""

import json

from langchain_core.tools import tool

from bioagents.sandbox.sandbox_manager import get_sandbox


@tool
def create_virtual_environment(env_name: str = "bioagents_env") -> str:
    """Create a Python virtual environment in the sandbox.

    Args:
        env_name: Name for the virtual environment (default 'bioagents_env').

    Returns:
        Confirmation message with activation instructions.
    """
    try:
        sandbox = get_sandbox()
        result = sandbox.run_command(f"python -m venv {env_name}", timeout=60)
        if result["success"]:
            activate_path = f"{env_name}/bin/activate"
            return (
                f"Virtual environment '{env_name}' created successfully.\n"
                f"Activate with: source {activate_path}\n"
                f"Location: {sandbox.workdir / env_name}"
            )
        return f"Failed to create virtual environment: {result['stderr']}"
    except Exception as e:
        return f"Error creating virtual environment: {e}"


@tool
def install_requirements(requirements: str) -> str:
    """Install Python packages from a requirements string or file.

    Args:
        requirements: Either a path to a requirements.txt file, or a
                      newline/comma-separated list of package specifiers
                      (e.g. 'numpy>=1.20,pandas,scipy').

    Returns:
        Installation results with success/failure details.
    """
    try:
        sandbox = get_sandbox()

        if sandbox.file_exists(requirements):
            cmd = f"pip install -r {requirements}"
        else:
            packages = [p.strip() for p in requirements.replace(",", "\n").split("\n") if p.strip()]
            if not packages:
                return "Error: No packages specified."
            req_content = "\n".join(packages)
            sandbox.write_file("_tmp_requirements.txt", req_content)
            cmd = "pip install -r _tmp_requirements.txt"

        result = sandbox.run_command(cmd, timeout=180)

        if result["success"]:
            output = result["stdout"]
            if len(output) > 3000:
                lines = output.strip().split("\n")
                installed = [
                    line
                    for line in lines
                    if "Successfully installed" in line or "already satisfied" in line.lower()
                ]
                output = "\n".join(installed) if installed else lines[-20:]
            return f"Installation successful.\n{output}"
        return f"Installation failed:\n{result['stderr']}"
    except Exception as e:
        return f"Error installing requirements: {e}"


@tool
def check_gpu_available() -> str:
    """Check if a GPU is available and return GPU information.

    Returns:
        JSON string with GPU availability, device name, memory, and CUDA version.
    """
    try:
        sandbox = get_sandbox()

        nvidia_result = sandbox.run_command(
            "nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader",
            timeout=15,
        )
        if nvidia_result["success"] and nvidia_result["stdout"].strip():
            lines = nvidia_result["stdout"].strip().split("\n")
            gpus = []
            for i, line in enumerate(lines):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpus.append(
                        {
                            "index": i,
                            "name": parts[0],
                            "memory_total": parts[1],
                            "memory_free": parts[2],
                            "driver_version": parts[3],
                        }
                    )

            cuda_result = sandbox.run_command("nvcc --version 2>/dev/null | tail -1", timeout=10)
            cuda_version = cuda_result["stdout"].strip() if cuda_result["success"] else "unknown"

            return json.dumps(
                {
                    "gpu_available": True,
                    "num_gpus": len(gpus),
                    "gpus": gpus,
                    "cuda_info": cuda_version,
                },
                indent=2,
            )

        torch_result = sandbox.run_command(
            "python -c \"import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')\"",
            timeout=15,
        )
        if torch_result["success"] and "True" in torch_result["stdout"]:
            lines = torch_result["stdout"].strip().split("\n")
            return json.dumps(
                {
                    "gpu_available": True,
                    "detected_via": "pytorch",
                    "device_name": lines[1] if len(lines) > 1 else "unknown",
                },
                indent=2,
            )

        return json.dumps({"gpu_available": False, "message": "No GPU detected."}, indent=2)
    except Exception as e:
        return f"Error checking GPU: {e}"


@tool
def get_system_info() -> str:
    """Get system information from the sandbox environment.

    Returns:
        JSON string with OS, Python version, CPU, memory, and disk usage.
    """
    try:
        sandbox = get_sandbox()
        info: dict = {}

        py_result = sandbox.run_command("python --version 2>&1", timeout=10)
        info["python_version"] = py_result["stdout"].strip() if py_result["success"] else "unknown"

        uname_result = sandbox.run_command("uname -srm 2>/dev/null", timeout=10)
        info["os"] = uname_result["stdout"].strip() if uname_result["success"] else "unknown"

        cpu_result = sandbox.run_command(
            "nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null", timeout=10
        )
        info["cpu_cores"] = cpu_result["stdout"].strip() if cpu_result["success"] else "unknown"

        mem_result = sandbox.run_command("free -h 2>/dev/null | head -2", timeout=10)
        if mem_result["success"] and mem_result["stdout"].strip():
            lines = mem_result["stdout"].strip().split("\n")
            if len(lines) >= 2:
                parts = lines[1].split()
                info["memory"] = {
                    "total": parts[1] if len(parts) > 1 else "unknown",
                    "used": parts[2] if len(parts) > 2 else "unknown",
                    "available": parts[6] if len(parts) > 6 else "unknown",
                }

        disk_result = sandbox.run_command(
            f"df -h {sandbox.workdir} 2>/dev/null | tail -1", timeout=10
        )
        if disk_result["success"] and disk_result["stdout"].strip():
            parts = disk_result["stdout"].strip().split()
            info["disk"] = {
                "total": parts[1] if len(parts) > 1 else "unknown",
                "used": parts[2] if len(parts) > 2 else "unknown",
                "available": parts[3] if len(parts) > 3 else "unknown",
                "use_pct": parts[4] if len(parts) > 4 else "unknown",
            }

        info["sandbox_workdir"] = str(sandbox.workdir)

        return json.dumps(info, indent=2)
    except Exception as e:
        return f"Error getting system info: {e}"


def get_environment_tools() -> list:
    """Return all environment tools."""
    return [create_virtual_environment, install_requirements, check_gpu_available, get_system_info]
