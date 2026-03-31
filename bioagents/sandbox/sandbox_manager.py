"""Sandbox environment manager for BioAgents.

Provides a complete sandboxed execution environment with:
- Shell command execution (git clone, pip install, wget, etc.)
- File system operations (read/write/list within sandbox)
- Network access (download files, fetch URLs)
- Package management (pip/conda install)
- Persistent volumes across agent calls
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

SANDBOX_BASE_DIR = Path(os.getenv("BIOAGENTS_SANDBOX_DIR", "sandbox_workdir"))
MAX_COMMAND_TIMEOUT = 300  # 5 minutes default


class SandboxManager:
    """Manages a sandboxed execution environment for agents."""

    def __init__(self, workspace_id: str | None = None):
        self.workspace_id = workspace_id or "default"
        self.workspace_dir = (SANDBOX_BASE_DIR / self.workspace_id).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self._command_history: list[dict] = []
        logger.info(f"Sandbox initialized at {self.workspace_dir}")

    @property
    def workdir(self) -> Path:
        return self.workspace_dir

    def run_command(
        self,
        command: str,
        timeout: int = MAX_COMMAND_TIMEOUT,
        cwd: str | None = None,
        env: dict | None = None,
    ) -> dict:
        """Execute a shell command in the sandbox.

        Args:
            command: Shell command to execute
            timeout: Max seconds to wait
            cwd: Working directory (relative to sandbox root)
            env: Additional environment variables

        Returns:
            Dict with stdout, stderr, returncode, success
        """
        if cwd:
            work_dir = Path(cwd)
            if not work_dir.is_absolute():
                work_dir = self.workspace_dir / work_dir
        else:
            work_dir = self.workspace_dir

        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        logger.info(f"Sandbox exec: {command[:200]}")

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=run_env,
            )
            record = {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout,
                "stderr": result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr,
                "success": result.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            record = {
                "command": command,
                "returncode": -1,
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "success": False,
            }
        except Exception as e:
            record = {
                "command": command,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "success": False,
            }

        self._command_history.append(record)
        return record

    def install_package(self, package: str, manager: str = "pip") -> dict:
        """Install a Python package in the sandbox."""
        if manager == "pip":
            return self.run_command(f"pip install {package}")
        elif manager == "conda":
            return self.run_command(f"conda install -y {package}")
        else:
            return {"success": False, "stderr": f"Unknown package manager: {manager}"}

    def git_clone(self, repo_url: str, target_dir: str | None = None) -> dict:
        """Clone a git repository into the sandbox."""
        cmd = f"git clone --depth 1 {repo_url}"
        if target_dir:
            cmd += f" {target_dir}"
        return self.run_command(cmd)

    def _normalize_path(self, path: str) -> str:
        """Strip workspace_dir prefix from a path so it won't be doubled.

        When the sandbox cwd is already workspace_dir, a relative path like
        ``sandbox_workdir/default/file.txt`` would resolve to
        ``sandbox_workdir/default/sandbox_workdir/default/file.txt``.
        """
        p = Path(path)
        if p.is_absolute():
            return path
        try:
            rel = p.relative_to(self.workspace_dir)
            return str(rel)
        except ValueError:
            pass
        try:
            rel = p.relative_to(SANDBOX_BASE_DIR)
            return str(rel)
        except ValueError:
            return path

    def download_file(self, url: str, output_path: str | None = None) -> dict:
        """Download a file from a URL."""
        if output_path:
            output_path = self._normalize_path(output_path)
            parent = Path(output_path).parent
            if str(parent) not in (".", ""):
                self.run_command(f"mkdir -p '{parent}'")
            cmd = f"wget -q -O '{output_path}' '{url}'"
        else:
            cmd = f"wget -q '{url}'"
        result = self.run_command(cmd)
        if not result["success"]:
            if output_path:
                cmd = f"curl -sL -o '{output_path}' '{url}'"
            else:
                cmd = f"curl -sLO '{url}'"
            result = self.run_command(cmd)
        return result

    def read_file(self, path: str) -> str:
        """Read a file from the sandbox."""
        file_path = Path(self._normalize_path(path))
        if not file_path.is_absolute():
            file_path = self.workspace_dir / file_path
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return file_path.read_text(errors="replace")

    def write_file(self, path: str, content: str) -> str:
        """Write content to a file in the sandbox."""
        file_path = Path(self._normalize_path(path))
        if not file_path.is_absolute():
            file_path = self.workspace_dir / file_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return str(file_path)

    def list_directory(self, path: str = ".") -> list[dict]:
        """List contents of a directory in the sandbox."""
        dir_path = Path(self._normalize_path(path))
        if not dir_path.is_absolute():
            dir_path = self.workspace_dir / dir_path
        if not dir_path.exists():
            return []

        entries = []
        for item in sorted(dir_path.iterdir()):
            entries.append(
                {
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else 0,
                }
            )
        return entries

    def file_exists(self, path: str) -> bool:
        """Check if a file exists in the sandbox."""
        file_path = Path(self._normalize_path(path))
        if not file_path.is_absolute():
            file_path = self.workspace_dir / file_path
        return file_path.exists()

    def get_command_history(self) -> list[dict]:
        """Return the command execution history."""
        return list(self._command_history)

    def cleanup(self):
        """Remove the sandbox workspace."""
        if self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir, ignore_errors=True)
            logger.info(f"Sandbox cleaned up: {self.workspace_dir}")


_default_sandbox: SandboxManager | None = None


def get_sandbox(workspace_id: str | None = None) -> SandboxManager:
    """Get or create the default sandbox manager."""
    global _default_sandbox
    if workspace_id:
        return SandboxManager(workspace_id)
    if _default_sandbox is None:
        _default_sandbox = SandboxManager()
    return _default_sandbox
