"""Tests for sandbox manager."""

from pathlib import Path
from unittest.mock import patch

import pytest

from bioagents.sandbox.sandbox_manager import SandboxManager, get_sandbox


class TestSandboxManager:
    def test_init_creates_workspace(self, tmp_path):
        with patch("bioagents.sandbox.sandbox_manager.SANDBOX_BASE_DIR", tmp_path):
            manager = SandboxManager("test_init")
            assert manager.workspace_dir.exists()
            assert manager.workspace_dir == tmp_path / "test_init"

    def test_default_workspace_id(self, tmp_path):
        with patch("bioagents.sandbox.sandbox_manager.SANDBOX_BASE_DIR", tmp_path):
            manager = SandboxManager()
            assert manager.workspace_id == "default"
            assert manager.workspace_dir == tmp_path / "default"

    def test_workdir_property(self, tmp_path):
        with patch("bioagents.sandbox.sandbox_manager.SANDBOX_BASE_DIR", tmp_path):
            manager = SandboxManager("test_prop")
            assert manager.workdir == manager.workspace_dir

    def test_write_and_read_file(self, tmp_path):
        with patch("bioagents.sandbox.sandbox_manager.SANDBOX_BASE_DIR", tmp_path):
            manager = SandboxManager("test_rw")
            manager.write_file("test.txt", "hello world")
            content = manager.read_file("test.txt")
            assert content == "hello world"

    def test_write_creates_subdirectories(self, tmp_path):
        with patch("bioagents.sandbox.sandbox_manager.SANDBOX_BASE_DIR", tmp_path):
            manager = SandboxManager("test_subdir")
            result = manager.write_file("sub/dir/file.txt", "nested")
            assert Path(result).exists()
            assert manager.read_file("sub/dir/file.txt") == "nested"

    def test_read_file_not_found(self, tmp_path):
        with patch("bioagents.sandbox.sandbox_manager.SANDBOX_BASE_DIR", tmp_path):
            manager = SandboxManager("test_notfound")
            with pytest.raises(FileNotFoundError):
                manager.read_file("nonexistent.txt")

    def test_list_directory(self, tmp_path):
        with patch("bioagents.sandbox.sandbox_manager.SANDBOX_BASE_DIR", tmp_path):
            manager = SandboxManager("test_ls")
            manager.write_file("a.txt", "a")
            manager.write_file("b.txt", "b")
            entries = manager.list_directory()
            assert len(entries) == 2
            names = {e["name"] for e in entries}
            assert names == {"a.txt", "b.txt"}

    def test_list_directory_empty(self, tmp_path):
        with patch("bioagents.sandbox.sandbox_manager.SANDBOX_BASE_DIR", tmp_path):
            manager = SandboxManager("test_ls_empty")
            entries = manager.list_directory()
            assert entries == []

    def test_list_directory_nonexistent(self, tmp_path):
        with patch("bioagents.sandbox.sandbox_manager.SANDBOX_BASE_DIR", tmp_path):
            manager = SandboxManager("test_ls_none")
            entries = manager.list_directory("no_such_dir")
            assert entries == []

    def test_file_exists_true(self, tmp_path):
        with patch("bioagents.sandbox.sandbox_manager.SANDBOX_BASE_DIR", tmp_path):
            manager = SandboxManager("test_exists")
            manager.write_file("present.txt", "data")
            assert manager.file_exists("present.txt") is True

    def test_file_exists_false(self, tmp_path):
        with patch("bioagents.sandbox.sandbox_manager.SANDBOX_BASE_DIR", tmp_path):
            manager = SandboxManager("test_exists_no")
            assert manager.file_exists("absent.txt") is False

    def test_run_command(self, tmp_path):
        with patch("bioagents.sandbox.sandbox_manager.SANDBOX_BASE_DIR", tmp_path):
            manager = SandboxManager("test_cmd")
            result = manager.run_command("echo hello")
            assert result["success"] is True
            assert "hello" in result["stdout"]
            assert result["returncode"] == 0

    def test_run_command_failure(self, tmp_path):
        with patch("bioagents.sandbox.sandbox_manager.SANDBOX_BASE_DIR", tmp_path):
            manager = SandboxManager("test_cmd_fail")
            result = manager.run_command("false")
            assert result["success"] is False
            assert result["returncode"] != 0

    def test_run_command_timeout(self, tmp_path):
        with patch("bioagents.sandbox.sandbox_manager.SANDBOX_BASE_DIR", tmp_path):
            manager = SandboxManager("test_timeout")
            result = manager.run_command("sleep 10", timeout=1)
            assert result["success"] is False
            assert "timed out" in result["stderr"].lower()

    def test_command_history(self, tmp_path):
        with patch("bioagents.sandbox.sandbox_manager.SANDBOX_BASE_DIR", tmp_path):
            manager = SandboxManager("test_hist")
            manager.run_command("echo test1")
            manager.run_command("echo test2")
            history = manager.get_command_history()
            assert len(history) == 2
            assert history[0]["command"] == "echo test1"
            assert history[1]["command"] == "echo test2"

    def test_install_package_pip(self, tmp_path):
        with patch("bioagents.sandbox.sandbox_manager.SANDBOX_BASE_DIR", tmp_path):
            manager = SandboxManager("test_pip")
            with patch.object(
                manager, "run_command", return_value={"success": True, "stdout": "ok", "stderr": ""}
            ) as mock_run:
                result = manager.install_package("numpy")
                mock_run.assert_called_once_with("pip install numpy")
                assert result["success"] is True

    def test_install_package_unknown_manager(self, tmp_path):
        with patch("bioagents.sandbox.sandbox_manager.SANDBOX_BASE_DIR", tmp_path):
            manager = SandboxManager("test_bad_mgr")
            result = manager.install_package("numpy", manager="unknown")
            assert result["success"] is False

    def test_git_clone(self, tmp_path):
        with patch("bioagents.sandbox.sandbox_manager.SANDBOX_BASE_DIR", tmp_path):
            manager = SandboxManager("test_clone")
            with patch.object(
                manager, "run_command", return_value={"success": True, "stdout": "", "stderr": ""}
            ) as mock_run:
                result = manager.git_clone("https://github.com/user/repo.git")
                mock_run.assert_called_once_with(
                    "git clone --depth 1 https://github.com/user/repo.git"
                )
                assert result["success"] is True

    def test_git_clone_with_target(self, tmp_path):
        with patch("bioagents.sandbox.sandbox_manager.SANDBOX_BASE_DIR", tmp_path):
            manager = SandboxManager("test_clone_dir")
            with patch.object(
                manager, "run_command", return_value={"success": True, "stdout": "", "stderr": ""}
            ) as mock_run:
                manager.git_clone("https://github.com/user/repo.git", "my_repo")
                mock_run.assert_called_once_with(
                    "git clone --depth 1 https://github.com/user/repo.git my_repo"
                )

    def test_cleanup(self, tmp_path):
        with patch("bioagents.sandbox.sandbox_manager.SANDBOX_BASE_DIR", tmp_path):
            manager = SandboxManager("test_cleanup")
            manager.write_file("temp.txt", "temp")
            assert manager.workspace_dir.exists()
            manager.cleanup()
            assert not manager.workspace_dir.exists()


class TestGetSandbox:
    def test_get_sandbox_singleton(self, tmp_path):
        with patch("bioagents.sandbox.sandbox_manager.SANDBOX_BASE_DIR", tmp_path):
            import bioagents.sandbox.sandbox_manager as sm

            sm._default_sandbox = None
            try:
                sandbox1 = get_sandbox()
                sandbox2 = get_sandbox()
                assert sandbox1 is sandbox2
            finally:
                sm._default_sandbox = None

    def test_get_sandbox_with_id_returns_new(self, tmp_path):
        with patch("bioagents.sandbox.sandbox_manager.SANDBOX_BASE_DIR", tmp_path):
            import bioagents.sandbox.sandbox_manager as sm

            sm._default_sandbox = None
            try:
                sandbox1 = get_sandbox("workspace_a")
                sandbox2 = get_sandbox("workspace_b")
                assert sandbox1 is not sandbox2
                assert sandbox1.workspace_id == "workspace_a"
                assert sandbox2.workspace_id == "workspace_b"
            finally:
                sm._default_sandbox = None
