"""Existing sandbox images must start Jupyter even if their CMD was changed."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
import smolagents.remote_executors as remote

import docker
from bioagents.sandbox import coder_executor, resilient_executor


@pytest.mark.parametrize("agent_type", ["coder", "ml", "dl"])
def test_existing_image_cmd_is_overridden_without_build_or_pull(monkeypatch, tmp_path, agent_type):
    monkeypatch.setenv("USE_LOCAL_EXECUTOR", "false")
    monkeypatch.setenv("BIOAGENTS_SANDBOX_IMAGE", f"bioagents-{agent_type}")
    monkeypatch.setenv("BIOAGENTS_DOCKER_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("BIOAGENTS_DOCKER_INHERIT_VOLUMES", "true")
    monkeypatch.setenv("HOSTNAME", "parent-container")
    monkeypatch.setattr(coder_executor.shutil, "which", lambda name: "/usr/bin/docker")
    inspect = Mock(return_value=SimpleNamespace(stdout="sha256:existing-image\n"))
    monkeypatch.setattr(coder_executor.subprocess, "run", inspect)
    monkeypatch.setattr(coder_executor, "find_available_port", lambda: 8892)
    client = Mock()
    image = SimpleNamespace(attrs={"Config": {"Cmd": ["pip", "install", "openbabel-wheel"]}})
    client.images.get.return_value = image

    def run(image_id, **kwargs):
        # Model Docker's command selection: without an override the one-off pip
        # command exits rather than serving a persistent Jupyter API.
        effective_command = kwargs.get("command", image.attrs["Config"]["Cmd"])
        assert effective_command[:2] == ["jupyter", "kernelgateway"]
        assert "--KernelGatewayApp.ip=0.0.0.0" in effective_command
        assert "--KernelGatewayApp.port=8888" in effective_command
        assert image_id == "sha256:existing-image"
        assert kwargs["volumes"] == {str(tmp_path): {"bind": "/app", "mode": "rw"}}
        assert kwargs["volumes_from"] == ["parent-container"]
        assert kwargs["ports"] == {"8888/tcp": ("127.0.0.1", 8892)}
        assert kwargs["detach"] is True
        return SimpleNamespace(status="running", short_id="test-container")

    client.containers.run.side_effect = run
    monkeypatch.setattr(docker, "from_env", lambda: client)
    monkeypatch.setattr(requests, "get", Mock(return_value=SimpleNamespace(status_code=200)))
    kernel = Mock(return_value="test-kernel")
    monkeypatch.setattr(remote, "_create_kernel_http", kernel)
    monkeypatch.setattr(resilient_executor, "_create_kernel_http", kernel)

    executor = coder_executor.create_executor(agent_type, additional_imports=["numpy"])

    assert not executor.started
    assert executor.kernel_id == "test-kernel"
    assert executor.installed_packages == []
    assert executor.ws_url == "ws://127.0.0.1:8892/api/kernels/test-kernel/channels"
    assert inspect.call_args.args[0] == [
        "/usr/bin/docker",
        "image",
        "inspect",
        "--format",
        "{{.Id}}",
        f"bioagents-{agent_type}",
    ]
    client.images.get.assert_called_once_with("sha256:existing-image")
    client.images.build.assert_not_called()
    client.images.pull.assert_not_called()
    client.images.remove.assert_not_called()
    kernel.assert_called_once_with("http://127.0.0.1:8892/api/kernels", executor.logger)


@pytest.mark.parametrize("driver", [None, "auto", "cdi", "nvidia"])
def test_gpu_uses_configured_driver(monkeypatch, driver):
    from bioagents.sandbox import resilient_executor

    monkeypatch.setenv("USE_LOCAL_EXECUTOR", "false")
    if driver is None:
        monkeypatch.delenv("BIOAGENTS_SANDBOX_GPU_DRIVER", raising=False)
    else:
        monkeypatch.setenv("BIOAGENTS_SANDBOX_GPU_DRIVER", driver)
    monkeypatch.setattr(coder_executor, "_SANDBOX_GPU", True)
    monkeypatch.setattr(coder_executor, "_existing_image_id", lambda _: "sha256:existing")
    monkeypatch.setattr(coder_executor, "find_available_port", lambda: 8892)
    remote = Mock()
    monkeypatch.setattr(resilient_executor, "ResilientDockerExecutor", remote)
    coder_executor.create_executor("coder", [])("pass")
    request = remote.call_args.kwargs["container_run_kwargs"]["device_requests"][0]
    assert request["Driver"] == (driver if driver in ("cdi", "nvidia") else "")
    if driver == "cdi":
        assert request["DeviceIDs"] == ["nvidia.com/gpu=all"]
        assert request["Count"] == 0
        assert request["Capabilities"] == []
    else:
        assert request["Count"] == -1
        assert request["Capabilities"] == [["gpu"]]
