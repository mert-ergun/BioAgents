"""The code sandbox: one shared image, built CPU-only, started only when used."""

from __future__ import annotations

import pytest

from bioagents.sandbox.coder_executor import (
    LazyExecutor,
    build_dockerfile,
    create_executor,
    sandbox_packages,
)


class FakeExecutor:
    def __init__(self):
        self.cleaned = False
        self.fatal_error = None
        self.state = {"_print_outputs": ""}

    def __call__(self, code):
        return f"ran {code}"

    def cleanup(self):
        self.cleaned = True


def test_executor_is_not_started_until_code_runs():
    started = []

    lazy = LazyExecutor(lambda: started.append(1) or FakeExecutor())

    assert lazy.started is False
    assert started == []

    lazy("1 + 1")

    assert lazy.started is True
    assert len(started) == 1


def test_status_reads_do_not_start_a_container():
    """Callers probe fatal_error to see whether anything ran; probing must not spawn."""
    started = []

    lazy = LazyExecutor(lambda: started.append(1) or FakeExecutor())

    assert lazy.fatal_error is None
    assert lazy.state == {}
    assert started == []


def test_cleanup_of_an_unused_executor_starts_nothing():
    started = []

    lazy = LazyExecutor(lambda: started.append(1) or FakeExecutor())
    lazy.cleanup()

    assert started == []


def test_cleanup_releases_a_started_executor():
    executor = FakeExecutor()
    lazy = LazyExecutor(lambda: executor)

    lazy("x = 1")
    lazy.cleanup()

    assert executor.cleaned is True


def test_attributes_pass_through_once_started():
    executor = FakeExecutor()
    executor.fatal_error = "kernel gone"
    lazy = LazyExecutor(lambda: executor)

    lazy("x = 1")

    assert lazy.fatal_error == "kernel gone"


def test_one_image_covers_coder_ml_and_dl():
    from bioagents.sandbox.coder_executor import _pip_packages
    from bioagents.sandbox.coder_helpers import (
        DEFAULT_CODER_IMPORTS,
        DEFAULT_DL_IMPORTS,
        DEFAULT_ML_IMPORTS,
    )

    packages = set(sandbox_packages())

    for imports in (DEFAULT_CODER_IMPORTS, DEFAULT_ML_IMPORTS, DEFAULT_DL_IMPORTS):
        assert set(_pip_packages(imports)) <= packages


def test_stdlib_and_project_modules_are_not_installed_as_packages():
    from bioagents.sandbox.coder_executor import _pip_packages

    packages = _pip_packages(["os.path", "json", "bioagents.*", "pandas", "Bio"])

    assert packages == ["pandas", "biopython"]


def test_image_installs_cpu_torch_by_default(monkeypatch):
    """The container gets no GPU, so CUDA wheels are 5+ GB that can never execute."""
    from bioagents.sandbox import coder_executor

    monkeypatch.setattr(coder_executor, "_SANDBOX_GPU", False)
    dockerfile = build_dockerfile()

    assert "download.pytorch.org/whl/cpu" in dockerfile
    assert "python:3.12-slim" in dockerfile
    assert "/opt/bioagents/install_environments.py" in dockerfile


def test_image_uses_cuda_wheels_when_the_sandbox_is_given_a_gpu(monkeypatch):
    import bioagents.sandbox.coder_executor as mod

    monkeypatch.setattr(mod, "_SANDBOX_GPU", True)

    dockerfile = mod.build_dockerfile()

    assert "download.pytorch.org/whl/cpu" not in dockerfile
    assert "download.pytorch.org/whl/cu130" in dockerfile


def test_automatic_image_contains_the_same_locked_inputs_as_explicit_build():
    import ast
    import base64
    import json
    import shlex
    from importlib.resources import files

    recipe = build_dockerfile()
    line = next(line.strip() for line in recipe.splitlines() if "python -c " in line)
    script = ast.parse(shlex.split(line)[3])
    encoded = next(
        node.args[0].value
        for node in ast.walk(script)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "b64decode"
    )
    payload = json.loads(base64.b64decode(encoded))
    resources = files("bioagents.sandbox").joinpath("requirements")
    for name in ("coder", "boltz", "pytdc"):
        filename = f"{name}.lock.txt"
        assert (
            payload[f"/opt/bioagents/requirements/{filename}"]
            == resources.joinpath(filename).read_text()
        )
    assert "/usr/local/bin/bioagents-python" in payload


def test_bioagents_is_never_pip_installed_into_the_image():
    """Installing the project in the kernel re-resolved it and pulled a 554 MB torch."""
    assert not any(p.startswith("bioagents") for p in sandbox_packages())


@pytest.mark.parametrize("agent_type", ["coder", "ml", "dl"])
def test_every_code_agent_gets_a_lazy_executor(agent_type, monkeypatch):
    monkeypatch.delenv("USE_LOCAL_EXECUTOR", raising=False)

    executor = create_executor(agent_type, ["pandas"])

    assert isinstance(executor, LazyExecutor)
    assert executor.started is False
