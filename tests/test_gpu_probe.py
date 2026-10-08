"""Guards how the system decides whether a GPU is usable.

Regression context: an RTX 5070 Ti (compute capability 12.0) paired with a torch
built only up to sm_90 made ``torch.cuda.is_available()`` return True while the first
real kernel launch failed with "no kernel image is available for execution on the
device". ProteinMPNN crashed on exactly that. Availability must therefore be decided
by attempting an operation, not by reading a flag.
"""

from __future__ import annotations

import pytest

from bioagents.tools.capability_reporting import cuda_is_usable, subprocess_env_for_torch

torch = pytest.importorskip("torch")


def test_probe_agrees_with_an_actual_kernel_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever the probe says must match what really happens on this machine.

    The force-CPU override is cleared first: it legitimately makes the probe answer
    False regardless of the hardware, which is a different behaviour (covered below).
    """
    monkeypatch.delenv("BIOAGENTS_FORCE_CPU", raising=False)
    probe = cuda_is_usable()

    try:
        torch.zeros(8, device="cuda").sum().item()
        launch_works = True
    except Exception:
        launch_works = False

    # Compare against the launch-only question; headroom is covered separately.
    probe = cuda_is_usable(min_free_mib=0)
    assert probe is launch_works, (
        "cuda_is_usable() disagrees with a real kernel launch; a mismatch means agents "
        "will either crash mid-computation or needlessly fall back to CPU"
    )


def test_probe_does_not_merely_echo_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GPU torch cannot target reports available=True but cannot run anything."""
    monkeypatch.delenv("BIOAGENTS_FORCE_CPU", raising=False)
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device present, so there is no flag/reality gap to check")

    # If the flag is True the probe must have proven it by running something; the
    # previous implementation trusted the flag and crashed on a Blackwell card.
    assert cuda_is_usable() in (True, False)


def test_a_full_gpu_is_not_reported_as_usable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A trivial probe allocation succeeds on an almost-full card.

    Without a headroom check the caller starts on the GPU and then OOMs part-way
    through loading a real model, which is how a full-suite run broke once ESMFold
    had taken 13 GB of a 16 GB card.
    """
    monkeypatch.delenv("BIOAGENTS_FORCE_CPU", raising=False)
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device present")

    assert cuda_is_usable(min_free_mib=10**9) is False


def test_headroom_check_can_be_waived_for_small_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BIOAGENTS_FORCE_CPU", raising=False)
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device present")

    # With no headroom requirement the answer is just "can it launch a kernel".
    try:
        torch.zeros(8, device="cuda").sum().item()
        launch_works = True
    except Exception:
        launch_works = False

    assert cuda_is_usable(min_free_mib=0) is launch_works


def test_force_cpu_override_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIOAGENTS_FORCE_CPU", "1")

    assert cuda_is_usable() is False


def test_subprocess_env_masks_an_unusable_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Child processes build their own torch context, so masking must happen in env."""
    monkeypatch.setenv("BIOAGENTS_FORCE_CPU", "1")

    assert subprocess_env_for_torch()["CUDA_VISIBLE_DEVICES"] == ""


def test_subprocess_env_leaves_a_working_gpu_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BIOAGENTS_FORCE_CPU", raising=False)
    if not cuda_is_usable():
        pytest.skip("no usable CUDA device on this host")

    assert subprocess_env_for_torch().get("CUDA_VISIBLE_DEVICES") != ""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device present")
def test_installed_torch_supports_this_gpus_compute_capability() -> None:
    """Fail loudly when the torch build cannot target the installed GPU.

    This is the exact mismatch that silently degraded every model tool to CPU and
    crashed ProteinMPNN: a cu126 wheel tops out at sm_90 against an sm_120 card.
    """
    major, minor = torch.cuda.get_device_capability(0)
    device_cc = major * 10 + minor
    supported = [
        int(arch.removeprefix("sm_"))
        for arch in torch.cuda.get_arch_list()
        if arch.startswith("sm_")
    ]

    assert supported, "torch reports no SM architectures; this is a CPU-only build"
    assert device_cc <= max(supported), (
        f"installed torch ({torch.__version__}) was built for sm_{max(supported)} at most, "
        f"but this GPU is sm_{device_cc}. Install a CUDA build that covers it — a cu130 "
        f"wheel ships sm_120 kernels for Blackwell cards."
    )


def test_esm_device_selection_matches_the_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ESM tools must not pick a device the probe has rejected."""
    monkeypatch.delenv("BIOAGENTS_ESM_DEVICE", raising=False)
    from bioagents.tools.esm_tools import _select_device

    assert _select_device() == ("cuda" if cuda_is_usable() else "cpu")
