"""Honest capability reporting for tools that cannot do the work they advertise.

A tool must never return a success string for a computation that did not happen.
When a model or binary is unavailable, the agent needs three things to behave
correctly: (1) an unambiguous failure status, (2) what exactly is missing, and
(3) what to do instead. This module provides that contract so every unavailable
capability reports it the same way.
"""

from __future__ import annotations

import json
from typing import Any

# Appended to every unavailable-capability payload. The agents' system prompts
# reinforce this, but repeating it in the tool result puts the instruction in the
# same place as the evidence, which is where it is most likely to be followed.
NO_FABRICATION_DIRECTIVE = (
    "NO computation was performed. Do not report, estimate, approximate or infer "
    "results for this capability. Either use the suggested alternative, or state "
    "plainly to the user that this capability is unavailable."
)


def capability_unavailable(
    capability: str,
    reason: str,
    *,
    use_instead: str | None = None,
    how_to_enable: str | None = None,
    **context: Any,
) -> str:
    """Return a structured 'this did not run' result.

    Args:
        capability: Human-readable name of what was requested, e.g. 'RFdiffusion'.
        reason: Precise reason it could not run (missing dependency, key, binary, GPU).
        use_instead: A real tool that can produce a comparable result, if one exists.
        how_to_enable: Concrete steps an operator can take to make this work.
        **context: Extra fields echoed back (paths, lengths, provider names).

    Returns:
        A JSON string with status='error' and error_type='capability_unavailable'.
    """
    payload: dict[str, Any] = {
        "status": "error",
        "error_type": "capability_unavailable",
        "capability": capability,
        "message": reason,
        "performed_any_computation": False,
        "directive": NO_FABRICATION_DIRECTIVE,
    }
    if use_instead:
        payload["use_instead"] = use_instead
    if how_to_enable:
        payload["how_to_enable"] = how_to_enable
    payload.update(context)
    return json.dumps(payload, indent=2)


def missing_dependency(
    capability: str,
    modules: list[str],
    *,
    install_hint: str,
    use_instead: str | None = None,
    **context: Any,
) -> str:
    """Report that a capability is blocked by missing Python dependencies."""
    return capability_unavailable(
        capability,
        reason=(
            f"{capability} cannot run: required Python module(s) not installed: "
            f"{', '.join(modules)}."
        ),
        use_instead=use_instead,
        how_to_enable=install_hint,
        missing_modules=modules,
        **context,
    )


def missing_binary(
    capability: str,
    binary: str,
    *,
    install_hint: str,
    use_instead: str | None = None,
    **context: Any,
) -> str:
    """Report that a capability is blocked by a missing external executable."""
    return capability_unavailable(
        capability,
        reason=f"{capability} cannot run: executable '{binary}' was not found on PATH.",
        use_instead=use_instead,
        how_to_enable=install_hint,
        missing_binary=binary,
        **context,
    )


def hosted_model_unavailable(
    capability: str,
    provider: str,
    *,
    use_instead: str | None = None,
    **context: Any,
) -> str:
    """Report that a hosted-only model has no client wired up in this deployment.

    Deliberately does not prompt for an API key: with no client implemented, a key
    would not make the call work, and asking for one implies a capability that does
    not exist.
    """
    return capability_unavailable(
        capability,
        reason=(
            f"{capability} has no local implementation and no API client for provider "
            f"'{provider}' is wired up in this deployment."
        ),
        use_instead=use_instead,
        how_to_enable=(
            f"Implement an API client for {provider} and register it, or run a local "
            f"equivalent model."
        ),
        provider=provider,
        **context,
    )


def check_modules(modules: list[str]) -> list[str]:
    """Return the subset of ``modules`` that cannot be imported."""
    import importlib.util

    missing: list[str] = []
    for name in modules:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ModuleNotFoundError, ValueError):
            missing.append(name)
    return missing


def cuda_is_usable() -> bool:
    """Return True only if torch can actually launch a kernel on the local GPU.

    ``torch.cuda.is_available()`` reports True for a GPU whose compute capability the
    installed torch build was not compiled for (e.g. an sm_120 card on a build topping
    out at sm_90). Such a device then fails at the first real kernel launch with
    ``no kernel image is available for execution on the device``. Probing with an
    actual operation is the only reliable check.
    """
    import os

    override = os.getenv("BIOAGENTS_FORCE_CPU")
    if override and override.lower() in ("1", "true", "yes"):
        return False
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        torch.zeros(8, device="cuda").sum().item()
        return True
    except Exception:
        return False


def subprocess_env_for_torch() -> dict:
    """Build an environment for a torch subprocess, hiding an unusable GPU.

    Child processes get their own torch context, so an unusable CUDA device must be
    masked via ``CUDA_VISIBLE_DEVICES`` before the child starts.
    """
    import os

    env = dict(os.environ)
    if not cuda_is_usable():
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env
