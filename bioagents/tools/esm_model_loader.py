"""Loading of ESM protein language models for the real ESM tools.

Responsibilities:

* Resolve a model alias to a public checkpoint. Gated models (ESM3 / ESM C from
  EvolutionaryScale) have no client here, so they are mapped to the public ESM-1b /
  ESM-2 checkpoint and the substitution is recorded in a ``fallback_note``.
* Load the model WITHOUT requiring a Hugging Face token. ESM-1b and ESM-2 are
  public; a token is only passed when one is present in the environment, and a
  rejected token is retried anonymously.
* Fall back from HuggingFace ``transformers`` to the ``fair-esm`` torch.hub weights
  (dl.fbaipublicfiles.com) when the Hub is unreachable and ``fair-esm`` is installed.
* Never install packages at runtime. Presence of torch / transformers / fair-esm is
  checked with ``importlib`` and a missing package is reported by name.
* Keep the weight caches (``HF_HOME`` / ``HF_HUB_CACHE`` and ``TORCH_HOME``) healthy:
  orphaned ``*.partial`` torch.hub downloads are removed (torch never resumes them),
  and a checkpoint that fails to deserialise is deleted / force-re-downloaded once
  instead of failing forever on a truncated file.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ESM1B = "facebook/esm1b_t33_650M_UR50S"
ESM2_650M = "facebook/esm2_t33_650M_UR50D"
DEFAULT_ESM_MODEL = ESM2_650M

# Canonical aliases so an agent asking for "esm-1b" or "esm1b" gets the right checkpoint.
ESM_MODEL_ALIASES: dict[str, str] = {
    "esm1b": ESM1B,
    "esm-1b": ESM1B,
    "esm1b_t33_650m_ur50s": ESM1B,
    "esm1v": "facebook/esm1v_t33_650M_UR90S_1",
    "esm-1v": "facebook/esm1v_t33_650M_UR90S_1",
    "esm2": ESM2_650M,
    "esm-2": ESM2_650M,
    "esm2_t33_650m_ur50d": ESM2_650M,
    "esm2-small": "facebook/esm2_t12_35M_UR50D",
    "esm2-tiny": "facebook/esm2_t6_8M_UR50D",
    "esm2-large": "facebook/esm2_t36_3B_UR50D",
}

# Gated / licence-restricted models with no client in this deployment.
GATED_MODEL_ALIASES: dict[str, str] = {
    "esm3": "ESM3 (EvolutionaryScale)",
    "esm-3": "ESM3 (EvolutionaryScale)",
    "esm3-open": "ESM3-open (EvolutionaryScale)",
    "esm3_sm_open_v1": "ESM3-open (EvolutionaryScale)",
    "esm3-sm-open-v1": "ESM3-open (EvolutionaryScale)",
    "esmc": "ESM C (EvolutionaryScale)",
    "esm-c": "ESM C (EvolutionaryScale)",
    "esmc-300m": "ESM C 300M (EvolutionaryScale)",
    "esmc_300m": "ESM C 300M (EvolutionaryScale)",
    "esmc-600m": "ESM C 600M (EvolutionaryScale)",
    "esmc_600m": "ESM C 600M (EvolutionaryScale)",
}

# HuggingFace id -> fair-esm (torch.hub) loader name, for the offline/Hub-down fallback.
FAIR_ESM_NAMES: dict[str, str] = {
    ESM1B: "esm1b_t33_650M_UR50S",
    "facebook/esm1v_t33_650M_UR90S_1": "esm1v_t33_650M_UR90S_1",
    "facebook/esm2_t6_8M_UR50D": "esm2_t6_8M_UR50D",
    "facebook/esm2_t12_35M_UR50D": "esm2_t12_35M_UR50D",
    "facebook/esm2_t33_650M_UR50D": "esm2_t33_650M_UR50D",
    "facebook/esm2_t36_3B_UR50D": "esm2_t36_3B_UR50D",
}

# Pinned Hub revisions: an unpinned ref lets the Hub serve different weights later, which
# would silently change every score. Override with BIOAGENTS_ESM_REVISION to move a pin.
ESM_MODEL_REVISIONS: dict[str, str] = {
    "facebook/esm2_t6_8M_UR50D": "c731040fcd8d73dceaa04b0a8e6329b345b0f5df",
    "facebook/esm2_t12_35M_UR50D": "6fbf070e65b0b7291e7bbcd451118c216cff79d8",
    "facebook/esm2_t33_650M_UR50D": "08e4846e537177426273712802403f7ba8261b6c",
    "facebook/esm2_t36_3B_UR50D": "476b639933c8baad5ad09a60ac1a87f987b656fc",
    ESM1B: "7b37824baec4d3658e1df7479222a7c79b465b76",
}

HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_API_KEY")

# Substrings that indicate a truncated / corrupt checkpoint rather than a real error.
_CORRUPTION_MARKERS = (
    "pytorchstreamreader",
    "unexpected eof",
    "invalid load key",
    "headertoolarge",
    "incomplete metadata",
    "error while deserializing",
    "failed finding central directory",
    "not a zip file",
    "truncated",
    "eoferror",
    "ran out of input",
)
_AUTH_MARKERS = ("401", "403", "unauthorized", "forbidden", "invalid user token", "gated")


class EsmLoadError(RuntimeError):
    """Raised when no backend could load the requested ESM model."""

    def __init__(self, message: str, attempts: list[str], missing: list[str]):
        super().__init__(message)
        self.attempts = attempts
        self.missing = missing


@dataclass(frozen=True)
class ModelResolution:
    """Outcome of mapping a requested model name to a loadable public checkpoint."""

    requested: str
    model_name: str
    fallback_note: str | None = None


@dataclass
class EsmModelHandle:
    """Backend-agnostic view of a loaded ESM masked language model."""

    backend: str  # "transformers" or "fair-esm"
    model_name: str
    device: str
    model: Any
    tokenizer: Any = None  # transformers tokenizer
    alphabet: Any = None  # fair-esm alphabet
    notes: list[str] = field(default_factory=list)

    @property
    def mask_id(self) -> int:
        if self.backend == "transformers":
            return int(self.tokenizer.mask_token_id)
        return int(self.alphabet.mask_idx)

    def aa_token_id(self, aa: str) -> int:
        if self.backend == "transformers":
            return int(self.tokenizer.convert_tokens_to_ids(aa))
        return int(self.alphabet.get_idx(aa))

    def encode(self, sequence: str):
        """Token ids [1, L+2]: <cls>/<bos>, residues, <eos>. Residue i (1-based) is at index i."""
        if self.backend == "transformers":
            ids = self.tokenizer(sequence, return_tensors="pt")["input_ids"]
        else:
            _, _, ids = self.alphabet.get_batch_converter()([("seq", sequence)])
        return ids.to(self.device)

    def logits(self, token_ids):
        """Return logits [B, L+2, vocab] for a batch of token ids."""
        if self.backend == "transformers":
            return self.model(input_ids=token_ids).logits
        return self.model(token_ids, repr_layers=[], return_contacts=False)["logits"]

    def last_hidden(self, token_ids):
        """Return final-layer representations [B, L+2, dim]."""
        if self.backend == "transformers":
            return self.model.esm(input_ids=token_ids).last_hidden_state
        layer = self.model.num_layers
        out = self.model(token_ids, repr_layers=[layer], return_contacts=False)
        return out["representations"][layer]


def package_available(module: str) -> bool:
    """True when ``module`` is importable — checked without importing or installing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def hf_token() -> str | None:
    """Return a Hugging Face token from the environment, or None. Never required."""
    for var in HF_TOKEN_ENV_VARS:
        value = os.getenv(var, "").strip()
        if value:
            return value
    return None


def cache_dirs() -> dict[str, str]:
    """Resolved weight cache locations (honours HF_HOME / HF_HUB_CACHE / TORCH_HOME)."""
    dirs: dict[str, str] = {}
    hf_home = os.getenv("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    dirs["hf_hub_cache"] = os.getenv("HF_HUB_CACHE") or str(Path(hf_home) / "hub")
    torch_home = os.getenv("TORCH_HOME") or str(
        Path(os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache") / "torch"
    )
    dirs["torch_hub_checkpoints"] = str(Path(torch_home) / "hub" / "checkpoints")
    return dirs


def _model_revision(model_name: str) -> str:
    override = os.getenv("BIOAGENTS_ESM_REVISION")
    if override:
        return override
    return ESM_MODEL_REVISIONS.get(model_name, "main")


def resolve_model(model: str | None, purpose: str = "scoring") -> ModelResolution:
    """Map an alias to a public checkpoint, substituting gated models with a note.

    ``purpose`` picks the substitute for a gated model: ESM-1b for variant scoring
    (the checkpoint used by published variant-effect benchmarks), ESM-2 650M for
    embeddings.
    """
    requested = (model or "").strip()
    if not requested:
        return ModelResolution(requested="", model_name=DEFAULT_ESM_MODEL)

    key = requested.lower()
    gated_label = GATED_MODEL_ALIASES.get(key)
    if gated_label is None and key.startswith("evolutionaryscale/"):
        gated_label = f"{requested} (EvolutionaryScale)"
    if gated_label is not None:
        substitute = ESM1B if purpose == "scoring" else ESM2_650M
        note = (
            f"Requested model '{requested}' ({gated_label}) is gated: it needs an "
            f"EvolutionaryScale licence/Forge API client that is not available in this "
            f"deployment. Fell back to the public model '{substitute}' (no token "
            f"required). All numbers below are {substitute} results, NOT {gated_label} results."
        )
        return ModelResolution(requested=requested, model_name=substitute, fallback_note=note)

    return ModelResolution(requested=requested, model_name=ESM_MODEL_ALIASES.get(key, requested))


def _is_corruption_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _CORRUPTION_MARKERS)


def _is_auth_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _AUTH_MARKERS)


def _select_device() -> str:
    """Pick a torch device that can actually run kernels (see capability_reporting)."""
    if os.getenv("BIOAGENTS_ESM_DEVICE"):
        return os.environ["BIOAGENTS_ESM_DEVICE"]
    from bioagents.tools.capability_reporting import cuda_is_usable

    if cuda_is_usable():
        return "cuda"
    logger.info("CUDA unusable on this host; running ESM on CPU.")
    return "cpu"


def clean_partial_torch_hub_downloads(fair_name: str) -> list[str]:
    """Delete orphaned torch.hub ``*.partial`` files for a fair-esm model.

    ``torch.hub.download_url_to_file`` writes to ``<dst>.<uuid>.partial`` and renames
    on success; an interrupted download leaves the partial behind and is never
    resumed, so it only wastes disk.
    """
    ckpt_dir = Path(cache_dirs()["torch_hub_checkpoints"])
    if not ckpt_dir.is_dir():
        return []
    removed: list[str] = []
    for path in ckpt_dir.glob(f"{fair_name}*.partial"):
        try:
            path.unlink()
            removed.append(path.name)
        except OSError as exc:
            logger.warning("Could not remove partial download %s: %s", path, exc)
    return removed


def _remove_torch_hub_checkpoints(fair_name: str) -> list[str]:
    ckpt_dir = Path(cache_dirs()["torch_hub_checkpoints"])
    removed: list[str] = []
    for path in ckpt_dir.glob(f"{fair_name}*.pt"):
        try:
            path.unlink()
            removed.append(path.name)
        except OSError as exc:
            logger.warning("Could not remove corrupt checkpoint %s: %s", path, exc)
    return removed


def _hf_from_pretrained(model_name: str, **kwargs: Any):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)  # nosec B615
    model = AutoModelForMaskedLM.from_pretrained(model_name, **kwargs)  # nosec B615
    return tokenizer, model


def _load_transformers(model_name: str, device: str) -> EsmModelHandle:
    """Load via HuggingFace; token only if present, anonymous retry if it is rejected."""
    notes: list[str] = []
    base: dict[str, Any] = {"revision": _model_revision(model_name)}
    token = hf_token()
    kwargs = {**base, "token": token} if token else dict(base)
    try:
        tokenizer, model = _hf_from_pretrained(model_name, **kwargs)
    except Exception as exc:
        if token and _is_auth_error(exc):
            notes.append(
                "The HF token in the environment was rejected; reloaded the public "
                "model anonymously (no token is needed for ESM-1b/ESM-2)."
            )
            tokenizer, model = _hf_from_pretrained(model_name, **base, token=False)
        elif _is_corruption_error(exc):
            notes.append(
                "Cached weights were corrupt or truncated; re-downloaded them "
                "(force_download=True)."
            )
            tokenizer, model = _hf_from_pretrained(model_name, **kwargs, force_download=True)
        else:
            raise
    model.eval()
    model.to(device)
    return EsmModelHandle(
        backend="transformers",
        model_name=model_name,
        device=device,
        model=model,
        tokenizer=tokenizer,
        notes=notes,
    )


def _load_fair_esm(model_name: str, fair_name: str, device: str) -> EsmModelHandle:
    """Load via fair-esm / torch.hub weights from dl.fbaipublicfiles.com (no token)."""
    import esm

    notes: list[str] = []
    removed = clean_partial_torch_hub_downloads(fair_name)
    if removed:
        notes.append(f"Removed interrupted torch.hub downloads: {', '.join(removed)}.")
    try:
        model, alphabet = esm.pretrained.load_model_and_alphabet(fair_name)
    except Exception as exc:
        if not _is_corruption_error(exc):
            raise
        deleted = _remove_torch_hub_checkpoints(fair_name)
        notes.append(
            f"Cached torch.hub checkpoint was corrupt ({', '.join(deleted) or 'unknown'}); "
            "deleted and re-downloaded it."
        )
        model, alphabet = esm.pretrained.load_model_and_alphabet(fair_name)
    model.eval()
    model.to(device)
    return EsmModelHandle(
        backend="fair-esm",
        model_name=model_name,
        device=device,
        model=model,
        alphabet=alphabet,
        notes=notes,
    )


@lru_cache(maxsize=2)
def load_esm_model(model_name: str) -> EsmModelHandle:
    """Load (and cache) an ESM masked LM, trying transformers then fair-esm.

    Raises:
        EsmLoadError: with ``missing`` (packages / resources that are absent) and
            ``attempts`` (what each backend reported), so the caller can say exactly
            what is missing. Nothing is installed at runtime.
    """
    missing: list[str] = []
    attempts: list[str] = []
    if not package_available("torch"):
        missing.append("python package 'torch' (PyTorch) is not installed")
        raise EsmLoadError(
            "Cannot run ESM: PyTorch is not installed in this environment. It must be "
            "provided by the environment (e.g. `uv sync`); it is not installed at runtime.",
            attempts,
            missing,
        )

    import torch

    torch.set_grad_enabled(False)
    device = _select_device()

    if package_available("transformers"):
        try:
            handle = _load_transformers(model_name, device)
            logger.info("Loaded ESM model %s via transformers on %s", model_name, device)
            return handle
        except Exception as exc:
            attempts.append(f"transformers (huggingface.co, model '{model_name}'): {exc}")
    else:
        missing.append("python package 'transformers' is not installed")

    fair_name = FAIR_ESM_NAMES.get(model_name)
    if fair_name is None:
        attempts.append(f"fair-esm: no torch.hub checkpoint is known for '{model_name}'")
    elif not package_available("esm"):
        missing.append(
            "python package 'fair-esm' (module 'esm') is not installed, so the torch.hub "
            "fallback (dl.fbaipublicfiles.com) is unavailable"
        )
    else:
        try:
            handle = _load_fair_esm(model_name, fair_name, device)
            handle.notes.append(
                "HuggingFace load failed; used the fair-esm torch.hub weights instead."
            )
            logger.info("Loaded ESM model %s via fair-esm on %s", model_name, device)
            return handle
        except Exception as exc:
            attempts.append(f"fair-esm torch.hub ('{fair_name}'): {exc}")

    raise EsmLoadError(
        f"Could not load ESM model '{model_name}' with any backend.", attempts, missing
    )


def describe_load_failure(model_name: str, exc: BaseException) -> dict[str, Any]:
    """Structured, specific explanation of a load failure for tool error payloads."""
    attempts = getattr(exc, "attempts", None) or [str(exc)]
    missing = getattr(exc, "missing", None) or []
    return {
        "message": f"Could not load ESM model '{model_name}'. NO scores were computed.",
        "missing": missing,
        "attempts": attempts,
        "hf_token_present": hf_token() is not None,
        "hf_token_required": False,
        "cache_dirs": cache_dirs(),
        "hint": (
            "ESM-1b/ESM-2 are public and need no HF token. Check network access to "
            "huggingface.co (or dl.fbaipublicfiles.com for the fair-esm fallback) and "
            "free disk space in the cache dirs. Do not pip install torch at runtime; "
            "missing packages must be added to the environment."
        ),
    }
