"""ESM-2 protein sequence embeddings via ``fair-esm`` (optional dependency)."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, ClassVar

from bioagents.workflows.esm_models import ALLOWED_ESM2_MODEL_NAMES, DEFAULT_ESM2_MODEL_NAME
from bioagents.workflows.node import WorkflowNode
from bioagents.workflows.schemas import NodeMetadata

logger = logging.getLogger(__name__)

# Reasonable default cap; ESM-2 uses ~1 residue ≈ 1 token (+ BOS/EOS).
_MAX_AA = 1020


def _ensure_torch_hub_cache() -> None:
    """Use a repo-local PyTorch hub cache when TORCH_HOME is unset (CI, sandboxes)."""
    if os.environ.get("TORCH_HOME"):
        return
    root = Path(__file__).resolve().parents[3]
    cache = root / ".bioagents_cache" / "torch"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(cache)


_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any, Any]] = {}


def _require_esm() -> tuple[Any, Any]:
    try:
        import esm
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "ESM-2 embeddings require optional dependencies. Install with:\n"
            "  uv sync --extra esm\n"
            "or: pip install 'bioagents[esm]'"
        ) from exc
    return torch, esm


def _normalize_sequence(raw: str) -> str:
    s = raw.upper().strip()
    s = re.sub(r"\s+", "", s)
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    return "".join(c if c in allowed else "X" for c in s)


def _pick_device(requested: str | None) -> str:
    torch, _ = _require_esm()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("CUDA requested for ESM-2 but not available; using CPU")
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


class Esm2EmbedderNode(WorkflowNode):
    """
    Mean-pooled embedding from the final transformer layer of an ESM-2 checkpoint.

    Models are loaded lazily and cached per (model_name, device). Requires the
    ``esm`` extra (PyTorch + fair-esm).
    """

    workflow_type_id: ClassVar[str] = "esm2_embedder"

    def __init__(
        self, model_name: str = DEFAULT_ESM2_MODEL_NAME, device: str | None = None
    ) -> None:
        if model_name not in ALLOWED_ESM2_MODEL_NAMES:
            raise ValueError(
                f"Unsupported ESM-2 model {model_name!r}; "
                f"allowed: {sorted(ALLOWED_ESM2_MODEL_NAMES)}"
            )
        self._model_name = model_name
        self._device_pref = device

    @property
    def params(self) -> dict[str, Any]:
        return {"model_name": self._model_name, "device": self._device_pref}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="ESM-2 protein embedder",
            description=f"Sequence embedding via facebookresearch/esm checkpoint {self._model_name!r} (mean pool, last layer).",
            version="1.0.0",
            category="model",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"embedding": "list[float]"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        sequence = inputs["sequence"]
        if not isinstance(sequence, str):
            raise TypeError("sequence must be str")
        seq = _normalize_sequence(sequence)
        if not seq:
            raise ValueError("sequence is empty after normalization")

        if len(seq) > _MAX_AA:
            logger.warning(
                "Truncating sequence from %d to %d residues for ESM-2",
                len(seq),
                _MAX_AA,
            )
            seq = seq[:_MAX_AA]

        _ensure_torch_hub_cache()
        torch, esm = _require_esm()
        device = _pick_device(self._device_pref)
        cache_key = (self._model_name, device)
        if cache_key not in _MODEL_CACHE:
            model, alphabet = esm.pretrained.load_model_and_alphabet(self._model_name)
            model = model.to(device)
            model.eval()
            batch_converter = alphabet.get_batch_converter()
            _MODEL_CACHE[cache_key] = (model, alphabet, batch_converter)
        model, alphabet, batch_converter = _MODEL_CACHE[cache_key]

        _, _, batch_tokens = batch_converter([("protein", seq)])
        batch_tokens = batch_tokens.to(device)

        layer = model.num_layers
        with torch.no_grad():
            out = model(batch_tokens, repr_layers=[layer], return_contacts=False)
        token_repr = out["representations"][layer]

        mask = (batch_tokens != alphabet.padding_idx).unsqueeze(-1)
        summed = (token_repr * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = (summed / counts).squeeze(0)

        vec = pooled.detach().float().cpu().tolist()
        return {"embedding": vec}
