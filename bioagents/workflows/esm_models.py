"""Allow-listed ESM-2 checkpoint names (passed to ``fair-esm`` loaders)."""

from __future__ import annotations

# Names accepted by ``esm.pretrained.load_model_and_alphabet`` for ESM-2.
ALLOWED_ESM2_MODEL_NAMES: frozenset[str] = frozenset(
    {
        "esm2_t6_8M_UR50D",
        "esm2_t12_35M_UR50D",
        "esm2_t33_650M_UR50D",
    }
)

DEFAULT_ESM2_MODEL_NAME = "esm2_t6_8M_UR50D"
