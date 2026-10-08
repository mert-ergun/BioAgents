"""Real ESM protein language model tools (embeddings and mutation fitness scoring).

These tools run ESM models locally via HuggingFace ``transformers``. They perform
actual computation and return actual numbers. If a model cannot be loaded or a
sequence cannot be scored, they return a structured error — they never report
success for work that did not happen.

Supported model families:
- ESM-2 (``facebook/esm2_*``) — general purpose, several sizes
- ESM-1b (``facebook/esm1b_t33_650M_UR50S``) — the model used by most published
  variant-effect / mutation-fitness benchmarks
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Canonical aliases so an agent asking for "esm-1b" or "esm1b" gets the right checkpoint.
ESM_MODEL_ALIASES: dict[str, str] = {
    "esm1b": "facebook/esm1b_t33_650M_UR50S",
    "esm-1b": "facebook/esm1b_t33_650M_UR50S",
    "esm1b_t33_650m_ur50s": "facebook/esm1b_t33_650M_UR50S",
    "esm2": "facebook/esm2_t33_650M_UR50D",
    "esm-2": "facebook/esm2_t33_650M_UR50D",
    "esm2-small": "facebook/esm2_t12_35M_UR50D",
    "esm2-tiny": "facebook/esm2_t6_8M_UR50D",
    "esm2-large": "facebook/esm2_t36_3B_UR50D",
}

DEFAULT_ESM_MODEL = "facebook/esm2_t33_650M_UR50D"

# ESM-1b was trained with a 1022-residue crop; longer inputs must be windowed.
MAX_ESM_SEQUENCE_LENGTH = 1022

# Pinned Hub revisions. An unpinned ref lets the Hub serve different weights later,
# which would silently change every score this module produces and make past results
# irreproducible. Override with BIOAGENTS_ESM_REVISION only to deliberately move a pin.
ESM_MODEL_REVISIONS: dict[str, str] = {
    "facebook/esm2_t6_8M_UR50D": "c731040fcd8d73dceaa04b0a8e6329b345b0f5df",
    "facebook/esm2_t12_35M_UR50D": "6fbf070e65b0b7291e7bbcd451118c216cff79d8",
    "facebook/esm2_t33_650M_UR50D": "08e4846e537177426273712802403f7ba8261b6c",
    "facebook/esm2_t36_3B_UR50D": "476b639933c8baad5ad09a60ac1a87f987b656fc",
    "facebook/esm1b_t33_650M_UR50S": "7b37824baec4d3658e1df7479222a7c79b465b76",
}


def _model_revision(model_name: str) -> str:
    """Return the pinned revision for a model, or 'main' for an unlisted checkpoint."""
    override = os.getenv("BIOAGENTS_ESM_REVISION")
    if override:
        return override
    return ESM_MODEL_REVISIONS.get(model_name, "main")


VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

_MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def _resolve_model_name(model: str) -> str:
    """Map a user/agent-supplied model alias to a HuggingFace checkpoint id."""
    if not model:
        return DEFAULT_ESM_MODEL
    return ESM_MODEL_ALIASES.get(model.strip().lower(), model.strip())


def _select_device() -> str:
    """Pick a usable torch device.

    A CUDA device is only used when torch can actually run kernels on it. Some
    GPUs (e.g. an sm_120 card on a torch build compiled for <= sm_90) report
    ``is_available() == True`` but fail at the first kernel launch, so we probe
    with a real operation instead of trusting the availability flag.
    """
    from bioagents.tools.capability_reporting import cuda_is_usable

    if os.getenv("BIOAGENTS_ESM_DEVICE"):
        return os.environ["BIOAGENTS_ESM_DEVICE"]
    if cuda_is_usable():
        return "cuda"
    logger.info("CUDA unusable on this host; running ESM on CPU.")
    return "cpu"


@lru_cache(maxsize=2)
def _load_masked_lm(model_name: str):
    """Load (and cache) an ESM masked-LM plus its tokenizer on the chosen device."""
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    device = _select_device()
    revision = _model_revision(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)  # nosec B615
    model = AutoModelForMaskedLM.from_pretrained(model_name, revision=revision)  # nosec B615
    model.eval()
    model.to(device)
    torch.set_grad_enabled(False)
    logger.info("Loaded ESM model %s on %s", model_name, device)
    return tokenizer, model, device


def _validate_sequence(sequence: str) -> tuple[str, str | None]:
    """Normalise a protein sequence and return (sequence, error_message)."""
    seq = "".join(sequence.split()).upper()
    if not seq:
        return "", "Sequence is empty."
    if seq.startswith(">"):
        seq = "".join(line for line in seq.splitlines() if not line.startswith(">"))
    invalid = sorted(set(seq) - VALID_AMINO_ACIDS)
    if invalid:
        return seq, f"Sequence contains non-standard amino acid(s): {', '.join(invalid)}"
    return seq, None


def _error(message: str, **extra: Any) -> str:
    payload: dict[str, Any] = {"status": "error", "message": message}
    payload.update(extra)
    return json.dumps(payload, indent=2)


@tool
def run_esm_embedding(
    sequence: str,
    model: str = "esm2",
    return_per_residue: bool = False,
) -> str:
    """Compute a real ESM protein language-model embedding for a protein sequence.

    Runs the model locally and returns actual embedding numbers. Use this to get a
    numeric vector representation of a protein for clustering, similarity search, or
    as input features to a downstream ML model.

    Args:
        sequence: Protein sequence in single-letter amino acid code (FASTA header allowed).
        model: Model alias or HuggingFace id. Accepts 'esm2' (default, 650M),
            'esm2-small', 'esm2-tiny', 'esm1b', or any 'facebook/esm...' checkpoint id.
        return_per_residue: If True, also return the per-residue embedding matrix
            (large). Default False returns only the mean-pooled sequence embedding.

    Returns:
        JSON with: status, model, device, sequence_length, embedding_dim,
        mean_embedding (list of floats), and optionally per_residue_embedding.
        On failure returns status='error' with the reason — never a fake success.
    """
    seq, err = _validate_sequence(sequence)
    if err:
        return _error(err)

    model_name = _resolve_model_name(model)
    try:
        tokenizer, mdl, device = _load_masked_lm(model_name)
    except Exception as exc:
        return _error(
            f"Could not load ESM model '{model_name}': {exc}",
            hint="Check network access to huggingface.co and available disk space.",
        )

    if len(seq) > MAX_ESM_SEQUENCE_LENGTH:
        return _error(
            f"Sequence length {len(seq)} exceeds the {MAX_ESM_SEQUENCE_LENGTH}-residue "
            f"limit for {model_name}. Split the sequence into domains and embed each.",
            sequence_length=len(seq),
        )

    try:
        import torch

        encoded = tokenizer(seq, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = mdl.esm(**encoded).last_hidden_state
        # Drop the leading <cls> and trailing <eos> tokens before pooling.
        residues = hidden[0, 1 : len(seq) + 1]
        mean_embedding = residues.mean(dim=0)

        result: dict[str, Any] = {
            "status": "success",
            "model": model_name,
            "device": device,
            "sequence_length": len(seq),
            "embedding_dim": int(mean_embedding.shape[-1]),
            "mean_embedding": [round(float(x), 6) for x in mean_embedding.tolist()],
        }
        if return_per_residue:
            result["per_residue_embedding"] = [
                [round(float(x), 6) for x in row] for row in residues.tolist()
            ]
        return json.dumps(result)
    except Exception as exc:
        logger.exception("ESM embedding failed")
        return _error(f"ESM embedding computation failed: {exc}", model=model_name)


@tool
def score_mutations_esm(
    sequence: str,
    mutations: str,
    model: str = "esm1b",
) -> str:
    """Score protein point mutations for fitness/variant effect with a real ESM model.

    Computes the standard masked-marginal score used in ESM variant-effect papers:
    for each mutation the position is masked, the model's log-probabilities are read
    out, and the score is log P(mutant) - log P(wild-type). Negative scores mean the
    mutation is predicted to be deleterious; the more negative, the worse. This is a
    real computation — the numbers come from the model, not from a template.

    Args:
        sequence: Wild-type protein sequence in single-letter code (FASTA header allowed).
        mutations: Comma-separated mutations in the form 'G11A' (wild-type residue,
            1-based position, mutant residue), e.g. 'G11A,K33M,D145N'. To run a full
            saturation scan at a position, pass every substitution explicitly.
        model: Model alias or HuggingFace id. Defaults to 'esm1b', the checkpoint used
            by published mutation-fitness benchmarks. 'esm2' and sizes are also accepted.

    Returns:
        JSON with: status, model, device, scored mutation records each containing
        mutation, position, wild_type, mutant, score (delta log-likelihood),
        wt_logprob and mt_logprob; plus 'ranked_worst' listing the most deleterious
        mutations first. Mutations whose wild-type residue does not match the sequence
        are reported in 'rejected' rather than silently scored.
    """
    seq, err = _validate_sequence(sequence)
    if err:
        return _error(err)

    requested = [m.strip().upper() for m in mutations.replace(";", ",").split(",") if m.strip()]
    if not requested:
        return _error("No mutations supplied. Provide e.g. 'G11A,K33M'.")

    parsed: list[tuple[str, str, int, str]] = []
    rejected: list[dict[str, str]] = []
    for mut in requested:
        match = _MUTATION_RE.match(mut)
        if not match:
            rejected.append({"mutation": mut, "reason": "Malformed; expected format like 'G11A'."})
            continue
        wt_aa, pos_str, mt_aa = match.group(1), match.group(2), match.group(3)
        pos = int(pos_str)
        if mt_aa not in VALID_AMINO_ACIDS:
            rejected.append({"mutation": mut, "reason": f"'{mt_aa}' is not a standard amino acid."})
            continue
        if pos < 1 or pos > len(seq):
            rejected.append(
                {"mutation": mut, "reason": f"Position {pos} outside sequence (1-{len(seq)})."}
            )
            continue
        if seq[pos - 1] != wt_aa:
            rejected.append(
                {
                    "mutation": mut,
                    "reason": f"Wild-type mismatch: sequence has {seq[pos - 1]} at position {pos}, not {wt_aa}.",
                }
            )
            continue
        parsed.append((mut, wt_aa, pos, mt_aa))

    if not parsed:
        return _error(
            "No valid mutations to score; all were rejected.",
            rejected=rejected,
            sequence_length=len(seq),
        )

    if len(seq) > MAX_ESM_SEQUENCE_LENGTH:
        return _error(
            f"Sequence length {len(seq)} exceeds the {MAX_ESM_SEQUENCE_LENGTH}-residue limit "
            f"for ESM. Score a domain window containing the positions of interest instead.",
            sequence_length=len(seq),
        )

    model_name = _resolve_model_name(model)
    try:
        tokenizer, mdl, device = _load_masked_lm(model_name)
    except Exception as exc:
        return _error(
            f"Could not load ESM model '{model_name}': {exc}",
            hint="Check network access to huggingface.co and available disk space.",
        )

    try:
        import torch

        input_ids = tokenizer(seq, return_tensors="pt")["input_ids"].to(device)
        # One masked forward pass per distinct position, reused across substitutions.
        positions = sorted({pos for _, _, pos, _ in parsed})
        logprobs_by_position: dict[int, Any] = {}
        for pos in positions:
            masked = input_ids.clone()
            masked[0, pos] = tokenizer.mask_token_id  # +1 offset for <cls> == index pos
            with torch.no_grad():
                logits = mdl(input_ids=masked).logits[0, pos]
            logprobs_by_position[pos] = torch.log_softmax(logits, dim=-1)

        scored: list[dict[str, Any]] = []
        for mut, wt_aa, pos, mt_aa in parsed:
            lp = logprobs_by_position[pos]
            wt_lp = float(lp[tokenizer.convert_tokens_to_ids(wt_aa)])
            mt_lp = float(lp[tokenizer.convert_tokens_to_ids(mt_aa)])
            scored.append(
                {
                    "mutation": mut,
                    "position": pos,
                    "wild_type": wt_aa,
                    "mutant": mt_aa,
                    "score": round(mt_lp - wt_lp, 4),
                    "wt_logprob": round(wt_lp, 4),
                    "mt_logprob": round(mt_lp, 4),
                }
            )

        ranked = sorted(scored, key=lambda r: r["score"])
        return json.dumps(
            {
                "status": "success",
                "model": model_name,
                "device": device,
                "scoring_method": "masked-marginal log-likelihood ratio: log P(mutant) - log P(wild-type)",
                "interpretation": "More negative = more deleterious / lower predicted fitness.",
                "sequence_length": len(seq),
                "num_scored": len(scored),
                "mutations": scored,
                "ranked_worst": [r["mutation"] for r in ranked],
                "rejected": rejected,
            },
            indent=2,
        )
    except Exception as exc:
        logger.exception("ESM mutation scoring failed")
        return _error(f"ESM mutation scoring failed: {exc}", model=model_name)


@tool
def saturation_scan_esm(
    sequence: str,
    positions: str,
    model: str = "esm1b",
    top_n: int = 10,
) -> str:
    """Run a full in-silico saturation mutagenesis scan at given positions with ESM.

    For each requested position this substitutes all 19 alternative amino acids and
    scores every variant with the same masked-marginal method as `score_mutations_esm`.
    Use this when asked to "mutate residues to the other 19 amino acids" and rank the
    resulting variants by predicted fitness.

    Args:
        sequence: Wild-type protein sequence in single-letter code.
        positions: Comma-separated 1-based positions to scan, e.g. '31,33,145'.
            Ranges are supported, e.g. '80-86,145'.
        model: Model alias or HuggingFace id; defaults to 'esm1b'.
        top_n: How many of the most deleterious variants to highlight in 'worst_variants'.

    Returns:
        JSON with every scored variant plus 'worst_variants' (the top_n most
        deleterious, most negative first) and per-position summary statistics.
        Returns status='error' on failure rather than fabricating scores.
    """
    seq, err = _validate_sequence(sequence)
    if err:
        return _error(err)

    wanted: list[int] = []
    for chunk in positions.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            try:
                start, end = (int(x) for x in chunk.split("-", 1))
            except ValueError:
                return _error(f"Malformed position range: '{chunk}'. Use e.g. '80-86'.")
            wanted.extend(range(start, end + 1))
        else:
            try:
                wanted.append(int(chunk))
            except ValueError:
                return _error(f"Malformed position: '{chunk}'. Use 1-based integers.")

    wanted = sorted(set(wanted))
    out_of_range = [p for p in wanted if p < 1 or p > len(seq)]
    if out_of_range:
        return _error(
            f"Position(s) outside sequence (1-{len(seq)}): {out_of_range}",
            sequence_length=len(seq),
        )
    if not wanted:
        return _error("No positions supplied. Provide e.g. '31,33,145' or '80-86'.")

    mutations = [
        f"{seq[p - 1]}{p}{mt}" for p in wanted for mt in sorted(VALID_AMINO_ACIDS - {seq[p - 1]})
    ]

    raw = str(
        score_mutations_esm.invoke(
            {"sequence": seq, "mutations": ",".join(mutations), "model": model}
        )
    )
    payload = json.loads(raw)
    if payload.get("status") != "success":
        return raw

    scored = payload["mutations"]
    ranked = sorted(scored, key=lambda r: r["score"])

    per_position: list[dict[str, Any]] = []
    for p in wanted:
        at_pos = [r for r in scored if r["position"] == p]
        if not at_pos:
            continue
        worst = min(at_pos, key=lambda r: r["score"])
        per_position.append(
            {
                "position": p,
                "wild_type": seq[p - 1],
                "mean_score": round(sum(r["score"] for r in at_pos) / len(at_pos), 4),
                "worst_mutation": worst["mutation"],
                "worst_score": worst["score"],
            }
        )

    return json.dumps(
        {
            "status": "success",
            "model": payload["model"],
            "device": payload["device"],
            "scoring_method": payload["scoring_method"],
            "interpretation": payload["interpretation"],
            "positions_scanned": wanted,
            "num_variants": len(scored),
            "worst_variants": ranked[:top_n],
            "per_position_summary": per_position,
            "all_variants": scored,
        },
        indent=2,
    )


def get_esm_tools() -> list:
    """Return the real ESM language-model tools."""
    return [run_esm_embedding, score_mutations_esm, saturation_scan_esm]
