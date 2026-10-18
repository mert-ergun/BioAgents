"""Real ESM protein language model tools (embeddings and mutation fitness scoring).

These tools run ESM models locally. They perform actual computation and return
actual numbers. If a model cannot be loaded or a sequence cannot be scored, they
return a structured error naming exactly what is missing — they never report
success for work that did not happen.

No Hugging Face token is needed: ESM-1b and ESM-2 are public checkpoints. A token
is only used if one happens to be set. Gated models (ESM3 / ESM C) are replaced by
the public ESM-1b / ESM-2 checkpoint and the substitution is stated in the result's
``fallback_note``. Model loading, backend fallback (transformers -> fair-esm
torch.hub) and cache hygiene live in :mod:`bioagents.tools.esm_model_loader`.

Supported model families:
- ESM-2 (``facebook/esm2_*``) — general purpose, several sizes
- ESM-1b (``facebook/esm1b_t33_650M_UR50S``) — the model used by most published
  variant-effect / mutation-fitness benchmarks
- ESM-1v (``facebook/esm1v_t33_650M_UR90S_1``) — variant-effect specialist
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from bioagents.tools.esm_model_loader import (
    DEFAULT_ESM_MODEL,
    ESM_MODEL_ALIASES,
    ESM_MODEL_REVISIONS,
    EsmModelHandle,
    ModelResolution,
    describe_load_failure,
    load_esm_model,
    resolve_model,
)
from bioagents.tools.esm_model_loader import _model_revision as _model_revision
from bioagents.tools.esm_model_loader import _select_device as _select_device

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_ESM_MODEL",
    "ESM_MODEL_ALIASES",
    "ESM_MODEL_REVISIONS",
    "get_esm_tools",
    "run_esm_embedding",
    "saturation_scan_esm",
    "score_mutations_esm",
]

# ESM-1b was trained with a 1022-residue crop; longer inputs are scored in windows.
MAX_ESM_SEQUENCE_LENGTH = 1022

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
AMINO_ACID_ORDER = "ACDEFGHIKLMNPQRSTVWY"

SCORING_METHODS = {
    "masked_marginal": (
        "masked-marginal log-likelihood ratio: the position is masked and "
        "score = log P(mutant) - log P(wild-type) (Meier et al. 2021)"
    ),
    "wt_marginal": (
        "wild-type-marginal log-likelihood ratio: one unmasked forward pass and "
        "score = log P(mutant) - log P(wild-type) at the position (Meier et al. 2021)"
    ),
}
_SCORING_ALIASES = {
    "masked": "masked_marginal",
    "masked-marginal": "masked_marginal",
    "masked_marginal": "masked_marginal",
    "mm": "masked_marginal",
    "wt": "wt_marginal",
    "wt-marginal": "wt_marginal",
    "wt_marginal": "wt_marginal",
    "wildtype_marginal": "wt_marginal",
    "wild-type-marginal": "wt_marginal",
}

# Masked copies of the sequence scored per forward pass.
_MASKED_BATCH_SIZE = 8

_MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def _resolve_model_name(model: str) -> str:
    """Map a user/agent-supplied model alias to a public checkpoint id."""
    return resolve_model(model).model_name


def _validate_sequence(sequence: str) -> tuple[str, str | None]:
    """Normalise a protein sequence and return (sequence, error_message)."""
    text = (sequence or "").strip()
    if text.startswith(">"):
        text = "\n".join(line for line in text.splitlines() if not line.startswith(">"))
    seq = "".join(text.split()).upper()
    if not seq:
        return "", "Sequence is empty."
    invalid = sorted(set(seq) - VALID_AMINO_ACIDS)
    if invalid:
        return seq, f"Sequence contains non-standard amino acid(s): {', '.join(invalid)}"
    return seq, None


def _error(message: str, **extra: Any) -> str:
    payload: dict[str, Any] = {"status": "error", "message": message}
    payload.update(extra)
    return json.dumps(payload, indent=2)


def _load(resolution: ModelResolution) -> tuple[EsmModelHandle | None, str | None]:
    """Load the resolved model; return (handle, error_json)."""
    try:
        return load_esm_model(resolution.model_name), None
    except Exception as exc:
        logger.warning("ESM model load failed for %s: %s", resolution.model_name, exc)
        details = describe_load_failure(resolution.model_name, exc)
        message = details.pop("message")
        if resolution.fallback_note:
            details["fallback_note"] = resolution.fallback_note
        return None, _error(message, requested_model=resolution.requested, **details)


def _model_fields(resolution: ModelResolution, handle: EsmModelHandle) -> dict[str, Any]:
    """Common provenance fields for every successful result."""
    fields: dict[str, Any] = {
        "model": handle.model_name,
        "requested_model": resolution.requested or handle.model_name,
        "backend": handle.backend,
        "device": handle.device,
        "hf_token_required": False,
    }
    if resolution.fallback_note:
        fields["fallback_note"] = resolution.fallback_note
    if handle.notes:
        fields["load_notes"] = list(handle.notes)
    return fields


def _window(seq_len: int, pos: int) -> tuple[int, int]:
    """0-based [start, end) window of <= MAX_ESM_SEQUENCE_LENGTH residues around ``pos``."""
    if seq_len <= MAX_ESM_SEQUENCE_LENGTH:
        return 0, seq_len
    half = MAX_ESM_SEQUENCE_LENGTH // 2
    start = min(max(pos - 1 - half, 0), seq_len - MAX_ESM_SEQUENCE_LENGTH)
    return start, start + MAX_ESM_SEQUENCE_LENGTH


def _position_logprobs(
    handle: EsmModelHandle, seq: str, positions: list[int], method: str
) -> dict[int, Any]:
    """Return {position: log-softmax vector over the vocab} for each 1-based position."""
    import torch

    by_window: dict[tuple[int, int], list[int]] = {}
    for pos in positions:
        by_window.setdefault(_window(len(seq), pos), []).append(pos)

    out: dict[int, Any] = {}
    for (start, end), window_positions in by_window.items():
        ids = handle.encode(seq[start:end])
        # Residue at 1-based ``pos`` sits at token index (pos - start) (index 0 is <cls>).
        if method == "wt_marginal":
            with torch.no_grad():
                logits = handle.logits(ids)[0]
            for pos in window_positions:
                out[pos] = torch.log_softmax(logits[pos - start], dim=-1)
            continue
        for i in range(0, len(window_positions), _MASKED_BATCH_SIZE):
            chunk = window_positions[i : i + _MASKED_BATCH_SIZE]
            batch = ids.repeat(len(chunk), 1)
            for row, pos in enumerate(chunk):
                batch[row, pos - start] = handle.mask_id
            with torch.no_grad():
                logits = handle.logits(batch)
            for row, pos in enumerate(chunk):
                out[pos] = torch.log_softmax(logits[row, pos - start], dim=-1)
    return out


def _parse_mutations(
    seq: str, mutations: str
) -> tuple[list[tuple[str, str, int, str]], list[dict[str, str]]]:
    requested = [m.strip().upper() for m in mutations.replace(";", ",").split(",") if m.strip()]
    parsed: list[tuple[str, str, int, str]] = []
    rejected: list[dict[str, str]] = []
    for mut in requested:
        match = _MUTATION_RE.match(mut)
        if not match:
            rejected.append({"mutation": mut, "reason": "Malformed; expected format like 'G11A'."})
            continue
        wt_aa, pos, mt_aa = match.group(1), int(match.group(2)), match.group(3)
        if mt_aa not in VALID_AMINO_ACIDS:
            rejected.append({"mutation": mut, "reason": f"'{mt_aa}' is not a standard amino acid."})
        elif pos < 1 or pos > len(seq):
            rejected.append(
                {"mutation": mut, "reason": f"Position {pos} outside sequence (1-{len(seq)})."}
            )
        elif seq[pos - 1] != wt_aa:
            rejected.append(
                {
                    "mutation": mut,
                    "reason": (
                        f"Wild-type mismatch: sequence has {seq[pos - 1]} at position "
                        f"{pos}, not {wt_aa}."
                    ),
                }
            )
        else:
            parsed.append((mut, wt_aa, pos, mt_aa))
    return parsed, rejected


def _parse_positions(seq: str, positions: str) -> tuple[list[int], str | None]:
    wanted: list[int] = []
    for raw_chunk in str(positions).replace(";", ",").split(","):
        chunk = raw_chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            try:
                start, end = (int(x) for x in chunk.split("-", 1))
            except ValueError:
                return [], f"Malformed position range: '{chunk}'. Use e.g. '80-86'."
            wanted.extend(range(start, end + 1))
        else:
            try:
                wanted.append(int(chunk))
            except ValueError:
                return [], f"Malformed position: '{chunk}'. Use 1-based integers."
    wanted = sorted(set(wanted))
    if not wanted:
        return [], "No positions supplied. Provide e.g. '31,33,145' or '80-86'."
    out_of_range = [p for p in wanted if p < 1 or p > len(seq)]
    if out_of_range:
        return [], f"Position(s) outside sequence (1-{len(seq)}): {out_of_range}"
    return wanted, None


def _normalise_method(scoring_method: str) -> str | None:
    return _SCORING_ALIASES.get((scoring_method or "masked_marginal").strip().lower())


def _score(
    handle: EsmModelHandle, seq: str, parsed: list[tuple[str, str, int, str]], method: str
) -> list[dict[str, Any]]:
    """Score parsed mutations; returns one record per mutation."""
    logprobs = _position_logprobs(handle, seq, sorted({p for _, _, p, _ in parsed}), method)
    scored: list[dict[str, Any]] = []
    for mut, wt_aa, pos, mt_aa in parsed:
        lp = logprobs[pos]
        wt_lp = float(lp[handle.aa_token_id(wt_aa)])
        mt_lp = float(lp[handle.aa_token_id(mt_aa)])
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
    return scored


@tool
def run_esm_embedding(
    sequence: str,
    model: str = "esm2",
    return_per_residue: bool = False,
) -> str:
    """Compute a real ESM protein language-model embedding for a protein sequence.

    Runs the model locally (no API key or HF token needed) and returns actual
    embedding numbers. Use this to get a numeric vector representation of a protein
    for clustering, similarity search, or as input features to a downstream ML model.

    Args:
        sequence: Protein sequence in single-letter amino acid code (FASTA header allowed).
        model: Model alias or HuggingFace id. Accepts 'esm2' (default, 650M),
            'esm2-small', 'esm2-tiny', 'esm1b', or any 'facebook/esm...' checkpoint id.
            Gated models ('esm3', 'esmc') fall back to public ESM-2 with a note.
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
    if len(seq) > MAX_ESM_SEQUENCE_LENGTH:
        return _error(
            f"Sequence length {len(seq)} exceeds the {MAX_ESM_SEQUENCE_LENGTH}-residue "
            "limit for ESM embeddings. Split the sequence into domains and embed each.",
            sequence_length=len(seq),
        )

    resolution = resolve_model(model, purpose="embedding")
    handle, load_err = _load(resolution)
    if handle is None:
        return load_err or _error("ESM model could not be loaded.")

    try:
        import torch

        with torch.no_grad():
            hidden = handle.last_hidden(handle.encode(seq))
        # Drop the leading <cls> and trailing <eos> tokens before pooling.
        residues = hidden[0, 1 : len(seq) + 1]
        mean_embedding = residues.mean(dim=0)
        result: dict[str, Any] = {
            "status": "success",
            **_model_fields(resolution, handle),
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
        return _error(f"ESM embedding computation failed: {exc}", model=handle.model_name)


@tool
def score_mutations_esm(
    sequence: str,
    mutations: str,
    model: str = "esm1b",
    scoring_method: str = "masked_marginal",
) -> str:
    """Score protein point mutations for fitness/variant effect with a real ESM model.

    Runs locally; no HF token or API key is needed (ESM-1b/ESM-2 are public). Default
    is the masked-marginal score used in ESM variant-effect papers: the position is
    masked and the score is log P(mutant) - log P(wild-type). 'wt_marginal' uses one
    unmasked pass instead (faster, slightly less accurate). Negative scores mean the
    mutation is predicted to be deleterious; the more negative, the worse.

    Args:
        sequence: Wild-type protein sequence in single-letter code (FASTA header allowed).
        mutations: Comma-separated mutations in the form 'G11A' (wild-type residue,
            1-based position, mutant residue), e.g. 'G11A,K33M,D145N'. For a full
            19-substitution scan use `saturation_scan_esm` instead.
        model: Model alias or HuggingFace id. Defaults to 'esm1b'; also 'esm1v',
            'esm2' and sizes. Gated 'esm3'/'esmc' fall back to ESM-1b with a note.
        scoring_method: 'masked_marginal' (default) or 'wt_marginal'.

    Returns:
        JSON with: status, model, backend, device, scored mutation records (mutation,
        position, wild_type, mutant, score, wt_logprob, mt_logprob) and 'ranked_worst'
        (most deleterious first). Wild-type mismatches are listed in 'rejected'.
    """
    seq, err = _validate_sequence(sequence)
    if err:
        return _error(err)
    method = _normalise_method(scoring_method)
    if method is None:
        return _error(
            f"Unknown scoring_method '{scoring_method}'. Use one of: {sorted(SCORING_METHODS)}."
        )
    if not mutations or not mutations.strip():
        return _error("No mutations supplied. Provide e.g. 'G11A,K33M'.")

    parsed, rejected = _parse_mutations(seq, mutations)
    if not parsed:
        return _error(
            "No valid mutations to score; all were rejected.",
            rejected=rejected,
            sequence_length=len(seq),
        )

    resolution = resolve_model(model, purpose="scoring")
    handle, load_err = _load(resolution)
    if handle is None:
        return load_err or _error("ESM model could not be loaded.")

    try:
        scored = _score(handle, seq, parsed, method)
    except Exception as exc:
        logger.exception("ESM mutation scoring failed")
        return _error(f"ESM mutation scoring failed: {exc}", model=handle.model_name)

    ranked = sorted(scored, key=lambda r: r["score"])
    result: dict[str, Any] = {
        "status": "success",
        **_model_fields(resolution, handle),
        "scoring_method": SCORING_METHODS[method],
        "interpretation": "More negative = more deleterious / lower predicted fitness.",
        "sequence_length": len(seq),
        "num_scored": len(scored),
        "mutations": scored,
        "ranked_worst": [r["mutation"] for r in ranked],
        "rejected": rejected,
    }
    if len(seq) > MAX_ESM_SEQUENCE_LENGTH:
        result["windowing"] = (
            f"Sequence longer than {MAX_ESM_SEQUENCE_LENGTH}; each position was scored "
            f"in a {MAX_ESM_SEQUENCE_LENGTH}-residue window centred on it."
        )
    return json.dumps(result, indent=2)


def _per_position_summary(seq: str, wanted: list[int], scored: list[dict]) -> list[dict]:
    summary: list[dict[str, Any]] = []
    for p in wanted:
        at_pos = [r for r in scored if r["position"] == p]
        if not at_pos:
            continue
        worst = min(at_pos, key=lambda r: r["score"])
        summary.append(
            {
                "position": p,
                "wild_type": seq[p - 1],
                "mean_score": round(sum(r["score"] for r in at_pos) / len(at_pos), 4),
                "worst_mutation": worst["mutation"],
                "worst_score": worst["score"],
            }
        )
    return summary


def _write_scan_csv(
    output_dir: str, seq: str, model_name: str, method: str, scored: list[dict]
) -> str:
    """Write every scored variant (with its full mutant sequence) to a CSV; return path."""
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = model_name.split("/")[-1]
    path = out_dir / f"esm_saturation_scan_{tag}_{method}.csv"
    ranked = sorted(scored, key=lambda r: r["score"])
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "mutation",
                "position",
                "wild_type",
                "mutant",
                "score",
                "fitness_ratio",
                "wt_logprob",
                "mt_logprob",
                "mutant_sequence",
            ]
        )
        for rank, rec in enumerate(ranked, start=1):
            pos = rec["position"]
            mutant_seq = seq[: pos - 1] + rec["mutant"] + seq[pos:]
            ratio = math.exp(rec["score"])
            writer.writerow(
                [
                    rank,
                    rec["mutation"],
                    pos,
                    rec["wild_type"],
                    rec["mutant"],
                    rec["score"],
                    f"{ratio:.6g}",
                    rec["wt_logprob"],
                    rec["mt_logprob"],
                    mutant_seq,
                ]
            )
    return str(path)


@tool
def saturation_scan_esm(
    sequence: str,
    positions: str,
    model: str = "esm1b",
    top_n: int = 10,
    scoring_method: str = "masked_marginal",
    output_dir: str = "esm_scan_output",
) -> str:
    """Deep mutational scan: score all 19 substitutions at given positions with real ESM.

    Use this (not hand-written code) when asked to "mutate residues to the other 19
    amino acids" and rank variants by predicted fitness, e.g. "call ESM-1b on each
    mutant and pick the worst 5". Runs locally; no HF token or API key needed.

    Args:
        sequence: Wild-type protein sequence in single-letter code (FASTA header allowed).
        positions: Comma-separated 1-based positions to scan, e.g. '31,33,145'.
            Ranges are supported, e.g. '80-86,145'.
        model: Model alias or HuggingFace id; defaults to 'esm1b'. Gated 'esm3'/'esmc'
            fall back to public ESM-1b with a 'fallback_note'.
        top_n: How many of the most deleterious variants to return in 'worst_variants'.
        scoring_method: 'masked_marginal' (default) or 'wt_marginal'.
        output_dir: Directory for the CSV of all variants (one row per mutant, with
            score, fitness_ratio = exp(score) and the full mutant sequence).

    Returns:
        JSON with csv_path, num_variants, 'worst_variants' (top_n most deleterious,
        most negative first) and per-position summary. status='error' on failure —
        scores are never fabricated.
    """
    seq, err = _validate_sequence(sequence)
    if err:
        return _error(err)
    wanted, pos_err = _parse_positions(seq, positions)
    if pos_err:
        return _error(pos_err, sequence_length=len(seq))
    method = _normalise_method(scoring_method)
    if method is None:
        return _error(
            f"Unknown scoring_method '{scoring_method}'. Use one of: {sorted(SCORING_METHODS)}."
        )
    top_n = max(1, int(top_n or 10))

    parsed = [
        (f"{seq[p - 1]}{p}{mt}", seq[p - 1], p, mt)
        for p in wanted
        for mt in AMINO_ACID_ORDER
        if mt != seq[p - 1]
    ]

    resolution = resolve_model(model, purpose="scoring")
    handle, load_err = _load(resolution)
    if handle is None:
        return load_err or _error("ESM model could not be loaded.")

    try:
        scored = _score(handle, seq, parsed, method)
    except Exception as exc:
        logger.exception("ESM saturation scan failed")
        return _error(f"ESM saturation scan failed: {exc}", model=handle.model_name)

    result: dict[str, Any] = {
        "status": "success",
        **_model_fields(resolution, handle),
        "scoring_method": SCORING_METHODS[method],
        "interpretation": (
            "score = log P(mutant) - log P(wild-type); more negative = more deleterious. "
            "fitness_ratio in the CSV = exp(score) = P(mutant)/P(wild-type)."
        ),
        "sequence_length": len(seq),
        "positions_scanned": wanted,
        "num_variants": len(scored),
        "worst_variants": sorted(scored, key=lambda r: r["score"])[:top_n],
        "per_position_summary": _per_position_summary(seq, wanted, scored),
    }
    try:
        result["csv_path"] = _write_scan_csv(output_dir, seq, handle.model_name, method, scored)
    except OSError as exc:
        # Scores are real; only the file could not be written. Return them inline.
        result["csv_error"] = f"Could not write CSV to '{output_dir}': {exc}"
        result["all_variants"] = scored
    return json.dumps(result, indent=2)


def get_esm_tools() -> list:
    """Return the real ESM language-model tools."""
    return [run_esm_embedding, score_mutations_esm, saturation_scan_esm]
