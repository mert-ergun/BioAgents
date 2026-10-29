"""
esm1b_fitness_scores - Generate a saturation-mutagenesis library at given
positions and score every mutant with ESM-1b zero-shot fitness.

Category: ml

Method (Meier et al. 2021, "Language models enable zero-shot prediction of
the effects of mutations on protein function"; masked-marginal probability):
  For each position p in binding_positions:
      - mask residue p in the wild-type sequence
      - run ONE forward pass (positions are batched)
      - log-softmax the logits at the masked position
      - for each of the 19 alternative amino acids:
            LLR           = log P(mut_aa | context) - log P(wt_aa | context)
            fitness_ratio = exp(LLR)
  WT rows (mut == wt) are included with LLR = 0 / ratio = 1 so the result
  forms the full 20 x N matrix.

Backends (model_backend='auto' tries in order): fair-esm
(esm1b_t33_650M_UR50S), then HuggingFace transformers
('facebook/esm1b_t33_650M_UR50S', pinned to a repository commit). Model weights are downloaded to the
standard local cache on first use. Nothing is pip-installed by this module:
if no backend is importable, or weight loading fails, the call returns
status 'capability_unavailable' with the exact missing package and no
results are fabricated.

Outputs (idempotent; existing outputs are reused unless force_recompute=True):
  mutation_library.csv       - 20 x N mutated sequences (incl. WT reference rows)
  esm1b_fitness_scores.csv   - position, wt_aa, mut_aa, mutation, LLR, fitness_ratio
  esm1b_fitness_summary.json - per-position best/worst + global worst-5 mutants

Example:
    esm1b_fitness_scores('MENFQK...', [33, 64], '/app/out/esm')
"""

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
FAIR_MODEL = "esm1b_t33_650M_UR50S"
HF_MODEL = "facebook/esm1b_t33_650M_UR50S"
HF_MODEL_REVISION = "7b37824baec4d3658e1df7479222a7c79b465b76"


def _diag() -> dict[str, Any]:
    """Environment diagnostics: which ESM backends are importable."""
    d: dict[str, Any] = {"torch": None, "fair_esm": None, "transformers": None, "cuda": False}
    try:
        import torch

        d["torch"] = torch.__version__
        d["cuda"] = bool(torch.cuda.is_available())
    except Exception as exc:
        d["torch"] = f"unavailable: {exc}"
    try:
        import esm

        d["fair_esm"] = getattr(esm, "__version__", "installed")
    except Exception as exc:
        d["fair_esm"] = f"unavailable: {exc}"
    try:
        import transformers

        d["transformers"] = transformers.__version__
    except Exception as exc:
        d["transformers"] = f"unavailable: {exc}"
    return d


def _is_avail(v: Any) -> bool:
    return bool(v) and not str(v).startswith("unavailable")


def _log(msg: str) -> None:
    print(f"[esm1b_fitness_scores] {msg}", file=sys.stderr)


def esm1b_fitness_scores(
    wild_type_sequence: str,
    binding_positions: list[int],
    output_dir: str,
    model_backend: str = "auto",
    batch_size: int = 8,
    device: str = "auto",
    check_only: bool = False,
    force_recompute: bool = False,
) -> dict[str, Any]:
    """Score every single-site mutant at the given positions with ESM-1b zero-shot fitness.

    Args:
        wild_type_sequence: WT amino acid sequence (uppercase, 20 standard AAs).
        binding_positions: 1-based positions to mutate (list of ints).
        output_dir: directory for mutation_library.csv, esm1b_fitness_scores.csv,
            esm1b_fitness_summary.json.
        model_backend: 'auto' (default) | 'fair-esm' | 'transformers'.
        batch_size: masked sequences per forward pass (default 8).
        device: 'auto' (default) | 'cpu' | 'cuda'.
        check_only: if True, return environment diagnostics without loading the model.
        force_recompute: if True, recompute even when outputs already exist.

    Returns:
        dict with status ('success' | 'check_only' | 'capability_unavailable'),
        n_mutations, output_files, worst_5_mutations and per_position summary.

    Raises:
        ValueError: invalid sequence characters or out-of-range positions.
    """
    seq = (wild_type_sequence or "").strip().upper()
    invalid = sorted({aa for aa in seq if aa not in AA_LIST})
    if not seq or invalid:
        raise ValueError(
            f"wild_type_sequence must contain only the 20 standard amino acids; invalid chars: {invalid}"
        )
    positions = sorted({int(p) for p in binding_positions})
    if not positions:
        raise ValueError("binding_positions must contain at least one 1-based position")
    for p in positions:
        if not 1 <= p <= len(seq):
            raise ValueError(f"position {p:d} out of range 1..{len(seq):d}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = str(Path(output_dir).joinpath("esm1b_fitness_scores.csv"))
    lib_path = str(Path(output_dir).joinpath("mutation_library.csv"))
    json_path = str(Path(output_dir).joinpath("esm1b_fitness_summary.json"))

    diag = _diag()
    if check_only:
        return {
            "status": "check_only",
            "diagnostics": diag,
            "backend_order": [f"fair-esm ({FAIR_MODEL})", f"transformers ({HF_MODEL})"],
            "mutations_would_score": len(positions) * 19,
        }

    if not force_recompute and Path(csv_path).exists() and Path(json_path).exists():
        with Path(json_path).open() as fh:
            summary = json.load(fh)
        return {
            "status": "success",
            "reused_existing": True,
            "output_files": {
                "mutation_library_csv": lib_path,
                "scores_csv": csv_path,
                "summary_json": json_path,
            },
            "n_mutations": summary.get("n_mutations"),
            "worst_5_mutations": summary.get("worst_5_mutations"),
            "summary": summary,
        }

    try:
        import torch
    except Exception as exc:
        return {
            "status": "capability_unavailable",
            "diagnostics": diag,
            "message": (
                "PyTorch (pip package 'torch') is required to run ESM-1b but is not importable in "
                f"this environment ({exc}). Reported as a missing dependency; nothing was installed "
                "and no fitness values were computed."
            ),
        }

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if dev == "cuda" and not torch.cuda.is_available():
        return {
            "status": "capability_unavailable",
            "diagnostics": diag,
            "message": "device='cuda' requested but CUDA is not available in this environment.",
        }
    if dev not in ("cpu", "cuda"):
        return {"status": "error", "message": "device must be 'auto', 'cpu' or 'cuda'"}

    backend = model_backend
    if backend == "auto":
        backend = (
            "fair-esm"
            if _is_avail(diag["fair_esm"]) and _is_avail(diag["torch"])
            else (
                "transformers"
                if _is_avail(diag["transformers"]) and _is_avail(diag["torch"])
                else "none"
            )
        )
    if backend == "none":
        return {
            "status": "capability_unavailable",
            "diagnostics": diag,
            "message": (
                "No ESM backend importable: need 'fair-esm' or 'transformers' together with "
                "'torch'. Reported as missing dependencies; nothing was installed or computed."
            ),
        }

    # -------- masked sequences: one per target position --------
    masked = []
    for p in positions:
        s = list(seq)
        s[p - 1] = "#"  # placeholder, replaced by the backend mask token below
        masked.append((p, "".join(s)))

    rows: list[dict[str, Any]] = []
    try:
        if backend == "fair-esm":
            import esm

            _log(f"loading fair-esm {FAIR_MODEL} weights (cached after first download)...")
            model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
            model = model.eval().to(dev)
            batch_converter = alphabet.get_batch_converter()
            mask_token = alphabet.mask_token
            mask_idx_token = alphabet.mask_token_idx
            aa_ids = {aa: alphabet.get_idx(aa) for aa in AA_LIST}
            with torch.no_grad():
                for start in range(0, len(masked), batch_size):
                    chunk = masked[start : start + batch_size]
                    data = [(f"pos{p:d}", mseq.replace("#", mask_token)) for p, mseq in chunk]
                    _, _, toks = batch_converter(data)
                    toks = toks.to(dev)
                    logits = model(toks)["logits"]
                    logprobs = torch.log_softmax(logits, dim=-1)
                    for bi, (p, _mseq) in enumerate(chunk):
                        mask_pos = int((toks[bi] == mask_idx_token).nonzero(as_tuple=True)[0][0])
                        lp = logprobs[bi, mask_pos]
                        wt_aa = seq[p - 1]
                        wt_lp = float(lp[aa_ids[wt_aa]].item())
                        for aa in AA_LIST:
                            llr = float(lp[aa_ids[aa]].item()) - wt_lp
                            rows.append(
                                {
                                    "position": p,
                                    "wt_aa": wt_aa,
                                    "mut_aa": aa,
                                    "mutation": f"{wt_aa}{p:d}{aa}",
                                    "llr": round(llr, 6),
                                    "fitness_ratio": round(math.exp(llr), 6),
                                }
                            )
                    _log(
                        f"scored {min(start + batch_size, len(masked)):d}/{len(masked):d} positions"
                    )
        else:
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            _log(f"loading transformers {HF_MODEL} weights (cached after first download)...")
            tok = AutoTokenizer.from_pretrained(
                HF_MODEL,
                revision="7b37824baec4d3658e1df7479222a7c79b465b76",
                trust_remote_code=False,
            )
            model = (
                AutoModelForMaskedLM.from_pretrained(
                    HF_MODEL,
                    revision="7b37824baec4d3658e1df7479222a7c79b465b76",
                    trust_remote_code=False,
                )
                .eval()
                .to(dev)
            )
            aa_ids = {aa: tok.convert_tokens_to_ids(aa) for aa in AA_LIST}
            with torch.no_grad():
                for start in range(0, len(masked), batch_size):
                    chunk = masked[start : start + batch_size]
                    seqs = [mseq.replace("#", tok.mask_token) for _, mseq in chunk]
                    enc = tok(seqs, return_tensors="pt", padding=True, add_special_tokens=True)
                    enc = {k: v.to(dev) for k, v in enc.items()}
                    logits = model(**enc).logits
                    logprobs = torch.log_softmax(logits, dim=-1)
                    for bi, (p, _mseq) in enumerate(chunk):
                        ids = enc["input_ids"][bi]
                        mask_pos = int((ids == tok.mask_token_id).nonzero(as_tuple=True)[0][0])
                        lp = logprobs[bi, mask_pos]
                        wt_aa = seq[p - 1]
                        wt_lp = float(lp[aa_ids[wt_aa]].item())
                        for aa in AA_LIST:
                            llr = float(lp[aa_ids[aa]].item()) - wt_lp
                            rows.append(
                                {
                                    "position": p,
                                    "wt_aa": wt_aa,
                                    "mut_aa": aa,
                                    "mutation": f"{wt_aa}{p:d}{aa}",
                                    "llr": round(llr, 6),
                                    "fitness_ratio": round(math.exp(llr), 6),
                                }
                            )
                    _log(
                        f"scored {min(start + batch_size, len(masked)):d}/{len(masked):d} positions"
                    )
    except Exception as exc:
        return {
            "status": "capability_unavailable",
            "diagnostics": diag,
            "backend": backend,
            "message": (
                f"ESM-1b could not be loaded or run ({type(exc).__name__}: {exc}). If this is a missing package or a "
                "network-blocked weight download, report it; nothing was installed and no fitness "
                "values were computed."
            ),
        }

    # -------- outputs --------
    mutants = [r for r in rows if r["mut_aa"] != r["wt_aa"]]
    mutants.sort(key=lambda r: r["llr"])  # most negative (most damaging) first
    worst5 = mutants[:5]
    per_position: dict[str, Any] = {}
    for p in positions:
        pos_sorted = sorted([r for r in mutants if r["position"] == p], key=lambda r: r["llr"])
        per_position[str(p)] = {
            "wt_aa": seq[p - 1],
            "worst": pos_sorted[:3],
            "best": pos_sorted[-3:][::-1],
        }
    summary = {
        "model": FAIR_MODEL if backend == "fair-esm" else HF_MODEL,
        "model_revision": HF_MODEL_REVISION if backend == "transformers" else None,
        "backend": backend,
        "sequence_length": len(seq),
        "positions": positions,
        "n_positions": len(positions),
        "n_mutations": len(mutants),
        "worst_5_mutations": worst5,
        "per_position": per_position,
        "note": (
            "LLR = log P(mut|masked context) - log P(wt|masked context), natural log; "
            "fitness_ratio = exp(LLR). More negative LLR = predicted more deleterious (zero-shot)."
        ),
    }
    with Path(json_path).open("w") as fh:
        json.dump(summary, fh, indent=2)
    with Path(csv_path).open("w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["position", "wt_aa", "mut_aa", "mutation", "llr", "fitness_ratio"]
        )
        w.writeheader()
        w.writerows(rows)
    with Path(lib_path).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["position", "wt_aa", "mut_aa", "mutation", "sequence"])
        w.writeheader()
        for p in positions:
            for aa in AA_LIST:
                mut_seq = seq if aa == seq[p - 1] else seq[: p - 1] + aa + seq[p:]
                w.writerow(
                    {
                        "position": p,
                        "wt_aa": seq[p - 1],
                        "mut_aa": aa,
                        "mutation": f"{seq[p - 1]}{p:d}{aa}",
                        "sequence": mut_seq,
                    }
                )
    return {
        "status": "success",
        "reused_existing": False,
        "backend": backend,
        "model": summary["model"],
        "device": dev,
        "n_positions": len(positions),
        "n_mutations": len(mutants),
        "output_files": {
            "mutation_library_csv": lib_path,
            "scores_csv": csv_path,
            "summary_json": json_path,
        },
        "worst_5_mutations": worst5,
        "per_position_worst_best": per_position,
    }
