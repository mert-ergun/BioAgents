import csv
import json
import time
from pathlib import Path


def _rp(p):
    return str(Path(p).expanduser().absolute())


def tsne_fingerprint_scatter(
    fingerprints_file,
    properties_file,
    output_dir,
    metrics=None,
    perplexity=15.0,
    metric="jaccard",
    random_seed=42,
    prefix="ecfp4_tsne",
):
    """Project ECFP4 fingerprints to 2D with t-SNE and scatter-plot colored by properties.

    Pairwise Tanimoto distances (1 - Jaccard similarity on binary ECFP4 vectors)
    are computed exactly with numpy and passed to t-SNE as a precomputed metric,
    avoiding scipy/pdist metric-availability issues.

    Args:
        fingerprints_file: JSON produced by generate_ecfp4_fingerprints.
        properties_file: CSV produced by calculate_molecular_properties.
        output_dir: directory for tsne_coordinates.csv and PNG scatter plots.
        metrics: property columns used as continuous color scales
            (default ['qed', 'sa_score', 'logp']).
        perplexity: t-SNE perplexity, clamped to (n-1)/3 if larger.
        metric: kept for API compatibility; Tanimoto distances are always used.
        random_seed: seed for reproducible t-SNE.
        prefix: filename prefix for outputs.

    Returns:
        dict with status, n_points, n_bits, perplexity_used, tsne_distance_metric,
        kl_divergence, metric_ranges, coordinates_file, plots, elapsed_seconds.
        If numpy/scikit-learn/matplotlib are missing, returns status
        'capability_unavailable' with missing_packages (no auto-install).

    Raises:
        RuntimeError if no valid fingerprints are present in the input;
        ValueError if metric is not 'jaccard'/'tanimoto'.
    """
    if metric not in ("jaccard", "tanimoto"):
        raise ValueError("only Tanimoto/Jaccard distance is supported for ECFP4 t-SNE")
    missing = []
    try:
        import numpy as np
    except ImportError:
        missing.append("numpy")
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        missing.append("scikit-learn")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        missing.append("matplotlib")
    if missing:
        return {
            "status": "capability_unavailable",
            "missing_packages": missing,
            "message": "Required packages are not preinstalled in the sandbox; auto-install is disabled.",
        }
    started = time.time()
    with Path(_rp(fingerprints_file)).open() as fh:
        fdata = json.load(fh)
    frows = [r for r in fdata.get("fingerprints", []) if "fingerprint_bitstring" in r]
    if not frows:
        raise RuntimeError("no valid fingerprints in input")
    Xb = np.array([[c == "1" for c in r["fingerprint_bitstring"]] for r in frows], dtype=bool)
    Xf = Xb.astype(np.float64)
    n = Xf.shape[0]
    inter = Xf @ Xf.T
    sizes = Xf.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - inter
    D = np.where(union > 0, 1.0 - inter / np.maximum(union, 1e-12), 0.0)
    np.fill_diagonal(D, 0.0)
    props = {}
    with Path(_rp(properties_file)).open() as fh:
        for row in csv.DictReader(fh):
            props[row["id"]] = row
    if metrics is None:
        metrics = ["qed", "sa_score", "logp"]
    ids = [r["id"] for r in frows]
    smis = [r["smiles"] for r in frows]
    perp = min(float(perplexity), (n - 1) / 3.0)
    embedder = TSNE(
        n_components=2,
        metric="precomputed",
        init="random",
        learning_rate=100.0,
        perplexity=perp,
        random_state=random_seed,
    )
    emb = embedder.fit_transform(D)
    metric_vals = {}
    for m in metrics:
        vals = []
        for mid in ids:
            raw = props.get(mid, {}).get(m, "")
            try:
                vals.append(float(raw))
            except (TypeError, ValueError):
                vals.append(float("nan"))
        metric_vals[m] = np.array(vals)
    outdir = _rp(output_dir)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    coord_path = str(Path(outdir).joinpath(f"{prefix}_coordinates.csv"))
    with Path(coord_path).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "smiles", "tsne_1", "tsne_2", *list(metrics)])
        for i, mid in enumerate(ids):
            w.writerow(
                [mid, smis[i], f"{emb[i, 0]:.4f}", f"{emb[i, 1]:.4f}"]
                + [
                    ("" if np.isnan(metric_vals[m][i]) else f"{metric_vals[m][i]:.4f}")
                    for m in metrics
                ]
            )
    plots = []
    ranges = {}
    for m in metrics:
        v = metric_vals[m]
        mask = ~np.isnan(v)
        if int(mask.sum()) < 2:
            continue
        p = str(Path(outdir).joinpath(f"{prefix}_{m}.png"))
        fig, ax = plt.subplots(figsize=(7.5, 6))
        sc = ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            c=v[mask],
            cmap="viridis",
            s=40,
            edgecolors="black",
            linewidths=0.4,
        )
        fig.colorbar(sc, ax=ax, label=m)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title(f"t-SNE of ECFP4 fingerprints colored by {m}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        plots.append(p)
        ranges[m] = {
            "min": round(float(np.nanmin(v)), 4),
            "max": round(float(np.nanmax(v)), 4),
            "mean": round(float(np.nanmean(v)), 4),
        }
    if len(metrics) > 1:
        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]
        for ax, m in zip(axes, metrics):
            v = metric_vals[m]
            mask = ~np.isnan(v)
            if int(mask.sum()) < 2:
                continue
            sc = ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                c=v[mask],
                cmap="viridis",
                s=30,
                edgecolors="black",
                linewidths=0.3,
            )
            fig.colorbar(sc, ax=ax, label=m)
            ax.set_title(m)
        panel = str(Path(outdir).joinpath(f"{prefix}_panel.png"))
        fig.savefig(panel, dpi=200, bbox_inches="tight")
        plt.close(fig)
        plots.append(panel)
    return {
        "status": "success",
        "n_points": n,
        "n_bits": int(Xb.shape[1]),
        "perplexity_used": round(float(perp), 2),
        "tsne_distance_metric": "precomputed Tanimoto distance (1 - Jaccard) on ECFP4",
        "kl_divergence": round(float(embedder.kl_divergence_), 4),
        "metric_ranges": ranges,
        "coordinates_file": coord_path,
        "plots": plots,
        "elapsed_seconds": round(time.time() - started, 1),
    }
