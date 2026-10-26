"""
uniprot_get_features - Compact UniProtKB feature extraction via the UniProt REST API.

Category: database

Fetches https://rest.uniprot.org/uniprotkb/{accession}.json and extracts
compact structured records for requested feature types:
  - binding sites (ligand name + ChEBI ID + evidence)
  - natural variants (WT aa, alternative aa, description, xrefs e.g. ClinVar/disease)
  - mutagenesis annotations (WT aa, alternative aa, description)
  - site / domain annotations
Returns short dicts (position, ligand, wild_type/alternative/description/xrefs)
instead of the huge raw UniProt JSON so downstream agents can use positions
directly. Stdlib-only (urllib), retries with backoff, optional JSON save.
"""

import json
import time
import urllib.error
from pathlib import Path
from typing import Any

from bioagents.tools.http_client import read_https_text

_FEATURE_TYPE_MAP = {
    "binding": "Binding site",
    "variant": "Natural variant",
    "mutagen": "Mutagenesis",
    "site": "Site",
    "domain": "Domain",
}
_UNIPROT_API = "https://rest.uniprot.org/uniprotkb/{accession}.json"
_DEFAULT_TIMEOUT = 60
_DEFAULT_RETRIES = 3
_BACKOFF = 2.0


def _fetch_json(url: str, timeout: int, retries: int) -> dict[str, Any]:
    last_err = None
    for attempt in range(retries):
        try:
            text = read_https_text(
                url,
                allowed_hosts={"rest.uniprot.org"},
                headers={"Accept": "application/json"},
                timeout=timeout,
            )
            result = json.loads(text)
            if not isinstance(result, dict):
                raise ValueError("UniProt JSON response must be an object")
            return result
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            last_err = exc
            if attempt < retries - 1:
                time.sleep(_BACKOFF * (attempt + 1))
    raise RuntimeError(f"UniProt REST request failed after {retries:d} attempts: {last_err}")


def _ev_strings(ev_list: Any) -> list[str]:
    out = []
    for ev in ev_list or []:
        if isinstance(ev, dict) and ev.get("evidenceCode"):
            src = ev.get("source") or ""
            idv = ev.get("id") or ""
            out.append(
                "{}|{}|{}".format(ev["evidenceCode"], src, idv) if idv else ev["evidenceCode"]
            )
    return out


def _xrefs(refs: Any) -> list[str]:
    out = []
    for ref in refs or []:
        db = ref.get("database") or ""
        idv = ref.get("id") or ""
        if db and idv:
            out.append(f"{db}:{idv}")
    return out


def _compact(feat: dict[str, Any]) -> dict[str, Any]:
    ftype = feat.get("type") or ""
    loc = feat.get("location") or {}
    start = loc.get("start", {}).get("value")
    end = loc.get("end", {}).get("value")
    desc = feat.get("description")
    rec: dict[str, Any] = {"type": ftype, "start": start, "end": end}
    lig = feat.get("ligand") or {}
    if lig.get("name"):
        rec["ligand"] = lig["name"]
        if lig.get("id"):
            rec["ligand_id"] = lig["id"]
    alt = feat.get("alternatives")
    if alt:
        rec["alternatives"] = alt
    for key in ("wildType", "alternativeSequence"):
        if feat.get(key) and feat[key] not in ("", None):
            rec["wild_type" if key == "wildType" else "alternative"] = feat[key]
    if desc:
        rec["description"] = desc
    if feat.get("evidences"):
        ev = _ev_strings(feat["evidences"])
        if ev:
            rec["evidence"] = ev[:4]
    if feat.get("crossReferences"):
        xr = _xrefs(feat["crossReferences"])
        if xr:
            rec["xrefs"] = xr[:6]
    return rec


def uniprot_get_features(
    accession: str,
    feature_types: str = "binding",
    output_file: str | None = None,
    timeout: int = _DEFAULT_TIMEOUT,
    retries: int = _DEFAULT_RETRIES,
) -> dict[str, Any]:
    """Extract compact structured feature records for a UniProtKB accession.

    Args:
        accession: UniProtKB accession, e.g. 'P24941' (human CDK2).
        feature_types: comma-separated keywords: binding, variant, mutagen, site, domain.
        output_file: optional path to save the compact JSON.
        timeout: HTTP timeout in seconds (default 60).
        retries: retries on transient network errors (default 3).

    Returns:
        dict with status, accession, n_features_total, n_features_returned,
        feature_types_requested, features (compact list sorted by type/position),
        and output_file when saved.

    Raises:
        ValueError: empty accession or no valid feature-type keyword.
        RuntimeError: persistent network failure.

    Example:
        uniprot_get_features('P24941', feature_types='binding,variant,mutagen',
                             output_file='cdk2_features.json')
    """
    accession = (accession or "").strip()
    if not accession:
        raise ValueError("accession must be a non-empty UniProtKB accession")
    keywords = [k.strip().lower() for k in (feature_types or "").split(",") if k.strip()]
    uniprot_types = [_FEATURE_TYPE_MAP[k] for k in keywords if k in _FEATURE_TYPE_MAP]
    if not uniprot_types:
        raise ValueError(
            f"no valid feature type keyword in {feature_types!r}; valid: {sorted(_FEATURE_TYPE_MAP)}"
        )
    url = _UNIPROT_API.format(accession=accession)
    data = _fetch_json(url, timeout, retries)
    all_feats = data.get("features") or []
    features = [_compact(f) for f in all_feats if f.get("type") in uniprot_types]
    features.sort(
        key=lambda r: (
            r.get("type", ""),
            r.get("start") if isinstance(r.get("start"), int) else 10**9,
        )
    )
    result = {
        "status": "success",
        "accession": accession,
        "protein_name": (data.get("proteinDescription", {}).get("recommendedName", {}) or {})
        .get("fullName", {})
        .get("value"),
        "sequence_length": (data.get("sequence") or {}).get("length"),
        "feature_types_requested": uniprot_types,
        "n_features_total": len(all_feats),
        "n_features_returned": len(features),
        "features": features,
    }
    if output_file:
        with Path(output_file).open("w") as fh:
            json.dump(result, fh, indent=2)
        result["output_file"] = output_file
    return result
