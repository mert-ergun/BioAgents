import csv
import io
import time

import requests

# GWAS Catalog uses its own EFO IDs; map common MONDO/other ontology IDs to EFO.
_MONDO_TO_EFO = {
    "MONDO_0004975": "EFO_0000249",  # Alzheimer disease
    "MONDO_0005301": "EFO_0003767",  # multiple sclerosis
    "MONDO_0004976": "EFO_0000249",  # Alzheimer disease (alt)
    "MONDO_0007256": "EFO_0000384",  # Crohn disease
    "MONDO_0005090": "EFO_0000692",  # schizophrenia
    "MONDO_0011561": "EFO_0000612",  # myocardial infarction
    "MONDO_0005148": "EFO_0001360",  # type 2 diabetes
}

_BASE = "https://www.ebi.ac.uk/gwas/rest/api"
_TIMEOUT = 15
_MAX_TOOL_SECONDS = 25


def _compute_pvalue(mantissa, exponent):
    try:
        return float(mantissa) * (10 ** int(exponent))
    except (TypeError, ValueError):
        return None


def _extract_genes(loci):
    genes = []
    for locus in loci:
        for g in locus.get("entrezMappedGenes", []):
            name = g.get("geneName", "").strip()
            if name:
                genes.append(name)
        if not genes:
            for g in locus.get("authorReportedGenes", []):
                name = g.get("geneName", "").strip()
                if name:
                    genes.append(name)
    return ", ".join(dict.fromkeys(genes)) if genes else "N/A"


def _extract_snp_allele(loci):
    for locus in loci:
        for allele in locus.get("strongestRiskAlleles", []):
            name = allele.get("riskAlleleName", "").strip()
            if name:
                if "-" in name:
                    parts = name.rsplit("-", 1)
                    return parts[0], parts[1]
                return name, "N/A"
    return "N/A", "N/A"


def _extract_trait(assoc):
    for t in assoc.get("efoTraits", []):
        trait = t.get("trait", "").strip()
        if trait:
            return trait
    return "N/A"


def _extract_study_accession(assoc):
    study_href = assoc.get("_links", {}).get("study", {}).get("href", "")
    if study_href:
        return study_href.rstrip("/").split("/")[-1]
    return "N/A"


def _fetch_pages(
    url: str,
    params: dict,
    p_threshold: float,
    *,
    max_pages: int = 2,
    deadline: float | None = None,
) -> list[dict]:
    """Fetch pages from a GWAS Catalog HAL endpoint and return filtered rows."""
    rows = []
    page = 0
    while True:
        if deadline and time.monotonic() > deadline:
            break
        params["page"] = page
        remaining = max(5, deadline - time.monotonic()) if deadline else _TIMEOUT
        per_req_timeout = min(_TIMEOUT, remaining)
        try:
            resp = requests.get(url, params=params, timeout=per_req_timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return rows
            return rows
        except Exception:
            return rows

        associations = data.get("_embedded", {}).get("associations", [])
        if not associations:
            break

        for assoc in associations:
            pval = _compute_pvalue(
                assoc.get("pvalueMantissa"), assoc.get("pvalueExponent")
            )
            if pval is None or pval > p_threshold:
                continue
            loci = assoc.get("loci", [])
            snp_id, risk_allele = _extract_snp_allele(loci)
            rows.append(
                {
                    "GWAS_Locus_ID": _extract_study_accession(assoc),
                    "Mapped_Gene_Symbol": _extract_genes(loci),
                    "P_value": pval,
                    "SNP_ID": snp_id,
                    "Risk_Allele": risk_allele,
                    "Trait": _extract_trait(assoc),
                }
            )

        page_info = data.get("page", {})
        total_pages = page_info.get("totalPages", 1)
        if page >= total_pages - 1 or not data.get("_links", {}).get("next"):
            break
        page += 1
        if page >= max_pages:
            break

    return rows


def gwas_get_associations_by_disease(
    efo_id: str,
    p_value_threshold: float = 5e-8,
) -> str:
    """
    Retrieve genome-wide significant GWAS associations for a disease from the
    NHGRI-EBI GWAS Catalog REST API (https://www.ebi.ac.uk/gwas/rest/api/).

    Args:
        efo_id: EFO ID of the disease (e.g., 'EFO_0000249' for Alzheimer's disease).
                Also accepts MONDO IDs such as 'MONDO_0004975' — these are
                automatically mapped to the corresponding GWAS Catalog EFO ID.
        p_value_threshold: P-value threshold for significance. Default: 5e-8.

    Returns:
        Tab-separated string with columns: GWAS_Locus_ID, Mapped_Gene_Symbol,
        P_value, SNP_ID, Risk_Allele, Trait. Returns an informative message when
        no associations are found.
    """
    # Resolve MONDO → EFO mapping if needed
    resolved_id = _MONDO_TO_EFO.get(efo_id, efo_id)
    deadline = time.monotonic() + _MAX_TOOL_SECONDS

    # Primary: search by EFO ID via /associations/search/findByEfoId
    params_by_id = {"efoId": resolved_id, "size": 100}
    rows = _fetch_pages(
        f"{_BASE}/associations/search/findByEfoId",
        dict(params_by_id),
        p_value_threshold,
        deadline=deadline,
    )

    # Fallback: search by trait name via /associations/search/findByEfoTrait
    if not rows and time.monotonic() < deadline:
        _EFO_TO_TRAIT = {
            "EFO_0000249": "Alzheimer disease",
            "EFO_0003767": "multiple sclerosis",
            "EFO_0000384": "Crohn disease",
            "EFO_0000692": "schizophrenia",
            "EFO_0000612": "myocardial infarction",
            "EFO_0001360": "type 2 diabetes",
        }
        trait_name = _EFO_TO_TRAIT.get(resolved_id)
        if trait_name:
            params_by_trait = {"efoTrait": trait_name, "size": 100}
            rows = _fetch_pages(
                f"{_BASE}/associations/search/findByEfoTrait",
                params_by_trait,
                p_value_threshold,
                deadline=deadline,
            )

    if not rows:
        return (
            f"No genome-wide significant associations (p ≤ {p_value_threshold:.0e}) "
            f"found for EFO ID '{efo_id}' (resolved to '{resolved_id}') "
            f"in the NHGRI-EBI GWAS Catalog. "
            f"Verify the EFO ID using the GWAS Catalog website: "
            f"https://www.ebi.ac.uk/gwas/search?query={resolved_id}"
        )

    out = io.StringIO()
    fieldnames = ["GWAS_Locus_ID", "Mapped_Gene_Symbol", "P_value", "SNP_ID", "Risk_Allele", "Trait"]
    writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()
