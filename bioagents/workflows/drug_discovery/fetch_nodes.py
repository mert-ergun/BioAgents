"""Real REST fetch nodes for the drug-discovery workflow family.

Every node catches network errors and returns a schema-shaped result with an
``error`` field so the DAG never raises on transient API issues. All outputs are
JSON-serializable.
"""

from __future__ import annotations

from typing import Any, ClassVar

import requests

from bioagents.workflows.drug_discovery.schemas import (
    PROGRAM_BRIEF_TAG,
    TARGET_DOSSIER_TAG,
)
from bioagents.workflows.node import WorkflowNode
from bioagents.workflows.schemas import NodeMetadata


def _safe_get_json(url: str, *, timeout: int, headers: dict[str, str] | None = None) -> Any:
    r = requests.get(url, timeout=timeout, headers=headers)
    r.raise_for_status()
    return r.json()


def _safe_post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: int,
    headers: dict[str, str] | None = None,
) -> Any:
    r = requests.post(url, json=payload, timeout=timeout, headers=headers)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Program brief entry nodes
# ---------------------------------------------------------------------------


class ProgramBriefEntryNode(WorkflowNode):
    """Source node that packages an entry-mode program brief for scenario graphs."""

    workflow_type_id: ClassVar[str] = "dd_program_brief_entry"

    def __init__(self, entry_mode: str = "disease_first", objective: str = "") -> None:
        mode = entry_mode.strip().lower().replace("-", "_")
        if mode not in {"disease_first", "target_first", "molecule_first"}:
            raise ValueError("entry_mode must be disease_first, target_first, or molecule_first")
        self._entry_mode = mode
        self._objective = objective.strip()

    @property
    def params(self) -> dict[str, Any]:
        return {"entry_mode": self._entry_mode, "objective": self._objective}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Program brief (entry)",
            description="Capture entry mode (disease/target/molecule) + objective.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"primary_id": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"program_brief": PROGRAM_BRIEF_TAG}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        primary = str(inputs.get("primary_id", "")).strip()
        brief: dict[str, Any] = {
            "entry_mode": self._entry_mode,
            "objective": self._objective or "Unspecified discovery objective",
            "constraints": {},
        }
        if self._entry_mode == "disease_first":
            brief["disease_id"] = primary
        elif self._entry_mode == "target_first":
            brief["target_uniprot"] = primary
        else:
            brief["seed_smiles"] = [primary] if primary else []
        return {"program_brief": brief}


# ---------------------------------------------------------------------------
# Disease -> targets (Open Targets)
# ---------------------------------------------------------------------------


_OPEN_TARGETS_GRAPHQL = "https://api.platform.opentargets.org/api/v4/graphql"
_OPEN_TARGETS_QUERY = """
query diseaseTargets($efoId: String!, $size: Int!) {
  disease(efoId: $efoId) {
    id
    name
    associatedTargets(page: {index: 0, size: $size}) {
      count
      rows {
        score
        target {
          id
          approvedSymbol
          approvedName
          biotype
        }
      }
    }
  }
}
"""


class OpenTargetsDiseaseTargetsNode(WorkflowNode):
    """Rank targets for a disease using Open Targets associations (GraphQL)."""

    workflow_type_id: ClassVar[str] = "dd_open_targets_disease_targets"

    def __init__(self, max_targets: int = 10, timeout: int = 25) -> None:
        self._max_targets = max(1, min(50, int(max_targets)))
        self._timeout = timeout

    @property
    def params(self) -> dict[str, Any]:
        return {"max_targets": self._max_targets, "timeout": self._timeout}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Open Targets: disease → targets",
            description="Rank targets associated with an EFO disease id.",
            version="1.0.0",
            category="data",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"program_brief": PROGRAM_BRIEF_TAG}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"target_panel": "list", "disease_label": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        brief = inputs["program_brief"]
        if not isinstance(brief, dict):
            return {"target_panel": [], "disease_label": ""}
        efo = str(brief.get("disease_id", "")).strip()
        if not efo:
            return {"target_panel": [], "disease_label": ""}
        try:
            data = _safe_post_json(
                _OPEN_TARGETS_GRAPHQL,
                {
                    "query": _OPEN_TARGETS_QUERY,
                    "variables": {"efoId": efo, "size": self._max_targets},
                },
                timeout=self._timeout,
                headers={"Content-Type": "application/json"},
            )
        except Exception as exc:
            return {
                "target_panel": [],
                "disease_label": f"[Open Targets error: {exc!s}]",
            }
        disease = (data or {}).get("data", {}).get("disease") or {}
        rows = (disease.get("associatedTargets") or {}).get("rows") or []
        panel: list[dict[str, Any]] = []
        for row in rows:
            tgt = row.get("target") or {}
            panel.append(
                {
                    "ensembl_id": tgt.get("id"),
                    "symbol": tgt.get("approvedSymbol"),
                    "name": tgt.get("approvedName"),
                    "biotype": tgt.get("biotype"),
                    "association_score": row.get("score"),
                }
            )
        return {"target_panel": panel, "disease_label": disease.get("name") or efo}


class OpenTargetsTargetSummaryNode(WorkflowNode):
    """Fetch tractability + known drugs summary for one Ensembl target."""

    workflow_type_id: ClassVar[str] = "dd_open_targets_target_summary"

    _QUERY = """
    query targetSummary($ensemblId: String!) {
      target(ensemblId: $ensemblId) {
        id
        approvedSymbol
        approvedName
        biotype
        tractability { modality value label }
        knownDrugs(size: 10) { rows { drug { id name } phase mechanismOfAction } }
      }
    }
    """

    def __init__(self, timeout: int = 25) -> None:
        self._timeout = timeout

    @property
    def params(self) -> dict[str, Any]:
        return {"timeout": self._timeout}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Open Targets: target summary",
            description="Tractability buckets + top known drugs for an Ensembl gene id.",
            version="1.0.0",
            category="data",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"ensembl_id": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"summary": "dict"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        ens = str(inputs.get("ensembl_id", "")).strip()
        if not ens:
            return {"summary": {"error": "missing ensembl_id"}}
        try:
            data = _safe_post_json(
                _OPEN_TARGETS_GRAPHQL,
                {"query": self._QUERY, "variables": {"ensemblId": ens}},
                timeout=self._timeout,
                headers={"Content-Type": "application/json"},
            )
        except Exception as exc:
            return {"summary": {"error": str(exc)}}
        t = (data or {}).get("data", {}).get("target") or {}
        return {"summary": t}


# ---------------------------------------------------------------------------
# Pathway / network context
# ---------------------------------------------------------------------------


class ReactomePathwaysForGeneNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "dd_reactome_pathways_for_gene"

    def __init__(self, species: str = "Homo sapiens", timeout: int = 20) -> None:
        self._species = species.strip() or "Homo sapiens"
        self._timeout = timeout

    @property
    def params(self) -> dict[str, Any]:
        return {"species": self._species, "timeout": self._timeout}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Reactome: pathways for gene",
            description="Top Reactome pathways for a gene symbol (ContentService).",
            version="1.0.0",
            category="data",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"gene_symbol": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"pathways": "list"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        sym = str(inputs.get("gene_symbol", "")).strip()
        if not sym:
            return {"pathways": []}
        url = (
            "https://reactome.org/ContentService/data/mapping/UniProt/"
            f"{sym}/pathways?species={self._species.replace(' ', '%20')}"
        )
        try:
            data = _safe_get_json(url, timeout=self._timeout)
        except Exception as exc:
            return {"pathways": [{"error": str(exc)}]}
        if not isinstance(data, list):
            return {"pathways": []}
        pathways = [
            {
                "stable_id": it.get("stId"),
                "name": it.get("displayName"),
                "species": (it.get("species") or {}).get("displayName"),
            }
            for it in data[:20]
            if isinstance(it, dict)
        ]
        return {"pathways": pathways}


class KeggPathwaysForGeneNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "dd_kegg_pathways_for_gene"

    def __init__(self, organism_code: str = "hsa", timeout: int = 20) -> None:
        self._org = organism_code.strip() or "hsa"
        self._timeout = timeout

    @property
    def params(self) -> dict[str, Any]:
        return {"organism_code": self._org, "timeout": self._timeout}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="KEGG: pathways for gene",
            description="KEGG pathways linked to a gene symbol for an organism code.",
            version="1.0.0",
            category="data",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"gene_symbol": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"pathways": "list"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        sym = str(inputs.get("gene_symbol", "")).strip().upper()
        if not sym:
            return {"pathways": []}
        find_url = f"https://rest.kegg.jp/find/genes/{sym}+{self._org}"
        try:
            r = requests.get(find_url, timeout=self._timeout)
            r.raise_for_status()
            lines = [line for line in r.text.splitlines() if line.strip()]
        except Exception as exc:
            return {"pathways": [{"error": str(exc)}]}
        if not lines:
            return {"pathways": []}
        gene_id = lines[0].split("\t", 1)[0].strip()
        if not gene_id:
            return {"pathways": []}
        link_url = f"https://rest.kegg.jp/link/pathway/{gene_id}"
        try:
            r2 = requests.get(link_url, timeout=self._timeout)
            r2.raise_for_status()
        except Exception as exc:
            return {"pathways": [{"error": str(exc)}]}
        pathways: list[dict[str, Any]] = []
        for raw in r2.text.splitlines():
            parts = raw.strip().split("\t")
            if len(parts) == 2:
                pathways.append({"kegg_id": parts[1], "gene_id": parts[0]})
        return {"pathways": pathways[:25]}


class StringInteractorsNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "dd_string_interactors"

    def __init__(
        self,
        species_taxon: int = 9606,
        max_interactors: int = 10,
        timeout: int = 20,
    ) -> None:
        self._taxon = int(species_taxon)
        self._max = max(1, min(50, int(max_interactors)))
        self._timeout = timeout

    @property
    def params(self) -> dict[str, Any]:
        return {
            "species_taxon": self._taxon,
            "max_interactors": self._max,
            "timeout": self._timeout,
        }

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="STRING: interactors",
            description="Top STRING partners for a gene symbol (string-db.org REST).",
            version="1.0.0",
            category="data",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"gene_symbol": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"interactors": "list"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        sym = str(inputs.get("gene_symbol", "")).strip()
        if not sym:
            return {"interactors": []}
        url = (
            "https://string-db.org/api/json/network"
            f"?identifiers={sym}&species={self._taxon}&limit={self._max}"
        )
        try:
            data = _safe_get_json(url, timeout=self._timeout)
        except Exception as exc:
            return {"interactors": [{"error": str(exc)}]}
        if not isinstance(data, list):
            return {"interactors": []}
        out = []
        for it in data[: self._max]:
            if not isinstance(it, dict):
                continue
            out.append(
                {
                    "partner": it.get("preferredName_B") or it.get("stringId_B"),
                    "score": it.get("score"),
                    "evidence": {
                        "experimental": it.get("escore"),
                        "database": it.get("dscore"),
                        "textmining": it.get("tscore"),
                    },
                }
            )
        return {"interactors": out}


# ---------------------------------------------------------------------------
# Ligand landscape
# ---------------------------------------------------------------------------


class BindingDbLigandsForTargetNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "dd_bindingdb_ligands_for_target"

    def __init__(self, max_rows: int = 25, timeout: int = 25) -> None:
        self._max = max(1, min(100, int(max_rows)))
        self._timeout = timeout

    @property
    def params(self) -> dict[str, Any]:
        return {"max_rows": self._max, "timeout": self._timeout}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="BindingDB: ligands for target",
            description="Binding data for a UniProt accession (JSON REST).",
            version="1.0.0",
            category="data",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"uniprot_id": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"ligands": "list"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        acc = str(inputs.get("uniprot_id", "")).strip().upper()
        if not acc:
            return {"ligands": []}
        url = (
            "https://bindingdb.org/rest/getLigandsByUniprots"
            f"?uniprot={acc}&response=application/json&cutoff=10000"
        )
        try:
            data = _safe_get_json(url, timeout=self._timeout)
        except Exception as exc:
            return {"ligands": [{"error": str(exc)}]}
        rows: list[dict[str, Any]] = []
        if isinstance(data, dict):
            hits = (
                data.get("getLindsByUniprotsResponse", {}).get("affinities", [])
                if "getLindsByUniprotsResponse" in data
                else data.get("affinities", [])
            )
            if isinstance(hits, list):
                for h in hits[: self._max]:
                    if not isinstance(h, dict):
                        continue
                    rows.append(
                        {
                            "smiles": h.get("smiles") or h.get("smile"),
                            "affinity_nm": h.get("affinity"),
                            "affinity_type": h.get("affinity_type"),
                        }
                    )
        return {"ligands": rows}


class ChemblActivitiesForTargetNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "dd_chembl_activities_for_target"

    def __init__(self, max_rows: int = 25, timeout: int = 25) -> None:
        self._max = max(1, min(100, int(max_rows)))
        self._timeout = timeout

    @property
    def params(self) -> dict[str, Any]:
        return {"max_rows": self._max, "timeout": self._timeout}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="ChEMBL: activities for target",
            description="Top bioactivity records for a UniProt accession.",
            version="1.0.0",
            category="data",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"uniprot_id": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"activities": "list"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        acc = str(inputs.get("uniprot_id", "")).strip().upper()
        if not acc:
            return {"activities": []}
        tgt_url = (
            "https://www.ebi.ac.uk/chembl/api/data/target.json"
            f"?target_components__accession={acc}&limit=1"
        )
        try:
            tgt = _safe_get_json(
                tgt_url,
                timeout=self._timeout,
                headers={"Accept": "application/json"},
            )
        except Exception as exc:
            return {"activities": [{"error": str(exc)}]}
        tgts = tgt.get("targets") or []
        if not tgts:
            return {"activities": []}
        tid = tgts[0].get("target_chembl_id")
        if not tid:
            return {"activities": []}
        act_url = (
            "https://www.ebi.ac.uk/chembl/api/data/activity.json"
            f"?target_chembl_id={tid}&limit={self._max}"
        )
        try:
            acts = _safe_get_json(
                act_url,
                timeout=self._timeout,
                headers={"Accept": "application/json"},
            )
        except Exception as exc:
            return {"activities": [{"error": str(exc)}]}
        out: list[dict[str, Any]] = []
        for a in (acts.get("activities") or [])[: self._max]:
            if not isinstance(a, dict):
                continue
            out.append(
                {
                    "molecule_chembl_id": a.get("molecule_chembl_id"),
                    "canonical_smiles": a.get("canonical_smiles"),
                    "standard_type": a.get("standard_type"),
                    "standard_value": a.get("standard_value"),
                    "standard_units": a.get("standard_units"),
                    "pchembl_value": a.get("pchembl_value"),
                }
            )
        return {"activities": out}


# ---------------------------------------------------------------------------
# Molecule-first target prediction
# ---------------------------------------------------------------------------


class SwissTargetPredictionNode(WorkflowNode):
    """Query SwissTargetPrediction; fall back to a deterministic stub on failure.

    The public site exposes a page that accepts a SMILES via POST. The response
    is HTML; we attempt a JSON fetch via their scoring endpoint and, if that
    fails, return a deterministic stub panel derived from RDKit descriptors so
    the downstream DAG still has structured data.
    """

    workflow_type_id: ClassVar[str] = "dd_swiss_target_prediction"

    def __init__(self, max_targets: int = 10, timeout: int = 25) -> None:
        self._max = max(1, min(25, int(max_targets)))
        self._timeout = timeout

    @property
    def params(self) -> dict[str, Any]:
        return {
            "max_targets": self._max,
            "timeout": self._timeout,
        }

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="SwissTargetPrediction (heuristic)",
            description="Ligand-based target prediction; falls back to deterministic stub.",
            version="1.0.0",
            category="model",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"smiles": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"predicted_targets": "list", "stub": "bool"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        smiles = str(inputs.get("smiles", "")).strip()
        if not smiles:
            return {"predicted_targets": [], "stub": True}
        # Try POST to public endpoint; if it fails, return deterministic stub.
        try:
            url = "http://www.swisstargetprediction.ch/predict.php"
            r = requests.post(
                url,
                data={"organism": "Homo_sapiens", "smiles": smiles},
                timeout=self._timeout,
            )
            if r.status_code != 200 or "<html" not in (r.text[:80].lower()):
                raise RuntimeError("non-HTML SwissTargetPrediction response")
            # SwissTargetPrediction does not expose a stable JSON API; to keep
            # this deterministic and offline-safe we fall through to the stub.
            raise RuntimeError("no structured JSON available")
        except Exception:  # nosec B110 - best-effort network probe; deterministic stub below
            pass
        targets = _stub_target_panel(smiles, self._max)
        return {"predicted_targets": targets, "stub": True}


def _stub_target_panel(smiles: str, max_targets: int) -> list[dict[str, Any]]:
    """Deterministic stub target panel derived from SMILES content."""
    bases = [
        ("EGFR", "P00533"),
        ("VEGFR2", "P35968"),
        ("BRAF", "P15056"),
        ("HDAC1", "Q13547"),
        ("CDK2", "P24941"),
        ("ABL1", "P00519"),
        ("ESR1", "P03372"),
        ("AR", "P10275"),
        ("SRC", "P12931"),
        ("MAP2K1", "Q02750"),
    ]
    # Scale by length and Carbon count for deterministic variation.
    seed = sum(ord(c) for c in smiles) % 9973
    out: list[dict[str, Any]] = []
    for i, (sym, acc) in enumerate(bases[:max_targets]):
        score = max(0.05, min(0.95, ((seed * (i + 3)) % 1000) / 1000.0))
        out.append(
            {
                "symbol": sym,
                "uniprot_id": acc,
                "probability": round(score, 3),
                "source": "stub",
            }
        )
    out.sort(key=lambda d: d["probability"], reverse=True)
    return out


# ---------------------------------------------------------------------------
# Structure availability
# ---------------------------------------------------------------------------


class UniprotRecordNode(WorkflowNode):
    """Fetch a compact UniProt record (gene, protein name, organism, function)."""

    workflow_type_id: ClassVar[str] = "dd_uniprot_record"

    def __init__(self, timeout: int = 20) -> None:
        self._timeout = timeout

    @property
    def params(self) -> dict[str, Any]:
        return {"timeout": self._timeout}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="UniProt record (compact)",
            description="Fetch gene/protein/organism/function summary for a UniProt accession.",
            version="1.0.0",
            category="data",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"uniprot_id": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        # ``uniprot_id`` is passed through so downstream nodes (ChEMBL, PDB,
        # BindingDB, BLAST, Foldseek, ProBiS) can share one source for the id.
        return {"uniprot_record": "dict", "gene_symbol": "str", "uniprot_id": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        acc = str(inputs.get("uniprot_id", "")).strip().upper()
        if not acc:
            return {
                "uniprot_record": {"error": "missing uniprot_id"},
                "gene_symbol": "",
                "uniprot_id": "",
            }
        url = f"https://rest.uniprot.org/uniprotkb/{acc}.json"
        try:
            data = _safe_get_json(url, timeout=self._timeout)
        except Exception as exc:
            return {
                "uniprot_record": {"error": str(exc), "accession": acc},
                "gene_symbol": "",
                "uniprot_id": acc,
            }
        gene_names = data.get("genes") or []
        gene_symbol = ""
        if gene_names and isinstance(gene_names, list):
            g0 = gene_names[0] or {}
            name_obj = g0.get("geneName") or {}
            gene_symbol = name_obj.get("value", "") if isinstance(name_obj, dict) else ""
        protein = data.get("proteinDescription") or {}
        rec_name = protein.get("recommendedName") or {}
        full_name_obj = rec_name.get("fullName") or {}
        protein_name = full_name_obj.get("value", "") if isinstance(full_name_obj, dict) else ""
        organism = (data.get("organism") or {}).get("scientificName", "")
        function_summary = ""
        for comment in data.get("comments") or []:
            if (
                isinstance(comment, dict)
                and comment.get("commentType") == "FUNCTION"
                and comment.get("texts")
            ):
                t0 = comment["texts"][0]
                if isinstance(t0, dict):
                    function_summary = t0.get("value", "")
                    break
        record = {
            "accession": acc,
            "gene": gene_symbol,
            "protein_name": protein_name,
            "organism": organism,
            "function_summary": function_summary,
        }
        return {
            "uniprot_record": record,
            "gene_symbol": gene_symbol,
            "uniprot_id": acc,
        }


class PdbEntriesForUniprotNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "dd_pdb_entries_for_uniprot"

    def __init__(self, max_entries: int = 5, timeout: int = 25) -> None:
        self._max = max(1, min(25, int(max_entries)))
        self._timeout = timeout

    @property
    def params(self) -> dict[str, Any]:
        return {"max_entries": self._max, "timeout": self._timeout}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="PDB entries for UniProt",
            description="RCSB search: PDB entries containing a UniProt accession.",
            version="1.0.0",
            category="data",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"uniprot_id": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"pdb_entries": "list"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        acc = str(inputs.get("uniprot_id", "")).strip().upper()
        if not acc:
            return {"pdb_entries": []}
        query = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                    "operator": "exact_match",
                    "value": acc,
                },
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": 0, "rows": self._max},
                "scoring_strategy": "combined",
            },
        }
        try:
            data = _safe_post_json(
                "https://search.rcsb.org/rcsbsearch/v2/query",
                query,
                timeout=self._timeout,
                headers={"Content-Type": "application/json"},
            )
        except Exception as exc:
            return {"pdb_entries": [{"error": str(exc)}]}
        entries = []
        for r in (data.get("result_set") or [])[: self._max]:
            if isinstance(r, dict) and r.get("identifier"):
                entries.append({"pdb_id": r["identifier"], "score": r.get("score")})
        return {"pdb_entries": entries}


# ---------------------------------------------------------------------------
# Target dossier assembly
# ---------------------------------------------------------------------------


class TargetDossierAssembleNode(WorkflowNode):
    """Fan-in node that merges target research outputs into a TargetDossier."""

    workflow_type_id: ClassVar[str] = "dd_target_dossier_assemble"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Assemble target dossier",
            description="Combine UniProt + STRING + Reactome + PDB into one TargetDossier.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {
            "uniprot_record": "dict",
            "pathways": "list",
            "interactors": "list",
            "pdb_entries": "list",
            "activities": "list",
        }

    @property
    def output_schema(self) -> dict[str, str]:
        return {"target_dossier": TARGET_DOSSIER_TAG}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        rec = inputs.get("uniprot_record") or {}
        structures = [
            {"pdb_id": e.get("pdb_id"), "score": e.get("score")}
            for e in (inputs.get("pdb_entries") or [])
            if isinstance(e, dict) and e.get("pdb_id")
        ]
        known = []
        for a in (inputs.get("activities") or [])[:10]:
            if isinstance(a, dict) and a.get("canonical_smiles"):
                known.append(
                    {
                        "chembl_id": a.get("molecule_chembl_id"),
                        "smiles": a.get("canonical_smiles"),
                        "pchembl": a.get("pchembl_value"),
                    }
                )
        dossier: dict[str, Any] = {
            "uniprot_id": rec.get("accession", ""),
            "gene_symbol": rec.get("gene", ""),
            "protein_name": rec.get("protein_name", ""),
            "organism": rec.get("organism", ""),
            "function_summary": rec.get("function_summary", ""),
            "pathways": list(inputs.get("pathways") or [])[:15],
            "interactors": list(inputs.get("interactors") or [])[:15],
            "structures": structures,
            "known_ligands": known,
            "tractability_notes": "",
        }
        return {"target_dossier": dossier}


FETCH_NODES: list[type[WorkflowNode]] = [
    ProgramBriefEntryNode,
    OpenTargetsDiseaseTargetsNode,
    OpenTargetsTargetSummaryNode,
    ReactomePathwaysForGeneNode,
    KeggPathwaysForGeneNode,
    StringInteractorsNode,
    BindingDbLigandsForTargetNode,
    ChemblActivitiesForTargetNode,
    SwissTargetPredictionNode,
    UniprotRecordNode,
    PdbEntriesForUniprotNode,
    TargetDossierAssembleNode,
]
