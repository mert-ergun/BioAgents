"""Remote bioinformatics APIs as workflow nodes (graceful error strings in outputs)."""

from __future__ import annotations

import json
from typing import Any, ClassVar

import requests

from bioagents.workflows.node import WorkflowNode
from bioagents.workflows.schemas import NodeMetadata


class AlphafoldSummaryNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "alphafold_prediction_summary"

    def __init__(self, timeout: int = 20) -> None:
        self._timeout = timeout

    @property
    def params(self) -> dict[str, Any]:
        return {"timeout": self._timeout}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="AlphaFold DB summary",
            description="EBI AlphaFold API: model confidence for a UniProt accession.",
            version="1.0.0",
            category="data",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"protein_id": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"summary": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        acc = str(inputs["protein_id"]).strip()
        url = f"https://alphafold.ebi.ac.uk/api/prediction/{acc}"
        try:
            r = requests.get(url, timeout=self._timeout)
            if r.status_code == 404:
                return {"summary": f"No AlphaFold entry for {acc!r}"}
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            return {"summary": f"AlphaFold API error: {exc!s}"}
        if not isinstance(data, list) or not data:
            return {"summary": f"Unexpected payload for {acc}"}
        entry = data[0]
        mid = entry.get("modelIdentifier", "n/a")
        ver = entry.get("latestVersion", "n/a")
        avg = entry.get("globalMetricValue")
        org = entry.get("organismScientificName", "n/a")
        lines = [
            f"UniProt: {acc}",
            f"Organism: {org}",
            f"Model: {mid}",
            f"Version: {ver}",
            f"Global metric (pLDDT-like): {avg}",
        ]
        return {"summary": "\n".join(lines)}


class UniprotEntryLiteNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "uniprot_entry_lite"

    def __init__(self, timeout: int = 15) -> None:
        self._timeout = timeout

    @property
    def params(self) -> dict[str, Any]:
        return {"timeout": self._timeout}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="UniProt entry (lite)",
            description="Core fields from UniProtKB JSON for an accession.",
            version="1.0.0",
            category="data",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"protein_id": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"summary": "str", "record": "dict"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        acc = str(inputs["protein_id"]).strip()
        url = f"https://rest.uniprot.org/uniprotkb/{acc}.json"
        try:
            r = requests.get(url, timeout=self._timeout)
            if r.status_code == 404:
                return {"summary": f"UniProt: {acc!r} not found", "record": {}}
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            return {"summary": f"UniProt error: {exc!s}", "record": {}}
        rec: dict[str, Any] = {"accession": acc}
        if isinstance(data, dict):
            pd = data.get("proteinDescription")
            if isinstance(pd, dict):
                rn = pd.get("recommendedName")
                if isinstance(rn, dict):
                    fn = rn.get("fullName")
                    if isinstance(fn, dict):
                        full = fn.get("value")
                        if isinstance(full, str) and full:
                            rec["protein_name"] = full
            org_block = data.get("organism")
            if isinstance(org_block, dict):
                org = org_block.get("scientificName")
                if isinstance(org, str) and org:
                    rec["organism"] = org
            seq = data.get("sequence")
            if isinstance(seq, dict) and "length" in seq:
                rec["length"] = seq["length"]
            genes = data.get("genes")
            if isinstance(genes, list) and genes:
                g0 = genes[0]
                if isinstance(g0, dict):
                    gn = g0.get("geneName")
                    if isinstance(gn, dict):
                        val = gn.get("value")
                        if isinstance(val, str) and val:
                            rec["gene"] = val
        lines = [f"UniProt: {acc}", json.dumps(rec, indent=2, default=str)]
        return {"summary": "\n".join(lines), "record": rec}


class PubchemCompoundPropertiesNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "pubchem_compound_properties"

    def __init__(self, timeout: int = 15) -> None:
        self._timeout = timeout

    @property
    def params(self) -> dict[str, Any]:
        return {"timeout": self._timeout}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="PubChem compound (CID)",
            description="MolecularFormula, MolecularWeight, CanonicalSMILES by numeric CID.",
            version="1.0.0",
            category="data",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"compound_id": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"summary": "str", "record": "dict"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        raw = str(inputs["compound_id"]).strip()
        if not raw.isdigit():
            return {"summary": f"compound_id must be numeric CID, got {raw!r}", "record": {}}
        cid = raw
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            f"{cid}/property/MolecularFormula,MolecularWeight,CanonicalSMILES/JSON"
        )
        try:
            r = requests.get(url, timeout=self._timeout)
            if r.status_code == 404:
                return {"summary": f"PubChem CID {cid} not found", "record": {}}
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            return {"summary": f"PubChem error: {exc!s}", "record": {}}
        props = (((data.get("PropertyTable") or {}).get("Properties")) or [{}])[0]
        rec = {k.lower(): v for k, v in props.items() if isinstance(k, str)}
        return {"summary": json.dumps(rec, indent=2, default=str), "record": rec}


class NcbiGeneSummaryNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "ncbi_gene_summary"

    def __init__(self, organism: str = "human", timeout: int = 20) -> None:
        self._organism = organism.strip() or "human"
        self._timeout = timeout

    @property
    def params(self) -> dict[str, Any]:
        return {"organism": self._organism, "timeout": self._timeout}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="NCBI Gene summary",
            description="esearch + esummary for gene symbol + organism (NCBI E-utilities).",
            version="1.0.0",
            category="data",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"gene_symbol": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"summary": "str", "record": "dict"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        sym = str(inputs["gene_symbol"]).strip()
        term = f"{sym}[Gene Name] AND {self._organism}[Organism]"
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        esearch_params: dict[str, str | int] = {
            "db": "gene",
            "term": term,
            "retmode": "json",
            "retmax": 5,
        }
        try:
            r = requests.get(
                f"{base}/esearch.fcgi",
                params=esearch_params,
                timeout=self._timeout,
            )
            r.raise_for_status()
            search = r.json()
            ids = (search.get("esearchresult") or {}).get("idlist") or []
            if not ids:
                return {
                    "summary": f"No NCBI Gene hits for {sym!r} ({self._organism})",
                    "record": {},
                }
            gid = ids[0]
            esummary_params: dict[str, str] = {
                "db": "gene",
                "id": str(gid),
                "retmode": "json",
            }
            r2 = requests.get(
                f"{base}/esummary.fcgi",
                params=esummary_params,
                timeout=self._timeout,
            )
            r2.raise_for_status()
            summ = r2.json()
        except Exception as exc:
            return {"summary": f"NCBI error: {exc!s}", "record": {}}
        result = (summ.get("result") or {}).get(str(gid)) or {}
        name = result.get("name", sym)
        desc = result.get("description", "")
        org_raw = result.get("organism")
        if isinstance(org_raw, dict):
            org = org_raw.get("scientificname", self._organism)
        elif isinstance(org_raw, str):
            org = org_raw
        else:
            org = self._organism
        rec = {"gene_id": gid, "symbol": name, "description": desc, "organism": org}
        lines = [f"GeneId: {gid}", f"Symbol: {name}", f"Organism: {org}", desc]
        return {"summary": "\n".join(lines), "record": rec}


class ChemblMoleculeByInchikeyNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "chembl_molecule_by_inchikey"

    def __init__(self, timeout: int = 20) -> None:
        self._timeout = timeout

    @property
    def params(self) -> dict[str, Any]:
        return {"timeout": self._timeout}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="ChEMBL molecule (InChI Key)",
            description="Lookup molecule metadata by full Standard InChI Key.",
            version="1.0.0",
            category="data",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"inchi_key": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"summary": "str", "record": "dict"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        key = str(inputs["inchi_key"]).strip().upper()
        if len(key) < 14:
            return {"summary": "Provide full Standard InChI Key (27 chars)", "record": {}}
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?molecule_structures__standard_inchi_key={key}&limit=1"
        try:
            r = requests.get(url, timeout=self._timeout, headers={"Accept": "application/json"})
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            return {"summary": f"ChEMBL error: {exc!s}", "record": {}}
        mols = data.get("molecules") or []
        if not mols:
            return {"summary": f"No ChEMBL molecule for InChI key {key!r}", "record": {}}
        m = mols[0]
        rec = {
            "chembl_id": m.get("molecule_chembl_id"),
            "pref_name": m.get("pref_name"),
            "max_phase": m.get("max_phase"),
        }
        return {"summary": json.dumps(rec, indent=2, default=str), "record": rec}


class EnsemblLookupGeneNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "ensembl_lookup_symbol"

    def __init__(self, species: str = "homo_sapiens", timeout: int = 20) -> None:
        self._species = species.strip().lower().replace(" ", "_") or "homo_sapiens"
        self._timeout = timeout

    @property
    def params(self) -> dict[str, Any]:
        return {"species": self._species, "timeout": self._timeout}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Ensembl lookup (symbol)",
            description="Resolve gene symbol to Ensembl gene id (REST).",
            version="1.0.0",
            category="data",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"gene_symbol": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"summary": "str", "record": "dict"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        sym = str(inputs["gene_symbol"]).strip().upper()
        url = f"https://rest.ensembl.org/lookup/symbol/{self._species}/{sym}"
        try:
            r = requests.get(
                url,
                timeout=self._timeout,
                headers={"Content-Type": "application/json"},
            )
            if r.status_code == 400:
                return {"summary": f"Ensembl: bad request for {sym!r}", "record": {}}
            if r.status_code == 404:
                return {"summary": f"Ensembl: {sym!r} not found in {self._species}", "record": {}}
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            return {"summary": f"Ensembl error: {exc!s}", "record": {}}
        rec = {
            "id": data.get("id"),
            "display_name": data.get("display_name"),
            "biotype": data.get("biotype"),
            "species": data.get("species"),
        }
        return {"summary": json.dumps(rec, indent=2, default=str), "record": rec}


BIOINFORMATICS_FETCH_NODES: list[type[WorkflowNode]] = [
    AlphafoldSummaryNode,
    UniprotEntryLiteNode,
    PubchemCompoundPropertiesNode,
    NcbiGeneSummaryNode,
    ChemblMoleculeByInchikeyNode,
    EnsemblLookupGeneNode,
]
