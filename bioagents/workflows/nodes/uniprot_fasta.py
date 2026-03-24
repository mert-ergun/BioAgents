"""UniProt FASTA data source node."""

from typing import Any, ClassVar

from bioagents.tools.proteomics_tools import fetch_uniprot_fasta_impl
from bioagents.workflows.node import WorkflowNode
from bioagents.workflows.schemas import NodeMetadata


class UniprotFastaNode(WorkflowNode):
    """Fetch raw FASTA text for a UniProt accession."""

    workflow_type_id: ClassVar[str] = "uniprot_fasta"

    def __init__(self, timeout: int = 10) -> None:
        self._timeout = timeout

    @property
    def params(self) -> dict[str, Any]:
        return {"timeout": self._timeout}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="UniProt FASTA fetch",
            description="Download FASTA for a UniProt ID via the UniProt REST API.",
            version="1.0.0",
            category="data",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"protein_id": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"fasta": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        protein_id = inputs["protein_id"]
        if not isinstance(protein_id, str):
            raise TypeError("protein_id must be str")
        fasta = fetch_uniprot_fasta_impl(protein_id, timeout=self._timeout)
        return {"fasta": fasta}
