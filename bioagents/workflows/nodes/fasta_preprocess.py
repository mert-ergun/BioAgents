"""FASTA cleaning and simple sequence statistics."""

from typing import Any, ClassVar

from bioagents.workflows.node import WorkflowNode
from bioagents.workflows.schemas import NodeMetadata


class FastaPreprocessorNode(WorkflowNode):
    """Strip FASTA headers and return one-letter sequence plus residue count."""

    workflow_type_id: ClassVar[str] = "fasta_preprocess"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="FASTA preprocessor",
            description="Remove header lines and concatenate sequence records.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"fasta": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"sequence": "str", "residue_count": "int"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        raw = inputs["fasta"]
        if not isinstance(raw, str):
            raise TypeError("fasta must be str")
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        seq_parts: list[str] = []
        for ln in lines:
            if ln.startswith(">"):
                continue
            seq_parts.append(ln.replace(" ", "").upper())
        sequence = "".join(seq_parts)
        return {"sequence": sequence, "residue_count": len(sequence)}
