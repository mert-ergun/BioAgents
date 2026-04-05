"""Serialize structured workflow results to a JSON string."""

import json
from typing import Any, ClassVar

from bioagents.workflows.node import WorkflowNode
from bioagents.workflows.schemas import NodeMetadata


class ExportJsonNode(WorkflowNode):
    """Combine embedding and residue metadata into JSON text."""

    workflow_type_id: ClassVar[str] = "export_json"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="JSON export",
            description="Dump selected ports to a JSON string for downstream use.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"embedding": "list[float]", "residue_count": "int"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"json": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        embedding = inputs["embedding"]
        residue_count = inputs["residue_count"]
        if not isinstance(embedding, list):
            raise TypeError("embedding must be a list")
        if not isinstance(residue_count, int):
            raise TypeError("residue_count must be int")
        payload = {"embedding": embedding, "residue_count": residue_count}
        return {"json": json.dumps(payload)}
