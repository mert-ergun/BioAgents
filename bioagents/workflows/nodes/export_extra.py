"""Additional export nodes for workflow presets."""

from __future__ import annotations

import json
from typing import Any, ClassVar

from bioagents.workflows.node import WorkflowNode
from bioagents.workflows.schemas import NodeMetadata


class ExportFastaJsonNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "export_fasta_json"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Export FASTA JSON",
            description="Wrap raw FASTA text in JSON.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"fasta": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"json": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"json": json.dumps({"fasta": inputs["fasta"]})}


class ExportSequenceJsonNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "export_sequence_json"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Export sequence JSON",
            description="Sequence and residue count as JSON.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"sequence": "str", "residue_count": "int"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"json": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        payload = {"sequence": inputs["sequence"], "residue_count": inputs["residue_count"]}
        return {"json": json.dumps(payload)}


class ExportTextJsonNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "export_text_json"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Export text JSON",
            description="Single text field as JSON.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"text": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"json": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"json": json.dumps({"report": inputs["text"]})}


class ExportDictJsonNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "export_dict_json"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Export dict JSON",
            description="Serialize a JSON-compatible dict.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"record": "dict"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"json": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        rec = inputs["record"]
        if not isinstance(rec, dict):
            raise TypeError("record must be dict")
        return {"json": json.dumps(rec, default=str)}
