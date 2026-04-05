"""Merge multiple text reports for export."""

from __future__ import annotations

from typing import Any, ClassVar

from bioagents.workflows.node import WorkflowNode
from bioagents.workflows.schemas import NodeMetadata


class MergeTwoTextNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "merge_two_text"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Merge two reports",
            description="Concatenate two text blocks with separators.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"text_a": "str", "text_b": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"merged": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        a, b = inputs["text_a"], inputs["text_b"]
        merged = f"=== Section A ===\n{a}\n\n=== Section B ===\n{b}"
        return {"merged": merged}


class MergeThreeTextNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "merge_three_text"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Merge three reports",
            description="Concatenate three text blocks.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"text_a": "str", "text_b": "str", "text_c": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"merged": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        parts = [inputs["text_a"], inputs["text_b"], inputs["text_c"]]
        labels = ["A", "B", "C"]
        blocks = [f"=== Section {labels[i]} ===\n{parts[i]}" for i in range(3)]
        return {"merged": "\n\n".join(blocks)}
