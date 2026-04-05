"""Placeholder embedding model (no GPU, fixed-size zero vector)."""

from typing import Any, ClassVar

from bioagents.workflows.node import WorkflowNode
from bioagents.workflows.schemas import NodeMetadata


class DummyEmbedderNode(WorkflowNode):
    """Produces a fixed-dimensional zero embedding for pipeline testing."""

    workflow_type_id: ClassVar[str] = "dummy_embedder"

    def __init__(self, dim: int = 8) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self._dim = dim

    @property
    def params(self) -> dict[str, Any]:
        return {"dim": self._dim}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Dummy protein embedder",
            description="Returns a constant zero vector of configurable length (stand-in for ESM2/ProtTrans).",
            version="1.0.0",
            category="model",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"embedding": "list[float]"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        sequence = inputs["sequence"]
        if not isinstance(sequence, str):
            raise TypeError("sequence must be str")
        _ = len(sequence)  # could influence dummy output later
        return {"embedding": [0.0] * self._dim}
