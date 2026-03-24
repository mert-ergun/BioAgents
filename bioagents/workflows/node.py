"""Abstract workflow node contract."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from bioagents.workflows.schemas import NodeMetadata, PortSchema


class WorkflowNode(ABC):
    """
    A single computational unit in a scientific DAG.

    Subclasses implement ``run`` and declare port schemas for static validation
    when wiring edges.
    """

    workflow_type_id: ClassVar[str]

    @property
    @abstractmethod
    def metadata(self) -> NodeMetadata:
        """Human-readable identity and classification."""

    @property
    @abstractmethod
    def input_schema(self) -> PortSchema:
        """Required input port names and type tags."""

    @property
    @abstractmethod
    def output_schema(self) -> PortSchema:
        """Output port names and type tags."""

    @property
    def params(self) -> dict[str, Any]:
        """Keyword args to reconstruct this node from :mod:`bioagents.workflows.serialization`."""
        return {}

    @abstractmethod
    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute the node; keys must match ``output_schema``."""
