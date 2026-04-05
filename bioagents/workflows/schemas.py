"""Shared types for scientific workflow nodes and edges."""

from dataclasses import dataclass
from typing import Literal

NodeCategory = Literal["model", "agent", "tool", "data"]

# Port name -> type tag (e.g. "str", "int", "list[float]", "dict").
PortSchema = dict[str, str]

TYPE_ANY = "any"


@dataclass(frozen=True)
class NodeMetadata:
    """Declarative metadata for UI, catalogs, and serialization."""

    name: str
    description: str
    version: str
    category: NodeCategory


def types_compatible(source_output_type: str, target_input_type: str) -> bool:
    """Return True if an output port may feed an input port."""
    if source_output_type == TYPE_ANY or target_input_type == TYPE_ANY:
        return True
    return source_output_type == target_input_type
