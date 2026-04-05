"""Reference data types for citation tracking."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Reference:
    """Base reference class."""

    id: str
    type: str = ""  # Default empty, subclasses set in __post_init__
    title: str = ""
    url: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert reference to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "url": self.url,
            "timestamp": self.timestamp,
        }

    def get_unique_key(self) -> str:
        """Get unique key for deduplication."""
        return f"{self.type}:{self.title}"


@dataclass
class PaperReference(Reference):
    """Scientific paper reference."""

    doi: str | None = None
    pmid: str | None = None
    arxiv_id: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    journal: str | None = None
    abstract: str | None = None

    def __post_init__(self):
        """Set type after initialization."""
        if not self.type or self.type == "":
            self.type = "paper"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update(
            {
                "doi": self.doi,
                "pmid": self.pmid,
                "arxiv_id": self.arxiv_id,
                "authors": self.authors,
                "year": self.year,
                "journal": self.journal,
                "abstract": self.abstract,
            }
        )
        return base

    def get_unique_key(self) -> str:
        """Get unique key for deduplication."""
        if self.doi:
            return f"paper:doi:{self.doi}"
        if self.pmid:
            return f"paper:pmid:{self.pmid}"
        if self.arxiv_id:
            return f"paper:arxiv:{self.arxiv_id}"
        return f"paper:title:{self.title}"


@dataclass
class DatabaseReference(Reference):
    """Database query reference."""

    database_name: str = ""
    query: str | None = None
    identifiers: list[str] = field(default_factory=list)
    result_count: int | None = None
    query_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set type after initialization."""
        if not self.type or self.type == "":
            self.type = "database"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update(
            {
                "database_name": self.database_name,
                "query": self.query,
                "identifiers": self.identifiers,
                "result_count": self.result_count,
                "query_params": self.query_params,
            }
        )
        return base

    def get_unique_key(self) -> str:
        """Get unique key for deduplication."""
        if self.identifiers:
            ids_str = ",".join(sorted(self.identifiers[:3]))
            return f"database:{self.database_name}:{ids_str}"
        return f"database:{self.database_name}:{self.query}"


@dataclass
class ToolReference(Reference):
    """Tool/software reference."""

    tool_name: str = ""
    version: str | None = None
    source_paper: str | None = None
    source_url: str | None = None
    description: str | None = None

    def __post_init__(self):
        """Set type after initialization."""
        if not self.type or self.type == "":
            self.type = "tool"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update(
            {
                "tool_name": self.tool_name,
                "version": self.version,
                "source_paper": self.source_paper,
                "source_url": self.source_url,
                "description": self.description,
            }
        )
        return base

    def get_unique_key(self) -> str:
        """Get unique key for deduplication."""
        return f"tool:{self.tool_name}"


@dataclass
class StructureReference(Reference):
    """Protein structure reference."""

    pdb_id: str | None = None
    uniprot_id: str | None = None
    source: str | None = None  # "AlphaFold", "RCSB PDB", etc.
    method: str | None = None  # "X-ray", "NMR", "Predicted", etc.
    organism: str | None = None
    resolution: float | None = None

    def __post_init__(self):
        """Set type after initialization."""
        if not self.type or self.type == "":
            self.type = "structure"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update(
            {
                "pdb_id": self.pdb_id,
                "uniprot_id": self.uniprot_id,
                "source": self.source,
                "method": self.method,
                "organism": self.organism,
                "resolution": self.resolution,
            }
        )
        return base

    def get_unique_key(self) -> str:
        """Get unique key for deduplication."""
        if self.pdb_id:
            return f"structure:pdb:{self.pdb_id}"
        if self.uniprot_id:
            return f"structure:uniprot:{self.uniprot_id}"
        return f"structure:title:{self.title}"


@dataclass
class ArtifactReference(Reference):
    """Generated artifact reference."""

    file_name: str = ""
    file_type: str | None = None
    path: str | None = None
    generation_method: str | None = None
    size: int | None = None
    agent: str | None = None

    def __post_init__(self):
        """Set type after initialization."""
        if not self.type or self.type == "":
            self.type = "artifact"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update(
            {
                "file_name": self.file_name,
                "file_type": self.file_type,
                "path": self.path,
                "generation_method": self.generation_method,
                "size": self.size,
                "agent": self.agent,
            }
        )
        return base

    def get_unique_key(self) -> str:
        """Get unique key for deduplication."""
        return f"artifact:{self.file_name}"
