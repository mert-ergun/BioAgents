"""Reference manager for tracking and deduplicating references."""

import logging
from typing import Any

from bioagents.references.reference_types import Reference

logger = logging.getLogger(__name__)


class ReferenceManager:
    """Manages references for a session with deduplication."""

    def __init__(self):
        """Initialize the reference manager."""
        self._references: dict[str, Reference] = {}  # ref_id -> Reference
        self._unique_keys: dict[str, str] = {}  # unique_key -> ref_id
        self._display_numbers: dict[str, int] = {}  # ref_id -> display number
        self._counter = 0

    def add_reference(self, ref: Reference) -> str:
        """
        Add a reference, with automatic deduplication.

        Args:
            ref: Reference object to add

        Returns:
            Reference ID (existing if duplicate, new if unique)
        """
        try:
            # Check if this reference already exists
            unique_key = ref.get_unique_key()
            if unique_key in self._unique_keys:
                existing_id = self._unique_keys[unique_key]
                logger.debug(f"Duplicate reference found, using existing ID: {existing_id}")
                return existing_id

            # Add new reference
            self._references[ref.id] = ref
            self._unique_keys[unique_key] = ref.id

            # Assign display number
            self._counter += 1
            self._display_numbers[ref.id] = self._counter

            logger.debug(f"Added reference {ref.id} with display number {self._counter}")
            return ref.id

        except Exception as e:
            logger.warning(f"Error adding reference: {e}")
            return ref.id

    def add_references(self, refs: list[Reference]) -> list[str]:
        """
        Add multiple references.

        Args:
            refs: List of Reference objects

        Returns:
            List of reference IDs
        """
        return [self.add_reference(ref) for ref in refs]

    def get_reference(self, ref_id: str) -> Reference | None:
        """
        Get a reference by ID.

        Args:
            ref_id: Reference ID

        Returns:
            Reference object or None if not found
        """
        return self._references.get(ref_id)

    def get_display_number(self, ref_id: str) -> int | None:
        """
        Get the display number for a reference.

        Args:
            ref_id: Reference ID

        Returns:
            Display number or None if not found
        """
        return self._display_numbers.get(ref_id)

    def get_all_references(self) -> list[Reference]:
        """
        Get all references sorted by display number.

        Returns:
            List of Reference objects
        """
        sorted_refs = sorted(
            self._references.values(), key=lambda r: self._display_numbers.get(r.id, 0)
        )
        return sorted_refs

    def get_references_by_type(self, ref_type: str) -> list[Reference]:
        """
        Get all references of a specific type.

        Args:
            ref_type: Reference type (e.g., "paper", "database", "tool")

        Returns:
            List of Reference objects
        """
        return [ref for ref in self.get_all_references() if ref.type == ref_type]

    def deduplicate(self):
        """
        Deduplicate references (already done automatically in add_reference).

        This method is a no-op but kept for API compatibility.
        """
        pass

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize references to dictionary.

        Returns:
            Dictionary with references grouped by type
        """
        all_refs = self.get_all_references()
        return {
            "references": [ref.to_dict() for ref in all_refs],
            "by_type": {
                "papers": [ref.to_dict() for ref in all_refs if ref.type == "paper"],
                "databases": [ref.to_dict() for ref in all_refs if ref.type == "database"],
                "tools": [ref.to_dict() for ref in all_refs if ref.type == "tool"],
                "structures": [ref.to_dict() for ref in all_refs if ref.type == "structure"],
                "artifacts": [ref.to_dict() for ref in all_refs if ref.type == "artifact"],
            },
            "display_numbers": self._display_numbers,
            "count": len(self._references),
        }

    def get_reference_with_number(self, ref_id: str) -> dict[str, Any] | None:
        """
        Get reference with its display number.

        Args:
            ref_id: Reference ID

        Returns:
            Dictionary with reference data and display number
        """
        ref = self.get_reference(ref_id)
        if ref:
            ref_dict = ref.to_dict()
            ref_dict["display_number"] = self._display_numbers.get(ref_id)
            return ref_dict
        return None

    def clear(self):
        """Clear all references."""
        self._references.clear()
        self._unique_keys.clear()
        self._display_numbers.clear()
        self._counter = 0

    def __len__(self) -> int:
        """Get number of references."""
        return len(self._references)

    def __contains__(self, ref_id: str) -> bool:
        """Check if reference exists."""
        return ref_id in self._references
