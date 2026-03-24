"""RCSB PDB REST summary (no structure file download)."""

from __future__ import annotations

from typing import Any, ClassVar

import requests

from bioagents.workflows.node import WorkflowNode
from bioagents.workflows.schemas import NodeMetadata


class RcsbEntrySummaryNode(WorkflowNode):
    """Fetch core metadata for a PDB ID via data.rcsb.org."""

    workflow_type_id: ClassVar[str] = "rcsb_entry_summary"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="RCSB entry summary",
            description="REST summary for a PDB/mmCIF entry id (4-letter code).",
            version="1.0.0",
            category="data",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"pdb_id": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"summary": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        pid = str(inputs["pdb_id"]).strip().upper()
        if len(pid) < 4:
            return {"summary": f"Error: invalid pdb_id {pid!r}"}
        pid = pid[:4]
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pid}"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 404:
                return {"summary": f"Error: entry {pid} not found"}
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            return {"summary": f"Error fetching PDB {pid}: {exc!s}"}
        struct = data.get("struct", {}) or {}
        title = struct.get("title", "n/a")
        method = "n/a"
        if data.get("exptl") and isinstance(data["exptl"], list) and data["exptl"]:
            method = data["exptl"][0].get("method", "n/a")
        lines = [
            f"PDB ID: {pid}",
            f"Title: {title}",
            f"Method: {method}",
        ]
        return {"summary": "\n".join(lines)}
