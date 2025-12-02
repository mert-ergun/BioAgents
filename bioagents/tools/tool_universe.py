"""Wrappers that expose the ToolUniverse catalog and SDK as LangChain tools."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, ClassVar
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from tooluniverse import ToolUniverse as _ToolUniverse
except ImportError:  # pragma: no cover - optional dependency
    _ToolUniverse = None


@dataclass
class CatalogEntry:
    """Simple representation of a ToolUniverse entry parsed from the markdown catalog."""

    name: str
    url: str
    section: str | None = None

    def to_dict(self, score: int | None = None) -> dict[str, Any]:
        """Convert the entry to a serializable dict."""
        data: dict[str, Any] = {
            "name": self.name,
            "url": self.url,
        }
        if self.section:
            data["section"] = self.section
        if score is not None:
            data["score"] = score
        return data


class ToolUniverseCatalogue:
    """Fallback catalogue that is built from the local ``tool_universe.md`` document."""

    _link_pattern = re.compile(r"- \[(?P<name>[^\]]+)\]\((?P<url>[^)]+)\)")

    def __init__(self, catalog_path: Path | None = None):
        self.catalog_path = (
            catalog_path if catalog_path is not None else Path("bioagents/tools/tool_universe.md")
        )
        self._entries: list[CatalogEntry] | None = None

    def _load_entries(self) -> list[CatalogEntry]:
        entries: list[CatalogEntry] = []
        if not self.catalog_path.exists():
            logger.warning("ToolUniverse catalog file not found at %s", self.catalog_path)
            return entries

        current_section: str | None = None
        with self.catalog_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue

                if line.startswith("### "):
                    current_section = line.lstrip("# ").strip()
                    continue

                match = self._link_pattern.match(line)
                if match:
                    entries.append(
                        CatalogEntry(
                            name=match.group("name"),
                            url=match.group("url"),
                            section=current_section,
                        )
                    )
        return entries

    @property
    def entries(self) -> list[CatalogEntry]:
        """Return cached entries, loading them on demand."""
        if self._entries is None:
            self._entries = self._load_entries()
        return self._entries

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """
        Perform a lightweight string search over the local catalogue.

        Args:
            query: Free-text description of the capability.
            limit: Maximum number of hits to return.
        """
        if not self.entries:
            return []

        limit = max(1, limit)
        tokens = [token for token in re.split(r"\W+", query.lower()) if token]
        if not tokens:
            candidates = self.entries[:limit]
            return [entry.to_dict(score=0) for entry in candidates]

        scored: list[tuple[int, CatalogEntry]] = []
        for entry in self.entries:
            name_lower = entry.name.lower()
            section_lower = (entry.section or "").lower()
            score = 0

            for token in tokens:
                if token in name_lower:
                    score += 3
                elif token in section_lower:
                    score += 1

            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda item: (-item[0], item[1].name))
        top_hits = scored[:limit]
        return [entry.to_dict(score=score) for score, entry in top_hits]


class ToolUniverseWrapper:
    """Encapsulates the ToolUniverse SDK with a graceful fallback to the catalogue."""

    FINDER_TO_TOOL: ClassVar[dict[str, str]] = {
        "keyword": "Tool_Finder_Keyword",
        "llm": "Tool_Finder_LLM",
        "embedding": "Tool_Finder",
    }

    def __init__(
        self,
        tool_factory: Callable[[], Any] | None = _ToolUniverse,
        catalog: ToolUniverseCatalogue | None = None,
    ):
        self._tool_factory = tool_factory
        self._catalog = catalog or ToolUniverseCatalogue()
        self._client = None
        self._lock = Lock()
        self._default_finder = os.getenv("BIOAGENTS_TOOL_UNIVERSE_FINDER", "keyword").lower()

    @property
    def client_available(self) -> bool:
        """Whether the Python SDK is available in the runtime environment."""
        return self._tool_factory is not None

    def _ensure_client(self):
        if not self.client_available:
            raise RuntimeError(
                "The 'tooluniverse' package is not installed. Install it to call live tools."
            )

        if self._client is not None:
            return self._client

        with self._lock:
            if self._client is None:
                logger.info("Loading ToolUniverse SDK (this may take a moment)...")
                if not callable(self._tool_factory):
                    raise RuntimeError(
                        "Tool factory is not callable (tooluniverse package missing)."
                    )
                self._client = self._tool_factory()
                if self._client is not None:
                    self._client.load_tools()
                else:
                    raise RuntimeError("Failed to load ToolUniverse SDK")
        return self._client

    @staticmethod
    def _format_response(payload: Any) -> str:
        return json.dumps(payload, indent=2, ensure_ascii=False, default=str)

    @staticmethod
    def _parse_arguments(arguments: str | dict[str, Any] | None) -> dict[str, Any]:
        if arguments is None:
            return {}
        if isinstance(arguments, dict):
            return arguments

        if isinstance(arguments, str):
            text = arguments.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("arguments must be a JSON object string") from exc

            if not isinstance(parsed, dict):
                raise ValueError("arguments JSON must describe an object")
            return parsed

        raise TypeError("arguments must be a dict, JSON string, or None")

    def _resolve_finder(self, finder: str | None) -> str:
        mode = (finder or self._default_finder or "keyword").lower()
        if mode not in self.FINDER_TO_TOOL:
            raise ValueError(
                f"Unknown ToolUniverse finder strategy '{mode}'. "
                f"Choose from {', '.join(self.FINDER_TO_TOOL)}."
            )
        return mode

    def find_tools(self, description: str, limit: int = 5, finder: str | None = None) -> str:
        """
        Run the Tool Finder tool (or the local fallback) to identify relevant tools.

        Args:
            description: A natural language description of the needed capability.
            limit: Maximum number of tools to return.
            finder: One of ``keyword`` (default), ``llm``, or ``embedding``.
        """
        limit = max(1, int(limit))
        finder_mode = self._resolve_finder(finder)

        if self.client_available:
            client = self._ensure_client()
            payload = {
                "name": self.FINDER_TO_TOOL[finder_mode],
                "arguments": {
                    "description": description,
                    "limit": limit,
                },
            }
            result = client.run(payload)
            return self._format_response(
                {
                    "source": "sdk",
                    "finder": finder_mode,
                    "query": description,
                    "result": result,
                }
            )

        fallback_hits = self._catalog.search(description, limit)
        return self._format_response(
            {
                "source": "catalog",
                "finder": "markdown_fallback",
                "query": description,
                "results": fallback_hits,
                "note": "Install the 'tooluniverse' package to run live tool searches.",
            }
        )

    def execute_tool(self, tool_name: str, arguments: str | dict[str, Any] | None = None) -> str:
        """
        Call a ToolUniverse tool by name via the SDK.

        Args:
            tool_name: Exact identifier of the tool to invoke.
            arguments: JSON string or dictionary of tool arguments.
        """
        client = self._ensure_client()
        payload = {
            "name": tool_name,
            "arguments": self._parse_arguments(arguments),
        }
        result = client.run(payload)
        return self._format_response({"tool": tool_name, "result": result})


DEFAULT_WRAPPER = ToolUniverseWrapper()


@tool
def tool_universe_find_tools(description: str, limit: int = 5, strategy: str = "keyword") -> str:
    """
    Search ToolUniverse for tools that match the provided description.

    Uses the live ToolUniverse SDK when available. If the SDK is not installed, the tool falls
    back to the locally cached ``tool_universe.md`` catalogue so agents can still inspect the
    ecosystem enumerated at https://aiscientist.tools/.
    """
    try:
        return DEFAULT_WRAPPER.find_tools(description, limit=int(limit), finder=strategy)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("ToolUniverse search failed")
        return f"Error searching ToolUniverse: {exc}"


@tool
def tool_universe_call_tool(tool_name: str, arguments_json: str = "") -> str:
    """
    Execute a specific ToolUniverse tool.

    Args:
        tool_name: Exact tool identifier, e.g., ``OpenTargets_get_associated_targets_by_disease_efoId``.
        arguments_json: JSON string describing the tool arguments.
    """
    try:
        return DEFAULT_WRAPPER.execute_tool(tool_name, arguments_json or None)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("ToolUniverse execution failed")
        return f"Error executing ToolUniverse tool '{tool_name}': {exc}"
