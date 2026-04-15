"""Wrappers that expose the ToolUniverse catalog and SDK as LangChain tools."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import ClassVar

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from langchain_core.tools import tool

from bioagents.tools.tool_policy import ToolPolicy, get_default_policy

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
        policy: ToolPolicy | None = None,
    ):
        self._tool_factory = tool_factory
        self._catalog = catalog or ToolUniverseCatalogue()
        self._client = None
        self._lock = Lock()
        self._default_finder = os.getenv("BIOAGENTS_TOOL_UNIVERSE_FINDER", "keyword").lower()
        self._policy = policy or get_default_policy()

    # ------------------------------------------------------------------
    # Custom tool registry integration
    # ------------------------------------------------------------------

    @staticmethod
    def _search_custom_registry(query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search the local custom tool registry and return results in SDK format."""
        try:
            from bioagents.tools.tool_registry import get_registry

            registry = get_registry()
            hits = registry.search_tools(query, limit=limit)
            results: list[dict[str, Any]] = []
            for tool_def, _score in hits:
                params: dict[str, Any] = {}
                required: list[str] = []
                for p in tool_def.parameters:
                    params[p.name] = {
                        "type": p.type,
                        "description": p.description,
                        "required": p.required,
                    }
                    if p.required:
                        required.append(p.name)
                results.append(
                    {
                        "name": tool_def.name,
                        "description": tool_def.description,
                        "parameter": {"type": "object", "properties": params},
                        "required": required,
                        "source": "custom_registry",
                    }
                )
            return results
        except Exception as exc:
            logger.debug("Custom registry search failed (non-fatal): %s", exc)
            return []

    @staticmethod
    def _execute_custom_tool(tool_name: str, arguments: dict[str, Any]) -> tuple[bool, Any]:
        """Try executing a tool from the custom registry. Returns (found, result)."""
        try:
            from bioagents.tools.tool_registry import get_registry

            registry = get_registry()
            tool_def = registry.get_tool(tool_name)
            if tool_def is None:
                return False, None
            func = registry.load_tool_function(tool_name)
            if func is None:
                return True, {"error": f"Tool '{tool_name}' found but failed to load"}

            import time
            from concurrent.futures import TimeoutError as FuturesTimeout

            from bioagents.llms.timeout_llm import _workflow_deadline

            hard_cap: int | float = 30
            if _workflow_deadline is not None:
                remaining = _workflow_deadline - time.monotonic()
                hard_cap = max(5, min(hard_cap, remaining - 10))

            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(func, **arguments)
                try:
                    result = fut.result(timeout=hard_cap)
                except FuturesTimeout:
                    logger.warning("Custom tool '%s' timed out after %.0fs", tool_name, hard_cap)
                    result = f"Tool '{tool_name}' timed out after {hard_cap:.0f}s — partial results unavailable."

            registry.record_usage(tool_name)
            return True, result
        except Exception as exc:
            logger.warning("Custom tool execution failed for '%s': %s", tool_name, exc)
            return True, {"error": str(exc)}

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
    def _resolve_result(result: Any) -> Any:
        """Await the result if the SDK returned a coroutine instead of a value.

        The ToolUniverse SDK may delegate to ``_run_async`` internally.  When
        called from synchronous code the ``client.run()`` call can return a
        bare coroutine object.  This helper transparently awaits it so callers
        always receive a concrete value.
        """
        if not inspect.isawaitable(result):
            return result
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(result)  # type: ignore[arg-type]
        # Already inside an event loop (e.g. Jupyter kernel) — offload to a
        # background thread so we can call asyncio.run() without conflict.
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, result).result(timeout=60)  # type: ignore[arg-type]

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
            logger.warning(
                "Unknown ToolUniverse finder strategy '%s'; falling back to 'keyword'.",
                mode,
            )
            mode = "keyword"
        return mode

    @staticmethod
    def _coerce_tool_entry(item: Any) -> dict[str, Any]:
        """Build a consistent tool dict for JSON consumers (name, description, parameters)."""
        if isinstance(item, dict):
            name = item.get("name") or item.get("tool_name") or ""
            param = item.get("parameter")
            if param is None:
                param = item.get("parameters") or {}
            req = item.get("required")
            if not isinstance(req, list):
                req = []
            return {
                "name": str(name),
                "description": str(item.get("description", "") or ""),
                "parameter": param if isinstance(param, dict) else {},
                "required": req,
            }
        if isinstance(item, str):
            return {"name": item, "description": "", "parameter": {}, "required": []}
        return {"name": repr(item), "description": "", "parameter": {}, "required": []}

    def _normalize_find_tools_sdk_result(self, raw: Any) -> list[dict[str, Any]]:
        """
        Coerce ToolUniverse ``client.run`` output into a list of tool metadata dicts.

        The SDK returns different shapes depending on the finder: formatted prompt strings,
        JSON strings, tuples of (prompts, tool names), or lists of tool names. Callers
        (including generated notebook code) expect ``result`` to be a list of dicts with
        at least ``name`` and ``description``.
        """
        if raw is None:
            return []

        if isinstance(raw, tuple) and len(raw) == 2:
            prompts, names = raw[0], raw[1]
            if not isinstance(names, list):
                return []
            tools: list[dict[str, Any]] = []
            for i, name in enumerate(names):
                if not isinstance(name, str):
                    tools.append(self._coerce_tool_entry(name))
                    continue
                desc = ""
                parameter: dict[str, Any] = {}
                required: list[Any] = []
                if isinstance(prompts, list) and i < len(prompts):
                    p = prompts[i]
                    if isinstance(p, dict):
                        desc = str(p.get("description", "") or "")
                        parameter = p.get("parameter") or p.get("parameters") or {}
                        if not isinstance(parameter, dict):
                            parameter = {}
                        req = p.get("required")
                        required = list(req) if isinstance(req, list) else []
                tools.append(
                    {
                        "name": name,
                        "description": desc,
                        "parameter": parameter,
                        "required": required,
                    }
                )
            return tools

        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return []
            if text.startswith("{") or text.startswith("["):
                try:
                    parsed: Any = json.loads(text)
                except json.JSONDecodeError:
                    return []
                if isinstance(parsed, dict):
                    if "tools" in parsed and isinstance(parsed["tools"], list):
                        return [self._coerce_tool_entry(x) for x in parsed["tools"]]
                    if "error" in parsed and parsed.get("tools") == []:
                        return []
                    inner = parsed.get("result")
                    if inner is not None:
                        return self._normalize_find_tools_sdk_result(inner)
                    return []
                if isinstance(parsed, list):
                    return [self._coerce_tool_entry(x) for x in parsed]
            return []

        if isinstance(raw, dict):
            if "tools" in raw and isinstance(raw["tools"], list):
                return [self._coerce_tool_entry(x) for x in raw["tools"]]
            inner = raw.get("result")
            if inner is not None:
                return self._normalize_find_tools_sdk_result(inner)
            return []

        if isinstance(raw, list):
            return [self._coerce_tool_entry(x) for x in raw]

        return []

    def find_tools(self, description: str, limit: int = 5, finder: str | None = None) -> str:
        """
        Run the Tool Finder tool (or the local fallback) to identify relevant tools.

        Also searches the local custom tool registry and merges results so that
        tools created by the ToolBuilder agent are always discoverable.

        Args:
            description: A natural language description of the needed capability.
            limit: Maximum number of tools to return.
            finder: One of ``keyword`` (default), ``llm``, or ``embedding``.
        """
        limit = max(1, int(limit))
        finder_mode = self._resolve_finder(finder)

        # Always search custom registry first — these are purpose-built tools
        custom_hits = self._search_custom_registry(description, limit=limit)

        if self.client_available:
            client = self._ensure_client()

            logger.info(f"ToolUniverse search query: '{description}' (limit={limit})")

            payload = {
                "name": self.FINDER_TO_TOOL[finder_mode],
                "arguments": {
                    "description": description,
                    "limit": limit,
                    "return_call_result": True,
                },
            }
            result = self._resolve_result(client.run(payload))
            normalized = self._normalize_find_tools_sdk_result(result)

            # Merge: custom tools first, then SDK results (deduplicated)
            seen_names = {h["name"] for h in custom_hits}
            merged = list(custom_hits)
            for tool_entry in normalized:
                if tool_entry.get("name") not in seen_names:
                    merged.append(tool_entry)
            merged = merged[:limit]

            # Apply policy filter to remove unrelated tools
            merged = self._policy.filter_find_results(merged, limit=limit)

            return self._format_response(
                {
                    "source": "sdk",
                    "finder": finder_mode,
                    "query": description,
                    "result": merged,
                }
            )

        fallback_hits = self._catalog.search(description, limit)
        # Merge custom tools with fallback catalog
        seen_names = {h["name"] for h in custom_hits}
        merged = list(custom_hits)
        for hit in fallback_hits:
            if hit.get("name") not in seen_names:
                merged.append(hit)
        merged = merged[:limit]

        # Apply policy filter to catalog fallback results too
        merged = self._policy.filter_find_results(merged, limit=limit)

        return self._format_response(
            {
                "source": "catalog",
                "finder": "markdown_fallback",
                "query": description,
                "results": merged,
                "note": "Install the 'tooluniverse' package to run live tool searches.",
            }
        )

    def execute_tool(self, tool_name: str, arguments: str | dict[str, Any] | None = None) -> str:
        """
        Call a tool by name — checks policy first, then custom registry, then the SDK.

        Args:
            tool_name: Exact identifier of the tool to invoke.
            arguments: JSON string or dictionary of tool arguments.
        """
        parsed_args = self._parse_arguments(arguments)

        # --- Policy gate ---
        policy_result = self._policy.evaluate(tool_name, parsed_args)
        if not policy_result.allowed:
            logger.warning("Tool policy blocked tool '%s': %s", tool_name, policy_result.reason)
            return self._format_response(
                {
                    "tool": tool_name,
                    "error": f"Tool blocked by policy: {policy_result.reason}",
                    "policy": {
                        "allowed": False,
                        "reason": policy_result.reason,
                        "category": policy_result.category,
                    },
                }
            )

        # If policy requires approval, note it in the response metadata.
        # The actual approval gate is handled at the tool-node level (see
        # truncating_tool_node.py) which can pause execution for user input.
        # Here we just flag it so the caller knows.
        if policy_result.requires_approval:
            logger.info(
                "Tool '%s' requires approval (risk: %s): %s",
                tool_name,
                policy_result.risk_level,
                policy_result.reason,
            )

        # Try custom registry first
        found, result = self._execute_custom_tool(tool_name, parsed_args)
        if found:
            return self._format_response({"tool": tool_name, "result": result})

        # Fall back to ToolUniverse SDK
        client = self._ensure_client()
        payload = {
            "name": tool_name,
            "arguments": parsed_args,
        }
        result = self._resolve_result(client.run(payload))
        return self._format_response({"tool": tool_name, "result": result})


DEFAULT_WRAPPER = ToolUniverseWrapper()


@tool
def tool_universe_find_tools(description: str, limit: int = 5, strategy: str = "keyword") -> str:
    """
    Search ToolUniverse for tools that match the provided description.

    Uses the live ToolUniverse SDK when available. If the SDK is not installed, the tool falls
    back to the locally cached ``tool_universe.md`` catalogue so agents can still inspect the
    ecosystem enumerated at https://aiscientist.tools/.

    Args:
        description: Natural language description of the capability you need.
        limit: Maximum number of tools to return (default 5).
        strategy: Search method — must be one of ``keyword``, ``llm``, or ``embedding``.
            Defaults to ``keyword``. Any other value is silently replaced with ``keyword``.
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
