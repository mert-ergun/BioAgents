"""Tool Registry for storing, managing, and loading custom-generated tools.

This module provides a persistent registry system for custom tools created by
the Tool Builder Agent. It handles:
- Saving tool definitions and code to disk
- Loading tools dynamically at runtime
- Validating tool schemas
- Providing tool metadata for RAG retrieval
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, cast

from bioagents.tools.result_status import tool_result_status

try:  # POSIX advisory locking; unavailable on e.g. Windows.
    import fcntl
except ImportError:  # pragma: no cover - platform dependent
    fcntl = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

logger = logging.getLogger(__name__)

DEFAULT_TOOLS_DIR = Path(__file__).parent / "custom_tools"


def normalize_tool_type(type_name: str) -> str:
    """Translate Python type aliases to schema/tool API types."""
    normalized = type_name.strip().lower()
    return {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "dict": "object",
        "list": "array",
    }.get(normalized, normalized)


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    type: str  # "string", "integer", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None

    def to_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format."""
        schema: dict[str, Any] = {
            "type": normalize_tool_type(self.type),
            "description": self.description,
        }
        if self.default is not None:
            schema["default"] = self.default
        if self.enum:
            schema["enum"] = self.enum
        return schema


@dataclass
class ToolDefinition:
    """Complete definition of a custom tool."""

    name: str
    description: str
    category: str  # e.g., "genomics", "proteomics", "structural", "ml"
    parameters: list[ToolParameter]
    return_type: str  # e.g., "string", "dict", "DataFrame"
    return_description: str

    source_paper: str | None = None  # DOI or bioRxiv ID
    source_url: str | None = None  # GitHub, documentation URL
    source_software: str | None = None  # Original software name

    code: str = ""  # Python code for the tool wrapper
    dependencies: list[str] = field(default_factory=list)  # pip packages
    system_dependencies: list[str] = field(default_factory=list)  # apt packages

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0.0"
    validated: bool = False
    validation_error: str | None = None

    usage_count: int = 0
    last_used: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["parameters"] = [asdict(p) for p in self.parameters]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolDefinition:
        """Create from dictionary."""
        params = [ToolParameter(**p) for p in data.pop("parameters", [])]
        return cls(parameters=params, **data)

    def get_schema(self) -> dict[str, Any]:
        """Get JSON Schema for the tool's parameters."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def get_function_signature(self) -> str:
        """Generate Python function signature."""
        params = []
        for p in self.parameters:
            if p.required:
                params.append(f"{p.name}: {self._type_hint(p.type)}")
            else:
                default = repr(p.default) if p.default is not None else "None"
                params.append(f"{p.name}: {self._type_hint(p.type)} = {default}")

        return (
            f"def {self._safe_name()}({', '.join(params)}) -> {self._type_hint(self.return_type)}:"
        )

    def _safe_name(self) -> str:
        """Convert tool name to valid Python function name."""
        name = re.sub(r"[^a-zA-Z0-9_]", "_", self.name)
        if name[0].isdigit():
            name = "_" + name
        return name.lower()

    @staticmethod
    def _type_hint(type_str: str) -> str:
        """Convert type string to Python type hint."""
        type_map = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "list",
            "object": "dict",
            "DataFrame": "pd.DataFrame",
            "dict": "dict[str, Any]",
        }
        return type_map.get(type_str, "Any")


class ToolRegistry:
    """Registry for managing custom tools.

    Provides persistent storage and dynamic loading of custom tools
    generated by the Tool Builder Agent.
    """

    REGISTRY_FILE = "registry.json"
    BACKUP_SUFFIX = ".bak"
    LOCK_SUFFIX = ".lock"
    DEFAULT_FILE_MODE = 0o644

    def __init__(self, tools_dir: Path | None = None):
        """Initialize the tool registry.

        Args:
            tools_dir: Directory for storing tool files. Defaults to custom_tools/.
        """
        self.tools_dir = tools_dir or DEFAULT_TOOLS_DIR
        self.tools_dir.mkdir(parents=True, exist_ok=True)

        self._registry: dict[str, ToolDefinition] = {}
        self._loaded_modules: dict[str, Any] = {}
        # Re-entrant: public methods hold it while calling _save_registry().
        self._lock = RLock()
        self._file_lock_depth = 0

        self._load_registry()

    def _registry_path(self) -> Path:
        """Get path to registry JSON file."""
        return self.tools_dir / self.REGISTRY_FILE

    def _backup_path(self) -> Path:
        """Get path to the last-known-good registry copy."""
        return self.tools_dir / f"{self.REGISTRY_FILE}{self.BACKUP_SUFFIX}"

    def _lock_path(self) -> Path:
        """Get path to the cross-process advisory lock file."""
        return self.tools_dir / f"{self.REGISTRY_FILE}{self.LOCK_SUFFIX}"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Hold the thread lock plus an exclusive cross-process file lock.

        Registry writers live in several OS processes (supervisor agents,
        ToolBuilder, the Docker-mounted sandbox), so a threading lock alone
        cannot serialise read-modify-write cycles. The file lock is an
        advisory ``flock`` and degrades to a no-op where unavailable.
        """
        with self._lock:
            if self._file_lock_depth > 0:
                # flock is per open file description; re-locking from a new fd
                # in the same process would deadlock, so only lock once.
                self._file_lock_depth += 1
                try:
                    yield
                finally:
                    self._file_lock_depth -= 1
                return

            fd = self._acquire_file_lock()
            self._file_lock_depth = 1
            try:
                yield
            finally:
                self._file_lock_depth = 0
                if fd is not None:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    finally:
                        os.close(fd)

    def _acquire_file_lock(self) -> int | None:
        """Open the lock file and take an exclusive flock; None if unsupported."""
        if fcntl is None:
            return None
        lock_path = self._lock_path()
        fd: int | None = None
        try:
            try:
                fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
            except PermissionError:
                # Lock file owned by another uid (e.g. sandbox); flock works on
                # read-only descriptors too.
                fd = os.open(lock_path, os.O_RDONLY)
            fcntl.flock(fd, fcntl.LOCK_EX)
            return fd
        except OSError as e:
            if fd is not None:
                os.close(fd)
            logger.warning(
                "Could not acquire registry file lock %s (%s); "
                "continuing without cross-process locking",
                lock_path,
                e,
            )
            return None

    def _load_registry(self) -> None:
        """Load registry from disk, recovering from corruption if possible."""
        with self._locked():
            tools_data = self._read_registry_file()
            if tools_data is None:
                return
            self._registry = self._parse_tools(tools_data)
            logger.info(f"Loaded {len(self._registry)} tools from registry")

    @staticmethod
    def _parse_tools(
        tools_data: dict[str, Any],
        fallback: dict[str, ToolDefinition] | None = None,
    ) -> dict[str, ToolDefinition]:
        """Convert raw registry entries into ToolDefinitions."""
        parsed: dict[str, ToolDefinition] = {}
        for name, tool_data in tools_data.items():
            try:
                parsed[name] = ToolDefinition.from_dict(dict(tool_data))
            except Exception as e:
                logger.warning(f"Failed to load tool '{name}': {e}")
                if fallback and name in fallback:
                    parsed[name] = fallback[name]
        return parsed

    @staticmethod
    def _extract_tools(data: Any) -> dict[str, Any]:
        """Return the ``tools`` mapping from a registry document or raise."""
        if not isinstance(data, dict):
            raise ValueError("registry root is not a JSON object")
        tools = data.get("tools", {})
        if not isinstance(tools, dict):
            raise ValueError("registry 'tools' entry is not a JSON object")
        return tools

    def _read_registry_file(self) -> dict[str, Any] | None:
        """Read the on-disk ``tools`` mapping. Caller must hold ``_locked()``.

        Returns None when no registry exists or it is unrecoverable. On
        corruption the bad file is preserved, recovery is attempted (first
        JSON document for "Extra data" errors, then the ``.bak`` copy), and a
        repaired registry is written back.
        """
        registry_path = self._registry_path()
        if not registry_path.exists():
            logger.info("No existing registry found, starting fresh")
            return None

        try:
            text = registry_path.read_text(encoding="utf-8")
            return self._extract_tools(json.loads(text))
        except (OSError, UnicodeDecodeError) as e:
            logger.error("Failed to read registry %s: %s", registry_path, e)
            return None
        except ValueError as e:  # includes json.JSONDecodeError
            logger.error("Tool registry %s is corrupt (%s); attempting recovery", registry_path, e)
            error = e

        corrupt_path = self._preserve_corrupt_file(registry_path)
        recovered = self._recover(text, error)
        if recovered is None:
            logger.error(
                "Could not recover tool registry %s; custom tools are unavailable "
                "(corrupt copy kept at %s)",
                registry_path,
                corrupt_path,
            )
            return None

        tools, source = recovered
        logger.warning(
            "Recovered %d tools for registry %s from %s (corrupt copy kept at %s)",
            len(tools),
            registry_path,
            source,
            corrupt_path,
        )
        self._write_registry_data(tools)
        return tools

    def _recover(self, text: str, error: ValueError) -> tuple[dict[str, Any], str] | None:
        """Try to salvage registry contents after a parse failure."""
        if isinstance(error, json.JSONDecodeError) and error.msg == "Extra data":
            try:
                data, _ = json.JSONDecoder().raw_decode(text.lstrip())
                return self._extract_tools(data), "first JSON document"
            except ValueError as e:
                logger.warning("Could not salvage first JSON document: %s", e)

        backup_path = self._backup_path()
        if backup_path.exists():
            try:
                data = json.loads(backup_path.read_text(encoding="utf-8"))
                return self._extract_tools(data), str(backup_path)
            except (OSError, ValueError) as e:
                logger.error("Registry backup %s is also unusable: %s", backup_path, e)
        return None

    def _preserve_corrupt_file(self, registry_path: Path) -> Path | None:
        """Copy a corrupt registry aside so it can be inspected later."""
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        corrupt_path = registry_path.with_name(f"{registry_path.name}.corrupt-{stamp}")
        try:
            shutil.copy2(registry_path, corrupt_path)
            return corrupt_path
        except OSError as e:
            logger.error("Failed to preserve corrupt registry %s: %s", registry_path, e)
            return None

    def _sync_from_disk(self, keep: Iterable[str] = ()) -> None:
        """Merge the on-disk registry into memory. Caller must hold ``_locked()``.

        On-disk entries win so other processes' registrations and deletions are
        not clobbered, except for names in ``keep`` (the entries this process
        is changing right now): for those the in-memory state is authoritative,
        including absence, so a tool deleted in memory stays deleted.
        """
        tools_data = self._read_registry_file()
        if tools_data is None:
            return

        keep_names = set(keep)
        merged = self._parse_tools(
            {n: d for n, d in tools_data.items() if n not in keep_names},
            fallback=self._registry,
        )
        for name in keep_names:
            if name in self._registry:
                merged[name] = self._registry[name]
        self._registry = merged

    def _save_registry(self, changed: Iterable[str] = ()) -> None:
        """Merge with the on-disk registry and persist it atomically.

        Args:
            changed: Names of tools modified in memory by the current operation.
        """
        with self._locked():
            self._sync_from_disk(keep=changed)
            self._write_registry_data(
                {name: tool.to_dict() for name, tool in self._registry.items()}
            )

    def _write_registry_data(self, tools: dict[str, Any]) -> None:
        """Atomically write registry.json and refresh the ``.bak`` copy."""
        data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "tools": tools,
        }
        try:
            payload = json.dumps(data, indent=2, ensure_ascii=False)
            self._atomic_write(self._registry_path(), payload)
            self._atomic_write(self._backup_path(), payload)
        except Exception as e:
            logger.error(f"Failed to save registry {self._registry_path()}: {e}")

    def _atomic_write(self, path: Path, payload: str) -> None:
        """Write ``payload`` to a temp file in the same dir, fsync, then rename."""
        try:
            mode = path.stat().st_mode & 0o777
        except OSError:
            mode = self.DEFAULT_FILE_MODE

        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            tmp_path.chmod(mode)
            tmp_path.replace(path)
        except BaseException:
            with suppress(OSError):
                tmp_path.unlink()
            raise

    def register_tool(
        self,
        definition: ToolDefinition,
        overwrite: bool = False,
    ) -> bool:
        """Register a new tool in the registry.

        Args:
            definition: The tool definition to register
            overwrite: Whether to overwrite existing tool with same name

        Returns:
            True if registration succeeded
        """
        with self._locked():
            self._sync_from_disk()
            if definition.name in self._registry and not overwrite:
                logger.warning(f"Tool '{definition.name}' already exists")
                return False

            if definition.code:
                code_path = self._get_code_path(definition.name)
                try:
                    code_path.write_text(definition.code, encoding="utf-8")
                    logger.info(f"Saved tool code to {code_path}")
                except Exception as e:
                    logger.error(f"Failed to save tool code: {e}")
                    return False

            definition.updated_at = datetime.now().isoformat()
            self._registry[definition.name] = definition
            self._save_registry(changed=[definition.name])

            logger.info(f"Registered tool: {definition.name}")
            return True

    def unregister_tool(self, name: str) -> bool:
        """Remove a tool from the registry.

        Args:
            name: Name of the tool to remove

        Returns:
            True if tool was removed
        """
        with self._locked():
            self._sync_from_disk()
            if name not in self._registry:
                return False

            code_path = self._get_code_path(name)
            if code_path.exists():
                code_path.unlink()

            del self._registry[name]
            if name in self._loaded_modules:
                del self._loaded_modules[name]

            self._save_registry(changed=[name])
            logger.info(f"Unregistered tool: {name}")
            return True

    def get_tool(self, name: str) -> ToolDefinition | None:
        """Get a tool definition by name."""
        return self._registry.get(name)

    def list_tools(
        self,
        category: str | None = None,
        validated_only: bool = False,
    ) -> list[ToolDefinition]:
        """List all registered tools.

        Args:
            category: Filter by category
            validated_only: Only return validated tools

        Returns:
            List of tool definitions
        """
        tools = list(self._registry.values())

        if category:
            tools = [t for t in tools if t.category == category]

        if validated_only:
            tools = [t for t in tools if t.validated]

        return tools

    def search_tools(
        self,
        query: str,
        limit: int = 5,
    ) -> list[tuple[ToolDefinition, float]]:
        """Search tools by description similarity.

        Simple keyword-based search. For production, integrate with
        a vector database for semantic search.

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            List of (tool, score) tuples
        """
        query_tokens = set(query.lower().split())
        scored_tools: list[tuple[ToolDefinition, float]] = []

        for tool in self._registry.values():
            text = f"{tool.name} {tool.description} {tool.category}".lower()
            text_tokens = set(text.split())

            overlap = len(query_tokens & text_tokens)
            if overlap > 0:
                score = overlap / len(query_tokens)
                scored_tools.append((tool, score))

        scored_tools.sort(key=lambda x: x[1], reverse=True)
        return scored_tools[:limit]

    def _get_code_path(self, name: str) -> Path:
        """Get path to tool's Python code file."""
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
        return self.tools_dir / f"{safe_name}.py"

    def load_tool_function(self, name: str, force_reload: bool = False) -> Callable | None:
        """Dynamically load and return a tool's function.

        Args:
            name: Name of the tool to load
            force_reload: If True, force reload even if already loaded (clears cache)

        Returns:
            The callable tool function, or None if load failed
        """
        if force_reload and name in self._loaded_modules:
            module_name = f"bioagents_custom_tool_{name}"
            if module_name in sys.modules:
                del sys.modules[module_name]
            del self._loaded_modules[name]

        if name in self._loaded_modules:
            module = self._loaded_modules[name]
            func_name = self._registry[name]._safe_name()
            func = getattr(module, func_name, None)
            return cast("Callable[..., Any]", func) if callable(func) else None

        tool = self._registry.get(name)
        if not tool:
            logger.error(f"Tool '{name}' not found in registry")
            return None

        code_path = self._get_code_path(name)
        if not code_path.exists():
            logger.error(f"Tool code file not found: {code_path}")
            return None

        try:
            module_name = f"bioagents_custom_tool_{name}"

            if module_name in sys.modules:
                del sys.modules[module_name]

            spec = importlib.util.spec_from_file_location(module_name, code_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot create module spec for {code_path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            self._loaded_modules[name] = module

            func_name = tool._safe_name()
            func = getattr(module, func_name, None)

            if not callable(func):
                logger.error("Tool '%s' must define a callable named '%s'", name, func_name)
                return None
            return cast("Callable[..., Any]", func)

        except Exception as e:
            logger.error(f"Failed to load tool '{name}': {e}")
            tool.validation_error = f"{type(e).__name__}: {e}"
            return None

    def validate_tool(self, name: str, test_args: dict[str, Any] | None = None) -> bool:
        """Validate a tool by attempting to load and optionally run it.

        Args:
            name: Name of the tool to validate
            test_args: Optional test arguments to run the tool with

        Returns:
            True if validation passed
        """
        tool = self._registry.get(name)
        if not tool:
            return False

        try:
            tool.validation_error = None
            func = self.load_tool_function(name, force_reload=True)
            if func is None:
                tool.validated = False
                tool.validation_error = tool.validation_error or "Failed to load function"
                self._save_registry(changed=[name])
                return False

            if test_args is not None:
                if not isinstance(test_args, dict):
                    raise ValueError("Test arguments must be a JSON object")
                inspect.signature(func).bind(**test_args)
                result = func(**test_args)
                status = tool_result_status(result)
                if status != "success":
                    raise ValueError(f"Tool execution did not succeed (status={status}): {result}")

            tool.validated = True
            tool.validation_error = None
            tool.updated_at = datetime.now().isoformat()
            self._save_registry(changed=[name])

            logger.info(f"Tool '{name}' validated successfully")
            return True

        except Exception as e:
            tool.validated = False
            tool.validation_error = str(e)
            self._save_registry(changed=[name])
            logger.error(f"Tool '{name}' validation failed: {e}")
            return False

    def record_usage(self, name: str) -> None:
        """Record that a tool was used."""
        tool = self._registry.get(name)
        if tool:
            tool.usage_count += 1
            tool.last_used = datetime.now().isoformat()
            self._save_registry(changed=[name])

    def get_categories(self) -> list[str]:
        """Get all unique tool categories."""
        return list({t.category for t in self._registry.values()})

    def export_for_context(self, names: list[str] | None = None) -> str:
        """Export tool documentation for LLM context injection.

        Args:
            names: Specific tools to export, or None for all

        Returns:
            Formatted string with tool documentation
        """
        tools = (
            [self._registry[n] for n in names if n in self._registry]
            if names
            else list(self._registry.values())
        )

        if not tools:
            return "No custom tools available."

        lines = ["# Available Custom Tools\n"]

        for tool in tools:
            lines.append(f"## {tool.name}")
            lines.append(f"**Category:** {tool.category}")
            lines.append(f"**Description:** {tool.description}")

            if tool.parameters:
                lines.append("\n**Parameters:**")
                for p in tool.parameters:
                    req = "(required)" if p.required else "(optional)"
                    lines.append(f"- `{p.name}` ({p.type}) {req}: {p.description}")

            lines.append(f"\n**Returns:** {tool.return_type} - {tool.return_description}")

            if tool.source_software:
                lines.append(f"\n**Wraps:** {tool.source_software}")

            lines.append("")

        return "\n".join(lines)


_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """Get or create the global tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
