"""Persistence robustness tests for ToolRegistry (issue #61)."""

from __future__ import annotations

import json
import multiprocessing
import sys
from pathlib import Path

import pytest

from bioagents.tools.tool_registry import ToolDefinition, ToolParameter, ToolRegistry

PROCESS_COUNT = 4
TOOLS_PER_PROCESS = 5


def _make_tool(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Tool {name}",
        category="testing",
        parameters=[ToolParameter(name="value", type="integer", description="A value")],
        return_type="integer",
        return_description="The value",
        code=f"def {name}(value: int) -> int:\n    return value\n",
    )


def _register_many(tools_dir: str, worker_id: int) -> None:
    registry = ToolRegistry(tools_dir=Path(tools_dir))
    for i in range(TOOLS_PER_PROCESS):
        assert registry.register_tool(_make_tool(f"tool_w{worker_id}_{i}"))


def _registry_json(tools_dir: Path) -> dict:
    return json.loads((tools_dir / ToolRegistry.REGISTRY_FILE).read_text(encoding="utf-8"))


def test_save_writes_valid_json_backup_and_no_temp_files(tmp_path):
    registry = ToolRegistry(tools_dir=tmp_path)

    registry.register_tool(_make_tool("alpha"))
    registry.register_tool(_make_tool("beta"))

    data = _registry_json(tmp_path)
    assert set(data["tools"]) == {"alpha", "beta"}
    backup = json.loads((tmp_path / "registry.json.bak").read_text(encoding="utf-8"))
    assert backup["tools"] == data["tools"]
    assert not list(tmp_path.glob(".registry.json*.tmp"))


def test_recovers_from_extra_data_trailing_garbage(tmp_path, caplog):
    ToolRegistry(tools_dir=tmp_path).register_tool(_make_tool("alpha"))
    registry_path = tmp_path / "registry.json"
    good = registry_path.read_text(encoding="utf-8")
    registry_path.write_text(good + '\n  }\n}"tail of another write', encoding="utf-8")
    (tmp_path / "registry.json.bak").unlink()

    with caplog.at_level("WARNING"):
        registry = ToolRegistry(tools_dir=tmp_path)

    assert registry.get_tool("alpha") is not None
    assert "corrupt" in caplog.text
    assert _registry_json(tmp_path)["tools"].keys() == {"alpha"}
    corrupt_copies = list(tmp_path.glob("registry.json.corrupt-*"))
    assert len(corrupt_copies) == 1
    assert corrupt_copies[0].read_text(encoding="utf-8").endswith("another write")


def test_recovers_from_backup_when_registry_unparseable(tmp_path, caplog):
    seed = ToolRegistry(tools_dir=tmp_path)
    seed.register_tool(_make_tool("alpha"))
    seed.register_tool(_make_tool("beta"))
    (tmp_path / "registry.json").write_text('{"version": "1.0", "tools": {', encoding="utf-8")

    with caplog.at_level("WARNING"):
        registry = ToolRegistry(tools_dir=tmp_path)

    assert {t.name for t in registry.list_tools()} == {"alpha", "beta"}
    assert "registry.json.bak" in caplog.text
    assert set(_registry_json(tmp_path)["tools"]) == {"alpha", "beta"}
    assert list(tmp_path.glob("registry.json.corrupt-*"))


def test_unrecoverable_registry_logs_error_and_keeps_corrupt_copy(tmp_path, caplog):
    (tmp_path / "registry.json").write_text("not json", encoding="utf-8")

    with caplog.at_level("ERROR"):
        registry = ToolRegistry(tools_dir=tmp_path)

    assert registry.list_tools() == []
    assert "Could not recover tool registry" in caplog.text
    assert list(tmp_path.glob("registry.json.corrupt-*"))


def test_save_merges_registrations_from_other_instances(tmp_path):
    first = ToolRegistry(tools_dir=tmp_path)
    second = ToolRegistry(tools_dir=tmp_path)

    first.register_tool(_make_tool("alpha"))
    second.register_tool(_make_tool("beta"))
    first.unregister_tool("alpha")

    assert set(_registry_json(tmp_path)["tools"]) == {"beta"}
    assert first.get_tool("beta") is not None


@pytest.mark.skipif(sys.platform == "win32", reason="relies on fork and fcntl")
def test_concurrent_multiprocess_registration_keeps_all_tools(tmp_path):
    ctx = multiprocessing.get_context("fork")
    processes = [
        ctx.Process(target=_register_many, args=(str(tmp_path), worker_id))
        for worker_id in range(PROCESS_COUNT)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0

    data = _registry_json(tmp_path)
    expected = {f"tool_w{w}_{i}" for w in range(PROCESS_COUNT) for i in range(TOOLS_PER_PROCESS)}
    assert set(data["tools"]) == expected
    assert len(ToolRegistry(tools_dir=tmp_path).list_tools()) == len(expected)
