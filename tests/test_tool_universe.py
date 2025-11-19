"""Tests for the ToolUniverse wrapper and LangChain tools."""

import json
from pathlib import Path

import pytest

from bioagents.tools import tool_universe as tu_module
from bioagents.tools.tool_universe import ToolUniverseCatalogue, ToolUniverseWrapper


class DummyCatalogue:
    """Simple stub that emulates catalogue search behaviour."""

    def search(self, query: str, limit: int):
        return [{"name": f"mock-{query}", "url": "https://example.com", "score": 1}]


class FakeToolUniverse:
    """Test double for the ToolUniverse SDK."""

    def __init__(self):
        self.load_count = 0
        self.run_calls = []

    def load_tools(self):
        self.load_count += 1

    def run(self, payload):
        self.run_calls.append(payload)
        return {"echo": payload}


def test_catalogue_search_uses_local_markdown(tmp_path: Path):
    """The catalogue should parse headings and links from tool_universe.md."""
    sample = """
    ### **AgenticTool**
    - [Tool_A](https://example.com/a)
    - [Tool_B](https://example.com/b)

    ### **APITool**
    - [OtherTool](https://example.com/c)
    """
    catalog_file = tmp_path / "tool_universe.md"
    catalog_file.write_text(sample, encoding="utf-8")

    catalogue = ToolUniverseCatalogue(catalog_path=catalog_file)
    results = catalogue.search("Tool", limit=2)

    assert len(results) == 2
    names = {entry["name"] for entry in results}
    assert "Tool_A" in names
    assert any(name in {"Tool_B", "OtherTool"} for name in names)
    assert all(entry["section"] for entry in results)


def test_wrapper_falls_back_to_catalogue():
    """When the SDK is missing, results should come from the markdown catalogue."""
    wrapper = ToolUniverseWrapper(tool_factory=None, catalog=DummyCatalogue())

    payload = json.loads(wrapper.find_tools("sequence alignment", limit=1))
    assert payload["source"] == "catalog"
    assert payload["results"][0]["name"].startswith("mock-sequence")


def test_wrapper_uses_sdk_when_available():
    """SDK-backed searches should call the Tool Finder tool exactly once per invocation."""
    fake_client = FakeToolUniverse()
    wrapper = ToolUniverseWrapper(tool_factory=lambda: fake_client)

    wrapper.find_tools("drug discovery", limit=2, finder="llm")
    wrapper.find_tools("drug discovery", limit=1, finder="llm")

    assert fake_client.load_count == 1
    assert len(fake_client.run_calls) == 2
    assert fake_client.run_calls[0]["name"] == "Tool_Finder_LLM"


def test_wrapper_execute_tool_invokes_sdk():
    """execute_tool should send the payload to the SDK with parsed arguments."""
    fake_client = FakeToolUniverse()
    wrapper = ToolUniverseWrapper(tool_factory=lambda: fake_client)

    response = json.loads(wrapper.execute_tool("ExampleTool", '{"gene": "TP53"}'))
    assert response["tool"] == "ExampleTool"

    last_call = fake_client.run_calls[-1]
    assert last_call["name"] == "ExampleTool"
    assert last_call["arguments"] == {"gene": "TP53"}


def test_wrapper_execute_tool_without_sdk():
    """An informative error should be raised when execute_tool is used without the SDK."""
    wrapper = ToolUniverseWrapper(tool_factory=None, catalog=DummyCatalogue())

    with pytest.raises(RuntimeError):
        wrapper.execute_tool("ExampleTool", "{}")


def test_langchain_tool_functions_use_wrapper(monkeypatch):
    """The exported LangChain tools should delegate to the default wrapper."""

    class StubWrapper:
        def find_tools(self, description, limit, finder):
            return json.dumps({"source": "stub", "description": description})

        def execute_tool(self, tool, arguments):
            return json.dumps({"tool": tool, "arguments": arguments})

    monkeypatch.setattr(tu_module, "DEFAULT_WRAPPER", StubWrapper())

    search_result = tu_module.tool_universe_find_tools.invoke(
        {"description": "protein structure", "limit": 3, "strategy": "keyword"}
    )
    assert json.loads(search_result)["source"] == "stub"

    call_result = tu_module.tool_universe_call_tool.invoke(
        {"tool_name": "Example", "arguments_json": '{"id": 1}'}
    )
    assert json.loads(call_result)["tool"] == "Example"
