"""Tests for hybrid tool discovery, metadata audit, provenance and local tools (issue #63)."""

import json

import pytest

from bioagents.tools import tool_discovery
from bioagents.tools.tool_universe import ToolUniverseWrapper

# Shaped like the registry entry seen in the benchmark runs: the name says Boltz-2
# co-folding, the description and the only required parameter describe a chemical
# component lookup.
BOLTZ2_LIKE = {
    "name": "NvidiaNIM_boltz2",
    "description": (
        "Retrieve chemical component information from the PDB Chemical Component "
        "Dictionary by component identifier."
    ),
    "parameter": {
        "type": "object",
        "properties": {"comp_id": {"type": "string", "description": "Component ID"}},
        "required": ["comp_id"],
    },
}

CONSISTENT_ENTRIES = [
    {
        "name": "UniProt_get_entry_by_accession",
        "description": "Get the full UniProtKB entry for a protein accession.",
        "parameter": {
            "type": "object",
            "properties": {"accession": {"type": "string"}},
            "required": ["accession"],
        },
    },
    {
        "name": "vina_dock_ligands",
        "description": "Dock ligands into a receptor with AutoDock Vina.",
        "parameter": {
            "type": "object",
            "properties": {"receptor_pdb": {"type": "string"}, "ligands": {"type": "array"}},
        },
    },
    {
        "name": "EuropePMC_search_articles",
        "description": "Search Europe PMC for scientific literature articles.",
        "parameter": {"query": {"type": "string", "required": True}},
    },
]


class ScriptedSDK:
    """ToolUniverse double whose finders return scripted hits or raise."""

    def __init__(self, hits_by_finder, failing=()):
        self.hits_by_finder = hits_by_finder
        self.failing = set(failing)
        self.run_calls = []

    def load_tools(self):
        pass

    def run(self, payload):
        self.run_calls.append(payload["name"])
        if payload["name"] in self.failing:
            raise RuntimeError(f"{payload['name']} could not load its model")
        return {"tools": self.hits_by_finder.get(payload["name"], [])}


def _tool(name, description="does a thing with proteins"):
    return {"name": name, "description": description, "parameter": {}}


@pytest.fixture(autouse=True)
def _no_custom_registry(monkeypatch):
    monkeypatch.setattr(
        ToolUniverseWrapper, "_search_custom_registry", staticmethod(lambda query, limit=5: [])
    )
    monkeypatch.delenv("BIOAGENTS_TOOL_UNIVERSE_FINDER", raising=False)
    monkeypatch.delenv("BIOAGENTS_TOOL_UNIVERSE_SEMANTIC_FINDERS", raising=False)


# --- metadata audit ---------------------------------------------------------------


def test_audit_flags_boltz2_like_mismatch():
    warnings = tool_discovery.audit_tool_metadata(BOLTZ2_LIKE)

    assert any("shares no terms with the tool name" in w for w in warnings)
    assert any("comp_id" in w and "boltz" in w for w in warnings)
    assert all(w.startswith("metadata mismatch") for w in warnings)


@pytest.mark.parametrize("entry", CONSISTENT_ENTRIES, ids=lambda e: e["name"])
def test_audit_passes_consistent_entries(entry):
    assert tool_discovery.audit_tool_metadata(entry) == []


def test_audit_flags_required_param_missing_from_schema_and_empty_description():
    entry = {
        "name": "Example_lookup",
        "description": "",
        "parameter": {"type": "object", "properties": {"gene": {"type": "string"}}},
        "required": ["gene_symbol"],
    }
    warnings = tool_discovery.audit_tool_metadata(entry)

    assert any("empty description" in w for w in warnings)
    assert any("'gene_symbol' is not defined" in w for w in warnings)


def test_audit_is_deterministic():
    assert tool_discovery.audit_tool_metadata(BOLTZ2_LIKE) == tool_discovery.audit_tool_metadata(
        dict(BOLTZ2_LIKE)
    )


# --- local tools --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("list files in directory workspace artifacts", "list_local_directory"),
        ("save JSON data to file write molecules", "write_local_file"),
        ("check whether a file exists", "check_file_exists"),
        ("read a csv file", "read_local_file"),
    ],
)
def test_local_file_tools_are_found(query, expected):
    hits = tool_discovery.search_local_tools(query)

    assert hits[0]["name"] == expected
    assert hits[0]["source"] == "local"
    assert hits[0]["validated"] is True
    assert "properties" in hits[0]["parameter"]


@pytest.mark.parametrize(
    "query",
    ["protein ligand complex structure prediction cofolding Boltz", "sequence alignment"],
)
def test_local_tools_not_returned_for_unrelated_queries(query):
    assert tool_discovery.search_local_tools(query) == []


def test_local_tools_surface_in_catalog_fallback():
    class EmptyCatalogue:
        def search(self, query, limit):
            return []

    wrapper = ToolUniverseWrapper(tool_factory=None, catalog=EmptyCatalogue())
    payload = json.loads(wrapper.find_tools("list files in directory workspace", limit=3))

    assert payload["results"][0]["name"] == "list_local_directory"
    assert payload["results"][0]["source"] == "local"


def test_execute_tool_runs_local_tool(monkeypatch):
    calls = {}

    class FakeTool:
        name = "list_local_directory"

        def invoke(self, args):
            calls["args"] = args
            return "listing"

    monkeypatch.setattr(
        tool_discovery, "_local_tool_objects", lambda: {"list_local_directory": FakeTool()}
    )
    wrapper = ToolUniverseWrapper(tool_factory=None)
    response = json.loads(wrapper.execute_tool("list_local_directory", '{"path": "."}'))

    assert response == {
        "tool": "list_local_directory",
        "result": "listing",
        "implementation": "local",
    }
    assert calls["args"] == {"path": "."}


# --- strategy handling and hybrid search ---------------------------------------------


def test_default_strategy_is_hybrid_and_merges_semantic_with_keyword():
    sdk = ScriptedSDK(
        {
            "Tool_Finder": [_tool("Boltz2_cofold"), _tool("Shared_tool")],
            "Tool_Finder_Keyword": [_tool("Shared_tool"), _tool("Keyword_only")],
        }
    )
    wrapper = ToolUniverseWrapper(tool_factory=lambda: sdk)
    payload = json.loads(wrapper.find_tools("protein cofolding", limit=5))

    names = [entry["name"] for entry in payload["result"]]
    assert payload["finder"] == "hybrid"
    assert payload["finders_used"] == ["embedding", "keyword"]
    assert names[0] == "Shared_tool"  # found by both finders
    assert set(names) == {"Boltz2_cofold", "Shared_tool", "Keyword_only"}
    assert len(names) == len(set(names))
    assert sorted(payload["result"][0]["matched_by"]) == ["embedding", "keyword"]
    assert all(entry["source"] == "sdk" for entry in payload["result"])
    assert all("validated" in entry and "metadata_warnings" in entry for entry in payload["result"])


def test_hybrid_falls_back_to_keyword_when_semantic_finder_unavailable(caplog):
    sdk = ScriptedSDK({"Tool_Finder_Keyword": [_tool("Keyword_only")]}, failing={"Tool_Finder"})
    wrapper = ToolUniverseWrapper(tool_factory=lambda: sdk)

    with caplog.at_level("WARNING"):
        payload = json.loads(wrapper.find_tools("protein cofolding", limit=3))
        json.loads(wrapper.find_tools("protein cofolding", limit=3))

    assert payload["source"] == "sdk"
    assert [e["name"] for e in payload["result"]] == ["Keyword_only"]
    assert payload["finders_used"] == ["keyword"]
    assert any("keyword-only" in note for note in payload["warnings"])
    assert any("embedding" in r.message and "unavailable" in r.message for r in caplog.records)
    # The broken finder is not retried on every search.
    assert sdk.run_calls.count("Tool_Finder") == 1


def test_hybrid_uses_llm_when_configured_and_embedding_fails(monkeypatch):
    monkeypatch.setenv("BIOAGENTS_TOOL_UNIVERSE_SEMANTIC_FINDERS", "embedding,llm")
    sdk = ScriptedSDK(
        {"Tool_Finder_LLM": [_tool("Llm_hit")], "Tool_Finder_Keyword": []},
        failing={"Tool_Finder"},
    )
    wrapper = ToolUniverseWrapper(tool_factory=lambda: sdk)
    payload = json.loads(wrapper.find_tools("protein cofolding", limit=3))

    assert payload["finders_used"] == ["llm", "keyword"]
    assert payload["result"][0]["name"] == "Llm_hit"


def test_unknown_strategy_is_reported_not_silently_coerced(caplog):
    sdk = ScriptedSDK({"Tool_Finder": [], "Tool_Finder_Keyword": [_tool("K")]})
    wrapper = ToolUniverseWrapper(tool_factory=lambda: sdk)

    with caplog.at_level("WARNING"):
        payload = json.loads(wrapper.find_tools("protein", limit=2, finder="semantic"))

    assert payload["finder"] == "hybrid"
    assert any("Unknown finder strategy 'semantic'" in w for w in payload["warnings"])
    assert any("hybrid, keyword, llm, embedding" in w for w in payload["warnings"])
    assert any("semantic" in r.message for r in caplog.records)


def test_invalid_env_default_logs_warning(monkeypatch, caplog):
    monkeypatch.setenv("BIOAGENTS_TOOL_UNIVERSE_FINDER", "fuzzy")
    with caplog.at_level("WARNING"):
        wrapper = ToolUniverseWrapper(tool_factory=None)

    assert wrapper._default_finder == "hybrid"
    assert any("fuzzy" in r.message and "valid" in r.message for r in caplog.records)


def test_env_can_select_keyword_only(monkeypatch):
    monkeypatch.setenv("BIOAGENTS_TOOL_UNIVERSE_FINDER", "keyword")
    sdk = ScriptedSDK({"Tool_Finder_Keyword": [_tool("K")]})
    wrapper = ToolUniverseWrapper(tool_factory=lambda: sdk)
    payload = json.loads(wrapper.find_tools("protein", limit=2))

    assert payload["finder"] == "keyword"
    assert sdk.run_calls == ["Tool_Finder_Keyword"]


# --- metadata warnings and provenance in find results --------------------------------


def test_find_results_carry_metadata_mismatch_signal():
    sdk = ScriptedSDK({"Tool_Finder": [BOLTZ2_LIKE], "Tool_Finder_Keyword": []})
    wrapper = ToolUniverseWrapper(tool_factory=lambda: sdk)
    payload = json.loads(wrapper.find_tools("boltz cofolding", limit=3))

    entry = payload["result"][0]
    assert entry["name"] == "NvidiaNIM_boltz2"
    assert entry["validated"] is False
    assert entry["metadata_warnings"]
    assert "metadata_note" in payload
    # Backward compatible: original fields are still present.
    assert {"name", "description", "parameter", "required"} <= set(entry)


def test_custom_registry_provenance_and_validated_flag(monkeypatch):
    custom = {
        "name": "list_workspace_files",
        "description": "List files in the workspace directory.",
        "parameter": {"type": "object", "properties": {"path": {"type": "string"}}},
        "required": [],
        "source": "custom_registry",
        "validated": False,
    }
    monkeypatch.setattr(
        ToolUniverseWrapper,
        "_search_custom_registry",
        staticmethod(lambda query, limit=5: [dict(custom)]),
    )
    sdk = ScriptedSDK({"Tool_Finder": [], "Tool_Finder_Keyword": []})
    wrapper = ToolUniverseWrapper(tool_factory=lambda: sdk)
    payload = json.loads(wrapper.find_tools("list files in directory workspace", limit=5))

    by_name = {entry["name"]: entry for entry in payload["result"]}
    assert by_name["list_workspace_files"]["source"] == "custom_registry"
    assert by_name["list_workspace_files"]["validated"] is False
    assert by_name["list_local_directory"]["source"] == "local"
    assert by_name["list_local_directory"]["validated"] is True
