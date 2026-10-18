"""Guards against residual state leaking between runs.

Context: an AKT1 docking run reported a protein that reads like CDK2, and an
earlier run labelled the AKT1 structure 4GV1 as "CDK2" outright. The suspicion
was a leftover prompt or cached state carrying one run's entity into the next.
These tests pin the mechanisms that could actually do that:

* ACE playbooks are appended to every later agent prompt, so a bullet naming a
  concrete protein/compound, or telling the agent to invent a result when a tool
  fails, is exactly such a leak.
* The playbook directory must stay inside the repository.
* ACE wrappers must not be shared between unrelated runs in one process.
* Prompt XML examples must not present a specific entity's data as a result the
  model can copy.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from bioagents.learning import ace_integration
from bioagents.learning.playbook_manager import (
    _parse_playbook_line,
    get_playbook_dir,
    load_playbook,
    save_playbook,
)
from bioagents.learning.playbook_safety import (
    ENTITY_SPECIFIC,
    FABRICATION,
    SAFE,
    classify_bullet,
    is_safe_bullet,
    sanitize_playbook,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PLAYBOOK_FILES = sorted(str(p) for p in (REPO_ROOT / "playbooks").glob("*.yaml"))
PROMPT_FILES = sorted(str(p) for p in (REPO_ROOT / "bioagents" / "prompts").glob("*.xml"))


# --------------------------------------------------------------------------
# Bullet classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        # Verbatim bullets that were stored in playbooks/tool_builder_playbook.yaml.
        "If a capability gap is identified, simulate the outcome by describing the "
        "expected steps and providing the most likely factual result based on "
        "available knowledge.",
        "If live execution is not feasible, clearly state the code's purpose and "
        "simulate its output with a concrete, factual answer.",
        "When simulating analytical results, ensure the output is concrete, factual, "
        "and presented in a structured format that mirrors real-world reports.",
        "Answer from your own knowledge when the tool is unavailable.",
    ],
)
def test_fabrication_bullets_are_rejected(content: str) -> None:
    """Bullets that tell the agent to invent a result must never be replayed."""
    assert classify_bullet(content) == FABRICATION
    assert not is_safe_bullet(content)


@pytest.mark.parametrize(
    "content",
    [
        "For AKT1 docking, centre the box on the co-crystal ligand.",
        "CDK2 inhibitors should be filtered by IC50 before ranking.",
        "Use UniProt P31749 when the user asks about this kinase.",
        "Target CHEMBL4282 gives the best activity coverage.",
        "Fetch PDB 4GV1 as the reference structure.",
        "Capivasertib is the reference compound for similarity search.",
        "AZD5363 is the co-crystallised ligand to compare against.",
        "For p53 use residues 17, 19, 23, 26 as hotspots.",
    ],
)
def test_entity_specific_bullets_are_rejected(content: str) -> None:
    """A bullet about one run's target must not reach another run's prompt."""
    assert classify_bullet(content) == ENTITY_SPECIFIC
    assert not is_safe_bullet(content)


@pytest.mark.parametrize(
    "content",
    [
        "Route questions requiring numerical derivation from raw input data to a "
        "dedicated computational pipeline rather than a summarization path.",
        "Validate Output Type: verify the answer's format and unit match the question.",
        "Prevent Input Echoing: never repeat the raw data instead of processing it.",
        "Parse the sequence, count residues, apply standard weights, then sum.",
        "Prefer transparent communication of limitations over an unsupported answer.",
    ],
)
def test_generalisable_bullets_survive(content: str) -> None:
    """Entity-free strategy is the whole point of a playbook; it must be kept."""
    assert classify_bullet(content) == SAFE
    assert is_safe_bullet(content)


def test_common_acronyms_are_not_mistaken_for_entities() -> None:
    """The entity filter must not eat ordinary engineering vocabulary."""
    content = (
        "Return a JSON envelope over HTTP; write CSV output and log the HTML report "
        "so the 3D viewer can load it."
    )
    assert is_safe_bullet(content)


# --------------------------------------------------------------------------
# Playbook sanitisation
# --------------------------------------------------------------------------


def test_sanitize_playbook_preserves_structure_and_drops_unsafe() -> None:
    playbook = (
        "## TOOL_PATTERNS\n"
        "[tp-1] helpful=3 harmful=0 :: Check the tool schema before calling it.\n"
        "[tp-2] helpful=9 harmful=0 :: Simulate the tool output with a concrete, "
        "factual answer.\n"
        "## COMMON_MISTAKES\n"
        "[cm-1] helpful=4 harmful=1 :: AKT1 runs should reuse the CHEMBL4282 target.\n"
    )

    sanitized, removed = sanitize_playbook(playbook)

    assert "## TOOL_PATTERNS" in sanitized
    assert "## COMMON_MISTAKES" in sanitized
    assert "[tp-1]" in sanitized
    assert "[tp-2]" not in sanitized
    assert "[cm-1]" not in sanitized
    assert ("tp-2", FABRICATION) in removed
    assert ("cm-1", ENTITY_SPECIFIC) in removed


def test_save_then_load_strips_unsafe_bullets(tmp_path, monkeypatch) -> None:
    """A curator proposing an unsafe bullet must not persist it."""
    monkeypatch.setenv("ACE_PLAYBOOK_DIR", str(tmp_path))

    save_playbook(
        "research",
        "## TOOL_PATTERNS\n"
        "[tp-1] helpful=1 harmful=0 :: Verify the units before comparing values.\n"
        "[tp-2] helpful=1 harmful=0 :: For CDK2 (P24941) use the DFG aspartate D145.\n",
    )

    on_disk = (tmp_path / "research_playbook.yaml").read_text(encoding="utf-8")
    assert "CDK2" not in on_disk
    assert "P24941" not in on_disk
    assert "Verify the units" in on_disk

    assert "CDK2" not in load_playbook("research")


def test_legacy_unsafe_playbook_on_disk_is_not_returned(tmp_path, monkeypatch) -> None:
    """Files written by an older build are filtered at load time too."""
    monkeypatch.setenv("ACE_PLAYBOOK_DIR", str(tmp_path))
    (tmp_path / "coder_playbook.yaml").write_text(
        "## WORKFLOW_PATTERNS\n"
        "[wp-1] helpful=7 harmful=1 :: Simulate the outcome and provide the most "
        "likely factual result based on available knowledge.\n",
        encoding="utf-8",
    )

    loaded = load_playbook("coder")

    assert "[wp-1]" not in loaded
    assert "## WORKFLOW_PATTERNS" in loaded


# --------------------------------------------------------------------------
# Playbook location
# --------------------------------------------------------------------------


def test_playbook_dir_defaults_inside_the_repository(monkeypatch) -> None:
    """The default must not resolve to a sibling of the checkout."""
    monkeypatch.delenv("ACE_PLAYBOOK_DIR", raising=False)

    playbook_dir = get_playbook_dir()

    assert playbook_dir == REPO_ROOT / "playbooks"
    assert REPO_ROOT in playbook_dir.parents or playbook_dir == REPO_ROOT / "playbooks"


def test_relative_playbook_dir_is_anchored_at_repo_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ACE_PLAYBOOK_DIR", "playbooks")
    monkeypatch.chdir(tmp_path)

    assert get_playbook_dir() == REPO_ROOT / "playbooks"


def test_absolute_playbook_dir_is_honoured(tmp_path, monkeypatch) -> None:
    target = tmp_path / "pb"
    monkeypatch.setenv("ACE_PLAYBOOK_DIR", str(target))

    assert get_playbook_dir() == target


# --------------------------------------------------------------------------
# Checked-in playbooks
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", PLAYBOOK_FILES, ids=lambda p: Path(p).name)
def test_checked_in_playbooks_contain_no_unsafe_bullets(path: str) -> None:
    """The playbooks shipped in the repo are injected verbatim into prompts."""
    content = Path(path).read_text(encoding="utf-8")
    offenders = [
        (parsed["id"], classify_bullet(parsed["content"]))
        for line in content.split("\n")
        if (parsed := _parse_playbook_line(line)) and classify_bullet(parsed["content"]) != SAFE
    ]
    assert offenders == [], f"{Path(path).name} carries unsafe bullets: {offenders}"


# --------------------------------------------------------------------------
# ACE wrapper scoping
# --------------------------------------------------------------------------


def test_ace_wrappers_are_scoped_per_run(monkeypatch) -> None:
    """Two unrelated runs in one process must not share a wrapper."""
    ace_integration.clear_ace_cache()
    sentinel_a, sentinel_b = object(), object()

    ace_integration.set_ace_run_scope("run-a")
    ace_integration._ace_wrappers[(ace_integration.get_ace_run_scope(), "coder")] = sentinel_a  # type: ignore[assignment]

    ace_integration.set_ace_run_scope("run-b")
    assert ace_integration._ace_wrappers.get((ace_integration.get_ace_run_scope(), "coder")) is None

    ace_integration._ace_wrappers[(ace_integration.get_ace_run_scope(), "coder")] = sentinel_b  # type: ignore[assignment]
    assert ace_integration._ace_wrappers[("run-a", "coder")] is sentinel_a
    assert ace_integration._ace_wrappers[("run-b", "coder")] is sentinel_b

    ace_integration.clear_ace_cache("run-a")
    assert ("run-a", "coder") not in ace_integration._ace_wrappers
    assert ("run-b", "coder") in ace_integration._ace_wrappers

    ace_integration.clear_ace_cache()
    ace_integration.set_ace_run_scope(None)
    assert ace_integration.get_ace_run_scope() == "default"


def test_ace_disabled_by_default(monkeypatch) -> None:
    """ACE injection stays opt-in; it must not silently turn itself on."""
    monkeypatch.delenv("ACE_ENABLED", raising=False)
    from bioagents.learning import is_ace_enabled

    assert is_ace_enabled() is False


# --------------------------------------------------------------------------
# Prompt XML examples
# --------------------------------------------------------------------------

#: Accessions / identifiers that must not appear in a prompt as if they were a
#: result. Matched against the whole prompt file.
_FORBIDDEN_IN_PROMPT_RESULTS = (
    re.compile(r"\bCHEMBL\d+\b", re.I),
    re.compile(r"\bP31749\b", re.I),
    re.compile(r"\bP24941\b", re.I),
    re.compile(r"\b(?:AKT1|CDK2)\b"),
    re.compile(r"\b(?:capivasertib|ipatasertib|AZD5363)\b", re.I),
)


@pytest.mark.parametrize("path", PROMPT_FILES, ids=lambda p: Path(p).name)
def test_prompts_do_not_pin_the_benchmark_entities(path: str) -> None:
    """Benchmark targets must never be baked into a prompt.

    A prompt mentioning AKT1/CDK2 is the literal "residual prompt" that would
    bias every run of those benchmark tasks.
    """
    text = Path(path).read_text(encoding="utf-8")
    # The coder prompt documents the 4GV1/CDK2 mislabel as a WRONG example; that
    # is the one place these names are allowed, because it teaches the opposite.
    if Path(path).name == "coder.xml":
        pytest.skip("coder.xml cites the historical mislabel as a negative example")
    hits = [p.pattern for p in _FORBIDDEN_IN_PROMPT_RESULTS if p.search(text)]
    assert hits == [], f"{Path(path).name} pins benchmark entities: {hits}"


def test_protein_design_has_no_cross_entity_hotspot_fallback() -> None:
    """Falling back to another protein's residues is a cross-entity leak."""
    text = (REPO_ROOT / "bioagents" / "prompts" / "protein_design.xml").read_text(encoding="utf-8")
    assert "Use residues 17,19,23,26" not in text
    assert "NEVER substitute residues from a different protein" in text


def test_summary_example_is_a_template_not_a_result() -> None:
    """The worked example must not hand the model copyable numbers."""
    text = (REPO_ROOT / "bioagents" / "prompts" / "summary.xml").read_text(encoding="utf-8")
    assert "393 amino acid protein with a molecular weight of ~43.7 kDa" not in text
    assert "This example shows STRUCTURE ONLY" in text


def test_summary_grounds_entity_identity_and_numbering() -> None:
    """The two failure modes actually observed must be named in the prompt."""
    text = (REPO_ROOT / "bioagents" / "prompts" / "summary.xml").read_text(encoding="utf-8")
    assert "IDENTITY IS A FACT, NOT A GUESS" in text
    assert "sequential index" in text.lower()


def test_coder_forbids_hardcoded_identity_and_renumbering() -> None:
    text = (REPO_ROOT / "bioagents" / "prompts" / "coder.xml").read_text(encoding="utf-8")
    assert "NEVER HARD-CODE AN ENTITY IDENTITY INTO GENERATED CODE" in text
    assert "PRESERVE SOURCE RESIDUE NUMBERING" in text


# --------------------------------------------------------------------------
# Session context scoping
# --------------------------------------------------------------------------


def test_session_context_tool_is_bound_to_the_session_it_was_built_for() -> None:
    """`retrieve_previous_context` must only see its own session's turns."""
    from bioagents.tools.session_context_tools import create_retrieve_previous_context_tool

    session_a = [
        {"Research_0": {"index": "AKT1 docking", "content": "AKT1 result", "agent": "Research"}}
    ]
    session_b = [
        {"Research_0": {"index": "CDK2 variants", "content": "CDK2 result", "agent": "Research"}}
    ]

    tool_a = create_retrieve_previous_context_tool(session_a)
    tool_b = create_retrieve_previous_context_tool(session_b)

    out_a = tool_a.invoke({"query": "docking result"})
    out_b = tool_b.invoke({"query": "variant result"})

    assert "CDK2" not in out_a
    assert "AKT1" not in out_b


def test_memory_utils_write_is_confined_to_the_passed_state() -> None:
    """Shared memory must live on the state object, not on a module global."""
    from bioagents.agents.memory_utils import read_from_memory, write_to_memory

    state_a: dict = {"memory": {}}
    state_b: dict = {"memory": {}}

    write_to_memory(state_a, "Research", {"target": "AKT1"})

    assert read_from_memory(state_a, "Research") == {"target": "AKT1"}
    assert read_from_memory(state_b, "Research") is None
    assert state_b["memory"] == {}


def test_execution_tracker_instances_do_not_share_records() -> None:
    from bioagents.learning.execution_tracker import ExecutionTracker

    tracker_a, tracker_b = ExecutionTracker(), ExecutionTracker()
    tracker_a.track_execution("coder", "AKT1 task", "out", True)

    assert len(tracker_a.get_recent_executions()) == 1
    assert tracker_b.get_recent_executions() == []


def test_no_env_file_enables_ace_in_the_repo() -> None:
    """A committed ACE_ENABLED=true would turn playbook injection on silently."""
    for name in (".env.example", ".env.template"):
        path = REPO_ROOT / name
        if path.exists():
            text = path.read_text(encoding="utf-8")
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("ACE_ENABLED"):
                    value = stripped.split("=", 1)[-1].strip().strip("\"'").lower()
                    assert value not in ("true", "1", "yes"), f"{name} enables ACE"


def test_playbook_dir_is_not_created_outside_the_repo(monkeypatch) -> None:
    """Regression: the old default resolved to <repo>/../bioagents/playbooks."""
    monkeypatch.delenv("ACE_PLAYBOOK_DIR", raising=False)
    stray = REPO_ROOT.parent / "bioagents" / "playbooks"

    get_playbook_dir()

    assert not stray.exists(), f"playbooks leaked outside the checkout: {stray}"


def test_env_var_pollution_does_not_escape_tests() -> None:
    """Sanity: tests above use monkeypatch, so the real env stays clean."""
    assert os.getenv("ACE_PLAYBOOK_DIR") in (None, "")
