"""Guards that authored prompt content actually reaches the model.

The loader used to render only a hardcoded whitelist of XML sections, silently
dropping anything else. A prompt file could then look correct on disk while most of
its guidance never reached the LLM (summary.xml was shipping 2.3KB of a 12.5KB file).
These tests fail if that class of bug returns.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from bioagents.prompts.prompt_loader import load_prompt

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "bioagents" / "prompts"
PROMPT_NAMES = sorted(p.stem for p in PROMPTS_DIR.glob("*.xml"))

# <metadata> holds model routing config, not instruction text, so it is not rendered.
NON_INSTRUCTIONAL_SECTIONS = {"metadata"}


def _section_texts(element: ET.Element) -> list[str]:
    """Collect every non-empty text fragment inside a section."""
    return [t.strip() for t in element.itertext() if t and t.strip()]


@pytest.mark.parametrize("prompt_name", PROMPT_NAMES)
def test_every_prompt_section_is_rendered(prompt_name: str) -> None:
    """No top-level section may be silently dropped from a loaded prompt."""
    root = ET.parse(PROMPTS_DIR / f"{prompt_name}.xml").getroot()
    rendered = load_prompt(prompt_name)

    dropped: list[str] = []
    for section in root:
        if section.tag in NON_INSTRUCTIONAL_SECTIONS:
            continue
        fragments = _section_texts(section)
        if not fragments:
            continue
        # A section is considered rendered if a distinctive fragment of it survives.
        longest = max(fragments, key=len)
        probe = " ".join(longest.split())[:60]
        if probe and probe not in " ".join(rendered.split()):
            dropped.append(section.tag)

    assert not dropped, (
        f"{prompt_name}.xml: section(s) {dropped} were authored but never reach the model. "
        "Add a formatter or ensure the generic renderer covers them."
    )


@pytest.mark.parametrize("prompt_name", PROMPT_NAMES)
def test_prompt_retains_most_authored_content(prompt_name: str) -> None:
    """The rendered prompt must retain the bulk of the authored instruction text."""
    root = ET.parse(PROMPTS_DIR / f"{prompt_name}.xml").getroot()
    authored = sum(
        len(" ".join(t.split()))
        for section in root
        if section.tag not in NON_INSTRUCTIONAL_SECTIONS
        for t in _section_texts(section)
    )
    rendered = len(" ".join(load_prompt(prompt_name).split()))

    assert rendered >= authored * 0.8, (
        f"{prompt_name}.xml: only {rendered} of ~{authored} authored characters are rendered. "
        "Guidance is being dropped before it reaches the model."
    )


def test_summary_prompt_carries_anti_fabrication_rules() -> None:
    """The summary agent must receive its grounding rules; it fabricated results without them."""
    rendered = load_prompt("summary")
    for required in (
        "traceable to a ToolMessage",
        "NEVER claim an agent performed work",
        "performed_any_computation",
        "Distinguish PLANNED from EXECUTED",
    ):
        assert required in rendered, f"summary prompt is missing grounding rule: {required!r}"


def test_research_merger_rejects_placeholder_findings() -> None:
    """The merger must be told that sub-agent status lines are not results."""
    rendered = load_prompt("research_merger")
    for required in (
        "SUB-AGENT STATUS LINES ARE NOT RESULTS",
        "NEVER invent a numeric result",
    ):
        assert required in rendered, f"research_merger prompt is missing rule: {required!r}"
