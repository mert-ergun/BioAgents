"""Manage playbook files (YAML storage only).

Playbooks are cross-run state: every bullet stored here is appended to the agent
prompt of *every later run* (see
:func:`bioagents.prompts.prompt_loader.load_prompt`). Bullets are therefore
filtered through :mod:`bioagents.learning.playbook_safety` on both save and load
so that fabrication directives and run-specific biological entities cannot be
replayed into an unrelated run.
"""

import logging
import os
from pathlib import Path
from typing import Any

from bioagents.learning.playbook_safety import sanitize_playbook

logger = logging.getLogger(__name__)

#: Repository root (``.../BioAgents``), three parents up from this file:
#: playbook_manager.py -> learning -> bioagents -> <repo root>.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: Playbook directory relative to the repository root when unset.
_DEFAULT_PLAYBOOK_DIR = "playbooks"


def get_playbook_dir() -> Path:
    """Get the playbook directory path.

    Honours ``ACE_PLAYBOOK_DIR``; a relative value is anchored at the repository
    root so playbooks never land outside the checkout.
    """
    configured = os.getenv("ACE_PLAYBOOK_DIR", "").strip() or _DEFAULT_PLAYBOOK_DIR
    candidate = Path(configured)
    playbook_path = candidate if candidate.is_absolute() else _REPO_ROOT / candidate
    playbook_path.mkdir(parents=True, exist_ok=True)
    return playbook_path


def get_playbook_path(agent_name: str) -> Path:
    """Get the playbook file path for an agent."""
    playbook_dir = get_playbook_dir()
    return playbook_dir / f"{agent_name}_playbook.yaml"


def load_playbook(agent_name: str) -> str:
    """
    Load playbook from YAML file.

    Unsafe bullets (fabrication directives, run-specific entities) are dropped
    before the content is returned, so a playbook written by an older build
    cannot leak into a prompt.

    Args:
        agent_name: Name of the agent

    Returns:
        Playbook content as string, or empty playbook if file doesn't exist
    """
    playbook_path = get_playbook_path(agent_name)

    if not playbook_path.exists():
        return _get_empty_playbook()

    with playbook_path.open(encoding="utf-8") as f:
        raw = f.read()

    return _sanitize_and_log(agent_name, raw, stage="load")


def save_playbook(agent_name: str, playbook: str) -> None:
    """
    Save playbook to YAML file.

    The playbook is sanitized first so a curator that proposes a fabrication
    directive or an entity-specific bullet never persists it.

    Args:
        agent_name: Name of the agent
        playbook: Playbook content to save
    """
    playbook_path = get_playbook_path(agent_name)
    sanitized = _sanitize_and_log(agent_name, playbook, stage="save")

    with playbook_path.open("w", encoding="utf-8") as f:
        f.write(sanitized)


def _sanitize_and_log(agent_name: str, playbook: str, stage: str) -> str:
    """Run the safety filter over *playbook*, logging anything it removes."""
    sanitized, removed = sanitize_playbook(playbook)
    for bullet_id, reason in removed:
        logger.warning(
            "Dropped %s playbook bullet %s on %s: %s",
            agent_name,
            bullet_id or "<unknown>",
            stage,
            reason,
        )
    return sanitized


def update_bullet_counts(playbook: str, tags: list[dict[str, str]]) -> str:
    """
    Update helpful/harmful counts in playbook based on tags.

    Args:
        playbook: Current playbook content
        tags: List of dicts with 'id' and 'tag' keys ('helpful', 'harmful', or 'neutral')

    Returns:
        Updated playbook content
    """
    lines = playbook.strip().split("\n")
    updated_lines = []

    tag_map = {}
    for tag in tags:
        if isinstance(tag, dict):
            bullet_id = tag.get("id", "")
            tag_value = tag.get("tag", "neutral")
            if bullet_id:
                tag_map[bullet_id] = tag_value

    if not tag_map:
        return playbook

    for line in lines:
        # Preserve section headers and empty lines
        if line.strip().startswith("#") or not line.strip():
            updated_lines.append(line)
            continue

        parsed = _parse_playbook_line(line)
        if parsed:
            parsed_bullet_id: Any = parsed.get("id")
            if isinstance(parsed_bullet_id, str) and parsed_bullet_id in tag_map:
                parsed_tag_value: str = tag_map[parsed_bullet_id]
                if parsed_tag_value == "helpful":
                    current_helpful: Any = parsed.get("helpful", 0)
                    parsed["helpful"] = (
                        current_helpful if isinstance(current_helpful, int) else 0
                    ) + 1
                elif parsed_tag_value == "harmful":
                    current_harmful: Any = parsed.get("harmful", 0)
                    parsed["harmful"] = (
                        current_harmful if isinstance(current_harmful, int) else 0
                    ) + 1

                # Reconstruct line with updated counts
                parsed_id: Any = parsed.get("id", "")
                parsed_helpful: Any = parsed.get("helpful", 0)
                parsed_harmful: Any = parsed.get("harmful", 0)
                parsed_content: Any = parsed.get("content", "")
                if (
                    isinstance(parsed_id, str)
                    and isinstance(parsed_helpful, int)
                    and isinstance(parsed_harmful, int)
                    and isinstance(parsed_content, str)
                ):
                    new_line = _format_playbook_line(
                        parsed_id, parsed_helpful, parsed_harmful, parsed_content
                    )
                    updated_lines.append(new_line)
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)
        else:
            updated_lines.append(line)

    return "\n".join(updated_lines)


def _parse_playbook_line(line: str) -> dict[str, Any] | None:
    """Parse a single playbook line to extract components."""
    import re

    # Pattern: [id] helpful=X harmful=Y :: content
    pattern = r"\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)"
    match = re.match(pattern, line.strip())

    if match:
        return {
            "id": match.group(1),
            "helpful": int(match.group(2)),
            "harmful": int(match.group(3)),
            "content": match.group(4),
        }
    return None


def _format_playbook_line(bullet_id: str, helpful: int, harmful: int, content: str) -> str:
    """Format a bullet into playbook line format."""
    return f"[{bullet_id}] helpful={helpful} harmful={harmful} :: {content}"


def _get_empty_playbook() -> str:
    """Get an empty playbook with standard sections."""
    return """## ROUTING_STRATEGIES

## TOOL_PATTERNS

## CODE_PATTERNS

## ERROR_PREVENTION

## WORKFLOW_PATTERNS

## COMMON_MISTAKES

## OTHERS
"""
