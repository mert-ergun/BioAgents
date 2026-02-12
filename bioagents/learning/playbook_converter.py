"""Convert between XML prompt instructions and ACE playbook format."""

import xml.etree.ElementTree as ET  # nosec B405
from pathlib import Path
from typing import Any


def extract_instructions(xml_root: ET.Element) -> list[dict[str, str]]:
    """
    Extract instructions from XML prompt.

    Args:
        xml_root: Root element of XML prompt

    Returns:
        List of instruction dicts with 'id', 'priority', and 'content'
    """
    instructions: list[dict[str, str]] = []
    instructions_elem = xml_root.find("instructions")

    if instructions_elem is None:
        return instructions

    for idx, inst_elem in enumerate(instructions_elem.findall("instruction")):
        content = inst_elem.text or ""
        priority = inst_elem.get("priority", "medium")
        # Generate a simple ID for tracking
        inst_id = f"inst-{idx:05d}"

        instructions.append(
            {
                "id": inst_id,
                "priority": priority,
                "content": content.strip(),
            }
        )

    return instructions


def xml_to_playbook(xml_prompt: str, agent_name: str) -> str:  # noqa: ARG001
    """
    Convert XML prompt to playbook format.

    Args:
        xml_prompt: XML prompt content as string
        agent_name: Name of the agent (for section mapping)

    Returns:
        Playbook content as string
    """
    root = ET.fromstring(xml_prompt)  # nosec B314
    instructions = extract_instructions(root)

    # Map instructions to playbook sections based on content
    sections: dict[str, list[dict[str, str]]] = {
        "ROUTING_STRATEGIES": [],
        "TOOL_PATTERNS": [],
        "CODE_PATTERNS": [],
        "ERROR_PREVENTION": [],
        "WORKFLOW_PATTERNS": [],
        "COMMON_MISTAKES": [],
        "OTHERS": [],
    }

    for inst in instructions:
        content = inst["content"].lower()
        section = "OTHERS"

        # Simple heuristics to categorize instructions
        if any(
            keyword in content for keyword in ["route", "routing", "direct", "send to", "agent"]
        ):
            section = "ROUTING_STRATEGIES"
        elif any(keyword in content for keyword in ["tool", "execute", "call"]):
            section = "TOOL_PATTERNS"
        elif any(keyword in content for keyword in ["code", "python", "script"]):
            section = "CODE_PATTERNS"
        elif any(
            keyword in content
            for keyword in ["error", "fail", "exception", "avoid", "don't", "never"]
        ):
            section = "ERROR_PREVENTION"
        elif any(keyword in content for keyword in ["workflow", "process", "step", "procedure"]):
            section = "WORKFLOW_PATTERNS"
        elif any(keyword in content for keyword in ["mistake", "wrong", "incorrect", "should not"]):
            section = "COMMON_MISTAKES"

        sections[section].append(inst)

    # Build playbook
    playbook_lines = []
    for section_name, section_instructions in sections.items():
        if section_instructions:
            playbook_lines.append(f"## {section_name}")
            playbook_lines.append("")

            for inst in section_instructions:
                # Convert to playbook format: [id] helpful=0 harmful=0 :: content
                bullet_id = inst["id"]
                content = inst["content"]
                line = f"[{bullet_id}] helpful=0 harmful=0 :: {content}"
                playbook_lines.append(line)

            playbook_lines.append("")

    return "\n".join(playbook_lines)


def apply_delta_updates(xml_root: ET.Element, operations: list[dict], next_id: int) -> ET.Element:
    """
    Apply curator operations (delta updates) to XML.

    Args:
        xml_root: Root element of XML prompt
        operations: List of curator operations (ADD, UPDATE, DELETE)
        next_id: Next available instruction ID

    Returns:
        Updated XML root element
    """
    instructions_elem = xml_root.find("instructions")
    if instructions_elem is None:
        # Create instructions element if it doesn't exist
        instructions_elem = ET.SubElement(xml_root, "instructions")

    current_id = next_id

    for op in operations:
        op_type = op.get("type", "").upper()
        content = op.get("content", "")
        priority = op.get("priority", "medium")

        if op_type == "ADD":
            # Add new instruction
            new_inst = ET.SubElement(instructions_elem, "instruction")
            new_inst.set("priority", priority)
            new_inst.text = content

            # Generate ID for tracking (will be mapped in playbook)
            add_inst_id = f"inst-{current_id:05d}"
            new_inst.set("ace_id", add_inst_id)  # Store for mapping
            current_id += 1

        elif op_type == "UPDATE":
            # Update existing instruction (by ID or content match)
            update_inst_id: str | None = op.get("id")
            if update_inst_id:
                for inst in instructions_elem.findall("instruction"):
                    if inst.get("ace_id") == update_inst_id:
                        if "content" in op:
                            inst.text = content
                        if "priority" in op:
                            inst.set("priority", priority)
                        break

        elif op_type == "DELETE":
            # Delete instruction by ID
            delete_inst_id: str | None = op.get("id")
            if delete_inst_id:
                for inst in instructions_elem.findall("instruction"):
                    if inst.get("ace_id") == delete_inst_id:
                        instructions_elem.remove(inst)
                        break

    return xml_root


def playbook_to_xml(playbook: str, xml_template: str) -> str:
    """
    Apply playbook updates to XML template.

    Extracts bullets from playbook and adds them as instructions to XML.
    Only includes bullets with helpful > harmful (learned best practices).

    Args:
        playbook: Current playbook content
        xml_template: Original XML template

    Returns:
        Updated XML content with playbook instructions merged
    """
    from bioagents.learning.playbook_manager import _parse_playbook_line

    # Parse XML template
    root = ET.fromstring(xml_template)  # nosec B314

    # Find or create instructions element
    instructions_elem = root.find("instructions")
    if instructions_elem is None:
        instructions_elem = ET.SubElement(root, "instructions")

    # Parse playbook to extract bullets
    playbook_lines = playbook.strip().split("\n")
    bullets_to_add = []

    for line in playbook_lines:
        # Skip section headers and empty lines
        if line.strip().startswith("#") or not line.strip():
            continue

        # Parse bullet line
        parsed = _parse_playbook_line(line)
        if parsed:
            helpful = parsed.get("helpful", 0) or 0
            harmful = parsed.get("harmful", 0) or 0
            content_str = parsed.get("content", "")
            content = content_str.strip() if isinstance(content_str, str) else ""

            # Only add bullets that are helpful (helpful > harmful)
            # This ensures we only add learned best practices
            if content and helpful > harmful:
                bullets_to_add.append(
                    {
                        "content": content,
                        "helpful": helpful,
                        "harmful": harmful,
                        "score": helpful - harmful,  # Sort by usefulness
                    }
                )

    # Sort by score (most helpful first)
    bullets_to_add.sort(key=lambda x: x["score"], reverse=True)

    # Add bullets as instructions to XML
    # Priority: high if score > 10, medium if score > 5, low otherwise
    for bullet in bullets_to_add:
        new_inst = ET.SubElement(instructions_elem, "instruction")

        # Set priority based on helpful score
        score_raw: Any = bullet.get("score", 0)
        score: int = score_raw if isinstance(score_raw, int) else 0
        if score > 10:
            priority = "high"
        elif score > 5:
            priority = "medium"
        else:
            priority = "low"

        new_inst.set("priority", priority)
        bullet_content = bullet.get("content", "")
        if isinstance(bullet_content, str):
            new_inst.text = bullet_content
        # Store playbook metadata
        bullet_helpful = bullet.get("helpful", 0)
        bullet_harmful = bullet.get("harmful", 0)
        new_inst.set("ace_helpful", str(bullet_helpful))
        new_inst.set("ace_harmful", str(bullet_harmful))

    # Convert back to string
    return ET.tostring(root, encoding="unicode")


def load_xml_prompt(agent_name: str) -> str:
    """
    Load XML prompt for an agent.

    Args:
        agent_name: Name of the agent

    Returns:
        XML prompt content as string
    """
    try:
        # PromptLoader.load_prompt returns formatted string, we need raw XML
        # So we'll load it directly
        base_path = Path(__file__).parent.parent.parent
        prompt_path = base_path / "bioagents" / "prompts" / f"{agent_name}.xml"
        if prompt_path.exists():
            with prompt_path.open(encoding="utf-8") as f:
                return f.read()
    except Exception:  # nosec B110
        pass

    # Fallback: try to load directly
    base_path = Path(__file__).parent.parent.parent
    prompt_path = base_path / "bioagents" / "prompts" / f"{agent_name}.xml"
    if prompt_path.exists():
        with prompt_path.open(encoding="utf-8") as f:
            return f.read()
    return ""
