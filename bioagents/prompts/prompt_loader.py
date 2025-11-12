"""Utility for loading and parsing XML-formatted system prompts."""

import xml.etree.ElementTree as ET  # nosec B405  # Internal trusted XML files only
from pathlib import Path


class PromptLoader:
    """Loads and parses XML-formatted system prompts."""

    def __init__(self, prompts_dir: Path | None = None):
        """
        Initialize the prompt loader.

        Args:
            prompts_dir: Directory containing prompt XML files.
                        If None, uses the default prompts directory.
        """
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent

        self.prompts_dir = Path(prompts_dir)

    def load_prompt(self, prompt_name: str) -> str:
        """
        Load and parse an XML prompt file into a formatted string.

        Args:
            prompt_name: Name of the prompt file (without .xml extension)

        Returns:
            The formatted prompt text ready for use as a system message

        Raises:
            FileNotFoundError: If the prompt file doesn't exist
            ET.ParseError: If the XML is malformed
        """
        prompt_file = self.prompts_dir / f"{prompt_name}.xml"

        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        # Parse the XML
        tree = ET.parse(prompt_file)  # nosec B314  # Internal trusted XML files only
        root = tree.getroot()

        # Build the prompt text
        sections = []

        # Add role section
        role = root.find("role")
        if role is not None and role.text:
            sections.append(role.text.strip())

        # Add capabilities (if present)
        capabilities = root.find("capabilities")
        if capabilities is not None:
            cap_text = self._format_capabilities(capabilities)
            if cap_text:
                sections.append(cap_text)

        # Add team information (for supervisor)
        team = root.find("team")
        if team is not None:
            team_text = self._format_team(team)
            if team_text:
                sections.append(team_text)

        # Add responsibilities (for report agent)
        responsibilities = root.find("responsibilities")
        if responsibilities is not None:
            resp_text = self._format_responsibilities(responsibilities)
            if resp_text:
                sections.append(resp_text)

        # Add instructions
        instructions = root.find("instructions")
        if instructions is not None:
            inst_text = self._format_instructions(instructions)
            if inst_text:
                sections.append(inst_text)

        # Add workflow (if present)
        workflow = root.find("workflow")
        if workflow is not None:
            workflow_text = self._format_workflow(workflow)
            if workflow_text:
                sections.append(workflow_text)

        # Add decision guidelines (for supervisor)
        decision_guidelines = root.find("decision_guidelines")
        if decision_guidelines is not None:
            guidelines_text = self._format_decision_guidelines(decision_guidelines)
            if guidelines_text:
                sections.append(guidelines_text)

        # Add communication style
        comm_style = root.find("communication_style")
        if comm_style is not None:
            style_text = self._format_communication_style(comm_style)
            if style_text:
                sections.append(style_text)

        # Join all sections with double newlines
        return "\n\n".join(sections)

    def _format_capabilities(self, capabilities: ET.Element) -> str:
        """Format the capabilities section."""
        lines = ["Your capabilities include:"]

        for capability in capabilities.findall("capability"):
            name = capability.find("name")
            desc = capability.find("description")
            tool = capability.find("tool")

            if name is not None and name.text:
                cap_line = f"- {name.text.strip()}"
                if desc is not None and desc.text:
                    cap_line += f": {desc.text.strip()}"
                if tool is not None and tool.text:
                    cap_line += f" (Tool: {tool.text.strip()})"
                lines.append(cap_line)
            elif capability.text:
                lines.append(f"- {capability.text.strip()}")

        return "\n".join(lines) if len(lines) > 1 else ""

    def _format_team(self, team: ET.Element) -> str:
        """Format the team section for supervisor."""
        lines = ["Your team consists of:"]

        for agent in team.findall("agent"):
            name = agent.get("name", "Unknown")
            desc = agent.find("description")
            use_when = agent.find("use_when")

            lines.append(f"\n**{name}**")
            if desc is not None and desc.text:
                lines.append(f"  - {desc.text.strip()}")
            if use_when is not None and use_when.text:
                lines.append(f"  - Use when: {use_when.text.strip()}")

        return "\n".join(lines)

    def _format_responsibilities(self, responsibilities: ET.Element) -> str:
        """Format the responsibilities section."""
        lines = ["Your responsibilities:"]

        for resp in responsibilities.findall("responsibility"):
            if resp.text:
                lines.append(f"- {resp.text.strip()}")

        return "\n".join(lines) if len(lines) > 1 else ""

    def _format_instructions(self, instructions: ET.Element) -> str:
        """Format the instructions section."""
        lines = ["Instructions:"]

        for instruction in instructions.findall("instruction"):
            if instruction.text:
                priority = instruction.get("priority", "")
                prefix = ""
                if priority == "high":
                    prefix = "! "
                elif priority == "medium":
                    prefix = "- "
                lines.append(f"{prefix}{instruction.text.strip()}")

        return "\n".join(lines) if len(lines) > 1 else ""

    def _format_workflow(self, workflow: ET.Element) -> str:
        """Format the workflow section."""
        lines = ["Workflow:"]

        steps = workflow.findall("step")
        for step in steps:
            if step.text:
                order = step.get("order", "")
                lines.append(f"{order}. {step.text.strip()}")

        return "\n".join(lines) if len(lines) > 1 else ""

    def _format_decision_guidelines(self, guidelines: ET.Element) -> str:
        """Format decision guidelines for supervisor."""
        lines = ["Decision Guidelines:"]

        for guideline in guidelines.findall("guideline"):
            condition = guideline.find("condition")
            action = guideline.find("action")

            if condition is not None and action is not None and condition.text and action.text:
                lines.append(f"- {condition.text.strip()} → {action.text.strip()}")

        return "\n".join(lines) if len(lines) > 1 else ""

    def _format_communication_style(self, comm_style: ET.Element) -> str:
        """Format communication style section."""
        lines = ["Communication Style:"]

        for principle in comm_style.findall("principle"):
            if principle.text:
                lines.append(f"- {principle.text.strip()}")

        return "\n".join(lines) if len(lines) > 1 else ""


# Global instance for convenience
_loader = PromptLoader()


def load_prompt(prompt_name: str) -> str:
    """
    Convenience function to load a prompt using the global loader.

    Args:
        prompt_name: Name of the prompt file (without .xml extension)

    Returns:
        The formatted prompt text
    """
    return _loader.load_prompt(prompt_name)
