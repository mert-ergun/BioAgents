"""Utility for loading and parsing XML-formatted system prompts."""

import xml.etree.ElementTree as ET  # nosec B405  # Internal trusted XML files only
from pathlib import Path

ModelMap = dict[str, str]


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

    def _get_prompt_root(self, prompt_name: str) -> ET.Element:
        """Parse a prompt XML file and return the root element."""
        prompt_file = self.prompts_dir / f"{prompt_name}.xml"

        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        tree = ET.parse(prompt_file)  # nosec B314  # Internal trusted XML files only
        return tree.getroot()

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
        root = self._get_prompt_root(prompt_name)

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

        # Add evaluation criteria (if present)
        evaluation_criteria = root.find("evaluation_criteria")
        if evaluation_criteria is not None:
            eval_text = self._format_evaluation_criteria(evaluation_criteria)
            if eval_text:
                sections.append(eval_text)

        # Add output format (if present)
        output_format = root.find("output_format")
        if output_format is not None:
            out_text = self._format_output_format(output_format)
            if out_text:
                sections.append(out_text)

        # Add data formats (if present)
        data_formats = root.find("data_formats")
        if data_formats is not None:
            df_text = self._format_data_formats(data_formats)
            if df_text:
                sections.append(df_text)

        # Add instructions
        instructions = root.find("instructions")
        if instructions is not None:
            inst_text = self._format_instructions(instructions)
            if inst_text:
                sections.append(inst_text)

        # Add error handling
        error_handling = root.find("error_handling")
        if error_handling is not None:
            err_text = self._format_error_handling(error_handling)
            if err_text:
                sections.append(err_text)

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

        # Add best practices
        best_practices = root.find("best_practices")
        if best_practices is not None:
            bp_text = self._format_best_practices(best_practices)
            if bp_text:
                sections.append(bp_text)

        # Add examples (if present)
        examples = root.find("examples")
        if examples is not None:
            examples_text = self._format_examples(examples)
            if examples_text:
                sections.append(examples_text)

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

    def get_llm_models(self, prompt_name: str) -> ModelMap:
        """
        Return the preferred LLM models for a given prompt keyed by provider.

        Args:
            prompt_name: Name of the prompt file (without .xml extension)

        Returns:
            Mapping of provider → model name (provider keys are lower-case)
        """
        root = self._get_prompt_root(prompt_name)
        metadata = root.find("metadata")
        if metadata is None:
            return {}

        llm_models = metadata.find("llm_models")
        if llm_models is None:
            return {}

        models: ModelMap = {}
        for model_elem in llm_models.findall("model"):
            provider = model_elem.get("provider")
            if not provider:
                continue
            provider_key = provider.lower()
            if model_elem.text:
                models[provider_key] = model_elem.text.strip()

        return models

    def get_llm_model(self, prompt_name: str, provider: str | None) -> str | None:
        """
        Return the preferred LLM model for a given prompt and provider.

        Args:
            prompt_name: Name of the prompt file (without .xml extension)
            provider: Provider identifier (e.g., 'openai', 'gemini', 'ollama')

        Returns:
            Model name if configured, otherwise None.
        """
        if not provider:
            return None

        provider_key = provider.lower()
        models = self.get_llm_models(prompt_name)
        return models.get(provider_key)

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

    def _format_evaluation_criteria(self, criteria: ET.Element) -> str:
        """Format the evaluation criteria section."""
        lines = ["Evaluation Criteria:"]

        for criterion in criteria.findall("criterion"):
            name = criterion.get("name")
            if name and criterion.text:
                lines.append(f"- {name}: {criterion.text.strip()}")
            elif criterion.text:
                lines.append(f"- {criterion.text.strip()}")

        return "\n".join(lines) if len(lines) > 1 else ""

    def _format_output_format(self, output_format: ET.Element) -> str:
        """Format the output format section."""
        lines = ["Output Format:"]

        # Handle requirements
        for req in output_format.findall("requirement"):
            if req.text:
                lines.append(f"- {req.text.strip()}")

        # Handle sections
        for section in output_format.findall("section"):
            name = section.get("name")
            desc = section.find("description")
            if name:
                line = f"- {name}"
                if desc is not None and desc.text:
                    line += f": {desc.text.strip()}"
                lines.append(line)

        return "\n".join(lines) if len(lines) > 1 else ""

    def _format_data_formats(self, data_formats: ET.Element) -> str:
        """Format the data formats section."""
        lines = ["Data Formats:"]

        for fmt in data_formats.findall("format"):
            name = fmt.get("name")
            if name:
                lines.append(f"\n{name}:")
                desc = fmt.find("description")
                if desc is not None and desc.text:
                    lines.append(f"  Description: {desc.text.strip()}")

                struct = fmt.find("structure")
                if struct is not None and struct.text:
                    struct_text = struct.text.strip()
                    indented_struct = "\n".join(f"    {line}" for line in struct_text.split("\n"))
                    lines.append(f"  Structure:\n{indented_struct}")

        return "\n".join(lines) if len(lines) > 1 else ""

    def _format_error_handling(self, error_handling: ET.Element) -> str:
        """Format the error handling section."""
        lines = ["Error Handling:"]

        for error in error_handling.findall("error"):
            err_type = error.get("type", "Error")
            lines.append(f"\n- {err_type}:")

            for child in error:
                if child.text:
                    lines.append(f"  {child.tag.capitalize()}: {child.text.strip()}")

        return "\n".join(lines) if len(lines) > 1 else ""

    def _format_best_practices(self, best_practices: ET.Element) -> str:
        """Format the best practices section."""
        lines = ["Best Practices:"]

        for practice in best_practices.findall("practice"):
            if practice.text:
                lines.append(f"- {practice.text.strip()}")

        return "\n".join(lines) if len(lines) > 1 else ""

    def _format_examples(self, examples: ET.Element) -> str:
        """Format the examples section."""
        lines = ["Examples:"]

        for i, example in enumerate(examples.findall("example"), 1):
            lines.append(f"\nExample {i}:")

            # Format all sub-elements of the example
            for child in example:
                tag = child.tag
                label = tag.capitalize()

                # If child has nested elements (like <actions><action>...</actions>)
                if len(child) > 0:
                    lines.append(f"  {label}:")
                    for subchild in child:
                        if subchild.text and subchild.text.strip():
                            text = subchild.text.strip()
                            order = subchild.get("order")
                            prefix = f"{order}. " if order else "- "

                            if "\n" in text:
                                indented_text = "\n".join(
                                    f"      {line}" for line in text.split("\n")
                                )
                                lines.append(f"    {prefix}{indented_text.lstrip()}")
                            else:
                                lines.append(f"    {prefix}{text}")
                # If child only has text
                elif child.text and child.text.strip():
                    text = child.text.strip()
                    if "\n" in text:
                        indented_text = "\n".join(f"    {line}" for line in text.split("\n"))
                        lines.append(f"  {label}:\n{indented_text}")
                    else:
                        lines.append(f"  {label}: {text}")

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


def get_prompt_llm_models(prompt_name: str) -> ModelMap:
    """
    Convenience function to fetch LLM model metadata for a prompt.
    """
    return _loader.get_llm_models(prompt_name)


def get_prompt_llm_model(prompt_name: str, provider: str | None) -> str | None:
    """
    Convenience function to fetch a provider-specific model for a prompt.
    """
    return _loader.get_llm_model(prompt_name, provider)
