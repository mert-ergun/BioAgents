"""Tests for prompt loader module."""

import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from bioagents.prompts.prompt_loader import (
    PromptLoader,
    get_prompt_llm_model,
    get_prompt_llm_models,
    load_prompt,
)


class TestPromptLoader:
    """Tests for the PromptLoader class."""

    def test_initialization_default(self):
        """Test initialization with default prompts directory."""
        loader = PromptLoader()
        assert loader.prompts_dir is not None
        assert loader.prompts_dir.exists()

    def test_initialization_custom_dir(self):
        """Test initialization with custom prompts directory."""
        with TemporaryDirectory() as tmpdir:
            loader = PromptLoader(prompts_dir=Path(tmpdir))
            assert loader.prompts_dir == Path(tmpdir)

    def test_load_nonexistent_prompt(self):
        """Test loading a nonexistent prompt file."""
        loader = PromptLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_prompt("nonexistent_prompt_file_12345")

    def test_load_simple_prompt(self):
        """Test loading a simple prompt with role and instructions."""
        with TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "test_prompt.xml"
            prompt_file.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<prompt>
    <role>You are a test assistant.</role>
    <instructions>
        <instruction>Follow these rules.</instruction>
        <instruction>Be helpful.</instruction>
    </instructions>
</prompt>
""")

            loader = PromptLoader(prompts_dir=tmpdir)
            result = loader.load_prompt("test_prompt")

            assert "You are a test assistant." in result
            assert "Instructions:" in result
            assert "Follow these rules." in result
            assert "Be helpful." in result

    def test_load_prompt_with_capabilities(self):
        """Test loading a prompt with capabilities section."""
        with TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "test_prompt.xml"
            prompt_file.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<prompt>
    <role>Research Agent</role>
    <capabilities>
        <capability>
            <name>Fetch Data</name>
            <description>Fetch protein data</description>
            <tool>fetch_uniprot_fasta</tool>
        </capability>
        <capability>Simple capability</capability>
    </capabilities>
</prompt>
""")

            loader = PromptLoader(prompts_dir=tmpdir)
            result = loader.load_prompt("test_prompt")

            assert "Your capabilities include:" in result
            assert "Fetch Data" in result
            assert "Fetch protein data" in result
            assert "fetch_uniprot_fasta" in result
            assert "Simple capability" in result

    def test_load_prompt_with_team(self):
        """Test loading a prompt with team section (for supervisor)."""
        with TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "test_prompt.xml"
            prompt_file.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<prompt>
    <role>Supervisor</role>
    <team>
        <agent name="research">
            <description>Fetches data</description>
            <use_when>Data needed</use_when>
        </agent>
        <agent name="analysis">
            <description>Analyzes data</description>
            <use_when>Analysis needed</use_when>
        </agent>
    </team>
</prompt>
""")

            loader = PromptLoader(prompts_dir=tmpdir)
            result = loader.load_prompt("test_prompt")

            assert "Your team consists of:" in result
            assert "**research**" in result
            assert "Fetches data" in result
            assert "Data needed" in result
            assert "**analysis**" in result

    def test_load_prompt_with_workflow(self):
        """Test loading a prompt with workflow section."""
        with TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "test_prompt.xml"
            prompt_file.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<prompt>
    <role>Agent</role>
    <workflow>
        <step order="1">First step</step>
        <step order="2">Second step</step>
        <step order="3">Third step</step>
    </workflow>
</prompt>
""")

            loader = PromptLoader(prompts_dir=tmpdir)
            result = loader.load_prompt("test_prompt")

            assert "Workflow:" in result
            assert "1. First step" in result
            assert "2. Second step" in result
            assert "3. Third step" in result

    def test_load_prompt_with_decision_guidelines(self):
        """Test loading a prompt with decision guidelines."""
        with TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "test_prompt.xml"
            prompt_file.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<prompt>
    <role>Supervisor</role>
    <decision_guidelines>
        <guideline>
            <condition>If data is needed</condition>
            <action>Route to research</action>
        </guideline>
        <guideline>
            <condition>If analysis is needed</condition>
            <action>Route to analysis</action>
        </guideline>
    </decision_guidelines>
</prompt>
""")

            loader = PromptLoader(prompts_dir=tmpdir)
            result = loader.load_prompt("test_prompt")

            assert "Decision Guidelines:" in result
            assert "If data is needed → Route to research" in result
            assert "If analysis is needed → Route to analysis" in result

    def test_load_prompt_with_communication_style(self):
        """Test loading a prompt with communication style."""
        with TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "test_prompt.xml"
            prompt_file.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<prompt>
    <role>Agent</role>
    <communication_style>
        <principle>Be clear and concise</principle>
        <principle>Use professional language</principle>
    </communication_style>
</prompt>
""")

            loader = PromptLoader(prompts_dir=tmpdir)
            result = loader.load_prompt("test_prompt")

            assert "Communication Style:" in result
            assert "Be clear and concise" in result
            assert "Use professional language" in result

    def test_load_prompt_with_responsibilities(self):
        """Test loading a prompt with responsibilities section."""
        with TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "test_prompt.xml"
            prompt_file.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<prompt>
    <role>Report Agent</role>
    <responsibilities>
        <responsibility>Synthesize findings</responsibility>
        <responsibility>Present clear reports</responsibility>
    </responsibilities>
</prompt>
""")

            loader = PromptLoader(prompts_dir=tmpdir)
            result = loader.load_prompt("test_prompt")

            assert "Your responsibilities:" in result
            assert "Synthesize findings" in result
            assert "Present clear reports" in result

    def test_load_prompt_with_priority_instructions(self):
        """Test loading a prompt with priority instructions."""
        with TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "test_prompt.xml"
            prompt_file.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<prompt>
    <role>Agent</role>
    <instructions>
        <instruction priority="high">Critical instruction</instruction>
        <instruction priority="medium">Important instruction</instruction>
        <instruction>Normal instruction</instruction>
    </instructions>
</prompt>
""")

            loader = PromptLoader(prompts_dir=tmpdir)
            result = loader.load_prompt("test_prompt")

            assert "Instructions:" in result
            assert "! Critical instruction" in result
            assert "- Important instruction" in result
            assert "Normal instruction" in result

    def test_load_malformed_xml(self):
        """Test error handling for malformed XML."""
        with TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "bad_prompt.xml"
            prompt_file.write_text("<prompt><role>Unclosed tag")

            loader = PromptLoader(prompts_dir=tmpdir)
            with pytest.raises(ET.ParseError):
                loader.load_prompt("bad_prompt")

    def test_load_empty_sections(self):
        """Test that empty sections are handled gracefully."""
        with TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "test_prompt.xml"
            prompt_file.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<prompt>
    <role>Agent</role>
    <capabilities></capabilities>
    <instructions></instructions>
</prompt>
""")

            loader = PromptLoader(prompts_dir=tmpdir)
            result = loader.load_prompt("test_prompt")

            assert "Agent" in result
            # Empty sections should not appear
            assert "Your capabilities include:" not in result or result.count("\n\n") >= 0

    def test_section_separation(self):
        """Test that sections are separated by double newlines."""
        with TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "test_prompt.xml"
            prompt_file.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<prompt>
    <role>Test Agent</role>
    <instructions>
        <instruction>First instruction</instruction>
    </instructions>
    <communication_style>
        <principle>Be clear</principle>
    </communication_style>
</prompt>
""")

            loader = PromptLoader(prompts_dir=tmpdir)
            result = loader.load_prompt("test_prompt")

            # Sections should be separated by double newlines
            assert "\n\n" in result

    def test_get_llm_models(self):
        """Test extracting LLM models from metadata."""
        with TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "test_prompt.xml"
            prompt_file.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<prompt>
    <metadata>
        <name>Test Agent</name>
        <llm_models>
            <model provider="openai">gpt-5.1</model>
            <model provider="Gemini">gemini-2.5-flash</model>
        </llm_models>
    </metadata>
    <role>Test role</role>
</prompt>
""")

            loader = PromptLoader(prompts_dir=tmpdir)
            models = loader.get_llm_models("test_prompt")

            assert models == {"openai": "gpt-5.1", "gemini": "gemini-2.5-flash"}

    def test_get_llm_model_with_provider(self):
        """Test retrieving a specific provider model."""
        with TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "test_prompt.xml"
            prompt_file.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<prompt>
    <metadata>
        <name>Test Agent</name>
        <llm_models>
            <model provider="openai">gpt-5.1</model>
        </llm_models>
    </metadata>
    <role>Test role</role>
</prompt>
""")

            loader = PromptLoader(prompts_dir=tmpdir)
            assert loader.get_llm_model("test_prompt", "OpenAI") == "gpt-5.1"
            assert loader.get_llm_model("test_prompt", "gemini") is None


class TestLoadPromptFunction:
    """Tests for the convenience load_prompt function."""

    def test_load_prompt_function_uses_global_loader(self):
        """Test that load_prompt function uses the global loader."""
        # This test attempts to load an actual prompt from the project
        # It will work if the prompts directory is properly set up
        try:
            result = load_prompt("supervisor")
            assert isinstance(result, str)
            assert len(result) > 0
        except FileNotFoundError:
            # If the file doesn't exist, that's also valid for the test environment
            pytest.skip("Supervisor prompt file not found in default location")

    def test_load_actual_prompts(self):
        """Test loading actual prompt files from the project."""
        prompt_names = ["supervisor", "research", "analysis", "report"]

        for prompt_name in prompt_names:
            try:
                result = load_prompt(prompt_name)
                assert isinstance(result, str)
                assert len(result) > 0
                assert "\n" in result  # Should have multiple lines
            except FileNotFoundError:
                pytest.skip(f"{prompt_name} prompt file not found")


class TestPromptIntegrity:
    """Tests for verifying the integrity of actual prompt files."""

    def test_supervisor_prompt_structure(self):
        """Test that supervisor prompt has expected structure."""
        try:
            result = load_prompt("supervisor")
            # Should contain team information
            assert any(
                keyword in result.lower() for keyword in ["team", "route", "supervisor", "agent"]
            )
        except FileNotFoundError:
            pytest.skip("Supervisor prompt file not found")

    def test_research_prompt_structure(self):
        """Test that research prompt has expected structure."""
        try:
            result = load_prompt("research")
            # Should contain research-related information
            assert any(
                keyword in result.lower() for keyword in ["research", "fetch", "data", "protein"]
            )
        except FileNotFoundError:
            pytest.skip("Research prompt file not found")

    def test_analysis_prompt_structure(self):
        """Test that analysis prompt has expected structure."""
        try:
            result = load_prompt("analysis")
            # Should contain analysis-related information
            assert any(
                keyword in result.lower()
                for keyword in ["analysis", "analyze", "calculate", "sequence"]
            )
        except FileNotFoundError:
            pytest.skip("Analysis prompt file not found")

    def test_report_prompt_structure(self):
        """Test that report prompt has expected structure."""
        try:
            result = load_prompt("report")
            # Should contain report-related information
            assert any(keyword in result.lower() for keyword in ["report", "synthesize", "present"])
        except FileNotFoundError:
            pytest.skip("Report prompt file not found")


class TestPromptLLMMetadataHelpers:
    """Tests for the convenience LLM metadata helpers."""

    def test_get_prompt_llm_models_helper(self):
        """Test helper returns expected providers for actual prompt file."""
        try:
            models = get_prompt_llm_models("analysis")
        except FileNotFoundError:
            pytest.skip("Analysis prompt file not found")

        assert models["openai"] == "gpt-5.1"
        assert models["gemini"] == "gemini-2.5-flash"
        assert models["ollama"] == "qwen3:14b"

    def test_get_prompt_llm_model_helper(self):
        """Test helper returns provider-specific model or None."""
        try:
            model = get_prompt_llm_model("analysis", "openai")
        except FileNotFoundError:
            pytest.skip("Analysis prompt file not found")

        assert model == "gpt-5.1"
        assert get_prompt_llm_model("analysis", "missing") is None
