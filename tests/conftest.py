"""Pytest configuration and shared fixtures for BioAgents tests."""

import os
from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_llm():
    """Fixture providing a mock LLM instance."""
    llm = Mock()
    llm.invoke = Mock(return_value="Mock LLM response")
    llm.bind_tools = Mock(return_value=llm)
    llm.with_structured_output = Mock(return_value=llm)
    return llm


@pytest.fixture
def sample_fasta_sequence():
    """Fixture providing a sample FASTA sequence for testing."""
    return """>sp|P04637|P53_HUMAN Cellular tumor antigen p53
MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP"""


@pytest.fixture
def sample_protein_id():
    """Fixture providing a sample protein ID."""
    return "P04637"


@pytest.fixture
def mock_uniprot_response():
    """Fixture providing a mock UniProt API response."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = """>sp|P04637|P53_HUMAN Cellular tumor antigen p53
MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP
DEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAK"""
    return mock_response


@pytest.fixture
def test_env_vars():
    """Fixture providing test environment variables."""
    return {
        "OPENAI_API_KEY": "test-openai-key",
        "GEMINI_API_KEY": "test-gemini-key",
        "OLLAMA_BASE_URL": "http://localhost:11434/v1",
        "LLM_PROVIDER": "openai",
    }


@pytest.fixture
def clean_env():
    """Fixture that cleans environment variables before and after test."""
    original_env = os.environ.copy()
    # Clear relevant env vars
    env_vars_to_clear = [
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "OLLAMA_BASE_URL",
        "LLM_PROVIDER",
        "OPENAI_RATE_LIMIT",
        "GEMINI_RATE_LIMIT",
        "OLLAMA_RATE_LIMIT",
    ]
    for var in env_vars_to_clear:
        os.environ.pop(var, None)

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def sample_agent_state():
    """Fixture providing a sample agent state."""
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content="Test query")],
        "next": "",
        "reasoning": "",
    }


@pytest.fixture
def mock_tools():
    """Fixture providing mock tools for agents."""
    tool1 = Mock()
    tool1.name = "mock_tool_1"
    tool1.description = "A mock tool for testing"

    tool2 = Mock()
    tool2.name = "mock_tool_2"
    tool2.description = "Another mock tool"

    return [tool1, tool2]


@pytest.fixture
def temp_prompt_dir(tmp_path):
    """Fixture providing a temporary directory with test prompt files."""
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()

    # Create a simple test prompt
    test_prompt = prompt_dir / "test_prompt.xml"
    test_prompt.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<prompt>
    <role>You are a test agent.</role>
    <instructions>
        <instruction>Follow test instructions.</instruction>
    </instructions>
</prompt>
""")

    return prompt_dir


@pytest.fixture
def sample_amino_acid_sequence():
    """Fixture providing various amino acid sequences for testing."""
    return {
        "simple": "MASLKGFVP",
        "all_standard": "ARNDCEQGHILKMFPSTWYV",
        "with_stop": "MASL*KGF",
        "with_gaps": "MASL-KGF",
        "basic": "KKKRRRHHHH",  # Basic (positively charged)
        "acidic": "DDDEEEEE",  # Acidic (negatively charged)
        "hydrophobic": "AVILMFWP",  # Hydrophobic residues
        "p53_fragment": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP",
    }


@pytest.fixture
def mock_rate_limiter():
    """Fixture providing a mock rate limiter."""
    from bioagents.llms.rate_limiter import RateLimiter

    limiter = RateLimiter(max_requests=100, time_window=60.0)
    return limiter


@pytest.fixture(autouse=True)
def reset_rate_limiters():
    """Fixture that resets global rate limiters after each test."""
    yield
    # Clear global rate limiters after each test to prevent interference
    from bioagents.llms import llm_provider

    llm_provider._rate_limiters.clear()
    llm_provider._rate_limiter_initialized.clear()


@pytest.fixture
def mock_graph_agents():
    """Fixture providing mocks for all graph agents."""
    from langchain_core.messages import AIMessage

    mock_supervisor = Mock()
    mock_supervisor.return_value = {
        "next": "FINISH",
        "reasoning": "Task complete",
        "messages": [],
    }

    mock_research = Mock()
    mock_research.return_value = {
        "messages": [AIMessage(content="Research complete", name="Research")]
    }

    mock_analysis = Mock()
    mock_analysis.return_value = {
        "messages": [AIMessage(content="Analysis complete", name="Analysis")]
    }

    mock_report = Mock()
    mock_report.return_value = {"messages": [AIMessage(content="Report complete", name="Report")]}

    return {
        "supervisor": mock_supervisor,
        "research": mock_research,
        "analysis": mock_analysis,
        "report": mock_report,
    }


@pytest.fixture
def sample_tool_calls():
    """Fixture providing sample tool call structures."""
    return {
        "fetch_protein": {
            "name": "fetch_uniprot_fasta",
            "args": {"protein_id": "P04637"},
            "id": "call_fetch_123",
        },
        "calc_mw": {
            "name": "calculate_molecular_weight",
            "args": {"fasta_sequence": ">test\nMASLKGFVP"},
            "id": "call_mw_456",
        },
        "analyze_comp": {
            "name": "analyze_amino_acid_composition",
            "args": {"fasta_sequence": ">test\nMASLKGFVP"},
            "id": "call_comp_789",
        },
    }


# Pytest configuration markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "requires_api: mark test as requiring external API access")


# Skip slow tests by default unless --runslow is passed
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")
    parser.addoption(
        "--runintegration",
        action="store_true",
        default=False,
        help="run integration tests",
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers and command line options."""
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if not config.getoption("--runintegration"):
        skip_integration = pytest.mark.skip(reason="need --runintegration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
