"""BioAgents test suite.

This package contains all tests for the BioAgents project, including:
- Unit tests for individual components
- Integration tests for multi-component workflows
- End-to-end tests for complete system functionality

Test Organization:
------------------
- test_rate_limiter.py: Tests for rate limiting functionality
- test_llm_provider.py: Tests for LLM provider configuration
- test_proteomics_tools.py: Tests for protein data fetching tools
- test_analysis_tools.py: Tests for protein sequence analysis tools
- test_prompt_loader.py: Tests for prompt loading and parsing
- test_agents.py: Tests for individual agent modules
- test_graph.py: Tests for graph workflow and routing
- test_integration.py: Integration tests for complete workflows
- conftest.py: Shared fixtures and pytest configuration

Running Tests:
-------------
Run all tests:
    pytest

Run specific test file:
    pytest tests/test_analysis_tools.py

Run specific test:
    pytest tests/test_analysis_tools.py::TestCalculateMolecularWeight::test_calculate_simple_sequence

Run tests with specific marker:
    pytest -m unit
    pytest -m integration

Run with coverage:
    pytest --cov=bioagents --cov-report=html

Run with verbose output:
    pytest -v

Run slow tests (skipped by default):
    pytest --runslow

Run integration tests (skipped by default):
    pytest --runintegration
"""

__version__ = "1.0.0"
