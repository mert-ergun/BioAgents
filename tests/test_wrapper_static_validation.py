"""Wrapper generation retries and static import/API drift checks (#65)."""

import json
import sys
import textwrap
from unittest.mock import Mock

import pytest

from bioagents.tools import tool_builder_tools as tools
from bioagents.tools.code_api_check import check_generated_code
from bioagents.tools.tool_registry import ToolRegistry

FAKE_PACKAGE = "bioagents_fake_chem_fixture"


@pytest.fixture
def fake_package(tmp_path, monkeypatch):
    package = tmp_path / FAKE_PACKAGE
    (package / "brics").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "brics" / "__init__.py").write_text(
        textwrap.dedent(
            """
            def BRICSBuild(fragments, onlyCompleteMols=True, seeds=None, maxDepth=3):
                return list(fragments)

            def flexible(*args, **kwargs):
                return kwargs

            class Builder:
                def __init__(self, size, *, strict=False):
                    self.size = size
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yield FAKE_PACKAGE
    for name in [m for m in sys.modules if m.startswith(FAKE_PACKAGE)]:
        del sys.modules[name]


def test_unknown_keyword_against_installed_signature_is_reported(fake_package):
    code = textwrap.dedent(
        f"""
        from {fake_package}.brics import BRICSBuild

        def generate(fragments):
            return BRICSBuild(fragments, minFragmentSize=3, maxDepth=2)
        """
    )
    problems = check_generated_code(code)
    assert len(problems) == 1
    assert "BRICSBuild() got an unexpected keyword argument 'minFragmentSize'" in problems[0]


def test_module_alias_attribute_calls_and_classes_are_checked(fake_package):
    code = textwrap.dedent(
        f"""
        import {fake_package}.brics as brics

        def generate():
            brics.Builder(3, strict=True)
            return brics.Builder(3, loose=True)
        """
    )
    problems = check_generated_code(code)
    assert len(problems) == 1
    assert "'loose'" in problems[0]


def test_var_kwargs_valid_keywords_and_stdlib_are_not_flagged(fake_package):
    code = textwrap.dedent(
        f"""
        import json
        from {fake_package}.brics import BRICSBuild, flexible

        def generate(fragments):
            flexible(anything=1)
            json.dumps({{}}, not_a_real_json_kwarg=True)
            return BRICSBuild(fragments, maxDepth=2, seeds=None)
        """
    )
    assert check_generated_code(code) == []


def test_unresolvable_imports_are_reported_unless_guarded():
    code = textwrap.dedent(
        """
        import definitely_missing_fixture_a
        from definitely_missing_fixture_b.sub import thing

        try:
            import definitely_missing_optional_fixture
        except ImportError:
            definitely_missing_optional_fixture = None

        def run():
            import definitely_missing_fixture_c
            return thing(x=1)
        """
    )
    problems = check_generated_code(code)
    assert any("definitely_missing_fixture_a" in p for p in problems)
    assert any("definitely_missing_fixture_b.sub" in p for p in problems)
    assert any("definitely_missing_fixture_c" in p for p in problems)
    assert not any("optional" in p for p in problems)
    assert len(problems) == 3


def test_syntax_error_is_a_problem():
    assert check_generated_code("def broken(:")[0].startswith("SyntaxError")


def _register(name, code):
    return json.loads(
        tools.register_custom_tool.invoke(
            {
                "name": name,
                "description": "Fixture",
                "category": "test",
                "code": code,
                "parameters_json": "[]",
                "return_type": "dict",
                "return_description": "Result",
            }
        )
    )


def test_registration_and_validation_record_static_failures(tmp_path, monkeypatch, fake_package):
    registry = ToolRegistry(tools_dir=tmp_path / "tools")
    monkeypatch.setattr(tools, "get_registry", lambda: registry)
    code = (
        f"from {fake_package}.brics import BRICSBuild\n"
        "def drifted():\n"
        "    return {'status': 'success', 'r': BRICSBuild([], minFragmentSize=3)}\n"
    )
    registered = _register("drifted", code)
    assert registered["status"] == "success"
    assert "minFragmentSize" in registered["static_check_errors"][0]

    validated = json.loads(tools.validate_custom_tool.invoke({"tool_name": "drifted"}))
    assert validated["status"] == "error"
    assert "minFragmentSize" in validated["message"]

    entry = ToolRegistry(tools_dir=tmp_path / "tools").get_tool("drifted")
    assert entry.validated is False
    assert "minFragmentSize" in entry.static_check_errors[0]
    assert entry.last_failure["phase"] == "static_check"


def test_load_failure_reason_is_returned_and_persisted(tmp_path, monkeypatch):
    registry = ToolRegistry(tools_dir=tmp_path / "tools")
    monkeypatch.setattr(tools, "get_registry", lambda: registry)
    _register(
        "needs_missing",
        "import definitely_missing_fixture_pkg\ndef needs_missing():\n    return {}\n",
    )
    result = json.loads(tools.execute_custom_tool.invoke({"tool_name": "needs_missing"}))
    assert result["status"] == "error"
    assert "definitely_missing_fixture_pkg" in result["message"]
    entry = ToolRegistry(tools_dir=tmp_path / "tools").get_tool("needs_missing")
    assert entry.last_failure["phase"] == "load"
    assert "definitely_missing_fixture_pkg" in entry.last_failure["reason"]


def test_clean_tool_validates_and_has_no_static_errors(tmp_path, monkeypatch, fake_package):
    registry = ToolRegistry(tools_dir=tmp_path / "tools")
    monkeypatch.setattr(tools, "get_registry", lambda: registry)
    code = (
        f"from {fake_package}.brics import BRICSBuild\n"
        "def clean():\n"
        "    return {'status': 'success', 'r': BRICSBuild([], maxDepth=2)}\n"
    )
    assert "static_check_errors" not in _register("clean", code)
    validated = json.loads(tools.validate_custom_tool.invoke({"tool_name": "clean"}))
    assert validated["status"] == "success"


class _FakeLLM:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def invoke(self, _prompt):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return Mock(content=outcome)


_WRAPPER_ARGS = {
    "tool_name": "example",
    "tool_description": "Test",
    "tool_category": "test",
    "parameters_json": "[]",
    "return_type": "dict",
    "return_description": "Result",
    "implementation_notes": "Test only",
}


def test_wrapper_generation_backs_off_with_bounded_delays(monkeypatch):
    llm = _FakeLLM(
        [TimeoutError("LLM invoke exceeded 300s"), "", '{"code": "def example(): pass"}']
    )
    delays = []
    monkeypatch.setattr(tools, "get_llm", lambda: llm)
    monkeypatch.setattr(tools, "remaining_workflow_seconds", lambda: None)
    monkeypatch.setattr(tools.time, "sleep", delays.append)
    monkeypatch.setattr(tools, "TOOL_WRAPPER_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(tools, "TOOL_WRAPPER_RETRY_BASE_DELAY_SEC", 2.0)
    monkeypatch.setattr(tools, "TOOL_WRAPPER_RETRY_MAX_DELAY_SEC", 3.0)
    result = json.loads(tools.generate_tool_wrapper.invoke(_WRAPPER_ARGS))
    assert result["code"] == "def example(): pass"
    assert llm.calls == 3
    assert delays == [2.0, 3.0]


def test_wrapper_generation_gives_up_after_configured_attempts(monkeypatch):
    llm = _FakeLLM([TimeoutError("t1"), TimeoutError("t2")])
    monkeypatch.setattr(tools, "get_llm", lambda: llm)
    monkeypatch.setattr(tools, "remaining_workflow_seconds", lambda: None)
    monkeypatch.setattr(tools.time, "sleep", lambda _: None)
    monkeypatch.setattr(tools, "TOOL_WRAPPER_MAX_ATTEMPTS", 2)
    result = json.loads(tools.generate_tool_wrapper.invoke(_WRAPPER_ARGS))
    assert result["error"] == "t2"
    assert llm.calls == 2


def test_generated_wrapper_carries_static_check_errors(monkeypatch):
    code = "import definitely_missing_fixture_gen\ndef example(): pass\n"
    llm = _FakeLLM([json.dumps({"code": code, "dependencies": []})])
    monkeypatch.setattr(tools, "get_llm", lambda: llm)
    result = json.loads(tools.generate_tool_wrapper.invoke(_WRAPPER_ARGS))
    assert "definitely_missing_fixture_gen" in result["static_check_errors"][0]
