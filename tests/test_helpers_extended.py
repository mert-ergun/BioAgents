"""Tests for extended helper functions added in Phase 0."""

from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from bioagents.agents.helpers import (
    create_retry_response,
    extract_best_content,
    extract_task_from_messages,
    get_content_text,
    get_message_content,
    invoke_with_retry,
    is_empty_response,
    prepare_messages_for_agent,
)
from bioagents.agents.supervisor_helpers import check_finish_if_code_agent_substantive_repeat

# =========================================================================
# get_content_text
# =========================================================================


class TestGetContentText:
    def test_string_content(self):
        assert get_content_text("hello") == "hello"

    def test_list_of_text_blocks(self):
        blocks = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
        assert get_content_text(blocks) == "hello world"

    def test_list_of_strings(self):
        assert get_content_text(["hello", "world"]) == "hello world"

    def test_empty_string(self):
        assert get_content_text("") == ""

    def test_none(self):
        assert get_content_text(None) == ""

    def test_mixed_list(self):
        blocks = [{"type": "text", "text": "a"}, "b", {"type": "image", "url": "x"}]
        assert get_content_text(blocks) == "a b"


# =========================================================================
# is_empty_response
# =========================================================================


class TestIsEmptyResponse:
    def test_empty_content_no_tools(self):
        response = AIMessage(content="")
        assert is_empty_response(response) is True

    def test_whitespace_only_no_tools(self):
        response = AIMessage(content="   ")
        assert is_empty_response(response) is True

    def test_non_empty_content(self):
        response = AIMessage(content="some data")
        assert is_empty_response(response) is False

    def test_empty_content_with_tool_calls(self):
        response = AIMessage(
            content="",
            tool_calls=[{"name": "t", "args": {}, "id": "1"}],
        )
        assert is_empty_response(response) is False

    def test_non_empty_content_with_tool_calls(self):
        response = AIMessage(
            content="data",
            tool_calls=[{"name": "t", "args": {}, "id": "1"}],
        )
        assert is_empty_response(response) is False


# =========================================================================
# prepare_messages_for_agent
# =========================================================================


class TestPrepareMessagesForAgent:
    def test_keeps_first_human_message(self):
        msgs = [
            HumanMessage(content="original query"),
            AIMessage(content="response 1"),
            AIMessage(content="response 2"),
        ]
        result = prepare_messages_for_agent(msgs, "test")
        assert any(isinstance(m, HumanMessage) and "original query" in m.content for m in result)

    def test_truncates_long_tool_results(self):
        msgs = [
            HumanMessage(content="query"),
            ToolMessage(content="x" * 5000, tool_call_id="1", name="tool"),
        ]
        result = prepare_messages_for_agent(msgs, "test", max_tool_result_len=100)
        for m in result:
            if isinstance(m, ToolMessage):
                assert len(m.content) < 5000

    def test_summary_mode_filters_tool_calls(self):
        msgs = [
            HumanMessage(content="query"),
            AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
            AIMessage(content="Final analysis result with enough substance to pass filter"),
        ]
        result = prepare_messages_for_agent(msgs, "test", summary_mode=True)
        has_tool_call_msg = any(
            isinstance(m, AIMessage) and hasattr(m, "tool_calls") and m.tool_calls for m in result
        )
        assert not has_tool_call_msg

    def test_empty_messages(self):
        result = prepare_messages_for_agent([], "test")
        assert result == []

    def test_preserves_supervisor_task_messages(self):
        msgs = [
            HumanMessage(content="query"),
            HumanMessage(content="[SUPERVISOR TASK] Analyze protein"),
            AIMessage(content="x" * 100),
        ]
        result = prepare_messages_for_agent(msgs, "test")
        supervisor_msgs = [
            m for m in result if isinstance(m, HumanMessage) and "[SUPERVISOR TASK]" in m.content
        ]
        assert len(supervisor_msgs) >= 1

    def test_windowing_respects_max_messages(self):
        msgs = [HumanMessage(content=f"msg {i}") for i in range(50)]
        result = prepare_messages_for_agent(msgs, "test", max_messages=5)
        assert len(result) <= 6  # max_messages + first_human if not in window

    def test_truncates_long_ai_content(self):
        msgs = [
            HumanMessage(content="query"),
            AIMessage(content="x" * 10000),
        ]
        result = prepare_messages_for_agent(msgs, "test", max_ai_content_len=100)
        for m in result:
            if isinstance(m, AIMessage):
                text = get_content_text(m.content)
                assert len(text) < 10000


# =========================================================================
# extract_best_content
# =========================================================================


class TestExtractBestContent:
    def test_finds_longest_content(self):
        msgs = [
            AIMessage(content="short"),
            AIMessage(content="a much longer content that should be selected as best"),
            AIMessage(content="medium length content"),
        ]
        best = extract_best_content(msgs)
        assert "much longer" in best

    def test_empty_messages(self):
        assert extract_best_content([]) == ""

    def test_ignores_empty_messages(self):
        msgs = [
            AIMessage(content=""),
            AIMessage(content="   "),
            AIMessage(content="actual content"),
        ]
        best = extract_best_content(msgs)
        assert best == "actual content"

    def test_ignores_non_ai_messages(self):
        msgs = [
            HumanMessage(content="this is very long user query that should not be selected"),
            AIMessage(content="short AI"),
        ]
        best = extract_best_content(msgs)
        assert best == "short AI"

    def test_truncates_very_long_content(self):
        msgs = [AIMessage(content="x" * 5000)]
        best = extract_best_content(msgs)
        assert len(best) <= 4100  # 4000 + truncation marker


# =========================================================================
# invoke_with_retry
# =========================================================================


class TestInvokeWithRetry:
    def test_returns_on_first_success(self):
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="response")
        msgs = [SystemMessage(content="system"), HumanMessage(content="query")]

        result = invoke_with_retry("Test", mock_llm, msgs)
        assert result["messages"][0].content == "response"
        assert mock_llm.invoke.call_count == 1

    def test_retries_on_empty(self):
        mock_llm = Mock()
        mock_llm.invoke.side_effect = [
            AIMessage(content=""),
            AIMessage(content="second try works"),
        ]
        msgs = [SystemMessage(content="system"), HumanMessage(content="query")]

        result = invoke_with_retry("Test", mock_llm, msgs)
        assert "second try" in result["messages"][0].content
        assert mock_llm.invoke.call_count == 2

    def test_returns_fallback_on_all_empty(self):
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="")
        msgs = [SystemMessage(content="system"), HumanMessage(content="query")]

        result = invoke_with_retry("Test", mock_llm, msgs, max_retries=1)
        assert len(result["messages"]) == 1
        assert result["messages"][0].content != ""

    def test_fallback_uses_best_content_from_history(self):
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="")
        msgs = [
            SystemMessage(content="system"),
            HumanMessage(content="query"),
            AIMessage(content="Some useful content from earlier"),
        ]

        result = invoke_with_retry("Test", mock_llm, msgs, max_retries=1)
        assert "useful content" in result["messages"][0].content

    def test_preserves_tool_call_responses(self):
        mock_llm = Mock()
        response = AIMessage(
            content="",
            tool_calls=[{"name": "tool", "args": {}, "id": "1"}],
        )
        mock_llm.invoke.return_value = response
        msgs = [SystemMessage(content="system"), HumanMessage(content="query")]

        result = invoke_with_retry("Test", mock_llm, msgs)
        assert len(result["messages"]) == 1
        assert result["messages"][0].tool_calls


# =========================================================================
# create_retry_response
# =========================================================================


class TestCreateRetryResponse:
    def test_returns_on_first_success(self):
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="done")
        msgs = [SystemMessage(content="system"), HumanMessage(content="query")]

        result = create_retry_response("Agent", msgs, ["tool1"], mock_llm)
        assert result["messages"][0].content == "done"
        assert mock_llm.invoke.call_count == 1

    def test_retries_on_empty(self):
        mock_llm = Mock()
        mock_llm.invoke.side_effect = [
            AIMessage(content=""),
            AIMessage(content="retry worked"),
        ]
        msgs = [SystemMessage(content="system"), HumanMessage(content="query")]

        result = create_retry_response("Agent", msgs, ["tool1"], mock_llm)
        assert result["messages"][0].content == "retry worked"

    def test_returns_error_message_on_exhausted_retries(self):
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="")
        msgs = [SystemMessage(content="system"), HumanMessage(content="query")]

        result = create_retry_response("Agent", msgs, ["tool1"], mock_llm)
        assert "unable to process" in result["messages"][0].content.lower()

    def test_with_task_extractor(self):
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="")
        msgs = [SystemMessage(content="system"), HumanMessage(content="query")]

        def extractor(messages):
            return "custom task hint"

        result = create_retry_response("Agent", msgs, ["tool1"], mock_llm, task_extractor=extractor)
        assert "custom task hint" in result["messages"][0].content


# =========================================================================
# extract_task_from_messages
# =========================================================================


class TestExtractTaskFromMessages:
    def test_detects_uniprot_id(self):
        msgs = [HumanMessage(content="Fetch protein P04637")]
        hint = extract_task_from_messages(msgs)
        assert "P04637" in hint

    def test_detects_fasta_request(self):
        msgs = [HumanMessage(content="Get the FASTA sequence")]
        hint = extract_task_from_messages(msgs)
        assert "fasta" in hint.lower() or "sequence" in hint.lower()

    def test_detects_search_request(self):
        msgs = [HumanMessage(content="Search for kinase inhibitors")]
        hint = extract_task_from_messages(msgs)
        assert "search" in hint.lower()

    def test_default_hint(self):
        msgs = [HumanMessage(content="Do something")]
        hint = extract_task_from_messages(msgs)
        assert len(hint) > 0


# =========================================================================
# get_message_content
# =========================================================================


class TestGetMessageContent:
    def test_string_content(self):
        msg = AIMessage(content="hello")
        assert get_message_content(msg) == "hello"

    def test_list_content_blocks(self):
        msg = AIMessage(content=[{"type": "text", "text": "hello"}])
        assert get_message_content(msg) == "hello"

    def test_dict_message(self):
        msg = {"content": "hello"}
        assert get_message_content(msg) == "hello"

    def test_no_content(self):
        assert get_message_content(42) == ""


# =========================================================================
# check_finish_if_code_agent_substantive_repeat
# =========================================================================


class TestCheckFinishIfCodeAgentSubstantiveRepeat:
    def test_no_finish_on_single_substantive_dl_turn(self):
        code = "```python\nimport torch\n" + ("x = 1\n" * 120) + "```"
        msgs = [
            HumanMessage(content="Design a model"),
            AIMessage(content=code, name="DL"),
        ]
        ok, _ = check_finish_if_code_agent_substantive_repeat(msgs)
        assert ok is False

    def test_finish_on_second_substantive_dl_turn(self):
        code = "```python\nimport torch\nimport torch.nn as nn\nclass M(nn.Module):\n    pass\n```"
        block = code + "\n" + ("explanation " * 80)
        msgs = [
            HumanMessage(content="Design a model"),
            AIMessage(content="stub", name="DL"),
            HumanMessage(content="[SUPERVISOR TASK] refine"),
            AIMessage(content=block, name="DL"),
        ]
        ok, reason = check_finish_if_code_agent_substantive_repeat(msgs)
        assert ok is True
        assert "DL" in reason

    def test_no_finish_if_latest_not_substantive(self):
        msgs = [
            HumanMessage(content="x"),
            AIMessage(content="```python\n" + "a\n" * 200 + "```", name="DL"),
            HumanMessage(content="[SUPERVISOR TASK] again"),
            AIMessage(content="short ack", name="DL"),
        ]
        ok, _ = check_finish_if_code_agent_substantive_repeat(msgs)
        assert ok is False
