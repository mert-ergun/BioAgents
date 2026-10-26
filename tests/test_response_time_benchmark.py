"""Offline checks for latency measurements; never call real model providers."""

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest

import benchmark_response_time as benchmark


class Stream(httpx.AsyncByteStream):
    def __init__(self, events):
        self.events = events

    async def __aiter__(self):
        for event in self.events:
            await asyncio.sleep(0.002)
            yield event.encode()


def mock_http(monkeypatch, events, status=200):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(status, stream=Stream(events))

    original = httpx.AsyncClient
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        benchmark.httpx, "AsyncClient", lambda **kw: original(transport=transport, **kw)
    )
    return requests


def sse(event):
    return f"data: {json.dumps(event)}\n\n"


def test_direct_ignores_role_reasoning_and_empty_chunks(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-secret")
    requests = mock_http(
        monkeypatch,
        [
            ": keepalive\r\n\r\n",
            sse({"choices": [{"delta": {"role": "assistant", "content": ""}}]}),
            sse({"choices": [{"delta": {"reasoning": "internal reasoning"}}]}),
            sse({"choices": [{"delta": {"content": "Merhaba"}}]}),
            sse({"choices": [{"delta": {"content": " dünya"}}]}),
            sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
            "data: [DONE]\n\n",
        ],
    )
    result = asyncio.run(benchmark.measure_direct("openrouter", "test-model", "hello", 5))
    assert result.status == "ok"
    assert 0 < result.ttft_s < result.last_text_s < result.total_s
    assert result.text_chunks == 2
    assert result.response_chars == len("Merhaba dünya")
    assert result.first_message_s is None
    assert len(requests) == 1
    assert json.loads(requests[0].content) == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    }


def test_truncated_stream_is_not_a_completed_answer(monkeypatch):
    mock_http(monkeypatch, [sse({"choices": [{"delta": {"content": "partial"}}]})])
    result = asyncio.run(benchmark.measure_direct("ollama", "test", "hello", 5))
    assert result.status == "error"
    assert result.ttft_s is not None
    assert result.total_s is None
    assert "completion marker" in result.error


def test_provider_error_after_first_text_is_reported(monkeypatch):
    mock_http(
        monkeypatch,
        [
            sse({"choices": [{"delta": {"content": "partial"}}]}),
            sse({"error": {"message": "upstream failed"}}),
        ],
    )
    result = asyncio.run(benchmark.measure_direct("ollama", "test", "hello", 5))
    assert result.status == "error"
    assert result.total_s is None
    assert "upstream failed" in result.error


def test_direct_timeout_preserves_partial_timing(monkeypatch):
    mock_http(monkeypatch, [sse({"choices": [{"delta": {"content": "part"}}]})] * 100)
    result = asyncio.run(benchmark.measure_direct("ollama", "test", "hello", 0.03))
    assert result.status == "timeout"
    assert result.ttft_s is not None
    assert result.total_s is None


def test_gemini_thoughts_are_excluded(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test-secret")
    requests = mock_http(
        monkeypatch,
        [
            sse({"candidates": [{"content": {"parts": [{"text": "hidden", "thought": True}]}}]}),
            sse(
                {
                    "candidates": [
                        {"content": {"parts": [{"text": "answer"}]}, "finishReason": "STOP"}
                    ]
                }
            ),
        ],
    )
    result = asyncio.run(benchmark.measure_direct("gemini", "test", "hello", 5))
    assert result.status == "ok"
    assert result.text_chunks == 1
    assert result.response_chars == 6
    assert 0 < result.ttft_s <= result.total_s
    assert requests[0].headers["x-goog-api-key"] == "offline-test-secret"
    assert "offline-test-secret" not in str(requests[0].url)


def test_sse_multiline_and_comment_parsing():
    async def parse():
        response = httpx.Response(
            200,
            stream=Stream(
                [
                    ': comment\r\ndata: {\r\ndata: "answer": true}\r\n\r\n',
                    "data: [DONE]\n\n",
                ]
            ),
        )
        return [event async for event in benchmark.sse_data(response)]

    assert asyncio.run(parse()) == ['{\n"answer": true}', "[DONE]"]


def mock_websocket(monkeypatch, events):
    sent = []

    class Socket:
        async def send(self, payload):
            sent.append(json.loads(payload))

        async def __aiter__(self):
            for event in events:
                await asyncio.sleep(0.002)
                yield json.dumps(event)

    @asynccontextmanager
    async def connect(*args, **kwargs):
        yield Socket()

    import websockets.asyncio.client

    monkeypatch.setattr(websockets.asyncio.client, "connect", connect)
    return sent


def test_websocket_status_messages_are_not_first_answer(monkeypatch):
    sent = mock_websocket(
        monkeypatch,
        [
            {"type": "log", "message": "Initializing agents..."},
            {"type": "agent_update", "agent": "supervisor"},
            {"type": "audit", "entries": []},
            {"type": "tool_result", "content": "tool text"},
            {"type": "message", "agent": "summary", "content": "answer"},
            {"type": "complete"},
        ],
    )
    result = asyncio.run(
        benchmark.measure_websocket("http://localhost:8000", "ollama", "test", "hello", 5)
    )
    assert result.status == "ok"
    assert result.ttft_s is None
    assert 0 < result.first_message_s < result.total_s
    assert result.response_chars == 6
    assert sent == [{"type": "query", "content": "hello", "provider": "ollama", "model": "test"}]


@pytest.mark.parametrize(
    "terminal",
    [
        {"type": "error", "message": "failed"},
        {"type": "engagement_request", "question": "enter key"},
        {"type": "tool_approval_request"},
        None,
    ],
)
def test_incomplete_websocket_never_reports_success(monkeypatch, terminal):
    mock_websocket(monkeypatch, [terminal] if terminal else [])
    result = asyncio.run(
        benchmark.measure_websocket("http://localhost:8000", "ollama", "test", "hello", 5)
    )
    assert result.status == "error"
    assert result.total_s is None
    assert result.ttft_s is None


def test_graph_distinguishes_routing_agent_and_final_summary(monkeypatch):
    clock = iter([11.0, 12.0, 13.0])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock))
    result = benchmark.Measurement("bioagents_graph")
    timing = benchmark.GraphTiming(result, start=10.0)
    timing.token("", "supervisor")
    timing.token("routing", "supervisor")
    timing.token("agent answer", "research")
    timing.token("final answer", "summary")
    assert result.first_llm_token_s == 1
    assert result.ttft_s == 2
    assert result.first_summary_token_s == 3
    assert result.text_chunks == 2


def test_instrumented_model_streams_through_real_timeout_and_rate_wrappers():
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from bioagents.llms.rate_limiter import RateLimitedLLM
    from bioagents.llms.timeout_llm import TimeoutBoundLLM

    tokens = []

    class Handler(BaseCallbackHandler):
        def on_llm_new_token(self, token, **kwargs):
            tokens.append(token)

    # FakeListChatModel implements streaming but has no streaming field. Add the
    # field used by the production ChatOpenAI and ChatGoogleGenerativeAI models.
    class FakeStreamingModel(FakeListChatModel):
        streaming: bool = False

        def _should_stream(self, *, async_api, run_manager=None, **kwargs):
            return self.streaming or super()._should_stream(
                async_api=async_api, run_manager=run_manager, **kwargs
            )

    base = FakeStreamingModel(responses=["abc"])
    wrapped = RateLimitedLLM(TimeoutBoundLLM(base, 5))
    benchmark.instrument_llm(wrapped, Handler())
    response = wrapped.invoke("hello")
    assert response.content == "abc"
    assert "".join(tokens) == "abc"
    assert len([token for token in tokens if token]) == 3


@pytest.mark.parametrize("fail", [False, True])
def test_graph_factory_instrumentation_and_completed_messages(monkeypatch, fail):
    from langchain_core.messages import AIMessage

    from bioagents.llms import llm_provider
    from bioagents.sandbox import coder_executor

    cleanup_calls = []
    fake_executor = SimpleNamespace(cleanup=lambda: cleanup_calls.append(True))
    monkeypatch.setattr(coder_executor, "create_executor", lambda *a, **kw: fake_executor)
    executor_factory = coder_executor.create_executor

    original = llm_provider.get_llm
    monkeypatch.setattr(
        llm_provider, "get_llm", lambda **kw: SimpleNamespace(streaming=False, callbacks=[])
    )
    factory = llm_provider.get_llm

    def create_graph():
        # During construction the script has replaced the central factory.
        assert coder_executor.create_executor("coder", []) is fake_executor
        if fail:
            raise RuntimeError("Graph construction failed after executor startup")
        model = llm_provider.get_llm(prompt_name="summary")
        assert model.streaming
        assert len(model.callbacks) == 1
        return model

    def steps(model, state):
        assert state["messages"][0].content == "hello"
        model.callbacks[0].on_llm_new_token("a")
        yield {"supervisor": {"messages": [AIMessage(content="routing")]}}
        yield {"summary": {"messages": [AIMessage(content="answer")]}}

    monkeypatch.setitem(
        sys.modules,
        "bioagents.graph",
        SimpleNamespace(ALL_MEMBERS=["summary"], create_graph=create_graph),
    )
    monkeypatch.setitem(
        sys.modules, "bioagents.graph_streaming", SimpleNamespace(iter_graph_stream=steps)
    )
    result = benchmark.measure_graph("ollama", "test", "hello")
    assert cleanup_calls == [True]
    assert coder_executor.create_executor is executor_factory
    assert llm_provider.get_llm is factory
    if fail:
        assert result.status == "error"
        assert result.total_s is None
        return
    assert result.status == "ok"
    assert result.ttft_s is not None
    assert result.first_summary_token_s == result.ttft_s
    assert result.response_chars == 6
    assert result.initialization_s <= result.ttft_s <= result.first_message_s <= result.total_s
    assert llm_provider.get_llm is factory
    assert original is not factory


def test_summary_excludes_errors_and_missing_ttft():
    result = benchmark.summarize(
        [
            benchmark.Measurement("direct_api", status="ok", ttft_s=2, total_s=4),
            benchmark.Measurement("direct_api", status="ok", ttft_s=4, total_s=8),
            benchmark.Measurement("direct_api", status="error", ttft_s=100, total_s=None),
            benchmark.Measurement(
                "bioagents_websocket", status="ok", first_message_s=3, total_s=10
            ),
        ]
    )
    assert result["direct_api"]["ttft_s"] == {"median": 3, "p95": 4, "n": 2}
    assert result["direct_api"]["failed_runs"] == 1
    assert "ttft_s" not in result["bioagents_websocket"]


def test_errors_redact_keys(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-do-not-print")
    assert benchmark.redact_error("bad token secret-do-not-print") == "bad token [REDACTED]"


def test_cli_writes_json_and_csv_without_prompt_or_key(monkeypatch, tmp_path, capsys):
    output = tmp_path / "results.json"
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-cli-secret")
    mock_http(
        monkeypatch,
        [
            sse({"choices": [{"delta": {"content": "answer"}}]}),
            "data: [DONE]\n\n",
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_response_time.py",
            "--target",
            "direct",
            "--provider",
            "openrouter",
            "--model",
            "test",
            "--prompt",
            "private prompt",
            "--runs",
            "2",
            "--env-file",
            str(tmp_path / "absent.env"),
            "--output",
            str(output),
        ],
    )
    assert benchmark.main() == 0
    report_text = output.read_text()
    report = json.loads(report_text)
    assert report["summary"]["direct_api"]["successful_runs"] == 2
    assert [row["run"] for row in report["results"]] == [1, 2]
    assert len(output.with_suffix(".csv").read_text().splitlines()) == 3
    assert "private prompt" not in report_text
    assert "offline-cli-secret" not in report_text
    assert "offline-cli-secret" not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("provider_model", "shared_model", "expected"),
    [
        ("provider-model", "shared-model", "provider-model"),
        ("  ", " shared-model ", "shared-model"),
    ],
)
def test_cli_uses_environment_model(monkeypatch, tmp_path, provider_model, shared_model, expected):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_MODEL", provider_model)
    monkeypatch.setenv("LLM_MODEL", shared_model)
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-key")
    requests = mock_http(
        monkeypatch, [sse({"choices": [{"delta": {"content": "ok"}}]}), "data: [DONE]\n\n"]
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_response_time.py",
            "--target",
            "direct",
            "--prompt",
            "hello",
            "--env-file",
            str(tmp_path / "absent.env"),
            "--output",
            str(tmp_path / "out.json"),
        ],
    )
    assert benchmark.main() == 0
    assert json.loads(requests[0].content)["model"] == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://localhost:8000/", "ws://localhost:8000/ws"),
        ("https://example.com/bioagents", "wss://example.com/bioagents/ws"),
        ("ws://localhost:8000/ws", "ws://localhost:8000/ws"),
    ],
)
def test_websocket_urls(url, expected):
    assert benchmark.websocket_url(url) == expected
