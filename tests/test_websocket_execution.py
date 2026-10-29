"""Regression coverage for Jupyter completion, deadlines, and failed sessions."""

import base64
import json
import pickle
import time
from queue import Empty, Queue
from threading import Event
from unittest.mock import Mock

import pytest
from smolagents.remote_executors import AgentError
from websocket import ABNF, WebSocket, WebSocketTimeoutException

from bioagents.sandbox import websocket_runner as runner
from bioagents.sandbox.resilient_executor import ResilientDockerExecutor, exec_timeout_seconds


def message(kind, content, parent="request"):
    return json.dumps({"msg_type": kind, "content": content, "parent_header": {"msg_id": parent}})


def completed(value="42"):
    return [
        message("execute_result", {"data": {"text/plain": value}}),
        message("execute_reply", {"status": "ok"}),
        message("status", {"execution_state": "idle"}),
    ]


class FakeSocket:
    def __init__(self, events=()):
        self.events = Queue()
        for event in events:
            self.events.put(event)
        self.aborted = Event()
        self.timeout = 0.01

    def settimeout(self, value):
        self.timeout = min(value, 0.01)

    def recv(self):
        try:
            event = self.events.get(timeout=self.timeout)
        except Empty as exc:
            raise WebSocketTimeoutException() from exc
        if isinstance(event, Exception):
            raise event
        return event

    def abort(self):
        self.aborted.set()
        self.events.put("")

    close = abort


@pytest.fixture(autouse=True)
def request_sender(monkeypatch):
    sender = Mock(return_value="request")
    monkeypatch.setattr(runner, "_websocket_send_execute_request", sender)
    return sender


def test_result_before_shell_reply_is_preserved():
    socket = FakeSocket(completed())
    out = runner.run_code_over_websocket("40 + 2", socket, Mock(), timeout=0.3)
    assert out.output == "42"
    assert socket.aborted.is_set()


def test_shell_reply_completes_without_idle_or_another_receive():
    socket = FakeSocket([message("execute_reply", {"status": "ok"})])
    out = runner.run_code_over_websocket("pass", socket, Mock(), timeout=0.3)
    assert out.output is None
    assert socket.aborted.is_set()


def test_jupyter_errors_propagate_without_the_local_dependency_debugger(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOAGENTS_SANDBOX_DIR", str(tmp_path))
    monkeypatch.setenv("BIOAGENTS_RUN_ID", "a" * 32)
    error = {
        "status": "error",
        "ename": "ModuleNotFoundError",
        "evalue": "No module named 'missing_package'",
        "traceback": ["ModuleNotFoundError: No module named 'missing_package'"],
    }
    socket = FakeSocket([message("execute_reply", error)])
    with pytest.raises(AgentError, match="missing_package"):
        runner.run_code_over_websocket(
            "!pip install missing_package\nimport missing_package", socket, Mock(), timeout=0.3
        )
    assert not (tmp_path / "dependency_events.jsonl").exists()


def test_unrelated_jupyter_error_is_ignored_without_debug_output(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOAGENTS_SANDBOX_DIR", str(tmp_path))
    monkeypatch.setenv("BIOAGENTS_RUN_ID", "a" * 32)
    socket = FakeSocket(
        [
            message(
                "error",
                {"ename": "ModuleNotFoundError", "evalue": "No module named 'other'"},
                parent="other-request",
            ),
            message("execute_reply", {"status": "ok"}),
        ]
    )
    runner.run_code_over_websocket("pass", socket, Mock(), timeout=0.3)
    assert not (tmp_path / "dependency_events.jsonl").exists()


def test_shell_reply_does_not_wait_for_later_messages():
    socket = FakeSocket([message("execute_reply", {"status": "ok"}), OSError("late disconnect")])
    out = runner.run_code_over_websocket("pass", socket, Mock(), timeout=0.3)
    assert out.output is None


def test_output_and_receive_timeout_do_not_imply_completion():
    socket = FakeSocket(
        [
            message("stream", {"text": "started\n"}),
            WebSocketTimeoutException(),
            *completed(),
        ]
    )
    out = runner.run_code_over_websocket("work()", socket, Mock(), timeout=0.3)
    assert out.output == "42"
    assert out.logs == "started\n"


@pytest.mark.parametrize(
    "events",
    [
        [],
        [message("stream", {"text": "started"})],
        [message("status", {"execution_state": "idle"})],
        [message("execute_reply", {"status": "ok"}, parent="other")],
        [message("execute_reply", {"status": "ok"}, parent=None)],
    ],
)
def test_incomplete_execution_raises_timeout(events):
    socket = FakeSocket(events)
    with pytest.raises(TimeoutError, match="completion was not confirmed"):
        runner.run_code_over_websocket("work()", socket, Mock(), timeout=0.03)
    assert socket.aborted.is_set()


@pytest.mark.parametrize("first_channel", ["execute_reply", "error"])
def test_final_answer_survives_both_channel_orders(first_channel):
    payload = base64.b64encode(pickle.dumps({"answer": 42})).decode()
    error = {"ename": "FinalAnswerException", "evalue": payload, "traceback": []}
    reply = message("execute_reply", {**error, "status": "error"})
    iopub = message("error", error)
    events = [reply, iopub] if first_channel == "execute_reply" else [iopub, reply]
    out = runner.run_code_over_websocket(
        "final_answer(answer=42)", FakeSocket(events), Mock(), timeout=0.3
    )
    assert out.is_final_answer
    assert out.output == {"answer": 42}


def test_parentless_or_unrelated_idle_cannot_complete_request():
    events = [
        message("status", {"execution_state": "idle"}, parent=None),
        message("status", {"execution_state": "idle"}, parent="other"),
        *completed(),
    ]
    assert (
        runner.run_code_over_websocket("40 + 2", FakeSocket(events), Mock(), timeout=0.3).output
        == "42"
    )


@pytest.mark.parametrize("state", ["dead", "restarting"])
def test_parentless_kernel_failure_raises(state):
    socket = FakeSocket([message("status", {"execution_state": state}, parent=None)])
    with pytest.raises(runner.RemoteExecutionError, match=state):
        runner.run_code_over_websocket("work()", socket, Mock(), timeout=0.3)


@pytest.mark.parametrize(
    "event",
    [
        "",
        OSError("disconnected"),
        "not json",
        "[]",
        message("stream", []),
        message("stream", {"text": 42}),
        message("execute_result", {"data": []}),
        message("error", {"traceback": [42]}),
    ],
)
def test_transport_failure_is_not_a_normal_code_output(event):
    with pytest.raises(runner.RemoteExecutionError):
        runner.run_code_over_websocket("work()", FakeSocket([event]), Mock(), timeout=0.3)


@pytest.mark.parametrize("kind", ["error", "execute_reply"])
def test_python_error_raises_agent_error(kind):
    error = {"status": "error", "ename": "ValueError", "evalue": "bad", "traceback": ["bad"]}
    with pytest.raises(AgentError, match="bad"):
        runner.run_code_over_websocket(
            "raise ValueError()", FakeSocket([message(kind, error)]), Mock(), timeout=0.3
        )


def test_real_websocket_ping_loop_cannot_bypass_deadline():
    socket = WebSocket()
    aborted = Event()
    frames = []

    def recv_frame():
        if aborted.wait(0.001):
            raise OSError("aborted")
        frames.append(1)
        return ABNF.create_frame(b"ping", ABNF.OPCODE_PING)

    socket.recv_frame = recv_frame
    socket.pong = lambda payload: None
    socket.abort = aborted.set
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        runner.run_code_over_websocket("work()", socket, Mock(), timeout=0.03)
    assert time.monotonic() - start < 1
    assert frames
    assert aborted.is_set()


def test_code_is_not_rewritten_with_a_restart_bootstrap(request_sender):
    code = 'from __future__ import annotations\nfinal_answer (answer="done")'
    runner.run_code_over_websocket(code, FakeSocket(completed()), Mock(), timeout=0.3)
    assert request_sender.call_args.args[0] == code


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "invalid"])
def test_invalid_configured_deadline_is_rejected(monkeypatch, value):
    monkeypatch.setenv("BIOAGENTS_CODE_EXECUTION_TIMEOUT_SEC", value)
    with pytest.raises(ValueError):
        runner.execution_timeout_seconds()


@pytest.fixture
def executor():
    value = object.__new__(ResilientDockerExecutor)
    value._init_resilience()
    value.ws_url = "ws://localhost/api/kernels/test/channels"
    value.base_url = "http://localhost"
    value.kernel_id = "test"
    value.logger = Mock()
    return value


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "invalid", ""])
def test_executor_rejects_invalid_timeout(monkeypatch, value):
    monkeypatch.setenv("BIOAGENTS_CODE_EXECUTION_TIMEOUT_SEC", value)
    with pytest.raises(ValueError):
        exec_timeout_seconds()


def test_timeout_alias_precedence(monkeypatch):
    monkeypatch.setenv("BIOAGENTS_CODER_EXEC_TIMEOUT", "900")
    monkeypatch.delenv("BIOAGENTS_CODE_EXECUTION_TIMEOUT_SEC", raising=False)
    assert exec_timeout_seconds() == 900
    monkeypatch.setenv("BIOAGENTS_CODE_EXECUTION_TIMEOUT_SEC", "300")
    assert exec_timeout_seconds() == 300


@pytest.mark.parametrize("recovery_fails", [False, True])
def test_timeout_restarts_restores_setup_and_allows_next_step(
    executor, monkeypatch, recovery_fails
):
    import requests

    executor._tools_code = "final_answer = restored_tool"
    executor._variables_code = "input_path = '/app/input.csv'"
    monkeypatch.setattr(executor, "_ensure_session", lambda: None)
    expected = runner.CodeOutput(output="done", logs="", is_final_answer=False)
    execute = Mock(side_effect=[TimeoutError("original"), expected, expected])
    monkeypatch.setattr(executor, "_raw_execute", execute)
    restart = Mock()
    if recovery_fails:
        restart.side_effect = requests.ConnectionError("gateway down")
    monkeypatch.setattr(requests, "post", restart)
    with pytest.raises(AgentError, match="original"):
        executor.resilient_run("work()")
    if recovery_fails:
        assert executor._execution_failure == "original"
        with pytest.raises(AgentError, match="Kernel recovery failed"):
            executor.resilient_run("next()")
        assert execute.call_count == 1
        restart.side_effect = None
    else:
        assert executor._execution_failure is None
    result = executor.resilient_run("next()")
    assert result.output == "done"
    assert executor._execution_failure is None
    assert execute.call_args_list[0].args[0] == "work()"
    assert execute.call_args_list[1].args[0] == executor.setup_code()
    assert execute.call_args_list[-1].args[0] == "next()"
    assert all(c.args[0].endswith("/restart") for c in restart.call_args_list)


def test_setup_failure_keeps_failure_until_next_recovery(executor, monkeypatch):
    import requests

    monkeypatch.setattr(executor, "_ensure_session", lambda: None)
    expected = runner.CodeOutput(output="done", logs="", is_final_answer=False)
    execute = Mock(
        side_effect=[TimeoutError("original"), TimeoutError("setup"), expected, expected]
    )
    monkeypatch.setattr(executor, "_raw_execute", execute)
    restart = Mock()
    monkeypatch.setattr(requests, "post", restart)
    with pytest.raises(AgentError, match="original"):
        executor.resilient_run("work()")
    assert executor._execution_failure == "original"
    assert executor.resilient_run("next()").output == "done"
    assert executor._execution_failure is None
    assert restart.call_count == 2


def test_ordinary_python_error_does_not_restart(executor, monkeypatch):
    import requests

    monkeypatch.setattr(executor, "_ensure_session", lambda: None)
    monkeypatch.setattr(executor, "_raw_execute", Mock(side_effect=AgentError("bad code", Mock())))
    restart = Mock()
    monkeypatch.setattr(requests, "post", restart)
    with pytest.raises(AgentError, match="bad code"):
        executor.resilient_run("bad_code()")
    restart.assert_not_called()


@pytest.mark.parametrize(
    "code",
    [
        'from __future__ import annotations\nfinal_answer(answer="done")',
        'final_answer (answer="done")',
        'text = "final_answer("',
    ],
)
def test_executor_preserves_code_and_completes_without_idle(
    executor, monkeypatch, code, request_sender
):
    monkeypatch.setattr(
        executor, "_connect", lambda url: FakeSocket([message("execute_reply", {"status": "ok"})])
    )
    executor._raw_execute(code, timeout=0.3)
    assert request_sender.call_args.args[0] == code
    compile(code, "<submitted>", "exec")


def test_lost_kernel_creates_one_fresh_session_and_restores_setup(executor, monkeypatch):
    from bioagents.sandbox.resilient_executor import KernelGoneError

    expected = runner.CodeOutput(output="done", logs="", is_final_answer=False)
    execute = Mock(side_effect=[KernelGoneError("404"), expected, expected])
    monkeypatch.setattr(executor, "_raw_execute", execute)
    create = Mock(return_value="new-kernel")
    monkeypatch.setattr(executor, "_create_kernel", create)
    with pytest.raises(AgentError, match="fresh kernel"):
        executor.resilient_run("work()")
    assert executor.kernel_id == "new-kernel"
    assert executor.ws_url.endswith("/api/kernels/new-kernel/channels")
    assert execute.call_args_list[1].args[0] == executor.setup_code()
    assert executor.resilient_run("next()").output == "done"
    create.assert_called_once()


def test_repeated_kernel_loss_stops_after_bounded_recoveries(executor, monkeypatch):
    from bioagents.sandbox.resilient_executor import KernelGoneError

    expected = runner.CodeOutput(output=None, logs="", is_final_answer=False)

    def execute(code, **kwargs):
        if code == "work()":
            raise KernelGoneError("404")
        return expected

    monkeypatch.setattr(executor, "_raw_execute", execute)
    create = Mock(return_value="fresh")
    monkeypatch.setattr(executor, "_create_kernel", create)
    for _ in range(2):
        with pytest.raises(AgentError):
            executor.resilient_run("work()")
    result = executor.resilient_run("work()")
    assert result.is_final_answer
    assert "sandbox_unrecoverable" in result.output
    assert create.call_count == 2


def test_invalid_transport_message_restarts_live_kernel(executor, monkeypatch):
    import requests

    sockets = iter(
        [FakeSocket(["not json"]), FakeSocket([message("execute_reply", {"status": "ok"})])]
    )
    monkeypatch.setattr(executor, "_connect", lambda _: next(sockets))
    monkeypatch.setattr(executor, "_kernel_alive", lambda: True)
    monkeypatch.setattr(executor, "_apply_setup", lambda: None)
    restart = Mock()
    monkeypatch.setattr(requests, "post", restart)
    with pytest.raises(AgentError, match="Invalid Jupyter"):
        executor.resilient_run("work()")
    assert executor._execution_failure is None
    assert not executor.resilient_run("next()").is_final_answer
    restart.assert_called_once()


@pytest.mark.parametrize("kind", ["error", "execute_reply"])
def test_final_answer_cannot_execute_pickle_on_host(tmp_path, kind):
    marker = tmp_path / "host-code-executed"

    class UnsafeAnswer:
        def __reduce__(self):
            return eval, (f"__import__('pathlib').Path({str(marker)!r}).write_text('unsafe')",)

    payload = base64.b64encode(pickle.dumps(UnsafeAnswer())).decode()
    error = {"status": "error", "ename": "FinalAnswerException", "evalue": payload}
    with pytest.raises(AgentError, match="plain Python values"):
        runner.run_code_over_websocket(
            "final_answer(...) ", FakeSocket([message(kind, error)]), Mock(), timeout=0.3
        )
    assert not marker.exists()


def test_fresh_websocket_connections_use_distinct_jupyter_sessions(executor, monkeypatch):
    from urllib.parse import parse_qs, urlsplit

    import websocket

    connect = Mock()
    monkeypatch.setattr(websocket, "create_connection", connect)
    for _ in range(2):
        executor._connect(
            "ws://localhost/api/kernels/test/channels?token=test-token&session_id=old"
        )
    queries = [parse_qs(urlsplit(call.args[0]).query) for call in connect.call_args_list]
    assert all(query["token"] == ["test-token"] for query in queries)
    assert queries[0]["session_id"] != queries[1]["session_id"]
    assert all(query["session_id"] != ["old"] for query in queries)


def test_setup_has_its_own_budget_when_user_cells_have_short_timeout(executor, monkeypatch):
    monkeypatch.setenv("BIOAGENTS_CODE_EXECUTION_TIMEOUT_SEC", "1")
    execute = Mock()
    monkeypatch.setattr(executor, "_raw_execute", execute)
    executor._apply_setup()
    assert execute.call_args.kwargs["timeout"] == 30
