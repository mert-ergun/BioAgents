"""Bound Jupyter execution without mistaking socket silence for completion."""

from __future__ import annotations

import base64
import io
import json
import math
import os

# Required by the upstream wire format; opcode checks and find_class reject objects.
import pickle  # nosec B403
import pickletools  # nosec B403
import time
from queue import Empty, Full, Queue
from threading import Event, Thread

from smolagents.local_python_executor import CodeOutput
from smolagents.remote_executors import (
    AgentError,
    RemotePythonExecutor,
    _websocket_send_execute_request,
)
from websocket import WebSocketTimeoutException


class RemoteExecutionError(RuntimeError):
    """The execution transport or kernel failed, leaving its state uncertain."""


class _PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, _module, _name):
        raise pickle.UnpicklingError("Final answers cannot instantiate Python classes")


def _decode_final_answer(payload):
    """Keep the upstream wire format without executing pickle instructions on the host."""
    if not isinstance(payload, str) or len(payload) > 8 * 1024 * 1024:
        raise ValueError("Invalid or oversized final-answer payload")
    data = base64.b64decode(payload, validate=True)
    forbidden = {
        "GLOBAL",
        "STACK_GLOBAL",
        "REDUCE",
        "INST",
        "OBJ",
        "NEWOBJ",
        "NEWOBJ_EX",
        "EXT1",
        "EXT2",
        "EXT4",
    }
    if any(op.name in forbidden for op, _, _ in pickletools.genops(data)):
        raise pickle.UnpicklingError("Only plain Python values are allowed in final answers")
    return _PrimitiveUnpickler(io.BytesIO(data)).load()


def execution_timeout_seconds() -> float:
    """Read a finite, positive per-execution budget (independent of agent steps)."""
    value = float(os.getenv("BIOAGENTS_CODE_EXECUTION_TIMEOUT_SEC", "300"))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("BIOAGENTS_CODE_EXECUTION_TIMEOUT_SEC must be finite and > 0")
    return value


def run_code_over_websocket(code, ws, logger, *, timeout=None) -> CodeOutput:
    """Complete on a matching shell reply without waiting for IOPub idle.

    A dedicated reader keeps the monotonic deadline outside ``ws.recv()``:
    websocket-client can process control frames internally without returning.
    The caller must provide a fresh connection for each execution. The connection
    is aborted on exit to wake the reader; socket silence never means success.
    """
    budget = execution_timeout_seconds() if timeout is None else float(timeout)
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("Execution timeout must be finite and > 0")
    deadline = time.monotonic() + budget
    ws.settimeout(min(1.0, budget))
    messages: Queue[str | bytes | Exception] = Queue(maxsize=64)
    stopped = Event()
    reader = None
    outputs = []

    def publish(value):
        while not stopped.is_set():
            try:
                messages.put(value, timeout=0.1)
                return
            except Full:
                continue

    def receive():
        while not stopped.is_set():
            try:
                raw = ws.recv()
                if not raw:
                    raise RemoteExecutionError("Jupyter WebSocket disconnected")
                publish(raw)
            except (WebSocketTimeoutException, TimeoutError):
                continue
            except Exception as exc:
                publish(exc)
                return

    try:
        msg_id = _websocket_send_execute_request(code, ws)
        reader = Thread(target=receive, name="bioagents-jupyter-reader", daemon=True)
        reader.start()
        result = None
        is_final_answer = False
        saw_reply = False
        while not saw_reply:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Jupyter execution exceeded {budget:g}s; completion was not confirmed"
                )
            try:
                raw = messages.get(timeout=remaining)
            except Empty as exc:
                raise TimeoutError(
                    f"Jupyter execution exceeded {budget:g}s; completion was not confirmed"
                ) from exc
            if isinstance(raw, Exception):
                raise RemoteExecutionError(f"Jupyter receive failed: {raw}") from raw
            try:
                msg = json.loads(raw)
                kind = msg.get("msg_type") or msg.get("header", {}).get("msg_type", "")
                content = msg.get("content", {})
                parent = msg.get("parent_header", {}).get("msg_id")
            except (ValueError, TypeError, AttributeError) as exc:
                raise RemoteExecutionError("Invalid Jupyter WebSocket message") from exc
            if not isinstance(content, dict):
                raise RemoteExecutionError("Invalid Jupyter message content")

            # Kernel lifecycle events affect the entire session, including events
            # that have no parent execution request. Parentless idle does not.
            if kind == "status" and content.get("execution_state") in ("dead", "restarting"):
                raise RemoteExecutionError(f"Kernel entered {content['execution_state']} state")
            if parent != msg_id:
                continue
            if kind == "stream":
                text = content.get("text", "")
                if not isinstance(text, str):
                    raise RemoteExecutionError("Invalid Jupyter stream text")
                outputs.append(text)
            elif kind == "execute_result":
                data = content.get("data", {})
                if not isinstance(data, dict):
                    raise RemoteExecutionError("Invalid Jupyter result data")
                if not is_final_answer:
                    result = data.get("text/plain")
            elif kind in ("execute_reply", "error"):
                if kind == "execute_reply":
                    saw_reply = True
                status = content.get("status")
                if kind == "error" or status in ("error", "aborted"):
                    if content.get("ename") == RemotePythonExecutor.FINAL_ANSWER_EXCEPTION:
                        try:
                            result = _decode_final_answer(content["evalue"])
                        except Exception as exc:
                            raise AgentError(
                                "Invalid final-answer payload. Return plain Python values "
                                "(strings, numbers, lists, tuples or dictionaries); save custom objects to files.",
                                logger,
                            ) from exc
                        is_final_answer = True
                    else:
                        traceback = content.get("traceback", [])
                        if not isinstance(traceback, list) or not all(
                            isinstance(line, str) for line in traceback
                        ):
                            raise RemoteExecutionError("Invalid Jupyter error traceback")
                        detail = "\n".join(traceback)
                        raise AgentError(
                            detail or content.get("evalue") or "Kernel execution failed", logger
                        )
                elif kind == "execute_reply" and status != "ok":
                    raise RemoteExecutionError(f"Unexpected execute_reply status: {status!r}")
        return CodeOutput(output=result, logs="".join(outputs), is_final_answer=is_final_answer)
    except Exception as exc:
        logger.log_error(f"Code execution failed: {exc}")
        if outputs:
            logger.log_error("Captured execution output:\n" + "".join(outputs)[-4000:])
        raise
    finally:
        stopped.set()
        # abort() interrupts a blocked recv without a WebSocket close handshake.
        try:
            ws.abort()
        except Exception:
            logger.log_error("Could not abort the Jupyter WebSocket reader")
        if reader is not None:
            reader.join(timeout=1.0)
            if reader.is_alive():
                logger.log_error("Jupyter reader did not stop after socket abort")
