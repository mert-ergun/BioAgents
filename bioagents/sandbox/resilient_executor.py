"""A DockerExecutor that survives kernel restarts and fails fast when it cannot.

Upstream ``smolagents.DockerExecutor`` defines ``final_answer`` and the agent tools in
the Jupyter kernel exactly once, at the start of ``agent.run()``. Two things then went
wrong in real runs (task_runs/test_runs_v1/task1):

1. A native crash during a step (a ``pip install`` replacing a loaded extension, an
   OOM) made the kernel restart. The fresh kernel had no ``final_answer``, so the
   agent's finished result died with ``NameError: name 'final_answer' is not defined``
   nine times in a row.
2. The kernel then disappeared from the gateway. Every websocket handshake returned
   ``404 Not Found`` and smolagents treated each one as an ordinary code error, so the
   agent burned 17 more steps against a kernel that no longer existed.

This executor:

* remembers the setup code (pip guard, tool definitions, variables) and re-applies it
  whenever ``final_answer`` is missing from the kernel namespace, so ``final_answer``
  always exists — including after a restart;
* detects a restart while a cell is running (iopub ``restarting``/``dead`` status, or
  the kernel vanishing from ``/api/kernels``) and tells the agent its variables are gone;
* on a 404 / dead kernel starts **one** fresh kernel and restores the setup; if that
  fails, or the sandbox keeps dying, it stops the agent with a clear final message
  instead of looping;
* bounds every execution with a timeout and interrupts the kernel when it is exceeded;
* never pip-installs the project package or core pinned packages on ``send_tools``.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import pickle  # nosec B403 - mirrors upstream smolagents final-answer transport
import time
import uuid
from typing import Any

import requests
from smolagents import DockerExecutor
from smolagents.local_python_executor import CodeOutput
from smolagents.remote_executors import RemotePythonExecutor, _create_kernel_http
from smolagents.tools import Tool, get_tools_definition_code
from smolagents.utils import AgentError

from bioagents.sandbox.pip_guard import filter_install_requests, kernel_guard_code

logger = logging.getLogger(__name__)

#: Marker put at the start of the final answer when the sandbox is unrecoverable.
SANDBOX_UNRECOVERABLE_MARKER = "[CODER_STATUS: sandbox_unrecoverable]"

DEFAULT_EXEC_TIMEOUT_S = 1800.0
#: How often a blocked ``recv`` wakes up to check the deadline and kernel liveness.
_POLL_INTERVAL_S = 15.0
#: Fresh kernels one executor may start per ``agent.run()`` before giving up.
MAX_KERNEL_RECOVERIES = 2

KERNEL_RESTARTED_MESSAGE = (
    "The Jupyter kernel died while executing this code (typically out-of-memory or a "
    "native crash, e.g. a pip install replacing an already-imported extension). A fresh "
    "kernel was started and final_answer plus the tools were restored, but ALL variables "
    "and imports from previous steps are gone: recompute what you need (results already "
    "written to files are still on disk)."
)


class KernelGoneError(RuntimeError):
    """The kernel no longer exists on the gateway (websocket 404 / dead kernel)."""


class KernelRestartedError(RuntimeError):
    """The kernel restarted while a cell was running; its namespace was wiped."""


class GatewayUnreachableError(RuntimeError):
    """The Jupyter gateway itself (the container) is not reachable."""


def exec_timeout_seconds() -> float:
    """Per-execution wall-clock limit, overridable via ``BIOAGENTS_CODER_EXEC_TIMEOUT``."""
    raw = os.getenv("BIOAGENTS_CODER_EXEC_TIMEOUT", "")
    try:
        value = float(raw) if raw else DEFAULT_EXEC_TIMEOUT_S
    except ValueError:
        return DEFAULT_EXEC_TIMEOUT_S
    return value if value > 0 else DEFAULT_EXEC_TIMEOUT_S


def _is_not_found(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    text = str(exc)
    return status == 404 or ("404" in text and "Handshake" in text)


def _is_connection_refused(exc: BaseException) -> bool:
    return isinstance(exc, ConnectionRefusedError | ConnectionResetError) or (
        "Connection refused" in str(exc)
    )


class ResilientKernelMixin:
    """Kernel-session management shared by remote Jupyter executors.

    The attributes below are supplied by the executor this mixes into; they are
    declared so the mixin type-checks on its own.
    """

    base_url: str
    kernel_id: str
    ws_url: str
    logger: Any

    FINAL_ANSWER_EXCEPTION = RemotePythonExecutor.FINAL_ANSWER_EXCEPTION

    def _init_resilience(self) -> None:
        self._tools_code: str = ""
        self._variables_code: str = ""
        self._recoveries = 0
        self.fatal_error: str | None = None
        self.kernel_restarts = 0

    # -- hooks overridable in tests -------------------------------------------------
    def _connect(self, url: str):
        from websocket import create_connection

        return create_connection(url, timeout=30)

    def _create_kernel(self) -> str:
        return str(_create_kernel_http(f"{self.base_url}/api/kernels", self.logger))

    def _kernel_alive(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/kernels/{self.kernel_id}", timeout=5)
        except requests.RequestException:
            return False
        if r.status_code != 200:
            return False
        try:
            return bool(r.json().get("execution_state") != "dead")
        except ValueError:
            return True

    def _interrupt_kernel(self) -> None:
        with contextlib.suppress(requests.RequestException):
            requests.post(f"{self.base_url}/api/kernels/{self.kernel_id}/interrupt", timeout=5)

    def setup_code(self) -> str:
        """Everything a fresh kernel needs before agent code can run."""
        return "\n\n".join(
            part for part in (kernel_guard_code(), self._tools_code, self._variables_code) if part
        )

    def _apply_setup(self) -> None:
        self._raw_execute(self.setup_code())

    def _final_answer_defined(self) -> bool:
        probe = self._raw_execute("print('final_answer' in globals())")
        return "True" in (probe.logs or "")

    def _ensure_session(self) -> None:
        """Make sure the kernel exists and still has final_answer defined."""
        if not self._tools_code:
            return  # send_tools not called yet: nothing to restore
        try:
            if self._final_answer_defined():
                return
            self.kernel_restarts += 1
            self.logger.log("final_answer missing from kernel namespace; restoring agent setup")
            self._apply_setup()
        except (KernelGoneError, GatewayUnreachableError, KernelRestartedError):
            self._recover_kernel()

    def _recover_kernel(self) -> None:
        """Start one fresh kernel and restore the setup, or mark the sandbox fatal."""
        if self._recoveries >= MAX_KERNEL_RECOVERIES:
            self._mark_fatal(
                f"the Jupyter kernel was lost again after {self._recoveries} fresh kernel(s) "
                "were started in this run"
            )
            return
        self._recoveries += 1
        try:
            new_id = self._create_kernel()
        except Exception as exc:  # gateway gone too: nothing left to try
            self._mark_fatal(f"the kernel was lost and a new one could not be started: {exc}")
            return
        self.kernel_id = new_id
        self.ws_url = self.ws_url.rsplit("/api/kernels/", 1)[0] + f"/api/kernels/{new_id}/channels"
        self.kernel_restarts += 1
        try:
            self._apply_setup()
        except Exception as exc:
            self._mark_fatal(f"a fresh kernel was started but setup failed: {exc}")
            return
        self.logger.log(f"Started fresh Jupyter kernel {new_id} and restored agent setup")

    def _mark_fatal(self, reason: str) -> None:
        self.fatal_error = (
            f"{SANDBOX_UNRECOVERABLE_MARKER} The code sandbox is unrecoverable: {reason}. "
            "Code execution was stopped instead of retrying against a dead kernel."
        )
        logger.error(self.fatal_error)

    def _raw_execute(self, code: str, timeout: float | None = None) -> CodeOutput:
        """Execute ``code`` once over a fresh websocket. Raises typed kernel errors."""
        try:
            ws = self._connect(self.ws_url)
        except Exception as exc:
            if _is_not_found(exc):
                raise KernelGoneError(str(exc)) from exc
            if _is_connection_refused(exc):
                raise GatewayUnreachableError(str(exc)) from exc
            raise
        with contextlib.closing(ws):
            return self._run_on_socket(code, ws, timeout or exec_timeout_seconds())

    def _run_on_socket(self, code: str, ws, timeout: float) -> CodeOutput:
        from websocket import WebSocketConnectionClosedException, WebSocketTimeoutException

        msg_id = str(uuid.uuid4())
        ws.send(
            json.dumps(
                {
                    "header": {
                        "msg_id": msg_id,
                        "username": "anonymous",
                        "session": str(uuid.uuid4()),
                        "msg_type": "execute_request",
                        "version": "5.0",
                    },
                    "parent_header": {},
                    "metadata": {},
                    "content": {
                        "code": code,
                        "silent": False,
                        "store_history": True,
                        "user_expressions": {},
                        "allow_stdin": False,
                    },
                }
            )
        )
        with_settimeout = getattr(ws, "settimeout", None)
        if with_settimeout:
            with_settimeout(min(_POLL_INTERVAL_S, timeout))

        deadline = time.monotonic() + timeout
        outputs: list[str] = []
        result: Any = None
        is_final_answer = False
        while True:
            if time.monotonic() > deadline:
                self._interrupt_kernel()
                raise AgentError(
                    f"Code execution exceeded {timeout:.0f}s and was interrupted. Run long "
                    "jobs in the background (subprocess.Popen writing to a log file) and "
                    "poll for completion in later steps.",
                    self.logger,
                )
            try:
                raw = ws.recv()
            except WebSocketTimeoutException:
                if not self._kernel_alive():
                    raise KernelGoneError("kernel disappeared while executing") from None
                continue
            except WebSocketConnectionClosedException as exc:
                if not self._kernel_alive():
                    raise KernelGoneError("kernel websocket closed and kernel is gone") from exc
                raise KernelRestartedError("kernel websocket closed mid-execution") from exc
            if not raw:
                continue
            msg = json.loads(raw)
            msg_type = msg.get("msg_type", "")
            content = msg.get("content", {}) or {}
            if msg_type == "status" and content.get("execution_state") in ("restarting", "dead"):
                raise KernelRestartedError(f"kernel entered state {content['execution_state']}")
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            if msg_type == "stream":
                outputs.append(content.get("text", ""))
            elif msg_type == "execute_result":
                result = content.get("data", {}).get("text/plain")
            elif msg_type == "error":
                if content.get("ename", "") == self.FINAL_ANSWER_EXCEPTION:
                    result = pickle.loads(base64.b64decode(content.get("evalue", "")))  # nosec B301
                    is_final_answer = True
                else:
                    raise AgentError("\n".join(content.get("traceback", [])), self.logger)
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break
        return CodeOutput(output=result, logs="".join(outputs), is_final_answer=is_final_answer)

    def _fatal_output(self) -> CodeOutput:
        return CodeOutput(output=self.fatal_error, logs="", is_final_answer=True)

    def resilient_run(self, code: str) -> CodeOutput:
        """Run agent code with session restoration and bounded kernel recovery."""
        if self.fatal_error:
            return self._fatal_output()
        restarts_before = self.kernel_restarts
        self._ensure_session()
        if self.fatal_error:
            return self._fatal_output()
        try:
            output = self._raw_execute(code)
            if self.kernel_restarts > restarts_before:
                # The kernel had died since the previous step; the code above ran in a
                # fresh namespace, so tell the agent why earlier variables are missing.
                notice = f"[sandbox notice] {KERNEL_RESTARTED_MESSAGE}\n"
                output = CodeOutput(
                    output=output.output,
                    logs=notice + (output.logs or ""),
                    is_final_answer=output.is_final_answer,
                )
            return output
        except KernelRestartedError:
            self.kernel_restarts += 1
            try:
                self._apply_setup()
            except (KernelGoneError, GatewayUnreachableError, KernelRestartedError):
                self._recover_kernel()
            if self.fatal_error:
                return self._fatal_output()
            raise AgentError(KERNEL_RESTARTED_MESSAGE, self.logger) from None
        except (KernelGoneError, GatewayUnreachableError):
            self._recover_kernel()
            if self.fatal_error:
                return self._fatal_output()
            raise AgentError(KERNEL_RESTARTED_MESSAGE, self.logger) from None


class ResilientDockerExecutor(ResilientKernelMixin, DockerExecutor):
    """``DockerExecutor`` with kernel-restart recovery and a pip guard."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._init_resilience()
        super().__init__(*args, **kwargs)

    def install_packages(self, additional_imports: list[str]) -> list[str]:
        allowed, skipped = filter_install_requests(additional_imports)
        if skipped:
            self.logger.log(f"Skipping pip install of pinned/project packages: {skipped}")
        if allowed:
            super().install_packages(allowed)
        return list(additional_imports)

    def send_tools(self, tools: dict[str, Tool]) -> None:
        # A new agent.run(): allow fresh recoveries and clear a previous fatal state,
        # but only if the kernel actually answers again.
        self._recoveries = 0
        self.fatal_error = None
        if "final_answer" in tools:
            self._patch_final_answer_with_exception(tools["final_answer"])
        requirements = {
            pkg
            for tool in tools.values()
            for pkg in tool.to_dict()["requirements"]
            if pkg not in self.installed_packages
        }
        if requirements:
            self.installed_packages += self.install_packages(sorted(requirements))
        self._tools_code = get_tools_definition_code(tools)
        try:
            self._apply_setup()
        except (KernelGoneError, GatewayUnreachableError, KernelRestartedError):
            self._recover_kernel()

    def send_variables(self, variables: dict[str, Any]) -> None:
        if not variables:
            self._variables_code = ""
            return
        pickled = base64.b64encode(pickle.dumps(variables)).decode()
        self._variables_code = (
            "import pickle, base64\n"
            f"vars_dict = pickle.loads(base64.b64decode('{pickled}'))\n"
            "locals().update(vars_dict)\n"
        )
        try:
            self._raw_execute(self._variables_code)
        except (KernelGoneError, GatewayUnreachableError, KernelRestartedError):
            self._recover_kernel()  # setup_code() replays the variables on the new kernel

    def run_code_raise_errors(self, code: str) -> CodeOutput:
        # Before send_tools has run (image build, install_packages during __init__),
        # behave exactly like upstream.
        if not getattr(self, "_tools_code", ""):
            return self._raw_execute(code)
        return self.resilient_run(code)
