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
import logging
import math
import os

# Serializes agent variables for transport into the trusted kernel.
import pickle  # nosec B403
import uuid
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests
from smolagents import DockerExecutor
from smolagents.local_python_executor import CodeOutput
from smolagents.remote_executors import RemotePythonExecutor, _create_kernel_http
from smolagents.tools import Tool, get_tools_definition_code
from smolagents.utils import AgentError

from bioagents.sandbox.dependency_events import record_dependency_event
from bioagents.sandbox.pip_guard import filter_install_requests, kernel_guard_code
from bioagents.sandbox.websocket_runner import RemoteExecutionError, run_code_over_websocket

logger = logging.getLogger(__name__)

#: Marker put at the start of the final answer when the sandbox is unrecoverable.
SANDBOX_UNRECOVERABLE_MARKER = "[CODER_STATUS: sandbox_unrecoverable]"

DEFAULT_EXEC_TIMEOUT_S = 1800.0
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
    """Read the execution budget; the newer setting overrides the legacy alias."""
    raw = os.getenv("BIOAGENTS_CODE_EXECUTION_TIMEOUT_SEC")
    if raw is None:
        raw = os.getenv("BIOAGENTS_CODER_EXEC_TIMEOUT")
    value = DEFAULT_EXEC_TIMEOUT_S if raw is None else float(raw)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Code execution timeout must be finite and > 0")
    return value


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
        self._execution_failure: str | None = None

    # -- hooks overridable in tests -------------------------------------------------
    def _connect(self, url: str):
        from websocket import create_connection

        # Fresh sockets need distinct client sessions. Reusing Jupyter's default
        # session can send the next shell reply into the previous socket's buffer.
        parts = urlsplit(url)
        query = [
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if key != "session_id"
        ]
        query.append(("session_id", uuid.uuid4().hex))
        target = urlunsplit(parts._replace(query=urlencode(query)))
        return create_connection(target, timeout=30)

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

    def _restart_kernel(self) -> None:
        """Restart the same kernel and only clear failure after setup succeeds."""
        response = requests.post(
            f"{self.base_url}/api/kernels/{self.kernel_id}/restart", timeout=30
        )
        response.raise_for_status()
        self.kernel_restarts += 1
        self._apply_setup()
        self._execution_failure = None
        self.logger.log("Jupyter kernel restarted and agent setup restored")

    def setup_code(self) -> str:
        """Everything a fresh kernel needs before agent code can run."""
        return "\n\n".join(
            part for part in (kernel_guard_code(), self._tools_code, self._variables_code) if part
        )

    def _apply_setup(self) -> None:
        # A short user-cell budget must not prevent a cold kernel from restoring
        # tools after restart. Setup has its own bounded recovery budget.
        self._raw_execute(self.setup_code(), timeout=30.0)

    def _final_answer_defined(self) -> bool:
        probe = self._raw_execute("print('final_answer' in globals())", timeout=30.0)
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
        self._execution_failure = None

    def _mark_fatal(self, reason: str) -> None:
        self.fatal_error = (
            f"{SANDBOX_UNRECOVERABLE_MARKER} The code sandbox is unrecoverable: {reason}. "
            "Code execution was stopped instead of retrying against a dead kernel."
        )
        logger.error(self.fatal_error)

    def _raw_execute(self, code: str, timeout: float | None = None) -> CodeOutput:
        """Execute ``code`` once over a fresh websocket. Raises typed kernel errors."""
        budget = exec_timeout_seconds() if timeout is None else timeout
        try:
            ws = self._connect(self.ws_url)
        except Exception as exc:
            if _is_not_found(exc):
                raise KernelGoneError(str(exc)) from exc
            from websocket import WebSocketException

            if _is_connection_refused(exc) or isinstance(exc, (OSError, WebSocketException)):
                raise GatewayUnreachableError(str(exc)) from exc
            raise
        with contextlib.closing(ws):
            return self._run_on_socket(code, ws, budget)

    def _run_on_socket(self, code: str, ws, timeout: float) -> CodeOutput:
        try:
            return run_code_over_websocket(code, ws, self.logger, timeout=timeout)
        except RemoteExecutionError as exc:
            if not self._kernel_alive():
                raise KernelGoneError(str(exc)) from exc
            if str(exc).startswith("Kernel entered "):
                raise KernelRestartedError(str(exc)) from exc
            raise

    def _fatal_output(self) -> CodeOutput:
        return CodeOutput(output=self.fatal_error, logs="", is_final_answer=True)

    def resilient_run(self, code: str) -> CodeOutput:
        """Run agent code with session restoration and bounded kernel recovery."""
        if self.fatal_error:
            return self._fatal_output()
        restarts_before = self.kernel_restarts
        try:
            if self._execution_failure:
                try:
                    self._restart_kernel()
                except requests.HTTPError as exc:
                    if exc.response is not None and exc.response.status_code == 404:
                        raise KernelGoneError(str(exc)) from exc
                    raise AgentError(
                        f"Kernel recovery failed; retry on the next step: {exc}", self.logger
                    ) from exc
                except Exception as exc:
                    raise AgentError(
                        f"Kernel recovery failed; retry on the next step: {exc}", self.logger
                    ) from exc
            self._ensure_session()
            if self.fatal_error:
                return self._fatal_output()
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
        except (TimeoutError, RemoteExecutionError) as exc:
            self._execution_failure = str(exc) or type(exc).__name__
            try:
                self._restart_kernel()
            except Exception as recovery_error:
                self.logger.log_error(f"Kernel restart/setup failed: {recovery_error}")
            notice = (
                KERNEL_RESTARTED_MESSAGE
                if not self._execution_failure
                else "Kernel recovery will be retried before the next execution."
            )
            raise AgentError(f"{exc}. {notice}", self.logger) from exc
        except KernelRestartedError:
            self.kernel_restarts += 1
            try:
                self._apply_setup()
            except (KernelGoneError, GatewayUnreachableError, KernelRestartedError):
                self._recover_kernel()
            except (TimeoutError, RemoteExecutionError, AgentError) as exc:
                self._execution_failure = str(exc) or type(exc).__name__
                self.logger.log_error(f"Kernel setup restoration failed: {exc}")
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
            record_dependency_event(
                "guard_skipped", component="framework", phase="setup", packages=skipped
            )
            self.logger.log(f"Skipping pip install of pinned/project packages: {skipped}")
        if allowed:
            code = (
                "import sys\n"
                "from bioagents.sandbox.dependency_events import dependency_scope, run_install_command\n"
                "with dependency_scope(component='framework', phase='setup', reason='tool_requirements'):\n"
                f"    _bioagents_install = run_install_command([sys.executable, '-m', 'pip', 'install', *{allowed!r}], reason='tool_requirements', capture_output=True, text=True, timeout=180)\n"
                "print(_bioagents_install.stdout)\n"
                "if _bioagents_install.returncode:\n"
                "    print(_bioagents_install.stderr)\n"
                "    raise RuntimeError('Dependency installation failed with exit code ' + str(_bioagents_install.returncode))\n"
            )
            self.run_code_raise_errors(code)
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
            self._raw_execute(self._variables_code, timeout=30.0)
        except (KernelGoneError, GatewayUnreachableError, KernelRestartedError):
            self._recover_kernel()  # setup_code() replays the variables on the new kernel

    def run_code_raise_errors(self, code: str) -> CodeOutput:
        # Before send_tools has run (image build, install_packages during __init__),
        # behave exactly like upstream.
        if not getattr(self, "_tools_code", ""):
            return self._raw_execute(code)
        return self.resilient_run(code)
