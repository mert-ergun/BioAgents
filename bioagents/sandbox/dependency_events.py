"""Runtime evidence for dependency installs, shared by tools and sandboxes.

Only executed installer commands count as attempts. Printed advice and code text
are not execution evidence. Unknown outcomes remain explicit rather than success.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import json
import logging
import os
import re
import shlex
import stat
import subprocess  # nosec B404
import sys
import tempfile
import threading
import uuid
from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

_P = ParamSpec("_P")
_R = TypeVar("_R")

logger = logging.getLogger(__name__)
EVENTS_FILE = "dependency_events.jsonl"
_scope: ContextVar[dict[str, str] | None] = ContextVar("dependency_scope", default=None)
_run: ContextVar[RunAudit | None] = ContextVar("dependency_run", default=None)
_suppressed: ContextVar[bool] = ContextVar("dependency_observer_suppressed", default=False)
_enabled = False
_enable_lock = threading.Lock()


def _scope_values() -> dict[str, str]:
    current = _scope.get()
    if current is not None:
        return current
    try:
        inherited = json.loads(os.getenv("BIOAGENTS_DEPENDENCY_SCOPE", "{}"))
        return {
            key: value
            for key, value in inherited.items()
            if key in {"component", "phase", "tool", "reason"} and isinstance(value, str)
        }
    except (ValueError, AttributeError):
        return {}


def _installer(command: Any) -> tuple[str, list[str], list[str]] | None:
    """Recognize direct pip/uv/conda/mamba/apt invocations without executing them."""
    if isinstance(command, bytes):
        command = os.fsdecode(command)
    try:
        tokens = shlex.split(command) if isinstance(command, str) else list(command)
    except (ValueError, TypeError):
        return None
    if not tokens or not all(isinstance(t, str) for t in tokens):
        return None
    executable = Path(tokens[0]).name
    if executable in {"sh", "bash", "dash"} and tokens[1:2] == ["-c"] and len(tokens) == 3:
        return _installer(tokens[2])
    if executable == "sudo":
        return _installer([t for t in tokens[1:] if t not in {"-n", "-H"}])
    # A shell compound is not proof that every branch executed. Child process
    # observations supply the evidence instead of counting static candidates.
    if any(t in {"&&", "||", ";", "|", "&"} for t in tokens):
        return None
    offset = 1
    if executable.startswith("python") and tokens[1:3] == ["-m", "pip"]:
        manager, offset = "pip", 3
    elif executable.startswith("python") and tokens[1:3] == ["-m", "ensurepip"]:
        return "ensurepip", ["pip"], _redact(tokens)
    elif executable == "uv" and tokens[1:2] == ["pip"]:
        manager, offset = "pip", 2
    elif re.fullmatch(r"pip\d*(?:\.\d+)*", executable):
        manager = "pip"
    elif executable in {"mamba", "conda", "apt", "apt-get"}:
        manager = executable
    else:
        return None
    install = offset
    option_values = {"--proxy", "--python", "-n", "--name", "-p", "--prefix", "-c", "--channel"}
    while install < len(tokens) and tokens[install].startswith("-"):
        flag = tokens[install]
        install += 2 if flag in option_values else 1
    if install >= len(tokens) or tokens[install] != "install":
        return None
    packages = []
    skip = False
    for token in tokens[install + 1 :]:
        if skip:
            skip = False
            continue
        if token.startswith("-"):
            skip = token in {
                "-c",
                "--channel",
                "-i",
                "--index-url",
                "--extra-index-url",
                "-r",
                "--requirement",
                "--constraint",
                "--cache-dir",
                "--target",
                "--prefix",
                "--password",
                "--token",
                "--api-key",
                "--proxy",
                "--python",
                "-n",
                "--name",
            }
            continue
        if "://" not in token and (
            match := re.match(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)(?:[\[<>=!~ @]|$)", token)
        ):
            packages.append(match[1])
    return manager, sorted(set(packages)), _redact(tokens)


def _redact(tokens: list[str]) -> list[str]:
    result = []
    private_next = False
    for token in tokens:
        if (
            private_next
            or "://" in token
            or re.search(r"(?i)(?:password|token|secret|api[_-]?key)=", token)
        ):
            result.append("<redacted>")
            private_next = False
            continue
        result.append(token[:200])
        private_next = token in {
            "--index-url",
            "--extra-index-url",
            "--password",
            "--token",
            "--api-key",
        }
    if result:
        result[0] = Path(result[0]).name
    return result[:80]


def _shell_install_candidates(command: Any) -> list[str]:
    """A compound shell may skip branches; report coverage gaps, not fake attempts."""
    if isinstance(command, (list, tuple)):
        if (
            len(command) == 3
            and Path(str(command[0])).name in {"sh", "bash", "dash"}
            and command[1] == "-c"
        ):
            command = command[2]
        else:
            return []
    if not isinstance(command, str):
        return []
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        parts = list(lexer)
    except ValueError:
        return []
    if not any(p in {";", "&&", "||", "|", "&"} for p in parts):
        return []
    candidates: list[str] = []
    segment: list[str] = []
    for part in [*parts, ";"]:
        if part in {";", "&&", "||", "|", "&"}:
            while segment and segment[0] in {"then", "do", "if", "!"}:
                segment.pop(0)
            if install := _installer(segment):
                candidates.extend(install[1])
            segment = []
        else:
            segment.append(part)
    return sorted(set(candidates))


@dataclass
class RunAudit:
    run_id: str
    directory: Path | None
    use_case_id: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    lock: Any = field(default_factory=threading.Lock, repr=False)

    def prepare(self) -> None:
        """Create the journal as the host user before a root kernel can create it."""
        if self.directory is None:
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                self.directory / EVENTS_FILE,
                os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                0o600,
            )
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise ValueError("Dependency events require a regular file")
            finally:
                os.close(descriptor)
        except (OSError, ValueError) as exc:
            self.errors.append(type(exc).__name__)
            logger.warning("Dependency journal preparation failed: %s", type(exc).__name__)

    def summary(self) -> dict[str, Any]:
        events = list(self.events)
        if self.directory:
            path = self.directory / EVENTS_FILE
            try:
                if path.is_symlink():
                    raise ValueError("Dependency event path is a symlink")
                if path.exists():
                    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                    with os.fdopen(descriptor) as handle:
                        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                            raise ValueError("Dependency events require a regular file")
                        import fcntl

                        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
                        for line in handle:
                            try:
                                event = json.loads(line)
                                if event.get("run_id") == self.run_id and (
                                    not self.use_case_id
                                    or event.get("use_case_id") == self.use_case_id
                                ):
                                    events.append(event)
                            except (ValueError, AttributeError):
                                self.errors.append("Malformed dependency event")
            except (OSError, ValueError, AttributeError, ImportError) as exc:
                self.errors.append(type(exc).__name__)
        unique = {e["event_id"]: e for e in events if isinstance(e, dict) and "event_id" in e}
        starts = {e["attempt_id"]: e for e in unique.values() if e["kind"] == "install_started"}
        finishes = {e["attempt_id"]: e for e in unique.values() if e["kind"] == "install_finished"}
        statuses = Counter(finishes.get(key, {}).get("outcome", "pending") for key in starts)
        return {
            "run_id": self.run_id,
            "use_case_id": self.use_case_id,
            "install_attempts": len(starts),
            "outcomes": dict(statuses),
            "event_counts": dict(Counter(e["kind"] for e in unique.values())),
            "by_component": dict(Counter(e.get("component", "unknown") for e in starts.values())),
            "by_phase": dict(Counter(e.get("phase", "unknown") for e in starts.values())),
            "by_reason": dict(Counter(e.get("reason", "unknown") for e in starts.values())),
            "by_tool": dict(Counter(e["tool"] for e in starts.values() if e.get("tool"))),
            "tracking_errors": list(self.errors),
            "unobserved_install_branches": sum(
                e["kind"] == "unobserved_install_branch" for e in unique.values()
            ),
            "interpretation": "Observed runtime attempts only; pending/unknown outcomes are not successful installs. No events do not prove dependency completeness.",
        }


def current_audit() -> RunAudit | None:
    current = _run.get()
    if current:
        return current
    run_id = os.getenv("BIOAGENTS_RUN_ID", "")
    if not re.fullmatch(r"[a-f0-9]{32}", run_id):
        return None
    directory = os.getenv("BIOAGENTS_DEPENDENCY_EVENTS_DIR") or os.getenv("BIOAGENTS_SANDBOX_DIR")
    return RunAudit(
        run_id, Path(directory) if directory else None, os.getenv("BIOAGENTS_USE_CASE_ID", "")
    )


def dependency_environment() -> dict[str, str]:
    audit = current_audit()
    if audit is None:
        return {}
    result = {"BIOAGENTS_RUN_ID": audit.run_id, "BIOAGENTS_USE_CASE_ID": audit.use_case_id}
    result["BIOAGENTS_DEPENDENCY_SCOPE"] = json.dumps(_scope_values())
    if audit.directory:
        result["BIOAGENTS_DEPENDENCY_EVENTS_DIR"] = str(audit.directory.resolve())
    return result


def record_dependency_event(kind: str, *, audit: RunAudit | None = None, **details: Any) -> None:
    audit = audit or current_audit()
    if audit is None:
        return
    if "packages" in details:
        details["packages"] = [
            match[1]
            for item in details["packages"]
            if isinstance(item, str)
            and (match := re.match(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)(?:[\[<>=!~ @]|$)", item))
        ]
    event = {
        **_scope_values(),
        **details,
        "kind": kind,
        "run_id": audit.run_id,
        "use_case_id": audit.use_case_id,
        "event_id": uuid.uuid4().hex,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    with audit.lock:
        audit.events.append(event)
    if audit.directory is None:
        return
    try:
        audit.directory.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(event, ensure_ascii=False) + "\n").encode()
        descriptor = os.open(
            audit.directory / EVENTS_FILE,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
            0o600,
        )
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("Dependency events require a regular file")
            # Hold a cross-process lock for the entire record, including short writes.
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
            view = memoryview(payload)
            while view:
                count = os.write(descriptor, view)
                if not count:
                    raise OSError("Short dependency event write")
                view = view[count:]
        finally:
            os.close(descriptor)
    except (OSError, ValueError, TypeError, ImportError) as exc:
        audit.errors.append(type(exc).__name__)
        logger.warning("Dependency event recording failed: %s", type(exc).__name__)


@contextlib.contextmanager
def dependency_scope(**details: str):
    token = _scope.set({**_scope_values(), **details})
    try:
        yield
    finally:
        _scope.reset(token)


def track_dependencies(
    *, component: str, phase: str
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        signature = inspect.signature(function)

        @functools.wraps(function)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            bound = signature.bind_partial(*args, **kwargs).arguments
            with dependency_scope(
                component=component,
                phase=phase,
                tool=str(
                    bound.get(
                        "tool_name", bound.get("name", getattr(bound.get("self"), "_tool_name", ""))
                    )
                ),
            ):
                enable_dependency_tracking()
                return function(*args, **kwargs)

        return wrapped

    return decorate


@dataclass
class _Attempt:
    audit: RunAudit
    identifier: str
    scope: dict[str, str]
    finished: bool = False
    lock: Any = field(default_factory=threading.Lock, repr=False)

    def finish(
        self, outcome: str, returncode: int | None = None, error_type: str | None = None
    ) -> None:
        with self.lock:
            if self.finished:
                return
            self.finished = True
        record_dependency_event(
            "install_finished",
            audit=self.audit,
            attempt_id=self.identifier,
            outcome=outcome,
            returncode=returncode,
            error_type=error_type,
            **self.scope,
        )


def _start(command: Any) -> _Attempt | None:
    audit, install = current_audit(), _installer(command)
    if audit is None:
        return None
    if install is None:
        if candidates := _shell_install_candidates(command):
            record_dependency_event(
                "unobserved_install_branch",
                audit=audit,
                packages=candidates,
                **_scope_values(),
            )
        return None
    manager, packages, safe_command = install
    attempt = _Attempt(
        audit,
        uuid.uuid4().hex,
        {
            "component": "sandbox",
            "phase": "execution",
            "reason": "executed_installer",
            **_scope_values(),
        },
    )
    record_dependency_event(
        "install_started",
        audit=audit,
        attempt_id=attempt.identifier,
        manager=manager,
        packages=packages,
        command=safe_command,
        **attempt.scope,
    )
    return attempt


DISABLE_INSTALLS_ENV = "BIOAGENTS_DISABLE_DYNAMIC_INSTALLS"


def dynamic_installs_disabled() -> bool:
    return os.getenv(DISABLE_INSTALLS_ENV, "").strip().lower() in {"1", "true", "yes"}


def record_blocked_install(command: list[str], *, reason: str, detail: str) -> None:
    """Count an install that was requested but deliberately not executed."""
    with dependency_scope(reason=reason):
        attempt = _start(command)
        if attempt:
            attempt.finish("blocked", error_type=detail)


def run_install_command(command: list[str], *, reason: str, **kwargs):
    """One audited boundary for controlled installs, including mockable failures.

    Every attempt is recorded with its component, phase, tool and reason. When
    ``BIOAGENTS_DISABLE_DYNAMIC_INSTALLS`` is set the command is not executed and
    the attempt is recorded as ``blocked``.
    """
    if dynamic_installs_disabled():
        record_blocked_install(command, reason=reason, detail="dynamic_installs_disabled")
        empty = "" if kwargs.get("text") or kwargs.get("universal_newlines") else b""
        message = f"Dynamic dependency installs are disabled ({DISABLE_INSTALLS_ENV})"
        return subprocess.CompletedProcess(
            command, 1, stdout=empty, stderr=message if isinstance(empty, str) else message.encode()
        )
    with dependency_scope(reason=reason):
        attempt = _start(command)
        token = _suppressed.set(True)
        try:
            result = subprocess.run(command, **kwargs)  # nosec B603
        except Exception as exc:
            if attempt:
                attempt.finish(
                    "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "failed",
                    getattr(exc, "returncode", None),
                    type(exc).__name__,
                )
            raise
        else:
            if attempt:
                attempt.finish("success" if result.returncode == 0 else "failed", result.returncode)
            return result
        finally:
            _suppressed.reset(token)


def enable_dependency_tracking() -> None:
    """Observe actual Python subprocess/system/exec calls, not submitted source.

    Popen/system outcomes are observed directly. exec/posix_spawn outcomes cannot
    be inferred here and are recorded explicitly as unknown. Installations made by
    unrelated native programs are outside this Python instrumentation.
    """
    global _enabled
    with _enable_lock:
        if _enabled:
            return
        original_system = os.system
        popen_signature = inspect.signature(subprocess.Popen)

        class AuditedPopen(subprocess.Popen):
            __signature__ = popen_signature

            def __init__(self, *args, **kwargs):
                inherited = dependency_environment()
                if inherited:
                    bound = popen_signature.bind_partial(*args, **kwargs)
                    environment = bound.arguments.get("env")
                    bound.arguments["env"] = (
                        {**os.environ, **inherited}
                        if environment is None
                        else {**inherited, **environment}
                    )
                    args, kwargs = bound.args, bound.kwargs
                self._dependency_attempt = (
                    None if _suppressed.get() else _start(args[0] if args else kwargs.get("args"))
                )
                token = _suppressed.set(True)
                try:
                    super().__init__(*args, **kwargs)
                except Exception as exc:
                    if self._dependency_attempt:
                        self._dependency_attempt.finish("failed", error_type=type(exc).__name__)
                    raise
                finally:
                    _suppressed.reset(token)

            def wait(self, *args, **kwargs):
                code = super().wait(*args, **kwargs)
                if self._dependency_attempt:
                    self._dependency_attempt.finish("success" if code == 0 else "failed", code)
                return code

            def poll(self):
                code = super().poll()
                if code is not None and self._dependency_attempt:
                    self._dependency_attempt.finish("success" if code == 0 else "failed", code)
                return code

        def audited_system(command):
            attempt = None if _suppressed.get() else _start(command)
            token = _suppressed.set(True)
            try:
                code = original_system(command)
            except Exception as exc:
                if attempt:
                    attempt.finish("failed", error_type=type(exc).__name__)
                raise
            else:
                if attempt:
                    attempt.finish("success" if code == 0 else "failed", code)
                return code
            finally:
                _suppressed.reset(token)

        def audit_hook(event, args):
            if (
                event not in {"os.exec", "os.posix_spawn", "os.system", "subprocess.Popen"}
                or _suppressed.get()
            ):
                return
            try:
                attempt = _start(args[0] if event == "os.system" else args[1])
                if attempt:
                    attempt.finish("unknown")
            except Exception as exc:
                logger.warning("Dependency observation failed: %s", type(exc).__name__)

        vars(subprocess)["Popen"] = AuditedPopen
        os.system = audited_system
        sys.addaudithook(audit_hook)
        _enabled = True


def audit_use_case(function):
    """Attach evidence to RunResult even when no code-agent steps were recorded."""

    @functools.wraps(function)
    def wrapped(use_case, config=None, *args, **kwargs):
        parent = current_audit()
        directory = parent.directory if parent else None
        if directory is None and config is not None:
            output = config.extra.get("output_dirs", {}).get(use_case.id)
            directory = Path(output) if output else None
        run_id = parent.run_id if parent else uuid.uuid4().hex
        if directory is None:
            directory = Path(".bioagents_cache/dependency-events") / run_id
        audit = RunAudit(run_id, directory, use_case.id)
        audit.prepare()
        token = _run.set(audit)
        try:
            enable_dependency_tracking()
            result = function(use_case, config, *args, **kwargs)
            if audit.directory:
                write_runtime_report(audit)
            result.dependency_audit = audit.summary()
            logger.info(
                "Dependency audit: install_attempts=%s outcomes=%s by_component=%s unobserved_branches=%s tracking_errors=%s",
                result.dependency_audit["install_attempts"],
                result.dependency_audit["outcomes"],
                result.dependency_audit["by_component"],
                result.dependency_audit["unobserved_install_branches"],
                len(result.dependency_audit["tracking_errors"]),
            )
            return result
        finally:
            _run.reset(token)

    return wrapped


def write_runtime_report(audit: RunAudit) -> None:
    """Atomically publish a run-wide snapshot; RunResult retains per-case counts."""
    if audit.directory is None:
        return
    temporary = None
    try:
        audit.directory.mkdir(parents=True, exist_ok=True)
        report = RunAudit(audit.run_id, audit.directory).summary()
        descriptor, temporary = tempfile.mkstemp(prefix=".dependency-audit-", dir=audit.directory)
        with os.fdopen(descriptor, "w") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        Path(temporary).replace(audit.directory / "dependency_audit.json")
    except (OSError, ValueError, TypeError) as exc:
        audit.errors.append(type(exc).__name__)
        logger.warning("Dependency summary recording failed: %s", type(exc).__name__)
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)
