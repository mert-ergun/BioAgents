"""Completion bookkeeping for the supervisor: stop re-delegating finished work.

Two signals decide that work is done:

* an explicit completion marker (``[CODER_STATUS: completed]``) from any agent in the
  current user turn — after it, only wrap-up routes (report once, then FINISH) are
  allowed;
* a completed subtask: a ``[SUPERVISOR TASK]`` whose agent answered without failing and
  either declared completion or needed no tool calls at all. Re-issuing the same (or a
  near-identical) instruction to any agent is blocked, because it would only re-run or
  re-write work that already exists.
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from langchain_core.messages import AIMessage, HumanMessage

from bioagents.agents.helpers import get_message_content
from bioagents.sandbox.coder_helpers import COMPLETION_MARKER

SUPERVISOR_TASK_TAG = "[SUPERVISOR TASK]"

# Two instructions at least this similar are the same subtask reworded.
DUPLICATE_TASK_SIMILARITY = 0.85

# Routes still allowed once the workflow carries an explicit completion marker.
WRAP_UP_ROUTES = frozenset({"report", "FINISH"})

# Human messages that continue the current turn rather than starting a new request.
_IN_TURN_HUMAN_TAGS = (SUPERVISOR_TASK_TAG, "[SYSTEM RESPONSE]")

# Forced returns and error stamps: the subtask was cut off, not finished.
_FAILURE_TAGS = (
    "[MAX_TOOL_ROUNDS]",
    "[LOOP_DETECTED]",
    "[MAX_TU_CALLS]",
    "[TIMEOUT]",
    "[AGENT_ERROR]",
    "[NO_OUTPUT]",
    "PARTIAL RESULT",
)

_FAILURE_LANGUAGE = re.compile(
    r"\b(unable to|cannot|can't|could ?n.t|failed|failure|error|not available"
    r"|no suitable|lacks?|missing|not (?:yet )?(?:complete|finished|done))\b",
    re.IGNORECASE,
)

_COMPLETION_CLAIM = re.compile(
    r"(TASK\s+(?:IS\s+)?COMPLETED?\b|ANALYSIS\s+COMPLETE\b|WORKFLOW\s+COMPLETE\b"
    r"|PIPELINE\s+COMPLETE\b|ALL\s+STEPS\s+COMPLETED?\b|successfully\s+(?:written|saved|created))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CompletedSubtask:
    """A delegated instruction that an agent already finished."""

    task: str
    agent: str
    explicit_marker: bool


def _is_ai(msg) -> bool:
    return isinstance(msg, AIMessage) or msg.__class__.__name__ == "AIMessage"


def _task_text(msg) -> str | None:
    if not isinstance(msg, HumanMessage):
        return None
    content = get_message_content(msg)
    if SUPERVISOR_TASK_TAG not in content:
        return None
    return content.replace(SUPERVISOR_TASK_TAG, "").strip()


def current_turn_start(messages) -> int:
    """Index of the human message that opened the current request.

    Supervisor handoffs and system replies stay inside a turn; a user query, a
    clarification answer or a steering directive starts a new one, so earlier
    completion evidence never blocks work the user has just asked for.
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, HumanMessage):
            continue
        content = get_message_content(msg)
        if not any(tag in content for tag in _IN_TURN_HUMAN_TAGS):
            return i
    return 0


def _is_failure(content: str) -> bool:
    if any(tag in content for tag in _FAILURE_TAGS):
        return True
    return "[CODER_STATUS:" in content and COMPLETION_MARKER not in content


def _span_is_completed(span: list) -> tuple[str, bool] | None:
    """Return ``(agent, explicit_marker)`` when the span's final answer finished the task."""
    replies = [m for m in span if _is_ai(m) and getattr(m, "name", "") not in ("", "Supervisor")]
    if not replies:
        return None
    final = replies[-1]
    if getattr(final, "tool_calls", None):
        return None
    content = get_message_content(final).strip()
    if not content or _is_failure(content):
        return None
    agent = getattr(final, "name", "")
    if COMPLETION_MARKER in content:
        return agent, True
    if _FAILURE_LANGUAGE.search(content):
        return None
    used_tools = any(getattr(m, "tool_calls", None) for m in replies)
    if _COMPLETION_CLAIM.search(content) or not used_tools:
        return agent, False
    return None


def find_completed_subtasks(messages) -> list[CompletedSubtask]:
    """Delegated instructions in the current turn that an agent already completed."""
    start = current_turn_start(messages)
    completed: list[CompletedSubtask] = []
    task: str | None = None
    span: list = []

    def close() -> None:
        if task:
            done = _span_is_completed(span)
            if done:
                completed.append(CompletedSubtask(task, done[0], done[1]))

    for msg in messages[start:]:
        text = _task_text(msg)
        if text is not None:
            close()
            task, span = text, []
        elif task is not None:
            span.append(msg)
    close()
    return completed


def has_completion_marker(messages) -> bool:
    """True if an agent stamped an explicit completion marker in the current turn."""
    start = current_turn_start(messages)
    return any(_is_ai(m) and COMPLETION_MARKER in get_message_content(m) for m in messages[start:])


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9_./]+", " ", text.lower()).split())


def _same_task(a: str, b: str) -> bool:
    na, nb = _normalise(a), _normalise(b)
    if not na or not nb:
        return False
    return na == nb or SequenceMatcher(None, na, nb).ratio() >= DUPLICATE_TASK_SIMILARITY


def find_duplicate_of_completed(messages, task_text: str) -> CompletedSubtask | None:
    """The completed subtask that ``task_text`` would re-issue, if any."""
    if not task_text:
        return None
    for done in reversed(find_completed_subtasks(messages)):
        if _same_task(task_text, done.task):
            return done
    return None


def _report_ran_since_completion(messages) -> bool:
    """True if the Report agent already answered after the latest completion marker."""
    start = current_turn_start(messages)
    marker_seen = False
    for msg in messages[start:]:
        if not _is_ai(msg):
            continue
        content = get_message_content(msg)
        if COMPLETION_MARKER in content:
            marker_seen = True
        elif marker_seen and getattr(msg, "name", "") == "Report" and content.strip():
            return True
    return False


def check_completed_work_guard(
    messages, next_agent: str, task_text: str, members
) -> tuple[str | None, str, str]:
    """Stop the supervisor from re-delegating work that is already done.

    Returns ``(replacement, reason, task)``: ``replacement`` is ``None`` when the route
    stands, otherwise ``"report"`` (with a read-only ``task``) or ``"FINISH"``.
    """
    if next_agent in ("FINISH", "user_input"):
        return None, "", ""

    if has_completion_marker(messages):
        if next_agent == "report" and not _report_ran_since_completion(messages):
            return None, "", ""
        if "report" in members and not _report_ran_since_completion(messages):
            return (
                "report",
                "An agent already marked the task completed; only a write-up remains.",
                read_only_report_task(),
            )
        return (
            "FINISH",
            "An agent already marked the task completed and it has been written up; "
            "finishing instead of delegating more work.",
            "",
        )

    duplicate = find_duplicate_of_completed(messages, task_text)
    if duplicate is not None:
        return (
            "FINISH",
            f"Agent '{duplicate.agent}' already completed this exact subtask; re-issuing it "
            "would only re-run or re-write finished work.",
            "",
        )
    return None, "", ""


def read_only_report_task() -> str:
    """Instruction for a wrap-up pass that must not redo any work."""
    return (
        "The requested work is already complete. Write up the results already present in "
        "the conversation and the WORKSPACE ARTIFACTS list. Verification is read-only: "
        "you may check that the reported files exist, but do not re-run analyses or "
        "re-write any file."
    )
