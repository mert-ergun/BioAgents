"""Central runtime limits (env-tunable) so workflows cannot hang indefinitely."""

import os


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


# Max characters stored in ToolMessage content after any tool runs (graph state).
# Prevents multi-MB payloads from slowing serialization and downstream LLM calls.
MAX_TOOL_OUTPUT_CHARS = max(1000, _env_int("BIOAGENTS_MAX_TOOL_OUTPUT_CHARS", 24_000))

# Hard cap on a single LLM API invoke (HTTP timeout is separate). 0 = disabled.
# Covers only the wait for the LLM response; sandbox code execution is untimed.
AGENT_LLM_INVOKE_TIMEOUT_SEC = _env_float("BIOAGENTS_AGENT_LLM_INVOKE_TIMEOUT_SEC", 600.0)

# Max seconds to wait for a rate-limit slot before failing fast (avoids “stuck” behind RPM).
RATE_LIMIT_MAX_WAIT_SEC = _env_float("BIOAGENTS_RATE_LIMIT_MAX_WAIT_SEC", 120.0)

# Total wall time for one graph.stream session (server / API). 0 = disabled.
GRAPH_STREAM_WALL_CLOCK_SEC = _env_float("BIOAGENTS_GRAPH_STREAM_WALL_CLOCK_SEC", 7200.0)

# Default LangGraph recursion cap (also set per call in server).
GRAPH_RECURSION_LIMIT = max(10, _env_int("BIOAGENTS_GRAPH_RECURSION_LIMIT", 100))

# Consecutive identical tool calls (same name + same args) before forced break.
MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS = max(
    1, _env_int("BIOAGENTS_MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS", 2)
)

# Max total tool_universe_call_tool calls per agent invocation.
MAX_TU_TOOL_CALLS_PER_AGENT = max(1, _env_int("BIOAGENTS_MAX_TU_TOOL_CALLS_PER_AGENT", 3))

# Seconds to wait for user approval before auto-rejecting.
TOOL_APPROVAL_TIMEOUT_SEC = _env_float("BIOAGENTS_TOOL_APPROVAL_TIMEOUT_SEC", 120.0)

# Whether to require approval for ALL tool_universe_call_tool calls.
TOOL_APPROVAL_ALL_TU_CALLS = os.getenv("BIOAGENTS_TOOL_APPROVAL_ALL_TU_CALLS", "").lower() in (
    "1",
    "true",
)

# Policy strictness: "strict", "moderate", or "permissive".
TOOL_POLICY_STRICTNESS = os.getenv("BIOAGENTS_TOOL_POLICY_STRICTNESS", "moderate").lower()

# Comma-separated extra tool categories to allow beyond defaults.
TOOL_POLICY_EXTRA_CATEGORIES = os.getenv("BIOAGENTS_TOOL_POLICY_EXTRA_CATEGORIES", "")

# Tool rounds one agent gets per supervisor task; uncapped, it never yields back
# and the supervisor's loop detection never runs.
MAX_AGENT_TOOL_ROUNDS = max(1, _env_int("BIOAGENTS_MAX_AGENT_TOOL_ROUNDS", 8))

# Attempts per tool call including the first; only transient failures retry, bad args fail fast.
TOOL_MAX_ATTEMPTS = max(1, _env_int("BIOAGENTS_TOOL_MAX_ATTEMPTS", 3))

# Backoff between attempts: base * 2**(attempt-1), capped, plus jitter.
TOOL_RETRY_BASE_DELAY_SEC = max(0.0, _env_float("BIOAGENTS_TOOL_RETRY_BASE_DELAY_SEC", 2.0))
TOOL_RETRY_MAX_DELAY_SEC = max(
    TOOL_RETRY_BASE_DELAY_SEC, _env_float("BIOAGENTS_TOOL_RETRY_MAX_DELAY_SEC", 30.0)
)

# Total wall clock for one BLAST search; remote nr searches routinely take 5-20 min.
# On timeout the tool returns the RID so the search can be resumed, not resubmitted.
BLAST_TIMEOUT_SEC = max(60.0, _env_float("BIOAGENTS_BLAST_TIMEOUT_SEC", 1200.0))

# NCBI usage guidelines: poll a single RID no more often than once per minute.
# Values below 60 are clamped up so the guideline cannot be configured away.
BLAST_POLL_INTERVAL_SEC = max(60.0, _env_float("BIOAGENTS_BLAST_POLL_INTERVAL_SEC", 60.0))

# Budget for a BLAST started from inside the coder sandbox. smolagents kills a
# Jupyter cell at 300 s and restarts the kernel, which would lose the RID mid-poll
# (the task3 failure). Capping below that lets the call return its RID instead.
BLAST_CELL_BUDGET_SEC = max(30.0, _env_float("BIOAGENTS_BLAST_CELL_BUDGET_SEC", 240.0))

# Exponential backoff for HTTP 429 / "CPU usage limit" responses: base delay, doubled
# on every consecutive rate-limit response, capped at the max.
BLAST_BACKOFF_BASE_SEC = max(1.0, _env_float("BIOAGENTS_BLAST_BACKOFF_BASE_SEC", 30.0))
BLAST_BACKOFF_MAX_SEC = max(
    BLAST_BACKOFF_BASE_SEC, _env_float("BIOAGENTS_BLAST_BACKOFF_MAX_SEC", 600.0)
)

# Read-only discovery tools. Repeating the same search over a large tool index is normal
# search behaviour, so these get their own (higher) consecutive-identical-call threshold.
# BIOAGENTS_DISCOVERY_TOOL_NAMES adds comma-separated names to the defaults.
DISCOVERY_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "tool_universe_find_tools",
        "find_tools",
        "search_custom_tools",
        "list_custom_tools",
        "get_tool_code",
    }
    | {
        name.strip()
        for name in os.getenv("BIOAGENTS_DISCOVERY_TOOL_NAMES", "").split(",")
        if name.strip()
    }
)

# Consecutive identical discovery calls (see DISCOVERY_TOOL_NAMES) before forced break.
MAX_CONSECUTIVE_IDENTICAL_DISCOVERY_CALLS = max(
    MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS,
    _env_int("BIOAGENTS_MAX_CONSECUTIVE_IDENTICAL_DISCOVERY_CALLS", 4),
)

# Rounds before the hard MAX_AGENT_TOOL_ROUNDS cut at which the agent is told to wrap up
# and report its findings. 0 disables the warning.
AGENT_TOOL_ROUNDS_WARNING_MARGIN = max(0, _env_int("BIOAGENTS_AGENT_TOOL_ROUNDS_WARNING_MARGIN", 1))
