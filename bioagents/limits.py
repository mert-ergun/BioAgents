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
# This ONLY covers the time waiting for the LLM HTTP response — code execution
# inside the Docker/Jupyter kernel (pip installs, model downloads, long computations)
# is entirely separate and has no timeout.
# 600s default accommodates Gemini 3 Flash with thinking enabled for long-form
# report generation. Increase via BIOAGENTS_AGENT_LLM_INVOKE_TIMEOUT_SEC if needed,
# or set to 0 to disable entirely.
AGENT_LLM_INVOKE_TIMEOUT_SEC = _env_float("BIOAGENTS_AGENT_LLM_INVOKE_TIMEOUT_SEC", 600.0)

# Max seconds to wait for a rate-limit slot before failing fast (avoids “stuck” behind RPM).
RATE_LIMIT_MAX_WAIT_SEC = _env_float("BIOAGENTS_RATE_LIMIT_MAX_WAIT_SEC", 120.0)

# Total wall time for one graph.stream session (server / API). 0 = disabled.
GRAPH_STREAM_WALL_CLOCK_SEC = _env_float("BIOAGENTS_GRAPH_STREAM_WALL_CLOCK_SEC", 7200.0)

# Max tool-call rounds a single agent may take before being forced back to the
# supervisor.  Prevents runaway agent↔tools loops (e.g. data_acquisition retrying
# downloads endlessly).  0 = disabled.
MAX_AGENT_TOOL_ROUNDS = _env_int("BIOAGENTS_MAX_AGENT_TOOL_ROUNDS", 5)

# Default LangGraph recursion cap (also set per call in server).
GRAPH_RECURSION_LIMIT = max(10, _env_int("BIOAGENTS_GRAPH_RECURSION_LIMIT", 100))

# --- Tool Policy & Approval limits ---

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
