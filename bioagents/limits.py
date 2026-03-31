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
# Prevents multi‑MB payloads from slowing serialization and downstream LLM calls.
MAX_TOOL_OUTPUT_CHARS = max(1000, _env_int("BIOAGENTS_MAX_TOOL_OUTPUT_CHARS", 24_000))

# Hard cap on a single LLM agent invoke (HTTP timeout is separate). 0 = disabled.
# Must be well below the benchmark/workflow timeout so one slow call doesn't
# consume the entire budget (benchmark default is 180s).
AGENT_LLM_INVOKE_TIMEOUT_SEC = _env_float("BIOAGENTS_AGENT_LLM_INVOKE_TIMEOUT_SEC", 90.0)

# Max seconds to wait for a rate‑limit slot before failing fast (avoids “stuck” behind RPM).
RATE_LIMIT_MAX_WAIT_SEC = _env_float("BIOAGENTS_RATE_LIMIT_MAX_WAIT_SEC", 120.0)

# Total wall time for one graph.stream session (server / API). 0 = disabled.
GRAPH_STREAM_WALL_CLOCK_SEC = _env_float("BIOAGENTS_GRAPH_STREAM_WALL_CLOCK_SEC", 7200.0)

# Default LangGraph recursion cap (also set per call in server).
GRAPH_RECURSION_LIMIT = max(10, _env_int("BIOAGENTS_GRAPH_RECURSION_LIMIT", 100))
