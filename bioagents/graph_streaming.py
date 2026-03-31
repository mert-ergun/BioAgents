"""Bounded iteration over LangGraph streams (wall-clock guard between steps)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from bioagents.limits import GRAPH_RECURSION_LIMIT, GRAPH_STREAM_WALL_CLOCK_SEC


def iter_graph_stream(
    graph: Any,
    initial_state: dict,
    *,
    recursion_limit: int | None = None,
    wall_clock_sec: float | None = None,
    config: dict | None = None,
) -> Iterator[Any]:
    """Yield graph.stream steps; raises TimeoutError if total elapsed exceeds wall_clock_sec."""
    rlim = recursion_limit if recursion_limit is not None else GRAPH_RECURSION_LIMIT
    wall = wall_clock_sec if wall_clock_sec is not None else GRAPH_STREAM_WALL_CLOCK_SEC
    cfg: dict[str, Any] = dict(config) if config else {}
    cfg["recursion_limit"] = rlim
    start = time.monotonic()
    for step in graph.stream(initial_state, cfg):
        if wall and wall > 0 and (time.monotonic() - start) > wall:
            raise TimeoutError(
                f"Workflow exceeded graph wall clock limit ({wall}s; "
                "BIOAGENTS_GRAPH_STREAM_WALL_CLOCK_SEC, or 0 to disable)"
            )
        yield step
