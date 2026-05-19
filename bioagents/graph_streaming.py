"""Bounded iteration over LangGraph streams (wall-clock guard between steps)."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import TYPE_CHECKING, Any

from bioagents.limits import GRAPH_RECURSION_LIMIT, GRAPH_STREAM_WALL_CLOCK_SEC

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator


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


async def aiter_graph_stream(
    graph: Any,
    initial_state: dict,
    *,
    recursion_limit: int | None = None,
    wall_clock_sec: float | None = None,
    config: dict | None = None,
) -> AsyncIterator[Any]:
    """Async wrapper around graph.stream() that doesn't block the event loop.

    Runs graph.stream() in a background thread and yields steps via an async
    queue. Uses a threading.Event for back-pressure so that the graph thread
    pauses after each step, making steering injection via ``graph.update_state``
    safe between yields.
    """
    rlim = recursion_limit if recursion_limit is not None else GRAPH_RECURSION_LIMIT
    wall = wall_clock_sec if wall_clock_sec is not None else GRAPH_STREAM_WALL_CLOCK_SEC
    cfg: dict[str, Any] = dict(config) if config else {}
    cfg["recursion_limit"] = rlim

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Any | None] = asyncio.Queue()
    step_consumed = threading.Event()
    stop = threading.Event()
    error_holder: list[BaseException | None] = [None]

    def _thread_run() -> None:
        try:
            start = time.monotonic()
            for step in graph.stream(initial_state, cfg):
                if stop.is_set():
                    return
                if wall and wall > 0 and (time.monotonic() - start) > wall:
                    error_holder[0] = TimeoutError(
                        f"Workflow exceeded graph wall clock limit ({wall}s; "
                        "BIOAGENTS_GRAPH_STREAM_WALL_CLOCK_SEC, or 0 to disable)"
                    )
                    return
                asyncio.run_coroutine_threadsafe(queue.put(step), loop)
                step_consumed.wait(timeout=600)
                step_consumed.clear()
                if stop.is_set():
                    return
        except BaseException as exc:
            error_holder[0] = exc
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    thread = threading.Thread(target=_thread_run, daemon=True, name="graph-stream")
    thread.start()

    try:
        while True:
            step = await queue.get()
            if step is None:
                break
            yield step
            step_consumed.set()
    finally:
        stop.set()
        step_consumed.set()
        thread.join(timeout=2)

    if error_holder[0] is not None:
        raise error_holder[0]
