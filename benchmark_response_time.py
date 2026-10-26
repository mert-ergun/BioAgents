"""Compare a plain streaming LLM request with a BioAgents prompt.

Usage: .venv/bin/python benchmark_response_time.py --prompt "DNA nedir?"
Run with --help for available options and output settings.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import multiprocessing
import os
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS = {
    "openrouter": "deepseek/deepseek-v4-flash",
    "openai": "gpt-5.1",
    "gemini": "gemini-3.1-flash-lite",
    "ollama": "qwen3:14b",
}


@dataclass
class Measurement:
    target: str
    run: int = 1
    status: str = "running"
    ttft_s: float | None = None
    first_llm_token_s: float | None = None
    first_summary_token_s: float | None = None
    first_message_s: float | None = None
    last_text_s: float | None = None
    total_s: float | None = None
    elapsed_s: float | None = None
    initialization_s: float | None = None
    response_chars: int = 0
    text_chunks: int = 0
    error: str | None = None


def text_content(content: Any) -> str:
    """Extract text only, excluding reasoning, tool arguments and metadata."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in content
            if isinstance(part, str)
            or (isinstance(part, dict) and part.get("type") in ("text", "output_text"))
        )
    return ""


def redact_error(error: Any) -> str:
    value = str(error)
    for name, secret in os.environ.items():
        if len(secret) >= 8 and any(
            word in name.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD")
        ):
            value = value.replace(secret, "[REDACTED]")
    return value[:1000]


async def sse_data(response: httpx.Response):
    """Parse SSE events, including comments, multiline data and CRLF streams."""
    parts = []
    async for line in response.aiter_lines():
        if not line:
            if parts:
                yield "\n".join(parts)
                parts = []
        elif line.startswith("data:"):
            parts.append(line[5:].removeprefix(" "))
    if parts:
        yield "\n".join(parts)


def direct_request(provider: str, model: str, prompt: str) -> tuple[str, dict, dict]:
    api_key = os.getenv(f"{provider.upper()}_API_KEY", "")
    if provider != "ollama" and not api_key:
        raise ValueError(f"{provider.upper()}_API_KEY is missing from the environment/.env")
    if provider == "gemini":
        return (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse",
            {"x-goog-api-key": api_key},
            {"contents": [{"role": "user", "parts": [{"text": prompt}]}]},
        )
    base = {
        "openrouter": "https://openrouter.ai/api/v1",
        "openai": "https://api.openai.com/v1",
        "ollama": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    }[provider]
    return (
        base.rstrip("/") + "/chat/completions",
        {"Authorization": f"Bearer {api_key or 'ollama'}"},
        {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True},
    )


async def measure_direct(provider: str, model: str, prompt: str, timeout: float) -> Measurement:
    result = Measurement("direct_api")
    start = time.perf_counter()
    try:
        url, headers, body = direct_request(provider, model, prompt)
        async with asyncio.timeout(timeout):
            async with httpx.AsyncClient(timeout=timeout) as client:
                # Includes connection/DNS/TLS and the HTTP request; no automatic retries.
                start = time.perf_counter()
                completed = False
                async with client.stream("POST", url, headers=headers, json=body) as response:
                    response.raise_for_status()
                    async for data in sse_data(response):
                        now = time.perf_counter() - start
                        if data == "[DONE]":
                            completed = True
                            result.total_s = now
                            break
                        event = json.loads(data)
                        if event.get("error"):
                            raise RuntimeError(event["error"])
                        if provider == "gemini":
                            candidates = event.get("candidates", [])
                            candidate = candidates[0] if candidates else {}
                            text = "".join(
                                part.get("text", "")
                                for part in candidate.get("content", {}).get("parts", [])
                                if not part.get("thought")
                            )
                            if candidate.get("finishReason"):
                                completed = True
                        else:
                            choices = event.get("choices", [])
                            choice = choices[0] if choices else {}
                            text = text_content(choice.get("delta", {}).get("content"))
                        if text:
                            if result.ttft_s is None:
                                result.ttft_s = now
                            result.last_text_s = now
                            result.response_chars += len(text)
                            result.text_chunks += 1
                    if not completed:
                        raise RuntimeError("Stream closed without a completion marker")
                    if result.total_s is None:
                        result.total_s = time.perf_counter() - start
                if result.ttft_s is None:
                    raise RuntimeError("Model completed without any answer text")
                result.status = "ok"
    except Exception as exc:
        result.status = (
            "timeout" if isinstance(exc, (TimeoutError, httpx.TimeoutException)) else "error"
        )
        result.error = redact_error(exc) or type(exc).__name__
        result.total_s = None
    result.elapsed_s = time.perf_counter() - start
    return result


class GraphTiming:
    """Thread-safe token timestamps: timeout wrappers invoke models in worker threads."""

    def __init__(self, result: Measurement, start: float):
        self.result = result
        self.start = start
        self.lock = threading.Lock()

    def token(self, token: str, agent: str | None):
        if not token:
            return
        now = time.perf_counter() - self.start
        with self.lock:
            if self.result.first_llm_token_s is None:
                self.result.first_llm_token_s = now
            # Supervisor routing tokens are internal, not an answer to the user.
            if agent != "supervisor":
                if self.result.ttft_s is None:
                    self.result.ttft_s = now
                self.result.last_text_s = now
                self.result.text_chunks += 1
            if agent == "summary" and self.result.first_summary_token_s is None:
                self.result.first_summary_token_s = now


def instrument_llm(llm: Any, handler: Any) -> None:
    """Attach to the base model so callbacks survive BioAgents' thread wrappers."""
    seen = set()
    while "_llm" in vars(llm):
        if id(llm) in seen:
            raise RuntimeError("Cyclic LLM wrapper")
        seen.add(id(llm))
        llm = vars(llm)["_llm"]
    if not hasattr(llm, "streaming"):
        raise RuntimeError(f"Streaming instrumentation unsupported for {type(llm).__name__}")
    llm.streaming = True
    existing = llm.callbacks
    if existing is None or isinstance(existing, list):
        llm.callbacks = [*(existing or []), handler]
    else:
        callbacks = existing.copy()
        callbacks.add_handler(handler, inherit=True)
        llm.callbacks = callbacks


def measure_graph(provider: str, model: str, prompt: str) -> Measurement:
    # All BioAgents imports happen after .env loading and before graph creation.
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import AIMessage, HumanMessage

    from bioagents.llms import llm_provider

    result = Measurement("bioagents_graph")
    start = time.perf_counter()
    timing = GraphTiming(result, start)
    original_get_llm = llm_provider.get_llm

    class TokenHandler(BaseCallbackHandler):
        run_inline = True

        def __init__(self, agent):
            self.agent = agent

        def on_llm_new_token(self, token: str, **_kwargs):
            timing.token(token, self.agent)

    @wraps(original_get_llm)
    def timed_get_llm(*args, **kwargs):
        llm = original_get_llm(*args, **kwargs)
        agent = kwargs.get("prompt_name")
        if agent is None and len(args) > 3:
            agent = args[3]
        instrument_llm(llm, TokenHandler(agent))
        return llm

    llm_provider.set_llm_overrides(provider, model)
    llm_provider.get_llm = timed_get_llm
    executors = []
    executor_module = None
    original_create_executor = None
    try:
        from bioagents.sandbox import coder_executor as executor_module

        original_create_executor = executor_module.create_executor

        @wraps(original_create_executor)
        def tracked_create_executor(*args, **kwargs):
            executor = original_create_executor(*args, **kwargs)
            executors.append(executor)
            return executor

        executor_module.create_executor = tracked_create_executor
        # Agent modules import the instrumented factory in this fresh subprocess.
        from bioagents.graph import ALL_MEMBERS, create_graph
        from bioagents.graph_streaming import iter_graph_stream

        graph = create_graph()
        result.initialization_s = time.perf_counter() - start
        for step in iter_graph_stream(graph, {"messages": [HumanMessage(content=prompt)]}):
            if "__interrupt__" in step:
                raise RuntimeError("Workflow requires user input; measurement is incomplete")
            for node, output in step.items():
                if node == "user_input":
                    raise RuntimeError("Workflow requires user input; measurement is incomplete")
                if node not in {*ALL_MEMBERS, "summary"} or not isinstance(output, dict):
                    continue
                for message in output.get("messages", []):
                    if not isinstance(message, AIMessage):
                        continue
                    if message.additional_kwargs.get("show_ui", True) is False:
                        continue
                    text = text_content(message.content)
                    if text:
                        now = time.perf_counter() - start
                        if result.first_message_s is None:
                            result.first_message_s = now
                        result.response_chars += len(text)
        result.total_s = time.perf_counter() - start
        if result.first_message_s is None:
            raise RuntimeError("Workflow completed without an answer message")
        result.status = "ok"
    except Exception as exc:
        result.status = "timeout" if isinstance(exc, TimeoutError) else "error"
        result.error = redact_error(exc)
        result.total_s = None
    finally:
        result.elapsed_s = time.perf_counter() - start
        llm_provider.get_llm = original_get_llm
        llm_provider.set_llm_overrides()
        if executor_module is not None and original_create_executor is not None:
            executor_module.create_executor = original_create_executor
        for executor in reversed(executors):
            cleanup = getattr(executor, "cleanup", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception as exc:
                    print(
                        f"Benchmark executor cleanup failed: {redact_error(exc)}", file=sys.stderr
                    )
    return result


def _graph_worker(connection, provider, model, prompt):
    try:
        result = measure_graph(provider, model, prompt)
        connection.send(asdict(result))
    except Exception as exc:
        connection.send(
            asdict(Measurement("bioagents_graph", status="error", error=redact_error(exc)))
        )
    finally:
        connection.close()
        # A timed-out model's ThreadPoolExecutor must not hold up process exit.
        os._exit(0)


def measure_graph_isolated(provider, model, prompt, timeout) -> Measurement:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_graph_worker, args=(sender, provider, model, prompt))
    process.start()
    sender.close()
    try:
        if receiver.poll(timeout):
            try:
                return Measurement(**receiver.recv())
            except EOFError:
                return Measurement(
                    "bioagents_graph", status="error", error="Graph worker exited without a result"
                )
        return Measurement(
            "bioagents_graph",
            status="timeout",
            elapsed_s=timeout,
            error=f"Graph exceeded {timeout:g}s; no completed-response measurement",
        )
    finally:
        receiver.close()
        process.join(timeout=1)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join()


def websocket_url(base_url: str) -> str:
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https", "ws", "wss") or not parts.netloc:
        raise ValueError("BioAgents URL must be an http(s) or ws(s) URL")
    scheme = {"http": "ws", "https": "wss"}.get(parts.scheme, parts.scheme)
    path = parts.path.rstrip("/")
    if not path.endswith("/ws"):
        path += "/ws"
    return urlunsplit((scheme, parts.netloc, path, parts.query, ""))


async def measure_websocket(base_url, provider, model, prompt, timeout) -> Measurement:
    from websockets.asyncio.client import connect

    result = Measurement("bioagents_websocket")
    start = time.perf_counter()
    try:
        async with asyncio.timeout(timeout):
            async with connect(
                websocket_url(base_url), open_timeout=min(timeout, 15), max_size=None
            ) as ws:
                # Start at prompt submission, matching an already connected browser.
                start = time.perf_counter()
                await ws.send(
                    json.dumps(
                        {
                            "type": "query",
                            "content": prompt,
                            "provider": provider,
                            "model": model,
                        }
                    )
                )
                async for payload in ws:
                    event = json.loads(payload)
                    now = time.perf_counter() - start
                    kind = event.get("type")
                    if kind == "error":
                        raise RuntimeError(event.get("message", "BioAgents error"))
                    if kind in ("engagement_request", "tool_approval_request"):
                        raise RuntimeError(
                            f"Workflow requires user input ({kind}); measurement is incomplete"
                        )
                    if kind == "message":
                        text = text_content(event.get("content"))
                        if text:
                            if result.first_message_s is None:
                                result.first_message_s = now
                            result.last_text_s = now
                            result.response_chars += len(text)
                    if kind == "complete":
                        result.total_s = now
                        if result.first_message_s is None:
                            raise RuntimeError("Workflow completed without an answer message")
                        result.status = "ok"
                        break
                if result.status != "ok":
                    raise RuntimeError("Connection closed before complete")
    except Exception as exc:
        result.status = "timeout" if isinstance(exc, TimeoutError) else "error"
        result.error = redact_error(exc) or type(exc).__name__
        result.total_s = None
    result.elapsed_s = time.perf_counter() - start
    return result


def summarize(results: list[Measurement]) -> dict:
    summary = {}
    for target in dict.fromkeys(item.target for item in results):
        rows = [item for item in results if item.target == target]
        successful = [item for item in rows if item.status == "ok"]
        stats = {"successful_runs": len(successful), "failed_runs": len(rows) - len(successful)}
        for name in ("ttft_s", "first_summary_token_s", "first_message_s", "total_s"):
            values = sorted(
                getattr(row, name) for row in successful if getattr(row, name) is not None
            )
            if values:
                # Nearest-rank p95; with small samples this is normally the maximum.
                index = max(0, (95 * len(values) + 99) // 100 - 1)
                stats[name] = {
                    "median": statistics.median(values),
                    "p95": values[index],
                    "n": len(values),
                }
        summary[target] = stats
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-file", type=Path)
    parser.add_argument("--provider", choices=list(DEFAULT_MODELS))
    parser.add_argument(
        "--model", help="Same model is used for direct API and all BioAgents agents"
    )
    parser.add_argument("--target", choices=("both", "direct", "bioagents"), default="both")
    parser.add_argument("--bioagents-mode", choices=("graph", "websocket"), default="graph")
    parser.add_argument("--bioagents-url", default="http://127.0.0.1:8000")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--timeout", type=float, default=600, help="Maximum seconds per measurement"
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--output", type=Path, help="JSON path; CSV is saved alongside it")
    args = parser.parse_args()
    if args.runs < 1 or args.timeout <= 0:
        parser.error("--runs and --timeout must be positive")
    load_dotenv(args.env_file, override=False)
    provider = args.provider or os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider not in DEFAULT_MODELS:
        parser.error(f"Unsupported provider: {provider}")
    from bioagents.llms.llm_provider import _environment_model

    model = (args.model or "").strip() or _environment_model(provider) or DEFAULT_MODELS[provider]
    prompt = (
        args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding="utf-8")
    )
    if not prompt.strip():
        parser.error("Prompt must not be empty")
    output = (
        args.output
        or ROOT / "benchmark_results" / f"latency_{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}.json"
    )
    if output.suffix.lower() != ".json":
        parser.error("--output must end with .json; the CSV file is created alongside it")
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    print(f"Provider: {provider} | Model: {model} | Runs: {args.runs}", flush=True)
    print("TTFT = first answer text chunk; reasoning/empty/status chunks excluded.", flush=True)
    if args.target != "direct":
        print(f"BioAgents mode: {args.bioagents_mode}", flush=True)
        if args.bioagents_mode == "websocket":
            print(
                "WebSocket emits complete messages: TTFT is unavailable; first_message_s is measured.",
                flush=True,
            )
        else:
            print(
                "Graph TTFT excludes supervisor routing; summary TTFT is recorded separately.",
                flush=True,
            )
    for run in range(1, args.runs + 1):
        targets = ["direct", "bioagents"] if args.target == "both" else [args.target]
        # Alternate order to reduce a consistent first-run/order bias.
        if run % 2 == 0:
            targets.reverse()
        for target in targets:
            print(f"[{run}/{args.runs}] {target} starting...", flush=True)
            if target == "direct":
                result = asyncio.run(measure_direct(provider, model, prompt, args.timeout))
            elif args.bioagents_mode == "graph":
                result = measure_graph_isolated(provider, model, prompt, args.timeout)
            else:
                result = asyncio.run(
                    measure_websocket(args.bioagents_url, provider, model, prompt, args.timeout)
                )
            result.run = run
            results.append(result)
            print(json.dumps(asdict(result), ensure_ascii=False), flush=True)
            report = {
                "created_at": datetime.now(UTC).isoformat(),
                "provider": provider,
                "model": model,
                "prompt_chars": len(prompt),
                "bioagents_mode": args.bioagents_mode,
                "results": [asdict(item) for item in results],
                "summary": summarize(results),
            }
            # Persist after every measurement, even if a later measurement fails.
            output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(asdict(result)))
                writer.writeheader()
                writer.writerows(asdict(item) for item in results)
    print("\nSummary (seconds; successful runs only):")
    print(json.dumps(summarize(results), ensure_ascii=False, indent=2))
    print(f"\nSaved: {output}\nSaved: {output.with_suffix('.csv')}")
    return int(any(result.status != "ok" for result in results))


if __name__ == "__main__":
    sys.exit(main())
