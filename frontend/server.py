"""
BioAgents Web Server
FastAPI backend for the BioAgents UI
"""

import asyncio
import contextlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Import BioAgents components
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from bioagents.graph import ALL_MEMBERS, create_graph
from bioagents.graph_streaming import aiter_graph_stream, iter_graph_stream
from bioagents.limits import TOOL_APPROVAL_TIMEOUT_SEC
from bioagents.llms.llm_provider import set_api_keys_override
from frontend.workflow_routes import include_workflow_routes

# Graph agent nodes whose assistant output is streamed to the chat (excludes supervisor + *_tools).
STREAM_UI_AGENTS = frozenset([*ALL_MEMBERS, "summary"])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _PollFilter(logging.Filter):
    """Suppress repetitive polling access-log entries."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not ("GET /api/experiments/runs?limit=" in msg and "200" in msg)


logging.getLogger("uvicorn.access").addFilter(_PollFilter())

# =====================================================
# APP CONFIGURATION
# =====================================================

app = FastAPI(
    title="BioAgents API",
    description="Multi-agent AI system for computational biology research",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
FRONTEND_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")
include_workflow_routes(app)

# =====================================================
# MODELS
# =====================================================


class QueryRequest(BaseModel):
    query: str
    api_keys: dict[str, str] | None = None


class QueryResponse(BaseModel):
    success: bool
    messages: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
    audit_log: list[dict[str, Any]]


# =====================================================
# WEBSOCKET CONNECTION MANAGER
# =====================================================


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._disconnected: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self._disconnected.discard(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self._disconnected.add(websocket)
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    def is_connected(self, websocket: WebSocket) -> bool:
        return websocket not in self._disconnected

    async def send_json(self, websocket: WebSocket, data: dict):
        await websocket.send_json(data)

    async def broadcast(self, data: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.error(f"Failed to broadcast to connection: {e}")


manager = ConnectionManager()

# Session-based reference storage: {session_id: ReferenceManager}
_session_references: dict[str, Any] = {}

# =====================================================
# ARTIFACT GENERATION
# =====================================================

ARTIFACTS_DIR = Path("generated_artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file for processing by agents."""
    try:
        file_path = UPLOADS_DIR / file.filename
        with file_path.open("wb") as buffer:
            import shutil

            shutil.copyfileobj(file.file, buffer)

        # Return path relative to project root
        try:
            rel_path = file_path.relative_to(Path.cwd())
        except ValueError:
            rel_path = file_path

        return {
            "status": "success",
            "filename": file.filename,
            "path": str(rel_path),
            "size": file_path.stat().st_size,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {e!s}") from e


def extract_artifacts_from_content(content: str, agent_name: str) -> list:
    """
    Extract and save artifacts from agent response content.
    Generates files for tables, code blocks, analysis results, etc.
    """
    artifacts = []
    timestamp = datetime.now().strftime("%H%M%S")

    # Extract markdown tables and save as CSV
    table_pattern = r"\|(.+\|)+\n\|[-:\s|]+\|\n((?:\|.+\|\n?)+)"
    tables = re.findall(table_pattern, content)
    for i, _table_match in enumerate(tables):
        try:
            # Parse markdown table to CSV
            full_match = re.search(r"(\|.+\|)\n(\|[-:\s|]+\|)\n((?:\|.+\|\n?)+)", content)
            if full_match:
                header = full_match.group(1)
                rows = full_match.group(3)

                # Convert to CSV
                csv_lines = []
                header_cells = [c.strip() for c in header.split("|")[1:-1]]
                csv_lines.append(",".join(header_cells))

                for row in rows.strip().split("\n"):
                    cells = [c.strip() for c in row.split("|")[1:-1]]
                    csv_lines.append(",".join(cells))

                csv_content = "\n".join(csv_lines)
                filename = f"{agent_name}_table_{timestamp}_{i + 1}.csv"
                filepath = ARTIFACTS_DIR / filename
                filepath.write_text(csv_content)

                artifacts.append(
                    {
                        "name": filename,
                        "path": str(filepath),
                        "type": "csv",
                        "size": len(csv_content),
                        "preview": csv_content[:500],
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to extract table: {e}")

    # Extract code blocks and save
    code_pattern = r"```(\w+)?\n([\s\S]*?)```"
    code_blocks = re.findall(code_pattern, content)
    for i, (lang, code) in enumerate(code_blocks):
        if len(code.strip()) > 50:  # Only save substantial code
            ext = {
                "python": "py",
                "javascript": "js",
                "json": "json",
                "bash": "sh",
                "shell": "sh",
                "sql": "sql",
                "r": "r",
                "html": "html",
                "css": "css",
            }.get(lang.lower() if lang else "", "txt")

            filename = f"{agent_name}_code_{timestamp}_{i + 1}.{ext}"
            filepath = ARTIFACTS_DIR / filename
            filepath.write_text(code.strip())

            artifacts.append(
                {
                    "name": filename,
                    "path": str(filepath),
                    "type": ext,
                    "size": len(code),
                    "preview": code[:300],
                }
            )

    # Extract JSON data blocks
    json_pattern = r'\{[\s\S]*?"[^"]+"\s*:[\s\S]*?\}'
    json_matches = re.findall(json_pattern, content)
    for i, json_str in enumerate(json_matches):
        try:
            # Validate it's proper JSON
            parsed = json.loads(json_str)
            if len(json_str) > 100:  # Only save substantial JSON
                filename = f"{agent_name}_data_{timestamp}_{i + 1}.json"
                filepath = ARTIFACTS_DIR / filename
                filepath.write_text(json.dumps(parsed, indent=2))

                artifacts.append(
                    {
                        "name": filename,
                        "path": str(filepath),
                        "type": "json",
                        "size": len(json_str),
                        "preview": json.dumps(parsed, indent=2)[:300],
                    }
                )
        except json.JSONDecodeError:
            pass  # Not valid JSON

    # Save full response as a report if it's substantial
    if len(content) > 500 and agent_name in ["report", "analysis", "research"]:
        filename = f"{agent_name}_report_{timestamp}.md"
        filepath = ARTIFACTS_DIR / filename
        filepath.write_text(content)

        artifacts.append(
            {
                "name": filename,
                "path": str(filepath),
                "type": "md",
                "size": len(content),
                "preview": content[:400] + "...",
            }
        )

    # Check for FASTA sequences
    fasta_pattern = r"(>[^\n]+\n[A-Z\n]+)"
    fasta_matches = re.findall(fasta_pattern, content)
    for i, fasta in enumerate(fasta_matches):
        if len(fasta) > 50:
            filename = f"{agent_name}_sequence_{timestamp}_{i + 1}.fasta"
            filepath = ARTIFACTS_DIR / filename
            filepath.write_text(fasta)

            artifacts.append(
                {
                    "name": filename,
                    "path": str(filepath),
                    "type": "fasta",
                    "size": len(fasta),
                    "preview": fasta[:200],
                }
            )

    return artifacts


# =====================================================
# BIOAGENTS EXECUTION
# =====================================================


def extract_metrics_from_content(content: str) -> dict | None:
    """Extract metrics like loss and accuracy from agent content."""
    metrics = {}
    # Look for "METRIC: loss=0.5, accuracy=0.8" or similar
    metric_match = re.search(
        r"METRIC:.*?loss=([0-9\.]+).*?accuracy=([0-9\.]+)", content, re.IGNORECASE
    )
    if metric_match:
        try:
            metrics["loss"] = float(metric_match.group(1))
            metrics["accuracy"] = float(metric_match.group(2))
            return metrics
        except (ValueError, IndexError):
            pass
    return None


# =====================================================
# ROUTES
# =====================================================


@app.get("/")
async def root():
    """Serve the main UI."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "bioagents"}


@app.post("/api/query")
async def query_bioagents(request: QueryRequest):
    """
    Execute a BioAgents query via REST API with SSE streaming.
    """

    async def generate():
        set_api_keys_override(request.api_keys)
        from bioagents.references.reference_manager import ReferenceManager

        graph = create_graph()
        reference_manager = ReferenceManager()
        initial_state = {
            "messages": [HumanMessage(content=request.query)],
            "references": reference_manager,
        }

        full_audit = []

        try:
            for step in iter_graph_stream(graph, initial_state):
                node_name = next(iter(step))
                node_output = step[node_name]

                if node_output is None:
                    node_output = {}

                # Send agent update
                yield f"data: {json.dumps({'type': 'agent_update', 'agent': node_name})}\n\n"

                # Process messages
                step_messages = []
                if "messages" in node_output:
                    for m in node_output["messages"]:
                        msg_info = {
                            "type": m.__class__.__name__,
                            "content": m.content if hasattr(m, "content") else str(m),
                        }
                        step_messages.append(msg_info)

                        # Check for PDB files (local paths and URLs)
                        content = msg_info["content"]
                        if isinstance(content, str) and ".pdb" in content.lower():
                            # Check local paths
                            local_matches = re.findall(
                                r'(/[^\s\'"]+\.pdb|(?:\.\/)?[^\s\'"]+\.pdb|(?:[a-zA-Z0-9_\-]+/)+[^\s\'"]+\.pdb)',
                                content,
                            )
                            for path_str in local_matches:
                                path = Path(path_str)
                                if path.exists():
                                    try:
                                        with path.open() as f:
                                            pdb_content = f.read()
                                        yield f"data: {json.dumps({'type': 'structure', 'pdbContent': pdb_content})}\n\n"
                                        yield f"data: {json.dumps({'type': 'artifact', 'artifact': {'name': path.name, 'path': str(path), 'type': 'pdb', 'size': len(pdb_content)}})}\n\n"
                                    except Exception as e:
                                        logger.error(f"Failed to load PDB: {e}")

                            # Check for PDB URLs
                            url_matches = re.findall(r'(https?://[^\s\'"<>]+\.pdb)', content)
                            for url in url_matches:
                                try:
                                    resp = requests.get(url, timeout=30)
                                    if resp.status_code == 200:
                                        pdb_content = resp.text
                                        filename = url.split("/")[-1]
                                        yield f"data: {json.dumps({'type': 'structure', 'pdbContent': pdb_content})}\n\n"
                                        yield f"data: {json.dumps({'type': 'artifact', 'artifact': {'name': filename, 'path': url, 'type': 'pdb', 'size': len(pdb_content)}})}\n\n"
                                except Exception as e:
                                    logger.error(f"Failed to fetch PDB from URL: {e}")

                # Audit entry
                audit_entry = {
                    "agent": node_name,
                    "decision": node_output.get("next", "Continue"),
                    "messages": step_messages,
                }
                full_audit.append(audit_entry)
                yield f"data: {json.dumps({'type': 'audit', 'entries': full_audit})}\n\n"

                # Send tool calls and tool results (consumed by advanced mode UI)
                if "messages" in node_output:
                    for m in node_output["messages"]:
                        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                            for tc in m.tool_calls:
                                yield f"data: {json.dumps({'type': 'tool_call', 'agent': node_name, 'tool_name': tc.get('name', 'unknown'), 'arguments': tc.get('args', {})})}\n\n"
                        elif isinstance(m, ToolMessage) and m.content:
                            tc_content = m.content
                            if isinstance(tc_content, list):
                                tc_content = str(tc_content)
                            tool_name = getattr(m, "name", "") or "tool"
                            tc_content = (
                                tc_content[:3000]
                                if isinstance(tc_content, str)
                                else str(tc_content)[:3000]
                            )
                            yield f"data: {json.dumps({'type': 'tool_result', 'agent': node_name, 'tool_name': tool_name, 'content': tc_content})}\n\n"

                # Send code steps if available
                if "code_steps" in node_output:
                    yield f"data: {json.dumps({'type': 'code_execution', 'agent': node_name, 'steps': node_output['code_steps']})}\n\n"

                # Send messages
                if node_name in STREAM_UI_AGENTS and "messages" in node_output:
                    for m in node_output["messages"]:
                        if (
                            isinstance(m, AIMessage)
                            and m.content
                            and getattr(m, "additional_kwargs", {}).get("show_ui", True)
                            is not False
                        ):
                            content = m.content
                            if isinstance(content, list):
                                text_parts = [
                                    item if isinstance(item, str) else item.get("text", "")
                                    for item in content
                                    if isinstance(item, (str, dict))
                                ]
                                content = "\n\n".join(text_parts)
                            if content:
                                # Get references for this message if available
                                message_refs = []
                                if reference_manager:
                                    all_refs = reference_manager.get_all_references()
                                    if len(all_refs) > 0:
                                        message_refs = [ref.to_dict() for ref in all_refs]
                                    else:
                                        # If no references but this looks like a research report, create a synthetic one
                                        if (
                                            node_name in ["report", "research"]
                                            and len(content) > 500
                                        ):
                                            # Extract year mentions
                                            import uuid

                                            from bioagents.references.reference_types import (
                                                PaperReference,
                                            )

                                            years = re.findall(r"\b(20\d{2})\b", content)
                                            year_str = (
                                                f" ({min(years)}-{max(years)})"
                                                if len(years) > 1
                                                else f" ({years[0]})"
                                                if years
                                                else ""
                                            )

                                            synth_ref = PaperReference(
                                                id=f"ref_{uuid.uuid4().hex[:8]}",
                                                title=f"Scientific Literature Review{year_str}",
                                                abstract="This response synthesizes information from peer-reviewed literature and scientific databases.",
                                                url=None,
                                            )
                                            reference_manager.add_reference(synth_ref)
                                            message_refs = [synth_ref.to_dict()]

                                yield f"data: {json.dumps({'type': 'message', 'agent': node_name, 'content': content, 'references': message_refs})}\n\n"

            # Send completion with references
            completion_data = {"type": "complete"}
            if reference_manager:
                completion_data["all_references"] = reference_manager.to_dict()
            yield f"data: {json.dumps(completion_data)}\n\n"

        except Exception as e:
            logger.error(f"Query error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/download")
async def download_file(path: str):
    """Download a generated artifact."""
    # Handle URLs - redirect to the URL
    if path.startswith("http://") or path.startswith("https://"):
        from fastapi.responses import RedirectResponse

        return RedirectResponse(url=path)

    path_obj = Path(path)
    if not path_obj.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Security check - only allow files in certain directories
    import tempfile

    allowed_dirs = [
        str(Path.cwd()),
        str(ARTIFACTS_DIR.resolve()),
        tempfile.gettempdir(),
        str(Path.home()),
    ]

    abs_path = str(path_obj.resolve())
    if not any(abs_path.startswith(d) for d in allowed_dirs):
        raise HTTPException(status_code=403, detail="Access denied")

    # Determine media type based on extension
    import mimetypes

    content_type, _ = mimetypes.guess_type(path)
    if not content_type:
        content_type = "application/octet-stream"

    return FileResponse(path, filename=path_obj.name, media_type=content_type)


@app.get("/api/artifact/{filename}")
async def get_artifact_content(filename: str):
    """Get artifact content for preview."""
    filepath = ARTIFACTS_DIR / filename

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")

    content = filepath.read_text()
    file_type = filepath.suffix[1:] if filepath.suffix else "txt"

    return {"name": filename, "type": file_type, "content": content, "size": len(content)}


@app.post("/api/clear-artifacts")
async def clear_artifacts():
    """Clear all generated artifacts for a fresh session."""
    try:
        count = 0
        for f in ARTIFACTS_DIR.glob("*"):
            if f.is_file():
                f.unlink()
                count += 1
        return {"status": "success", "cleared": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# =====================================================
# WEBSOCKET ENDPOINT
# =====================================================


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication with steering support."""
    await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "query":
                query = data.get("content", "")
                api_keys = data.get("api_keys")
                session_id = data.get("session_id")
                if query:
                    # Create a steering queue for this query execution
                    steering_queue: asyncio.Queue[str] = asyncio.Queue()

                    # Create an approval response queue for tool approval flow
                    approval_queue: asyncio.Queue[dict] = asyncio.Queue()

                    # Create an engagement response queue for user engagement flow
                    engagement_queue: asyncio.Queue[dict] = asyncio.Queue()

                    # Start a concurrent reader that puts steering/approval messages on their queues
                    async def _steering_reader(
                        _sq: asyncio.Queue[str] = steering_queue,
                        _aq: asyncio.Queue[dict] = approval_queue,
                        _eq: asyncio.Queue[dict] = engagement_queue,
                    ) -> None:
                        try:
                            while True:
                                msg = await websocket.receive_json()
                                if msg.get("type") == "steer":
                                    await _sq.put(msg.get("content", ""))
                                elif msg.get("type") == "tool_approval_response":
                                    await _aq.put(msg)
                                elif msg.get("type") == "engagement_response":
                                    await _eq.put(msg)
                                elif msg.get("type") == "ping":
                                    await websocket.send_json({"type": "pong"})
                        except WebSocketDisconnect:
                            raise
                        except Exception:  # nosec B110
                            pass  # Connection closed or other error

                    reader_task = asyncio.create_task(_steering_reader())
                    try:
                        await run_bioagents_streaming(
                            query,
                            websocket,
                            api_keys=api_keys,
                            session_id=session_id,
                            steering_queue=steering_queue,
                            approval_queue=approval_queue,
                            engagement_queue=engagement_queue,
                        )
                    except WebSocketDisconnect:
                        raise
                    except Exception as e:
                        logger.error(f"Query execution error: {e}")
                        with contextlib.suppress(Exception):
                            await websocket.send_json({"type": "error", "message": str(e)})
                    finally:
                        reader_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await reader_task

            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def run_bioagents_streaming(
    query: str,
    websocket: WebSocket,
    api_keys: dict[str, str] | None = None,
    session_id: str | None = None,
    steering_queue: asyncio.Queue[str] | None = None,
    approval_queue: asyncio.Queue[dict] | None = None,
    engagement_queue: asyncio.Queue[dict] | None = None,
):
    """Execute BioAgents query with WebSocket streaming and optional steering/approval support."""
    import uuid

    from langgraph.checkpoint.memory import MemorySaver

    from bioagents.references.reference_extractor import extract_references_from_messages
    from bioagents.references.reference_manager import ReferenceManager
    from bioagents.tools.tool_policy import ToolPolicy

    set_api_keys_override(api_keys)

    # Build a session-specific tool policy with available API keys
    available_keys: set[str] = set()
    if api_keys:
        available_keys = {k for k, v in api_keys.items() if v}
    session_policy = ToolPolicy(available_api_keys=available_keys)

    # Track whether the client is still connected
    _client_disconnected = False

    async def safe_send(data: dict) -> bool:
        """Send JSON to the WebSocket. Returns False if client is gone."""
        nonlocal _client_disconnected
        if _client_disconnected or not manager.is_connected(websocket):
            _client_disconnected = True
            return False
        try:
            await websocket.send_json(data)
            return True
        except WebSocketDisconnect:
            _client_disconnected = True
            return False
        except Exception:
            _client_disconnected = True
            return False

    # Create graph with checkpointer to support mid-execution state updates (steering)
    checkpointer = MemorySaver()
    await safe_send({"type": "log", "message": "Initializing agents..."})
    graph = await asyncio.to_thread(create_graph, checkpointer=checkpointer, policy=session_policy)

    # Thread ID for checkpoint-based state updates
    thread_id = f"session_{session_id or uuid.uuid4().hex[:8]}"
    stream_config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
    }

    # Initialize with reference manager
    # Note: ReferenceManager is NOT included in initial_state because MemorySaver
    # uses msgpack serialization which can't handle arbitrary Python objects.
    # References are extracted in the streaming loop instead.
    reference_manager = ReferenceManager()
    initial_state: dict[str, Any] = {
        "messages": [HumanMessage(content=query)],
    }

    full_audit: list[dict[str, Any]] = []
    sent_artifacts: set[str] = set()
    sent_structures: set[str] = set()
    sent_protein_ids: set[str] = set()

    def extract_protein_id(name_or_url: str) -> str | None:
        """Extract protein ID from filename or URL (e.g., AF-P01308-F1 -> P01308)"""
        match = re.search(r"([A-Z][A-Z0-9]{4,5})", name_or_url, re.IGNORECASE)
        return match.group(1).upper() if match else None

    def _drain_steering_queue() -> list[str]:
        """Non-blocking drain of all pending steering messages."""
        messages: list[str] = []
        if steering_queue is None:
            return messages
        while not steering_queue.empty():
            try:
                messages.append(steering_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return messages

    def _extract_engagement_from_interrupt(
        interrupt_value: Any,
    ) -> dict[str, Any] | None:
        """Convert an interrupt payload to a frontend-friendly engagement dict."""
        if isinstance(interrupt_value, dict):
            return {
                "id": interrupt_value.get("id", "unknown"),
                "engagement_type": interrupt_value.get("type", "clarification"),
                "question": interrupt_value.get("question", ""),
                "options": interrupt_value.get("options", []),
                "context": interrupt_value.get("context", ""),
                "agent": interrupt_value.get("agent", "Supervisor"),
            }
        return None

    try:
        from langgraph.types import Command

        _stream_input = initial_state

        while True:
            resume_after_engagement = False

            async for step in aiter_graph_stream(graph, _stream_input, config=stream_config):
                # Detect LangGraph interrupt steps
                if "__interrupt__" in step:
                    interrupts = step["__interrupt__"]
                    if interrupts:
                        interrupt_value = (
                            interrupts[0].value
                            if hasattr(interrupts[0], "value")
                            else interrupts[0]
                        )
                        engagement_data = _extract_engagement_from_interrupt(interrupt_value)
                        if engagement_data:
                            await safe_send({"type": "agent_update", "agent": "user_input"})
                            if not await safe_send(
                                {"type": "engagement_request", **engagement_data}
                            ):
                                break
                            logger.info(
                                f"Engagement request sent: type={engagement_data.get('engagement_type')}, "
                                f"question={engagement_data.get('question', '')[:80]}"
                            )
                            try:
                                response = await asyncio.wait_for(
                                    engagement_queue.get()
                                    if engagement_queue
                                    else asyncio.Future(),
                                    timeout=TOOL_APPROVAL_TIMEOUT_SEC,
                                )
                                resume_payload = {
                                    "content": response.get("content", ""),
                                    "selected_option": response.get("selected_option"),
                                }
                                if not await safe_send(
                                    {
                                        "type": "engagement_response_received",
                                        "id": engagement_data.get("id"),
                                    }
                                ):
                                    break
                                logger.info(
                                    f"Engagement response received: {resume_payload['content'][:80]}"
                                )
                            except TimeoutError:
                                resume_payload = {"content": ""}
                                if not await safe_send(
                                    {
                                        "type": "engagement_timeout",
                                        "id": engagement_data.get("id"),
                                    }
                                ):
                                    break
                                logger.info("Engagement timed out, proceeding with best judgment")

                            _stream_input = Command(resume=resume_payload)
                            resume_after_engagement = True
                    break  # Exit inner for-loop; outer while-loop will resume

                node_name = next(iter(step))
                node_output = step[node_name]

                if node_output is None:
                    node_output = {}

                # Send agent update — bail out if client is gone
                if not await safe_send({"type": "agent_update", "agent": node_name}):
                    break

                # Process messages
                step_messages = []
                if "messages" in node_output:
                    for m in node_output["messages"]:
                        if _client_disconnected:
                            break
                        msg_info = {
                            "type": m.__class__.__name__,
                            "content": m.content if hasattr(m, "content") else str(m),
                        }
                        step_messages.append(msg_info)

                        # Check for PDB files (local paths and URLs)
                        content = msg_info["content"]
                        if isinstance(content, str) and ".pdb" in content.lower():
                            # Check for local file paths
                            local_matches = re.findall(
                                r'(/[^\s\'"]+\.pdb|(?:\.\/)?[^\s\'"]+\.pdb|(?:[a-zA-Z0-9_\-]+/)+[^\s\'"]+\.pdb)',
                                content,
                            )
                            for path_str in local_matches:
                                path = Path(path_str)
                                if path.exists() and str(path) not in sent_structures:
                                    sent_structures.add(str(path))
                                    try:
                                        with path.open() as f:
                                            pdb_content = f.read()
                                        if not await safe_send(
                                            {"type": "structure", "pdbContent": pdb_content}
                                        ):
                                            break
                                        artifact_key = path.name
                                        if artifact_key not in sent_artifacts:
                                            sent_artifacts.add(artifact_key)
                                            if not await safe_send(
                                                {
                                                    "type": "artifact",
                                                    "artifact": {
                                                        "name": path.name,
                                                        "path": str(path),
                                                        "type": "pdb",
                                                        "size": path.stat().st_size,
                                                    },
                                                }
                                            ):
                                                break
                                    except Exception as e:
                                        logger.error(f"Failed to load PDB: {e}")

                            # Check for PDB URLs (AlphaFold, RCSB, etc.)
                            url_matches = re.findall(r'(https?://[^\s\'"<>]+\.pdb)', content)
                            for url in url_matches:
                                if _client_disconnected:
                                    break
                                # Extract protein ID to prevent duplicates
                                protein_id = extract_protein_id(url)

                                # Skip if we already sent this URL or protein
                                if url in sent_structures:
                                    continue
                                if protein_id and protein_id in sent_protein_ids:
                                    continue

                                sent_structures.add(url)
                                if protein_id:
                                    sent_protein_ids.add(protein_id)

                                try:
                                    logger.info(f"Fetching PDB from URL: {url}")
                                    resp = requests.get(url, timeout=30)
                                    if resp.status_code == 200:
                                        pdb_content = resp.text
                                        # Extract filename from URL
                                        filename = url.split("/")[-1]
                                        if not await safe_send(
                                            {"type": "structure", "pdbContent": pdb_content}
                                        ):
                                            break
                                        if filename not in sent_artifacts:
                                            sent_artifacts.add(filename)
                                            if not await safe_send(
                                                {
                                                    "type": "artifact",
                                                    "artifact": {
                                                        "name": filename,
                                                        "path": url,
                                                        "type": "pdb",
                                                        "size": len(pdb_content),
                                                    },
                                                }
                                            ):
                                                break
                                            if not await safe_send(
                                                {
                                                    "type": "log",
                                                    "message": f"Loaded 3D structure: {filename}",
                                                }
                                            ):
                                                break
                                except Exception as e:
                                    logger.error(f"Failed to fetch PDB from URL {url}: {e}")

                        # Check for other artifacts
                        if isinstance(content, str):
                            for ext in [".csv", ".png", ".jpg", ".json", ".pdf"]:
                                if ext in content.lower():
                                    matches = re.findall(
                                        rf'(/[^\s\'"]+{ext}|(?:\.\/)?[^\s\'"]+{ext}|(?:[a-zA-Z0-9_\-]+/)+[^\s\'"]+{ext})',
                                        content,
                                        re.IGNORECASE,
                                    )
                                    for path_str in matches:
                                        path = Path(path_str)
                                        if path.exists() and not await safe_send(
                                            {
                                                "type": "artifact",
                                                "artifact": {
                                                    "name": path.name,
                                                    "path": str(path),
                                                    "type": ext[1:],
                                                    "size": path.stat().st_size,
                                                },
                                            }
                                        ):
                                            break

                if _client_disconnected:
                    break

                # Build audit entry
                audit_entry = {
                    "agent": node_name,
                    "decision": node_output.get("next", "Continue"),
                    "reasoning": node_output.get("reasoning", ""),
                    "messages": step_messages,
                }
                full_audit.append(audit_entry)
                if not await safe_send({"type": "audit", "entries": full_audit}):
                    break

                # Extract references from step messages (ReferenceManager is kept
                # outside the graph state to avoid msgpack serialization errors).
                if "messages" in node_output:
                    refs = extract_references_from_messages(node_output["messages"])
                    if refs:
                        reference_manager.add_references(refs)

                # Send tool calls and tool results (consumed by advanced mode UI)
                if "messages" in node_output:
                    for m in node_output["messages"]:
                        if _client_disconnected:
                            break
                        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                            for tc in m.tool_calls:
                                if not await safe_send(
                                    {
                                        "type": "tool_call",
                                        "agent": node_name,
                                        "tool_name": tc.get("name", "unknown"),
                                        "arguments": tc.get("args", {}),
                                    }
                                ):
                                    break
                        elif isinstance(m, ToolMessage) and m.content:
                            tc_content = m.content
                            if isinstance(tc_content, list):
                                tc_content = str(tc_content)
                            tool_name = getattr(m, "name", "") or "tool"

                            # Check for approval-required or policy-blocked messages
                            if isinstance(tc_content, str):
                                if "[APPROVAL_REQUIRED]" in tc_content:
                                    # Extract request_id from message
                                    import re as _re

                                    req_match = _re.search(
                                        r"approval_request_id=([a-f0-9-]+)", tc_content
                                    )
                                    request_id = (
                                        req_match.group(1) if req_match else str(uuid.uuid4())
                                    )
                                    reason_match = _re.search(r"Reason: (.+?)\. Risk", tc_content)
                                    reason = (
                                        reason_match.group(1)
                                        if reason_match
                                        else "External API tool"
                                    )
                                    risk_match = _re.search(r"Risk level: (\w+)", tc_content)
                                    risk_level = risk_match.group(1) if risk_match else "medium"

                                    if not await safe_send(
                                        {
                                            "type": "tool_approval_request",
                                            "request_id": request_id,
                                            "tool_name": tool_name,
                                            "agent": node_name,
                                            "reason": reason,
                                            "risk_level": risk_level,
                                            "content": tc_content[:3000],
                                        }
                                    ):
                                        break

                                    # Drain any approval responses that arrived
                                    if approval_queue is not None:
                                        while not approval_queue.empty():
                                            try:
                                                resp = approval_queue.get_nowait()
                                                if resp.get("tool_name") == tool_name and resp.get(
                                                    "approved"
                                                ):
                                                    session_policy.mark_approved(tool_name)
                                                    await safe_send(
                                                        {
                                                            "type": "log",
                                                            "message": f"Tool '{tool_name}' approved for this session",
                                                        }
                                                    )
                                            except asyncio.QueueEmpty:
                                                break
                                    continue

                                if "[POLICY_BLOCKED]" in tc_content:
                                    if not await safe_send(
                                        {
                                            "type": "tool_policy_blocked",
                                            "agent": node_name,
                                            "tool_name": tool_name,
                                            "content": tc_content[:3000],
                                        }
                                    ):
                                        break
                                    continue

                            if not await safe_send(
                                {
                                    "type": "tool_result",
                                    "agent": node_name,
                                    "tool_name": tool_name,
                                    "content": tc_content[:3000]
                                    if isinstance(tc_content, str)
                                    else str(tc_content)[:3000],
                                }
                            ):
                                break

                if _client_disconnected:
                    break

                # Send code steps if available (from coder, ml, or dl agents)
                if "code_steps" in node_output and not await safe_send(
                    {
                        "type": "code_execution",
                        "agent": node_name,
                        "steps": node_output["code_steps"],
                    }
                ):
                    break

                # Send tool policy stats periodically
                stats = session_policy.stats
                if not await safe_send(
                    {
                        "type": "tool_policy_stats",
                        "auto_approved": stats.auto_approved,
                        "user_approved": stats.user_approved,
                        "blocked": stats.blocked,
                        "filtered_at_discovery": stats.filtered_at_discovery,
                    }
                ):
                    break

                # Send messages from user-facing agents
                if node_name in STREAM_UI_AGENTS and "messages" in node_output:
                    for m in node_output["messages"]:
                        if _client_disconnected:
                            break
                        # Skip HumanMessages (usually handoffs) and ToolMessages
                        if (
                            hasattr(m, "content")
                            and m.content
                            and not isinstance(m, (HumanMessage, ToolMessage))
                            and getattr(m, "additional_kwargs", {}).get("show_ui", True)
                            is not False
                        ):
                            content = m.content
                            if isinstance(content, list):
                                text_parts = [
                                    item if isinstance(item, str) else item.get("text", "")
                                    for item in content
                                    if isinstance(item, (str, dict))
                                ]
                                content = "\n\n".join(text_parts)
                            if content:
                                # Get references for this message if available
                                message_refs = []
                                if reference_manager:
                                    all_refs = reference_manager.get_all_references()
                                    if len(all_refs) > 0:
                                        message_refs = [ref.to_dict() for ref in all_refs]
                                    else:
                                        # If no references but this looks like a research report, create a synthetic one
                                        if (
                                            node_name in ["report", "research"]
                                            and len(content) > 500
                                        ):
                                            # Extract year mentions
                                            import uuid

                                            from bioagents.references.reference_types import (
                                                PaperReference,
                                            )

                                            years = re.findall(r"\b(20\d{2})\b", content)
                                            year_str = (
                                                f" ({min(years)}-{max(years)})"
                                                if len(years) > 1
                                                else f" ({years[0]})"
                                                if years
                                                else ""
                                            )

                                            synth_ref = PaperReference(
                                                id=f"ref_{uuid.uuid4().hex[:8]}",
                                                title=f"Scientific Literature Review{year_str}",
                                                abstract="This response synthesizes information from peer-reviewed literature and scientific databases.",
                                                url=None,
                                            )
                                            reference_manager.add_reference(synth_ref)
                                            message_refs = [synth_ref.to_dict()]

                                if not await safe_send(
                                    {
                                        "type": "message",
                                        "agent": node_name,
                                        "content": content,
                                        "references": message_refs,
                                    }
                                ):
                                    break

                                # Extract metrics
                                metrics = extract_metrics_from_content(content)
                                if metrics and not await safe_send(
                                    {"type": "metrics", "metrics": metrics}
                                ):
                                    break

                                # Extract and send artifacts from content (with deduplication)
                                extracted_artifacts = extract_artifacts_from_content(
                                    content, node_name
                                )
                                for artifact in extracted_artifacts:
                                    if _client_disconnected:
                                        break
                                    if artifact["name"] not in sent_artifacts:
                                        sent_artifacts.add(artifact["name"])
                                        if not await safe_send(
                                            {"type": "artifact", "artifact": artifact}
                                        ):
                                            break
                                        if not await safe_send(
                                            {
                                                "type": "log",
                                                "message": f"Generated: {artifact['name']}",
                                            }
                                        ):
                                            break

            if _client_disconnected:
                break

            # --- Steering injection: drain queue and inject into graph state ---
            steering_texts = _drain_steering_queue()
            for steering_text in steering_texts:
                steering_msg = HumanMessage(content=f"[USER STEERING] {steering_text}")
                graph.update_state(
                    stream_config,
                    {"messages": [steering_msg]},
                )
                if not await safe_send({"type": "steering_received", "content": steering_text}):
                    break
                logger.info(f"Steering message injected: {steering_text[:80]}")

            await asyncio.sleep(0.05)

            # If we broke out of the inner loop, stop the outer loop too
            if _client_disconnected:
                break

            # Persist references for this session
            if session_id and reference_manager:
                _session_references[session_id] = reference_manager

            # Send completion with all references (only if still connected)
            if not _client_disconnected:
                completion_data = {"type": "complete", "artifacts": [], "audit_log": full_audit}
                if reference_manager:
                    completion_data["all_references"] = reference_manager.to_dict()

                await safe_send(completion_data)

            # Continue outer while-loop if we broke out to resume after an interrupt
            if resume_after_engagement and not _client_disconnected:
                continue
            break  # Normal stream completion — exit outer while-loop

    except WebSocketDisconnect:
        # Client disconnected cleanly during streaming — nothing to send back.
        raise
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        if not _client_disconnected:
            with contextlib.suppress(Exception):
                await websocket.send_json({"type": "error", "message": str(e)})


# =====================================================
# EXPERIMENT ENDPOINTS
# =====================================================

from bioagents.benchmarks.experiment_runner import (  # noqa: E402
    DEFAULT_RUNS_DIR,
    list_runs,
    load_run,
    run_experiment,
)
from bioagents.benchmarks.use_case_loader import (  # noqa: E402
    filter_use_cases,
    load_experiment_configs_from_file,
    load_use_cases_from_dir,
)
from bioagents.benchmarks.use_case_models import ExperimentConfig  # noqa: E402


class ExperimentRunRequest(BaseModel):
    use_case_ids: list[str] | None = None  # None = run all loaded use cases
    config: dict | None = None  # Serialised ExperimentConfig dict; None = defaults
    skip_judge: bool = False
    use_llm_judge: bool = False  # Use LLM judge vs heuristic


_USE_CASES_CACHE: list | None = None
_CONFIGS_CACHE: list | None = None


def _get_use_cases():
    global _USE_CASES_CACHE
    if _USE_CASES_CACHE is None:
        _USE_CASES_CACHE = load_use_cases_from_dir()
    return _USE_CASES_CACHE


def _get_experiment_configs():
    global _CONFIGS_CACHE
    if _CONFIGS_CACHE is None:
        configs_path = DEFAULT_RUNS_DIR.parent / "use_cases" / "experiment_configs.yaml"
        if configs_path.exists():
            _CONFIGS_CACHE = load_experiment_configs_from_file(configs_path)
        else:
            _CONFIGS_CACHE = [ExperimentConfig(name="default")]
    return _CONFIGS_CACHE


@app.get("/api/experiments/use-cases")
async def get_use_cases(category: str | None = None, tags: str | None = None):
    """List all available use cases, optionally filtered."""
    use_cases = _get_use_cases()
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    filtered = filter_use_cases(use_cases, tags=tag_list, category=category)
    return {"use_cases": [uc.to_dict() for uc in filtered], "total": len(filtered)}


@app.get("/api/experiments/configs")
async def get_experiment_configs():
    """List available experiment configurations."""
    configs = _get_experiment_configs()
    return {"configs": [c.to_dict() for c in configs]}


@app.post("/api/experiments/run")
async def start_experiment_run(request: ExperimentRunRequest):
    """
    Run an experiment asynchronously in a background thread.

    Returns immediately with a run_id; poll /api/experiments/runs/{run_id}
    for results (they are persisted to disk when finished).
    """
    import asyncio

    use_cases = _get_use_cases()
    if request.use_case_ids is not None:
        from bioagents.benchmarks.use_case_loader import filter_use_cases as _filter

        use_cases = _filter(use_cases, ids=request.use_case_ids)

    if not use_cases:
        raise HTTPException(status_code=400, detail="No use cases found for the given ids.")

    config = (
        ExperimentConfig.from_dict(request.config)
        if request.config
        else ExperimentConfig(name="default")
    )

    judge_llm = None
    if not request.skip_judge and request.use_llm_judge:
        try:
            from bioagents.llms.llm_provider import get_llm

            judge_llm = get_llm()
        except Exception as exc:
            logger.warning("Could not instantiate judge LLM: %s. Using heuristic scoring.", exc)

    import uuid as _uuid

    pending_run_id = _uuid.uuid4().hex

    async def _run_in_background():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: run_experiment(
                use_cases=use_cases,
                config=config,
                skip_judge=request.skip_judge,
                judge_llm=judge_llm,
                show_trace=False,
                save_results=True,
                run_id=pending_run_id,
            ),
        )

    background_task = asyncio.create_task(_run_in_background())
    background_task.add_done_callback(
        lambda t: t.exception() if not t.cancelled() else None
    )  # consume result so exceptions don't go unobserved
    return {"status": "started", "run_id": pending_run_id, "use_case_count": len(use_cases)}


@app.get("/api/experiments/runs")
async def get_experiment_runs(limit: int = 20):
    """List recent experiment runs (summary only)."""
    runs = list_runs(limit=limit)
    return {"runs": runs, "total": len(runs)}


@app.get("/api/experiments/runs/{run_id}")
async def get_experiment_run(run_id: str):
    """Return full details for a specific experiment run."""
    data = load_run(run_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return data


@app.delete("/api/experiments/use-cases/cache")
async def invalidate_use_cases_cache():
    """Force reload of use case definitions from disk."""
    global _USE_CASES_CACHE, _CONFIGS_CACHE
    _USE_CASES_CACHE = None
    _CONFIGS_CACHE = None
    return {"status": "cache cleared"}


# =====================================================
# DEMO ENDPOINTS
# =====================================================


@app.get("/api/demo/structure/{identifier}")
async def demo_structure(identifier: str):
    """
    Demo endpoint to fetch and return a protein structure for 3D viewing.
    Supports UniProt IDs (e.g., P04637) or PDB IDs (e.g., 1YCR).
    """
    # Check if it's a UniProt ID (format: letter + 5 alphanumeric)
    is_uniprot = len(identifier) == 6 and identifier[0].isalpha()

    try:
        if is_uniprot:
            # AlphaFold structure
            url = f"https://alphafold.ebi.ac.uk/files/AF-{identifier}-F1-model_v4.pdb"
            response = requests.get(url, timeout=30)
            if response.status_code == 404:
                # Try v3
                url = f"https://alphafold.ebi.ac.uk/files/AF-{identifier}-F1-model_v3.pdb"
                response = requests.get(url, timeout=30)
        else:
            # PDB structure
            url = f"https://files.rcsb.org/download/{identifier.upper()}.pdb"
            response = requests.get(url, timeout=30)

        if response.status_code != 200:
            raise HTTPException(status_code=404, detail=f"Structure not found for {identifier}")

        return {
            "status": "success",
            "identifier": identifier,
            "source": "AlphaFold" if is_uniprot else "RCSB PDB",
            "pdbContent": response.text,
        }

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/demo/artifacts")
async def demo_artifacts():
    """
    Return list of demo artifacts that exist in the project.
    """
    demo_files = []

    # Check for PDB files in common locations
    search_paths = [
        Path.cwd(),
        Path.cwd() / "data",
        Path.cwd() / "binder_designs",
    ]

    for search_path in search_paths:
        if search_path.exists():
            for ext in ["*.pdb", "*.csv", "*.png", "*.json"]:
                for f in search_path.glob(ext):
                    demo_files.append(
                        {
                            "name": f.name,
                            "path": str(f),
                            "type": f.suffix[1:],
                            "size": f.stat().st_size,
                        }
                    )

    return {"artifacts": demo_files[:20]}  # Limit to 20


@app.get("/api/pdb/{filename}")
async def get_pdb_content(filename: str):
    """
    Return the content of a PDB file for 3D viewing.
    """
    # Search for the file in common locations
    search_paths = [
        Path.cwd() / filename,
        Path.cwd() / "data" / filename,
        Path.cwd() / "binder_designs" / filename,
    ]

    for path in search_paths:
        if path.exists() and path.suffix.lower() == ".pdb":
            return {"status": "success", "filename": filename, "pdbContent": path.read_text()}

    raise HTTPException(status_code=404, detail=f"PDB file not found: {filename}")


# References endpoint
@app.get("/api/references")
async def get_references(session_id: str | None = None):
    """Return references accumulated during a session."""
    if session_id and session_id in _session_references:
        ref_mgr = _session_references[session_id]
        return ref_mgr.to_dict()
    return {
        "references": [],
        "by_type": {
            "papers": [],
            "databases": [],
            "tools": [],
            "structures": [],
            "artifacts": [],
        },
        "display_numbers": {},
        "count": 0,
    }


# =====================================================
# AGENT & TOOL REGISTRY API
# =====================================================

AGENT_REGISTRY = {
    "research": {
        "name": "Research Agent",
        "category": "Research & Knowledge",
        "description": "Fetches biological data from databases (UniProt, PDB, ToolUniverse)",
        "tools": [
            "fetch_uniprot_fasta",
            "tool_universe_find_tools",
            "tool_universe_call_tool",
            "fetch_webpage_as_pdf_text",
            "extract_pdf_text_spacy_layout",
            "fetch_alphafold_structure",
            "fetch_pdb_structure",
            "download_structure_file",
        ],
        "status": "active",
    },
    "analysis": {
        "name": "Analysis Agent",
        "category": "Computation",
        "description": "Analyzes protein sequences and calculates biochemical properties",
        "tools": [
            "calculate_molecular_weight",
            "analyze_amino_acid_composition",
            "calculate_isoelectric_point",
        ],
        "status": "active",
    },
    "coder": {
        "name": "Coder Agent",
        "category": "Computation",
        "description": "Writes and executes Python code, creates visualizations",
        "tools": ["python_executor"],
        "status": "active",
    },
    "ml": {
        "name": "ML Agent",
        "category": "Computation",
        "description": "Traditional machine learning on tabular data",
        "tools": ["python_executor"],
        "status": "active",
    },
    "dl": {
        "name": "Deep Learning Agent",
        "category": "Computation",
        "description": "Neural networks, transformers, GNNs for biological data",
        "tools": ["python_executor"],
        "status": "active",
    },
    "report": {
        "name": "Report Agent",
        "category": "Quality",
        "description": "Synthesizes findings into comprehensive reports",
        "tools": [],
        "status": "active",
    },
    "critic": {
        "name": "Critic Agent",
        "category": "Quality",
        "description": "Validates scientific accuracy and biological plausibility",
        "tools": [],
        "status": "active",
    },
    "summary": {
        "name": "Summary Agent",
        "category": "Quality",
        "description": "Final user-facing output with workflow summaries",
        "tools": [],
        "status": "active",
    },
    "tool_builder": {
        "name": "Tool Builder Agent",
        "category": "Meta",
        "description": "Creates custom tools when existing tools are insufficient",
        "tools": [
            "create_tool",
            "search_custom_tools",
            "execute_custom_tool",
            "research_tool_documentation",
        ],
        "status": "active",
    },
    "protein_design": {
        "name": "Protein Design Agent",
        "category": "Domain",
        "description": "Computational protein design and structure analysis",
        "tools": [
            "fetch_target_structure",
            "analyze_interface",
            "run_rfdiffusion",
            "run_proteinmpnn",
            "predict_structure",
            "evaluate_binders",
        ],
        "status": "active",
    },
    "literature": {
        "name": "Literature Agent",
        "category": "Research & Knowledge",
        "description": "Searches and summarizes scientific literature",
        "tools": ["search_pubmed", "search_arxiv", "search_biorxiv", "fetch_paper_metadata"],
        "status": "active",
    },
    "web_browser": {
        "name": "Web Browser Agent",
        "category": "Research & Knowledge",
        "description": "Navigates web, fetches documentation, extracts data",
        "tools": ["fetch_url_content", "search_google_scholar", "download_file_from_url"],
        "status": "active",
    },
    "paper_replication": {
        "name": "Paper Replication Agent",
        "category": "Research & Knowledge",
        "description": "Orchestrates full paper replication workflows",
        "tools": [
            "fetch_url_content",
            "search_google_scholar",
            "download_file_from_url",
            "git_clone_repo",
            "list_repo_files",
            "read_repo_file",
        ],
        "status": "active",
    },
    "data_acquisition": {
        "name": "Data Acquisition Agent",
        "category": "Research & Knowledge",
        "description": "Downloads and manages datasets from biological databases",
        "tools": [
            "fetch_url_content",
            "download_file_from_url",
            "read_local_file",
            "write_local_file",
            "list_local_directory",
        ],
        "status": "active",
    },
    "genomics": {
        "name": "Genomics Agent",
        "category": "Domain",
        "description": "Sequence alignment, variant calling, genome annotation",
        "tools": [
            "run_blast_search",
            "parse_fasta_file",
            "reverse_complement",
            "translate_dna",
            "calculate_gc_content",
        ],
        "status": "active",
    },
    "transcriptomics": {
        "name": "Transcriptomics Agent",
        "category": "Domain",
        "description": "RNA-seq, differential expression, gene set enrichment",
        "tools": [
            "run_differential_expression",
            "run_gene_set_enrichment",
            "normalize_expression_data",
        ],
        "status": "active",
    },
    "structural_biology": {
        "name": "Structural Biology Agent",
        "category": "Domain",
        "description": "Protein structure prediction and structural analysis",
        "tools": ["fetch_pdb_structure", "fetch_alphafold_structure", "download_structure_file"],
        "status": "active",
    },
    "phylogenetics": {
        "name": "Phylogenetics Agent",
        "category": "Domain",
        "description": "Phylogenetic tree construction and evolutionary analysis",
        "tools": ["run_blast_search", "parse_fasta_file", "reverse_complement", "translate_dna"],
        "status": "active",
    },
    "docking": {
        "name": "Docking Agent",
        "category": "Domain",
        "description": "Molecular docking and virtual screening guidance",
        "tools": [],
        "status": "active",
    },
    "planner": {
        "name": "Planner Agent",
        "category": "Meta",
        "description": "Breaks complex tasks into step-by-step execution plans",
        "tools": [],
        "status": "active",
    },
    "tool_validator": {
        "name": "Tool Validator Agent",
        "category": "Meta",
        "description": "Validates tool calls and detects wrong tool usage",
        "tools": [],
        "status": "active",
    },
    "tool_discovery": {
        "name": "Tool Discovery Agent",
        "category": "Meta",
        "description": "Searches for available tools and assigns them to agents",
        "tools": ["create_tool", "search_custom_tools", "execute_custom_tool"],
        "status": "active",
    },
    "prompt_optimizer": {
        "name": "Prompt Optimizer Agent",
        "category": "Meta",
        "description": "Analyzes agent failures and suggests prompt improvements",
        "tools": [],
        "status": "active",
    },
    "result_checker": {
        "name": "Result Checker Agent",
        "category": "Meta",
        "description": "Validates outputs for scientific accuracy and reproducibility",
        "tools": [],
        "status": "active",
    },
    "shell": {
        "name": "Shell Agent",
        "category": "Infrastructure",
        "description": "Executes shell commands in the sandbox",
        "tools": ["run_shell_command", "install_python_package", "check_installed_packages"],
        "status": "active",
    },
    "git": {
        "name": "Git Agent",
        "category": "Infrastructure",
        "description": "Manages git repositories",
        "tools": ["git_clone_repo", "list_repo_files", "read_repo_file", "git_checkout_branch"],
        "status": "active",
    },
    "environment": {
        "name": "Environment Agent",
        "category": "Infrastructure",
        "description": "Sets up execution environments",
        "tools": [
            "create_virtual_environment",
            "install_requirements",
            "check_gpu_available",
            "get_system_info",
        ],
        "status": "active",
    },
    "visualization": {
        "name": "Visualization Agent",
        "category": "Infrastructure",
        "description": "Creates publication-quality plots and visualizations",
        "tools": [
            "create_bar_chart",
            "create_heatmap",
            "create_scatter_plot",
            "create_volcano_plot",
        ],
        "status": "active",
    },
}


@app.get("/api/agents")
async def list_agents():
    """List all agents with their capabilities."""
    return {"agents": AGENT_REGISTRY, "total": len(AGENT_REGISTRY)}


@app.get("/api/agents/{agent_name}")
async def get_agent(agent_name: str):
    """Get details for a specific agent."""
    if agent_name not in AGENT_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    return AGENT_REGISTRY[agent_name]


@app.get("/api/agents/{agent_name}/tools")
async def get_agent_tools(agent_name: str):
    """Get tools available to a specific agent."""
    if agent_name not in AGENT_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    return {"agent": agent_name, "tools": AGENT_REGISTRY[agent_name]["tools"]}


# =====================================================
# SESSION MANAGEMENT
# =====================================================

_sessions: dict[str, dict] = {}


@app.get("/api/sessions")
async def list_sessions():
    """List all sessions."""
    return {"sessions": list(_sessions.values()), "total": len(_sessions)}


class CreateSessionRequest(BaseModel):
    name: str = ""


@app.post("/api/sessions")
async def create_session(req: CreateSessionRequest):
    """Create a new session."""
    import uuid

    session_id = str(uuid.uuid4())
    session = {
        "id": session_id,
        "name": req.name or f"Session {len(_sessions) + 1}",
        "created_at": datetime.now().isoformat(),
        "messages": [],
    }
    _sessions[session_id] = session
    return session


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a specific session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return _sessions[session_id]


@app.get("/api/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get full execution history for a session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "messages": _sessions[session_id].get("messages", [])}


# =====================================================
# TOOL REGISTRY API
# =====================================================


@app.get("/api/tools/registry")
async def browse_tool_registry():
    """Browse the custom tool registry."""
    try:
        from bioagents.tools.tool_registry import ToolRegistry

        registry = ToolRegistry()
        tools = registry.list_tools() if hasattr(registry, "list_tools") else []
        return {"tools": tools, "total": len(tools)}
    except Exception as e:
        return {"tools": [], "total": 0, "error": str(e)}


class ValidateToolRequest(BaseModel):
    tool_name: str
    arguments: dict = {}


@app.post("/api/tools/validate")
async def validate_tool(req: ValidateToolRequest):
    """Validate a tool call."""
    try:
        from bioagents.tools.tool_registry import ToolRegistry

        registry = ToolRegistry()
        tool = registry.get_tool(req.tool_name) if hasattr(registry, "get_tool") else None
        if tool is None:
            return {"valid": False, "error": f"Tool '{req.tool_name}' not found"}
        return {"valid": True, "tool_name": req.tool_name}
    except Exception as e:
        return {"valid": False, "error": str(e)}


# =====================================================
# SANDBOX STATUS API
# =====================================================


@app.get("/api/sandbox/status")
async def sandbox_status():
    """Get sandbox health and resource usage."""
    try:
        from bioagents.sandbox.sandbox_manager import get_sandbox

        sandbox = get_sandbox()
        return {
            "status": "running",
            "workspace_dir": str(sandbox.workdir),
            "workspace_exists": sandbox.workdir.exists(),
            "command_history_count": len(sandbox.get_command_history()),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# =====================================================
# MAIN
# =====================================================


def main():
    """Run the server."""
    import uvicorn

    print("\n" + "=" * 60)
    print("🧬 BioAgents Web Interface")
    print("=" * 60)
    print("\n🌐 Open your browser and navigate to:")
    print("   http://localhost:8000")
    print("\n" + "=" * 60 + "\n")

    uvicorn.run("frontend.server:app", host="127.0.0.1", port=8000, reload=False, log_level="info")


if __name__ == "__main__":
    main()
