"""
BioAgents Web Server
FastAPI backend for the BioAgents UI
"""

import asyncio
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

from bioagents.graph import create_graph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# =====================================================
# MODELS
# =====================================================


class QueryRequest(BaseModel):
    query: str


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

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_json(self, websocket: WebSocket, data: dict):
        await websocket.send_json(data)

    async def broadcast(self, data: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.error(f"Failed to broadcast to connection: {e}")


manager = ConnectionManager()

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


async def run_bioagents_query(query: str, websocket: WebSocket | None = None):
    """
    Execute a BioAgents query and stream results.
    """
    graph = create_graph()
    initial_state = {"messages": [HumanMessage(content=query)]}

    full_audit = []
    generated_files = []

    async def send_update(data: dict):
        if websocket:
            try:
                await websocket.send_json(data)
            except Exception as e:
                logger.error(f"WebSocket send error: {e}")
        yield data

    try:
        # Stream through the graph execution
        for step in graph.stream(initial_state, {"recursion_limit": 100}):
            node_name = next(iter(step))
            node_output = step[node_name]

            # Send agent update
            await send_update({"type": "agent_update", "agent": node_name}).__anext__()

            # Send a notification that the agent is starting if it's ML or DL
            if node_name in ["ml", "dl"]:
                msg = (
                    "Initializing training environment and loading data..."
                    if node_name == "ml"
                    else "Designing neural network architecture and preparing data loaders..."
                )
                await send_update(
                    {"type": "message", "agent": node_name, "content": f"_System: {msg}_"}
                ).__anext__()

            # Process messages
            step_messages = []
            if "messages" in node_output:
                for m in node_output["messages"]:
                    msg_info = {
                        "type": m.__class__.__name__,
                        "content": m.content if hasattr(m, "content") else str(m),
                    }
                    if getattr(m, "additional_kwargs", {}).get("show_ui", True) is False:
                        continue

                    if hasattr(m, "tool_calls") and m.tool_calls:
                        msg_info["tool_calls"] = m.tool_calls
                    step_messages.append(msg_info)

                    # Check for artifacts in content
                    content = msg_info["content"]
                    if isinstance(content, str):
                        # Find PDB files
                        if ".pdb" in content.lower():
                            matches = re.findall(
                                r'(/[^\s\'"]+\.pdb|(?:\.\/)?[^\s\'"]+\.pdb)', content
                            )
                            for path_str in matches:
                                path = Path(path_str)
                                if path.exists():
                                    generated_files.append(
                                        {
                                            "name": path.name,
                                            "path": str(path),
                                            "type": "pdb",
                                            "size": path.stat().st_size,
                                        }
                                    )
                                    # Send structure update
                                    try:
                                        with path.open() as f:
                                            pdb_content = f.read()
                                        await send_update(
                                            {"type": "structure", "pdbContent": pdb_content}
                                        ).__anext__()
                                    except Exception as e:
                                        logger.error(f"Failed to load PDB: {e}")

                        # Find other file types
                        for ext in [".csv", ".png", ".jpg", ".json", ".txt", ".pdf"]:
                            if ext in content.lower():
                                matches = re.findall(
                                    rf'(/[^\s\'"]+{ext}|(?:\.\/)?[^\s\'"]+{ext}|(?:[a-zA-Z0-9_\-]+/)+[^\s\'"]+{ext})',
                                    content,
                                    re.IGNORECASE,
                                )
                                for path_str in matches:
                                    path = Path(path_str)
                                    if path.exists():
                                        generated_files.append(
                                            {
                                                "name": path.name,
                                                "path": str(path),
                                                "type": ext[1:],
                                                "size": path.stat().st_size,
                                            }
                                        )
                                        await send_update(
                                            {"type": "artifact", "artifact": generated_files[-1]}
                                        ).__anext__()

            # Build audit entry
            audit_entry = {
                "agent": node_name,
                "decision": node_output.get("next", "Continue"),
                "reasoning": node_output.get("reasoning", ""),
                "messages": step_messages,
            }
            full_audit.append(audit_entry)

            # Send audit update
            await send_update({"type": "audit", "entries": full_audit}).__anext__()

            # Send messages from user-facing agents
            if (
                node_name
                in [
                    "report",
                    "critic",
                    "analysis",
                    "research",
                    "protein_design",
                    "coder",
                    "tool_builder",
                    "ml",
                    "dl",
                ]
                and "messages" in node_output
            ):
                for m in node_output["messages"]:
                    # Skip HumanMessages (usually handoffs) and ToolMessages
                    if (
                        hasattr(m, "content")
                        and m.content
                        and not isinstance(m, (HumanMessage, ToolMessage))
                        and getattr(m, "additional_kwargs", {}).get("show_ui", True) is not False
                    ):
                        content = m.content
                        if isinstance(content, list):
                            text_parts = []
                            for item in content:
                                if isinstance(item, str):
                                    text_parts.append(item)
                                elif isinstance(item, dict) and item.get("type") == "text":
                                    text_parts.append(item.get("text", ""))
                            content = "\n\n".join(text_parts)
                        if content:
                            await send_update(
                                {"type": "message", "agent": node_name, "content": content}
                            ).__anext__()

                            # Extract metrics
                            metrics = extract_metrics_from_content(content)
                            if metrics:
                                await send_update(
                                    {"type": "metrics", "metrics": metrics}
                                ).__anext__()

            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.1)

        # Send completion
        await send_update(
            {"type": "complete", "artifacts": generated_files, "audit_log": full_audit}
        ).__anext__()

    except Exception as e:
        logger.error(f"BioAgents execution error: {e}")
        await send_update({"type": "error", "message": str(e)}).__anext__()
        raise


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
        graph = create_graph()
        initial_state = {"messages": [HumanMessage(content=request.query)]}

        full_audit = []

        try:
            for step in graph.stream(initial_state, {"recursion_limit": 100}):
                node_name = next(iter(step))
                node_output = step[node_name]

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

                # Send code steps if available
                if "code_steps" in node_output:
                    yield f"data: {json.dumps({'type': 'code_execution', 'agent': node_name, 'steps': node_output['code_steps']})}\n\n"

                # Send messages
                if (
                    node_name
                    in [
                        "report",
                        "critic",
                        "analysis",
                        "research",
                        "protein_design",
                        "coder",
                        "tool_builder",
                        "ml",
                        "dl",
                    ]
                    and "messages" in node_output
                ):
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
                                yield f"data: {json.dumps({'type': 'message', 'agent': node_name, 'content': content})}\n\n"

            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

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
    """WebSocket endpoint for real-time communication."""
    await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "query":
                query = data.get("content", "")
                if query:
                    try:
                        await run_bioagents_streaming(query, websocket)
                    except Exception as e:
                        logger.error(f"Query execution error: {e}")
                        await websocket.send_json({"type": "error", "message": str(e)})

            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def run_bioagents_streaming(query: str, websocket: WebSocket):
    """Execute BioAgents query with WebSocket streaming."""
    graph = create_graph()
    initial_state = {"messages": [HumanMessage(content=query)]}

    full_audit = []
    sent_artifacts = set()  # Track sent artifacts to prevent duplicates
    sent_structures = set()  # Track sent PDB URLs/paths
    sent_protein_ids = set()  # Track protein IDs to prevent duplicate structures

    def extract_protein_id(name_or_url: str) -> str | None:
        """Extract protein ID from filename or URL (e.g., AF-P01308-F1 -> P01308)"""
        import re

        match = re.search(r"([A-Z][A-Z0-9]{4,5})", name_or_url, re.IGNORECASE)
        return match.group(1).upper() if match else None

    try:
        for step in graph.stream(initial_state, {"recursion_limit": 100}):
            node_name = next(iter(step))
            node_output = step[node_name]

            # Send agent update
            await websocket.send_json({"type": "agent_update", "agent": node_name})

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
                                    await websocket.send_json(
                                        {"type": "structure", "pdbContent": pdb_content}
                                    )
                                    artifact_key = path.name
                                    if artifact_key not in sent_artifacts:
                                        sent_artifacts.add(artifact_key)
                                        await websocket.send_json(
                                            {
                                                "type": "artifact",
                                                "artifact": {
                                                    "name": path.name,
                                                    "path": str(path),
                                                    "type": "pdb",
                                                    "size": path.stat().st_size,
                                                },
                                            }
                                        )
                                except Exception as e:
                                    logger.error(f"Failed to load PDB: {e}")

                        # Check for PDB URLs (AlphaFold, RCSB, etc.)
                        url_matches = re.findall(r'(https?://[^\s\'"<>]+\.pdb)', content)
                        for url in url_matches:
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
                                    await websocket.send_json(
                                        {"type": "structure", "pdbContent": pdb_content}
                                    )
                                    if filename not in sent_artifacts:
                                        sent_artifacts.add(filename)
                                        await websocket.send_json(
                                            {
                                                "type": "artifact",
                                                "artifact": {
                                                    "name": filename,
                                                    "path": url,
                                                    "type": "pdb",
                                                    "size": len(pdb_content),
                                                },
                                            }
                                        )
                                        await websocket.send_json(
                                            {
                                                "type": "log",
                                                "message": f"Loaded 3D structure: {filename}",
                                            }
                                        )
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
                                    if path.exists():
                                        await websocket.send_json(
                                            {
                                                "type": "artifact",
                                                "artifact": {
                                                    "name": path.name,
                                                    "path": str(path),
                                                    "type": ext[1:],
                                                    "size": path.stat().st_size,
                                                },
                                            }
                                        )

            # Build audit entry
            audit_entry = {
                "agent": node_name,
                "decision": node_output.get("next", "Continue"),
                "reasoning": node_output.get("reasoning", ""),
                "messages": step_messages,
            }
            full_audit.append(audit_entry)
            await websocket.send_json({"type": "audit", "entries": full_audit})

            # Send code steps if available (from coder, ml, or dl agents)
            if "code_steps" in node_output:
                await websocket.send_json(
                    {
                        "type": "code_execution",
                        "agent": node_name,
                        "steps": node_output["code_steps"],
                    }
                )

            # Send messages from user-facing agents
            if (
                node_name
                in [
                    "report",
                    "critic",
                    "analysis",
                    "research",
                    "protein_design",
                    "coder",
                    "tool_builder",
                    "ml",
                    "dl",
                ]
                and "messages" in node_output
            ):
                for m in node_output["messages"]:
                    # Skip HumanMessages (usually handoffs) and ToolMessages
                    if (
                        hasattr(m, "content")
                        and m.content
                        and not isinstance(m, (HumanMessage, ToolMessage))
                        and getattr(m, "additional_kwargs", {}).get("show_ui", True) is not False
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
                            await websocket.send_json(
                                {"type": "message", "agent": node_name, "content": content}
                            )

                            # Extract metrics
                            metrics = extract_metrics_from_content(content)
                            if metrics:
                                await websocket.send_json({"type": "metrics", "metrics": metrics})

                            # Extract and send artifacts from content (with deduplication)
                            extracted_artifacts = extract_artifacts_from_content(content, node_name)
                            for artifact in extracted_artifacts:
                                if artifact["name"] not in sent_artifacts:
                                    sent_artifacts.add(artifact["name"])
                                    await websocket.send_json(
                                        {"type": "artifact", "artifact": artifact}
                                    )
                                    await websocket.send_json(
                                        {"type": "log", "message": f"Generated: {artifact['name']}"}
                                    )

            await asyncio.sleep(0.05)

        await websocket.send_json({"type": "complete", "artifacts": [], "audit_log": full_audit})

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        await websocket.send_json({"type": "error", "message": str(e)})


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

    uvicorn.run("frontend.server:app", host="127.0.0.1", port=8000, reload=True, log_level="info")


if __name__ == "__main__":
    main()
