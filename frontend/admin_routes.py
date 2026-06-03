"""Admin dashboard API routes.

Provides endpoints for login, dashboard stats, client browsing,
session inspection, chat history, experiment/workflow logs,
activity feeds, and search. All endpoints except /login require
admin authentication via Bearer token.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query

from frontend.admin_auth import (
    AdminLoginRequest,
    AdminLoginResponse,
    create_admin_token,
    get_admin_dependency,
)

if TYPE_CHECKING:
    from frontend.admin_database import AdminDatabase

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin")

_db: AdminDatabase | None = None


def _get_db() -> AdminDatabase:
    """Return the shared AdminDatabase instance."""
    if _db is None:
        raise HTTPException(status_code=500, detail="Admin database not initialized")
    return _db


# ------------------------------------------------------------------
# Login
# ------------------------------------------------------------------


@router.post("/login", response_model=AdminLoginResponse)
async def admin_login(req: AdminLoginRequest) -> AdminLoginResponse:
    """Authenticate admin and return a signed token."""
    try:
        import time

        token = create_admin_token(req.password)
        expires_at = time.time() + 86400  # matches _TOKEN_EXPIRY_SECONDS
        return AdminLoginResponse(token=token, expires_at=expires_at)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid password") from None


# ------------------------------------------------------------------
# Dashboard overview
# ------------------------------------------------------------------


@router.get("/dashboard")
async def get_dashboard(
    _token: str = Depends(get_admin_dependency),
) -> dict[str, Any]:
    """Get overview stats for the admin dashboard."""
    db = _get_db()
    return db.get_dashboard_stats()


# ------------------------------------------------------------------
# Clients
# ------------------------------------------------------------------


@router.get("/clients")
async def list_clients(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    sort: str = Query("last_seen"),
    order: str = Query("desc"),
    _token: str = Depends(get_admin_dependency),
) -> dict[str, Any]:
    """List all clients with pagination."""
    db = _get_db()
    return db.list_clients(page=page, limit=limit, sort=sort, order=order)


@router.get("/clients/{client_id}")
async def get_client_detail(
    client_id: str,
    _token: str = Depends(get_admin_dependency),
) -> dict[str, Any]:
    """Get client detail with session history."""
    db = _get_db()
    client = db.get_client(client_id)
    if client is None:
        raise HTTPException(status_code=404, detail="Client not found")
    sessions = db.list_sessions(client_id=client_id, limit=50)
    return {"client": client, "sessions": sessions}


# ------------------------------------------------------------------
# Sessions
# ------------------------------------------------------------------


@router.get("/sessions")
async def list_sessions(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    client_id: str | None = Query(None),
    sort: str = Query("started_at"),
    order: str = Query("desc"),
    _token: str = Depends(get_admin_dependency),
) -> dict[str, Any]:
    """List all sessions with optional client filter and pagination."""
    db = _get_db()
    return db.list_sessions(page=page, limit=limit, client_id=client_id, sort=sort, order=order)


@router.get("/sessions/{session_id}")
async def get_session_detail(
    session_id: str,
    _token: str = Depends(get_admin_dependency),
) -> dict[str, Any]:
    """Get session detail with all messages, tool events, decisions, artifacts."""
    db = _get_db()
    session = db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = db.get_chat_history(session_id=session_id, limit=500)
    tool_events = db.list_tool_events(session_id=session_id, limit=500)
    decisions = db.list_agent_decisions(session_id=session_id, limit=500)
    artifacts = db.list_artifact_events(session_id=session_id, limit=100)
    engagements = db.list_engagement_events(session_id=session_id, limit=100)
    return {
        "session": session,
        "messages": messages,
        "tool_events": tool_events,
        "decisions": decisions,
        "artifacts": artifacts,
        "engagements": engagements,
    }


# ------------------------------------------------------------------
# Chats
# ------------------------------------------------------------------


@router.get("/chats")
async def list_chats(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    session_id: str | None = Query(None),
    client_id: str | None = Query(None),
    search: str | None = Query(None),
    sort: str = Query("created_at"),
    order: str = Query("desc"),
    _token: str = Depends(get_admin_dependency),
) -> dict[str, Any]:
    """List chat messages with filters."""
    db = _get_db()
    return db.get_chat_history(
        session_id=session_id,
        client_id=client_id,
        page=page,
        limit=limit,
        search=search,
        sort=sort,
        order=order,
    )


# ------------------------------------------------------------------
# Experiments
# ------------------------------------------------------------------


@router.get("/experiments")
async def list_experiments(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    client_id: str | None = Query(None),
    status: str | None = Query(None),
    sort: str = Query("created_at"),
    order: str = Query("desc"),
    _token: str = Depends(get_admin_dependency),
) -> dict[str, Any]:
    """List experiment runs with optional filters."""
    db = _get_db()
    return db.list_experiments(
        page=page,
        limit=limit,
        client_id=client_id,
        status=status,
        sort=sort,
        order=order,
    )


# ------------------------------------------------------------------
# Workflows
# ------------------------------------------------------------------


@router.get("/workflows")
async def list_workflows(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    client_id: str | None = Query(None),
    status: str | None = Query(None),
    workflow_type: str | None = Query(None),
    sort: str = Query("created_at"),
    order: str = Query("desc"),
    _token: str = Depends(get_admin_dependency),
) -> dict[str, Any]:
    """List workflow runs with optional filters."""
    db = _get_db()
    return db.list_workflows(
        page=page,
        limit=limit,
        client_id=client_id,
        status=status,
        workflow_type=workflow_type,
        sort=sort,
        order=order,
    )


# ------------------------------------------------------------------
# Activity feed
# ------------------------------------------------------------------


@router.get("/activity")
async def get_activity(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    action: str | None = Query(None),
    client_id: str | None = Query(None),
    session_id: str | None = Query(None),
    cursor_id: int | None = Query(None),
    _token: str = Depends(get_admin_dependency),
) -> dict[str, Any]:
    """Get activity feed with pagination."""
    db = _get_db()
    return db.get_activity_feed(
        page=page,
        limit=limit,
        action=action,
        client_id=client_id,
        session_id=session_id,
        cursor_id=cursor_id,
    )


# ------------------------------------------------------------------
# Full log search
# ------------------------------------------------------------------


@router.get("/logs")
async def search_logs(
    query: str | None = Query(None),
    action: str | None = Query(None),
    client_id: str | None = Query(None),
    status: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    sort: str = Query("created_at"),
    order: str = Query("desc"),
    _token: str = Depends(get_admin_dependency),
) -> dict[str, Any]:
    """Full-text search across activity logs."""
    db = _get_db()
    return db.search_logs(
        query=query,
        action=action,
        client_id=client_id,
        status=status,
        date_from=date_from,
        date_to=date_to,
        page=page,
        limit=limit,
        sort=sort,
        order=order,
    )


# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------


@router.get("/stats/hourly")
async def get_hourly_stats(
    hours: int = Query(24, ge=1, le=168),
    _token: str = Depends(get_admin_dependency),
) -> dict[str, Any]:
    """Get hourly activity aggregation for charts."""
    db = _get_db()
    return {"hours": db.get_hourly_stats(hours=hours)}


@router.get("/stats/agents")
async def get_agent_stats(
    _token: str = Depends(get_admin_dependency),
) -> dict[str, Any]:
    """Get agent usage statistics."""
    db = _get_db()
    return {"agents": db.get_agent_stats()}


@router.get("/stats/providers")
async def get_provider_stats(
    _token: str = Depends(get_admin_dependency),
) -> dict[str, Any]:
    """Get LLM provider usage breakdown."""
    db = _get_db()
    return {"providers": db.get_provider_stats()}


# ------------------------------------------------------------------
# Tool events
# ------------------------------------------------------------------


@router.get("/tool-events")
async def list_tool_events(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    session_id: str | None = Query(None),
    agent: str | None = Query(None),
    tool_name: str | None = Query(None),
    _token: str = Depends(get_admin_dependency),
) -> dict[str, Any]:
    """List tool call/result events with filters."""
    db = _get_db()
    return db.list_tool_events(
        session_id=session_id,
        agent=agent,
        tool_name=tool_name,
        page=page,
        limit=limit,
    )


# ------------------------------------------------------------------
# Agent decisions
# ------------------------------------------------------------------


@router.get("/agent-decisions")
async def list_agent_decisions(
    page: int = Query(1, ge=1),
    limit: int = Query(100, ge=1, le=500),
    session_id: str | None = Query(None),
    agent: str | None = Query(None),
    _token: str = Depends(get_admin_dependency),
) -> dict[str, Any]:
    """List agent routing decisions with reasoning."""
    db = _get_db()
    return db.list_agent_decisions(
        session_id=session_id,
        agent=agent,
        page=page,
        limit=limit,
    )


# ------------------------------------------------------------------
# Engagement events
# ------------------------------------------------------------------


@router.get("/engagements")
async def list_engagements(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    session_id: str | None = Query(None),
    _token: str = Depends(get_admin_dependency),
) -> dict[str, Any]:
    """List engagement question/response events."""
    db = _get_db()
    return db.list_engagement_events(
        session_id=session_id,
        page=page,
        limit=limit,
    )


# ------------------------------------------------------------------
# Artifact events
# ------------------------------------------------------------------


@router.get("/artifacts")
async def list_artifacts(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    session_id: str | None = Query(None),
    _token: str = Depends(get_admin_dependency),
) -> dict[str, Any]:
    """List artifact generation/download events."""
    db = _get_db()
    return db.list_artifact_events(
        session_id=session_id,
        page=page,
        limit=limit,
    )


# ------------------------------------------------------------------
# Tool approval events
# ------------------------------------------------------------------


@router.get("/tool-approvals")
async def list_tool_approvals(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    session_id: str | None = Query(None),
    _token: str = Depends(get_admin_dependency),
) -> dict[str, Any]:
    """List tool approval/block events."""
    db = _get_db()
    return db.list_tool_approval_events(
        session_id=session_id,
        page=page,
        limit=limit,
    )


# ------------------------------------------------------------------
# Session timeline
# ------------------------------------------------------------------


@router.get("/sessions/{session_id}/timeline")
async def get_session_timeline(
    session_id: str,
    _token: str = Depends(get_admin_dependency),
) -> dict[str, Any]:
    """Get complete chronological timeline for a session."""
    db = _get_db()
    session = db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    timeline = db.get_session_timeline(session_id)
    return {"session": session, "timeline": timeline}


# ------------------------------------------------------------------
# Route registration (mirrors include_workflow_routes pattern)
# ------------------------------------------------------------------


def include_admin_routes(app: FastAPI, admin_db: AdminDatabase) -> None:
    """Register admin dashboard endpoints on the main FastAPI application."""
    global _db
    _db = admin_db
    app.include_router(router)
