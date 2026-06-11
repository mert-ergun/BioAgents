"""FastAPI middleware that logs all API requests to the admin database.

Skips static files, health checks, and admin endpoints to avoid noise
and recursive logging.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

    from frontend.admin_database import AdminDatabase

from frontend.client_tracker import generate_client_id

logger = logging.getLogger(__name__)

# Paths that should be skipped by the logging middleware
_SKIP_PREFIXES = ("/static", "/favicon", "/health")
_SKIP_EXACT = ("/",)
_SKIP_ADMIN_PREFIX = "/api/admin"


class ActivityLoggingMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that logs all API requests to the admin database."""

    def __init__(self, app: Any, db: AdminDatabase) -> None:
        super().__init__(app)
        self.db = db

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process the request, log activity, and pass through."""
        path = request.url.path

        # Skip static files, health checks, root, and admin endpoints
        if (
            any(path.startswith(p) for p in _SKIP_PREFIXES)
            or path in _SKIP_EXACT
            or path.startswith(_SKIP_ADMIN_PREFIX)
        ):
            return await call_next(request)

        start_time = time.monotonic()

        # Generate client ID from request headers
        ip_address = request.headers.get(
            "x-forwarded-for",
            request.client.host if request.client else "unknown",
        )
        user_agent = request.headers.get("user-agent", "unknown")
        client_id = generate_client_id(ip_address, user_agent)

        # Upsert client record
        try:
            self.db.upsert_client(
                client_id=client_id,
                ip_hash=generate_client_id(ip_address, ""),
                user_agent_hash=generate_client_id("", user_agent),
            )
        except Exception:
            logger.debug("Failed to upsert client", exc_info=True)

        # Process the request
        response = await call_next(request)
        duration_ms = (time.monotonic() - start_time) * 1000

        # Classify and log the action
        action = self._classify_action(path, request.method)

        try:
            self.db.log_activity(
                client_id=client_id,
                action=action,
                details={
                    "method": request.method,
                    "path": path,
                    "status_code": response.status_code,
                },
                duration_ms=duration_ms,
                status="success" if response.status_code < 400 else "error",
            )
        except Exception:
            logger.debug("Failed to log activity", exc_info=True)

        # Inject client ID header for downstream use
        response.headers["X-Client-ID"] = client_id
        return response

    @staticmethod
    def _classify_action(path: str, method: str) -> str:
        """Classify the request into an action type based on URL path."""
        if "/api/query" in path:
            return "query"
        if "/api/experiments" in path and method == "POST":
            return "experiment_run"
        if "/api/workflows" in path and method == "POST":
            return "workflow_run"
        if "/api/upload" in path:
            return "upload"
        if "/api/drug-discovery" in path:
            return "drug_discovery"
        if "/ws" in path:
            return "websocket_connect"
        if "/api/sessions" in path and method == "POST":
            return "session_create"
        if "/api/tools" in path:
            return "tool_call"
        return "api_call"
