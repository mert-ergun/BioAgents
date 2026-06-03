"""Tests for the activity logging middleware."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from frontend.admin_database import AdminDatabase
from frontend.logging_middleware import ActivityLoggingMiddleware


@pytest.fixture
def app_with_middleware() -> FastAPI:
    """Create a test app with the logging middleware."""
    application = FastAPI()

    @application.get("/api/query")
    async def query_endpoint() -> dict:
        return {"status": "ok"}

    @application.post("/api/experiments/run")
    async def experiment_endpoint() -> dict:
        return {"status": "started"}

    @application.get("/static/style.css")
    async def static_file() -> dict:
        return {"file": "style.css"}

    @application.get("/api/admin/dashboard")
    async def admin_endpoint() -> dict:
        return {"total_clients": 0}

    @application.get("/")
    async def root() -> dict:
        return {"message": "root"}

    @application.get("/api/sessions")
    async def sessions() -> dict:
        return {"sessions": []}

    db = AdminDatabase(":memory:")
    application.add_middleware(ActivityLoggingMiddleware, db=db)

    # Attach db for assertions
    application.state.db = db
    return application


@pytest.fixture
def client(app_with_middleware: FastAPI) -> TestClient:
    return TestClient(app_with_middleware)


class TestMiddlewareLogging:
    def test_logs_api_request(self, client: TestClient, app_with_middleware: FastAPI) -> None:
        resp = client.get("/api/query")
        assert resp.status_code == 200
        assert "X-Client-ID" in resp.headers

        db = app_with_middleware.state.db
        feed = db.get_activity_feed()
        assert feed["total"] == 1
        assert feed["items"][0]["action"] == "query"

    def test_classifies_experiment_run(
        self, client: TestClient, app_with_middleware: FastAPI
    ) -> None:
        resp = client.post("/api/experiments/run")
        assert resp.status_code == 200

        db = app_with_middleware.state.db
        feed = db.get_activity_feed()
        assert feed["total"] >= 1
        # Find the experiment_run action
        actions = [item["action"] for item in feed["items"]]
        assert "experiment_run" in actions

    def test_skips_static_files(self, client: TestClient, app_with_middleware: FastAPI) -> None:
        resp = client.get("/static/style.css")
        assert resp.status_code == 200

        db = app_with_middleware.state.db
        feed = db.get_activity_feed()
        assert feed["total"] == 0

    def test_skips_root(self, client: TestClient, app_with_middleware: FastAPI) -> None:
        resp = client.get("/")
        assert resp.status_code == 200

        db = app_with_middleware.state.db
        feed = db.get_activity_feed()
        assert feed["total"] == 0

    def test_skips_admin_endpoints(self, client: TestClient, app_with_middleware: FastAPI) -> None:
        resp = client.get("/api/admin/dashboard")
        assert resp.status_code == 200

        db = app_with_middleware.state.db
        feed = db.get_activity_feed()
        assert feed["total"] == 0

    def test_client_id_header_set(self, client: TestClient) -> None:
        resp = client.get("/api/query")
        assert resp.headers["X-Client-ID"]
        assert len(resp.headers["X-Client-ID"]) == 16

    def test_upserts_client_on_request(
        self, client: TestClient, app_with_middleware: FastAPI
    ) -> None:
        client.get("/api/query")

        db = app_with_middleware.state.db
        clients = db.list_clients()
        assert clients["total"] == 1

    def test_logs_duration(self, client: TestClient, app_with_middleware: FastAPI) -> None:
        client.get("/api/query")

        db = app_with_middleware.state.db
        feed = db.get_activity_feed()
        assert feed["items"][0]["duration_ms"] is not None
        assert feed["items"][0]["duration_ms"] >= 0

    def test_multiple_requests_increment_counter(
        self, client: TestClient, app_with_middleware: FastAPI
    ) -> None:
        client.get("/api/query")
        client.get("/api/query")
        client.get("/api/query")

        db = app_with_middleware.state.db
        clients = db.list_clients()
        assert clients["items"][0]["total_requests"] == 3
