"""Tests for admin dashboard API routes."""

from __future__ import annotations

import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Set password BEFORE importing admin_routes (which triggers admin_auth import)
os.environ["ADMIN_PASSWORD"] = "test_password_123"

from frontend.admin_database import AdminDatabase
from frontend.admin_routes import include_admin_routes


@pytest.fixture
def app() -> FastAPI:
    """Create a test FastAPI app with admin routes."""
    application = FastAPI()
    db = AdminDatabase(":memory:")
    include_admin_routes(application, admin_db=db)
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture
def auth_token(client: TestClient) -> str:
    """Get a valid admin token."""
    resp = client.post("/api/admin/login", json={"password": "test_password_123"})
    assert resp.status_code == 200
    return resp.json()["token"]


@pytest.fixture
def auth_headers(auth_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {auth_token}"}


class TestAdminLogin:
    def test_login_with_correct_password(self, client: TestClient) -> None:
        resp = client.post("/api/admin/login", json={"password": "test_password_123"})
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert "expires_at" in data

    def test_login_with_wrong_password(self, client: TestClient) -> None:
        resp = client.post("/api/admin/login", json={"password": "wrong"})
        assert resp.status_code == 401

    def test_login_with_empty_password(self, client: TestClient) -> None:
        resp = client.post("/api/admin/login", json={"password": ""})
        assert resp.status_code == 401


class TestProtectedEndpoints:
    def test_dashboard_without_token(self, client: TestClient) -> None:
        resp = client.get("/api/admin/dashboard")
        assert resp.status_code == 401

    def test_dashboard_with_valid_token(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        resp = client.get("/api/admin/dashboard", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "total_clients" in data
        assert "total_queries" in data

    def test_clients_without_token(self, client: TestClient) -> None:
        resp = client.get("/api/admin/clients")
        assert resp.status_code == 401

    def test_sessions_without_token(self, client: TestClient) -> None:
        resp = client.get("/api/admin/sessions")
        assert resp.status_code == 401

    def test_chats_without_token(self, client: TestClient) -> None:
        resp = client.get("/api/admin/chats")
        assert resp.status_code == 401

    def test_experiments_without_token(self, client: TestClient) -> None:
        resp = client.get("/api/admin/experiments")
        assert resp.status_code == 401

    def test_workflows_without_token(self, client: TestClient) -> None:
        resp = client.get("/api/admin/workflows")
        assert resp.status_code == 401

    def test_logs_without_token(self, client: TestClient) -> None:
        resp = client.get("/api/admin/logs")
        assert resp.status_code == 401

    def test_stats_without_token(self, client: TestClient) -> None:
        resp = client.get("/api/admin/stats/hourly")
        assert resp.status_code == 401


class TestDashboardEndpoint:
    def test_empty_dashboard(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        resp = client.get("/api/admin/dashboard", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_clients"] == 0
        assert data["active_sessions"] == 0

    def test_dashboard_with_activity(
        self, client: TestClient, auth_headers: dict[str, str], app: FastAPI
    ) -> None:
        # Access the db through the module-level _db
        from frontend import admin_routes

        db = admin_routes._db
        assert db is not None
        db.upsert_client("c1", "ip", "ua")
        db.log_activity(client_id="c1", action="query")

        resp = client.get("/api/admin/dashboard", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_clients"] == 1
        assert data["total_queries"] == 1


class TestClientsEndpoint:
    def test_list_clients_empty(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        resp = client.get("/api/admin/clients", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_list_clients_pagination(
        self, client: TestClient, auth_headers: dict[str, str], app: FastAPI
    ) -> None:
        from frontend import admin_routes

        db = admin_routes._db
        for i in range(5):
            db.upsert_client(f"c{i}", f"ip{i}", f"ua{i}")

        resp = client.get("/api/admin/clients?page=1&limit=2", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 2
        assert data["total"] == 5

    def test_get_client_detail_not_found(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        resp = client.get("/api/admin/clients/nonexistent", headers=auth_headers)
        assert resp.status_code == 404


class TestSessionsEndpoint:
    def test_list_sessions(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        resp = client.get("/api/admin/sessions", headers=auth_headers)
        assert resp.status_code == 200

    def test_session_detail_not_found(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        resp = client.get("/api/admin/sessions/nonexistent", headers=auth_headers)
        assert resp.status_code == 404


class TestStatsEndpoints:
    def test_hourly_stats(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        resp = client.get("/api/admin/stats/hourly", headers=auth_headers)
        assert resp.status_code == 200
        assert "hours" in resp.json()

    def test_agent_stats(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        resp = client.get("/api/admin/stats/agents", headers=auth_headers)
        assert resp.status_code == 200
        assert "agents" in resp.json()

    def test_provider_stats(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        resp = client.get("/api/admin/stats/providers", headers=auth_headers)
        assert resp.status_code == 200
        assert "providers" in resp.json()


# ------------------------------------------------------------------
# New endpoint tests (6 new routes + enhanced session detail)
# ------------------------------------------------------------------


class TestToolEventsEndpoint:
    def test_list_tool_events_empty(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        resp = client.get("/api/admin/tool-events", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_list_tool_events_with_data(
        self, client: TestClient, auth_headers: dict[str, str], app: FastAPI
    ) -> None:
        from frontend import admin_routes

        db = admin_routes._db
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        db.log_tool_event(
            client_id="c1",
            session_id="s1",
            agent="Research",
            tool_name="fetch_fasta",
            event_type="call",
        )

        resp = client.get("/api/admin/tool-events", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["tool_name"] == "fetch_fasta"

    def test_filter_by_session(
        self, client: TestClient, auth_headers: dict[str, str], app: FastAPI
    ) -> None:
        from frontend import admin_routes

        db = admin_routes._db
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        db.upsert_session("s2", "c1")
        db.log_tool_event(
            client_id="c1", session_id="s1", agent="Research", tool_name="t1", event_type="call"
        )
        db.log_tool_event(
            client_id="c1", session_id="s2", agent="Research", tool_name="t2", event_type="call"
        )

        resp = client.get("/api/admin/tool-events?session_id=s1", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_filter_by_agent_and_tool_name(
        self, client: TestClient, auth_headers: dict[str, str], app: FastAPI
    ) -> None:
        from frontend import admin_routes

        db = admin_routes._db
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        db.log_tool_event(
            client_id="c1", session_id="s1", agent="Research", tool_name="search", event_type="call"
        )
        db.log_tool_event(
            client_id="c1",
            session_id="s1",
            agent="Analysis",
            tool_name="analyze",
            event_type="call",
        )

        resp = client.get(
            "/api/admin/tool-events?agent=Research&tool_name=search", headers=auth_headers
        )
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_tool_events_without_token(self, client: TestClient) -> None:
        resp = client.get("/api/admin/tool-events")
        assert resp.status_code == 401


class TestAgentDecisionsEndpoint:
    def test_list_decisions_empty(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        resp = client.get("/api/admin/agent-decisions", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_list_decisions_with_data(
        self, client: TestClient, auth_headers: dict[str, str], app: FastAPI
    ) -> None:
        from frontend import admin_routes

        db = admin_routes._db
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        db.log_agent_decision(
            client_id="c1",
            session_id="s1",
            agent="supervisor",
            decision="Research",
            reasoning="Protein-related query",
        )

        resp = client.get("/api/admin/agent-decisions", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["decision"] == "Research"

    def test_filter_by_session(
        self, client: TestClient, auth_headers: dict[str, str], app: FastAPI
    ) -> None:
        from frontend import admin_routes

        db = admin_routes._db
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        db.upsert_session("s2", "c1")
        db.log_agent_decision(
            client_id="c1", session_id="s1", agent="supervisor", decision="Research"
        )
        db.log_agent_decision(
            client_id="c1", session_id="s2", agent="supervisor", decision="Analysis"
        )

        resp = client.get("/api/admin/agent-decisions?session_id=s1", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_filter_by_agent(
        self, client: TestClient, auth_headers: dict[str, str], app: FastAPI
    ) -> None:
        from frontend import admin_routes

        db = admin_routes._db
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        db.log_agent_decision(
            client_id="c1", session_id="s1", agent="supervisor", decision="Research"
        )
        db.log_agent_decision(
            client_id="c1", session_id="s1", agent="supervisor", decision="Analysis"
        )

        resp = client.get("/api/admin/agent-decisions?agent=supervisor", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["total"] == 2

    def test_decisions_without_token(self, client: TestClient) -> None:
        resp = client.get("/api/admin/agent-decisions")
        assert resp.status_code == 401


class TestEngagementsEndpoint:
    def test_list_engagements_empty(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        resp = client.get("/api/admin/engagements", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_list_engagements_with_data(
        self, client: TestClient, auth_headers: dict[str, str], app: FastAPI
    ) -> None:
        from frontend import admin_routes

        db = admin_routes._db
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        db.log_engagement_event(
            client_id="c1",
            session_id="s1",
            engagement_id="eng1",
            engagement_type="clarification",
            question="Which organism?",
        )

        resp = client.get("/api/admin/engagements", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["question"] == "Which organism?"

    def test_filter_by_session(
        self, client: TestClient, auth_headers: dict[str, str], app: FastAPI
    ) -> None:
        from frontend import admin_routes

        db = admin_routes._db
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        db.upsert_session("s2", "c1")
        db.log_engagement_event(client_id="c1", session_id="s1", engagement_id="eng1")
        db.log_engagement_event(client_id="c1", session_id="s2", engagement_id="eng2")

        resp = client.get("/api/admin/engagements?session_id=s1", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_engagements_without_token(self, client: TestClient) -> None:
        resp = client.get("/api/admin/engagements")
        assert resp.status_code == 401


class TestArtifactsEndpoint:
    def test_list_artifacts_empty(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        resp = client.get("/api/admin/artifacts", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_list_artifacts_with_data(
        self, client: TestClient, auth_headers: dict[str, str], app: FastAPI
    ) -> None:
        from frontend import admin_routes

        db = admin_routes._db
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        db.log_artifact_event(
            client_id="c1",
            session_id="s1",
            artifact_name="report.md",
            artifact_type="markdown",
            artifact_size=4200,
        )

        resp = client.get("/api/admin/artifacts", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["artifact_name"] == "report.md"

    def test_filter_by_session(
        self, client: TestClient, auth_headers: dict[str, str], app: FastAPI
    ) -> None:
        from frontend import admin_routes

        db = admin_routes._db
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        db.upsert_session("s2", "c1")
        db.log_artifact_event(client_id="c1", session_id="s1", artifact_name="a.md")
        db.log_artifact_event(client_id="c1", session_id="s2", artifact_name="b.md")

        resp = client.get("/api/admin/artifacts?session_id=s1", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_artifacts_without_token(self, client: TestClient) -> None:
        resp = client.get("/api/admin/artifacts")
        assert resp.status_code == 401


class TestToolApprovalsEndpoint:
    def test_list_approvals_empty(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        resp = client.get("/api/admin/tool-approvals", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_list_approvals_with_data(
        self, client: TestClient, auth_headers: dict[str, str], app: FastAPI
    ) -> None:
        from frontend import admin_routes

        db = admin_routes._db
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        db.log_tool_approval_event(
            client_id="c1",
            session_id="s1",
            request_id="req1",
            tool_name="shell_exec",
            outcome="pending",
            risk_level="high",
        )

        resp = client.get("/api/admin/tool-approvals", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["outcome"] == "pending"

    def test_filter_by_session(
        self, client: TestClient, auth_headers: dict[str, str], app: FastAPI
    ) -> None:
        from frontend import admin_routes

        db = admin_routes._db
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        db.upsert_session("s2", "c1")
        db.log_tool_approval_event(client_id="c1", session_id="s1", request_id="r1", tool_name="t1")
        db.log_tool_approval_event(client_id="c1", session_id="s2", request_id="r2", tool_name="t2")

        resp = client.get("/api/admin/tool-approvals?session_id=s1", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_approvals_without_token(self, client: TestClient) -> None:
        resp = client.get("/api/admin/tool-approvals")
        assert resp.status_code == 401


class TestSessionTimelineEndpoint:
    def test_timeline_not_found(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        resp = client.get("/api/admin/sessions/nonexistent/timeline", headers=auth_headers)
        assert resp.status_code == 404

    def test_timeline_empty(
        self, client: TestClient, auth_headers: dict[str, str], app: FastAPI
    ) -> None:
        from frontend import admin_routes

        db = admin_routes._db
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")

        resp = client.get("/api/admin/sessions/s1/timeline", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["session"]["session_id"] == "s1"
        assert data["timeline"]["total"] == 0
        assert data["timeline"]["items"] == []

    def test_timeline_with_mixed_events(
        self, client: TestClient, auth_headers: dict[str, str], app: FastAPI
    ) -> None:
        from frontend import admin_routes

        db = admin_routes._db
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        db.log_chat_message(client_id="c1", session_id="s1", role="user", content="Hello")
        db.log_agent_decision(
            client_id="c1", session_id="s1", agent="supervisor", decision="Research"
        )
        db.log_tool_event(
            client_id="c1", session_id="s1", agent="Research", tool_name="search", event_type="call"
        )
        db.log_artifact_event(client_id="c1", session_id="s1", artifact_name="report.md")
        db.log_tool_approval_event(
            client_id="c1", session_id="s1", request_id="r1", tool_name="exec"
        )

        resp = client.get("/api/admin/sessions/s1/timeline", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["timeline"]["total"] == 5
        types = {e["event_type"] for e in data["timeline"]["items"]}
        assert types == {"message", "decision", "tool", "artifact", "approval"}

    def test_timeline_without_token(self, client: TestClient) -> None:
        resp = client.get("/api/admin/sessions/s1/timeline")
        assert resp.status_code == 401


class TestEnhancedSessionDetail:
    def test_session_detail_includes_new_events(
        self, client: TestClient, auth_headers: dict[str, str], app: FastAPI
    ) -> None:
        from frontend import admin_routes

        db = admin_routes._db
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        db.log_chat_message(client_id="c1", session_id="s1", role="user", content="Hello")
        db.log_tool_event(
            client_id="c1", session_id="s1", agent="Research", tool_name="search", event_type="call"
        )
        db.log_agent_decision(
            client_id="c1", session_id="s1", agent="supervisor", decision="Research"
        )
        db.log_artifact_event(client_id="c1", session_id="s1", artifact_name="out.pdb")
        db.log_engagement_event(
            client_id="c1", session_id="s1", engagement_id="eng1", question="Continue?"
        )

        resp = client.get("/api/admin/sessions/s1", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()

        # Session detail now includes all event types
        assert "tool_events" in data
        assert "decisions" in data
        assert "artifacts" in data
        assert "engagements" in data
        assert data["tool_events"]["total"] == 1
        assert data["decisions"]["total"] == 1
        assert data["artifacts"]["total"] == 1
        assert data["engagements"]["total"] == 1


class TestExpiredToken:
    def test_expired_token_rejected(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tokens with past expiry should be rejected."""
        import time as time_mod

        from frontend import admin_auth

        # Save original password
        original_password = admin_auth.ADMIN_PASSWORD

        # Monkey-patch to control time
        real_time = time_mod.time
        fake_now = real_time()

        # Create token at current time
        token = admin_auth.create_admin_token(original_password)

        # Simulate token being 25 hours old (past 24h expiry)
        monkeypatch.setattr(time_mod, "time", lambda: fake_now + 90000)

        resp = client.get(
            "/api/admin/dashboard",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 401
