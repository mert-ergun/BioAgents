"""Tests for the admin database layer.

Uses in-memory SQLite for all tests to ensure isolation.
"""

from __future__ import annotations

import threading

import pytest

from frontend.admin_database import AdminDatabase


@pytest.fixture
def db() -> AdminDatabase:
    """Provide a fresh in-memory AdminDatabase."""
    return AdminDatabase(":memory:")


class TestSchemaCreation:
    def test_tables_created(self, db: AdminDatabase) -> None:
        """All expected tables should exist after init."""
        conn = db._get_conn()
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        expected = {
            "clients",
            "client_sessions",
            "activity_log",
            "chat_messages",
            "experiment_logs",
            "workflow_logs",
        }
        assert expected.issubset(tables)

    def test_indexes_created(self, db: AdminDatabase) -> None:
        """Key indexes should exist for query performance."""
        conn = db._get_conn()
        indexes = {
            row[1]
            for row in conn.execute("SELECT * FROM sqlite_master WHERE type='index'").fetchall()
        }
        assert "idx_activity_client" in indexes
        assert "idx_activity_created" in indexes
        assert "idx_chat_session" in indexes


class TestClientOperations:
    def test_upsert_new_client(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip_h", "ua_h")
        client = db.get_client("c1")
        assert client is not None
        assert client["client_id"] == "c1"
        assert client["total_requests"] == 1

    def test_upsert_existing_client_updates(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip_h", "ua_h")
        db.upsert_client("c1", "ip_h", "ua_h", metadata={"browser": "chrome"})
        client = db.get_client("c1")
        assert client is not None
        assert client["total_requests"] == 2

    def test_get_nonexistent_client(self, db: AdminDatabase) -> None:
        assert db.get_client("nope") is None

    def test_list_clients_pagination(self, db: AdminDatabase) -> None:
        for i in range(5):
            db.upsert_client(f"c{i}", f"ip{i}", f"ua{i}")
        result = db.list_clients(page=1, limit=2)
        assert len(result["items"]) == 2
        assert result["total"] == 5
        assert result["page"] == 1


class TestSessionOperations:
    def test_upsert_new_session(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1", provider="openai", model="gpt-4")
        session = db.get_session("s1")
        assert session is not None
        assert session["client_id"] == "c1"
        assert session["provider"] == "openai"

    def test_upsert_existing_session_updates(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        db.upsert_session("s1", "c1", provider="gemini")
        session = db.get_session("s1")
        assert session["provider"] == "gemini"

    def test_increment_session_counter(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        db.increment_session_counter("s1", "total_queries")
        db.increment_session_counter("s1", "total_queries")
        session = db.get_session("s1")
        assert session["total_queries"] == 2

    def test_end_session(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        db.end_session("s1")
        session = db.get_session("s1")
        assert session["ended_at"] is not None

    def test_list_sessions_by_client(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        db.upsert_client("c2", "ip2", "ua2")
        db.upsert_session("s1", "c1")
        db.upsert_session("s2", "c1")
        db.upsert_session("s3", "c2")
        result = db.list_sessions(client_id="c1")
        assert result["total"] == 2


class TestActivityLog:
    def test_log_activity(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        row_id = db.log_activity(
            client_id="c1",
            action="query",
            details={"method": "POST", "path": "/api/query"},
            duration_ms=123.4,
        )
        assert row_id > 0
        feed = db.get_activity_feed()
        assert feed["total"] == 1
        assert feed["items"][0]["action"] == "query"

    def test_log_activity_with_error(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        db.log_activity(
            client_id="c1",
            action="query",
            status="error",
            error_message="Timeout",
        )
        feed = db.get_activity_feed()
        assert feed["items"][0]["status"] == "error"

    def test_activity_feed_filters(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        db.log_activity(client_id="c1", action="query")
        db.log_activity(client_id="c1", action="upload")
        feed = db.get_activity_feed(action="query")
        assert feed["total"] == 1
        assert feed["items"][0]["action"] == "query"

    def test_activity_feed_pagination(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        for _i in range(10):
            db.log_activity(client_id="c1", action="query")
        page1 = db.get_activity_feed(page=1, limit=5)
        page2 = db.get_activity_feed(page=2, limit=5)
        assert len(page1["items"]) == 5
        assert len(page2["items"]) == 5
        assert page1["total"] == 10


class TestChatMessages:
    def test_log_chat_message(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        row_id = db.log_chat_message(
            client_id="c1",
            session_id="s1",
            role="user",
            content="What is protein folding?",
        )
        assert row_id > 0

    def test_get_chat_history(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        db.log_chat_message(client_id="c1", session_id="s1", role="user", content="Hello")
        db.log_chat_message(
            client_id="c1", session_id="s1", role="assistant", agent="Research", content="Hi there"
        )
        history = db.get_chat_history(session_id="s1")
        assert history["total"] == 2

    def test_chat_search(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        db.log_chat_message(client_id="c1", session_id="s1", role="user", content="protein folding")
        db.log_chat_message(client_id="c1", session_id="s1", role="user", content="DNA structure")
        result = db.get_chat_history(session_id="s1", search="protein")
        assert result["total"] == 1

    def test_long_content_stored(self, db: AdminDatabase) -> None:
        """Long content is stored as-is; truncation is the caller's responsibility."""
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        long_content = "x" * 10000
        db.log_chat_message(client_id="c1", session_id="s1", role="user", content=long_content)
        history = db.get_chat_history(session_id="s1")
        assert len(history["items"][0]["content"]) == 10000


class TestExperimentLogs:
    def test_log_experiment(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        row_id = db.log_experiment(
            client_id="c1",
            run_id="run_abc",
            use_case_ids=["uc1", "uc2"],
            config={"name": "test"},
        )
        assert row_id > 0

    def test_update_experiment(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        db.log_experiment(client_id="c1", run_id="run_abc")
        db.update_experiment(run_id="run_abc", status="completed", duration_ms=5000)
        exps = db.list_experiments()
        assert exps["items"][0]["status"] == "completed"
        assert exps["items"][0]["completed_at"] is not None

    def test_list_experiments_by_status(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        db.log_experiment(client_id="c1", run_id="r1")
        db.log_experiment(client_id="c1", run_id="r2")
        db.update_experiment(run_id="r1", status="completed")
        result = db.list_experiments(status="completed")
        assert result["total"] == 1


class TestWorkflowLogs:
    def test_log_workflow(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        row_id = db.log_workflow(
            client_id="c1",
            workflow_type="preset",
            preset_id="protein_embedding",
        )
        assert row_id > 0

    def test_update_workflow(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        wid = db.log_workflow(client_id="c1", workflow_type="preset", preset_id="p1")
        db.update_workflow(workflow_id=wid, status="completed", duration_ms=2000)
        wfs = db.list_workflows()
        assert wfs["items"][0]["status"] == "completed"


class TestDashboardStats:
    def test_empty_database_stats(self, db: AdminDatabase) -> None:
        stats = db.get_dashboard_stats()
        assert stats["total_clients"] == 0
        assert stats["total_queries"] == 0
        assert stats["total_activities"] == 0

    def test_dashboard_stats_with_data(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        db.upsert_client("c2", "ip2", "ua2")
        db.upsert_session("s1", "c1")
        db.log_activity(client_id="c1", action="query")
        db.log_activity(client_id="c2", action="upload")
        db.log_experiment(client_id="c1", run_id="r1")
        db.log_workflow(client_id="c2", workflow_type="preset")

        stats = db.get_dashboard_stats()
        assert stats["total_clients"] == 2
        assert stats["total_queries"] == 1
        assert stats["total_experiments"] == 1
        assert stats["total_workflows"] == 1

    def test_agent_stats(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        db.log_chat_message(
            client_id="c1", session_id="s1", role="assistant", agent="Research", content="test"
        )
        db.log_chat_message(
            client_id="c1", session_id="s1", role="assistant", agent="Research", content="test2"
        )
        db.log_chat_message(
            client_id="c1", session_id="s1", role="assistant", agent="Analysis", content="test3"
        )

        stats = db.get_agent_stats()
        assert len(stats) == 2
        assert stats[0]["agent"] == "Research"
        assert stats[0]["count"] == 2


class TestSearchLogs:
    def test_search_by_query(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        db.log_activity(client_id="c1", action="query", details={"path": "/api/query"})
        db.log_activity(client_id="c1", action="upload", details={"path": "/api/upload"})
        result = db.search_logs(query="/api/query")
        assert result["total"] == 1

    def test_search_by_status(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        db.log_activity(client_id="c1", action="query", status="success")
        db.log_activity(client_id="c1", action="query", status="error")
        result = db.search_logs(status="error")
        assert result["total"] == 1

    def test_sql_injection_prevention(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        db.log_activity(client_id="c1", action="query")
        # This should not crash or return unexpected data
        result = db.search_logs(query="'; DROP TABLE activity_log; --")
        assert result["total"] == 0


class TestThreadSafety:
    def test_concurrent_writes(self, db: AdminDatabase) -> None:
        """Multiple threads writing should not corrupt data."""
        db.upsert_client("c1", "ip", "ua")
        errors: list[str] = []

        def write_activity(idx: int) -> None:
            try:
                db.log_activity(client_id="c1", action=f"query_{idx}")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=write_activity, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        feed = db.get_activity_feed(limit=100)
        assert feed["total"] == 20


class TestToolEvents:
    def test_log_tool_call(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        row_id = db.log_tool_event(
            client_id="c1",
            session_id="s1",
            agent="Research",
            tool_name="fetch_uniprot_fasta",
            event_type="call",
            arguments='{"protein_id": "P04637"}',
        )
        assert row_id > 0
        events = db.list_tool_events(session_id="s1")
        assert events["total"] == 1
        assert events["items"][0]["tool_name"] == "fetch_uniprot_fasta"

    def test_log_tool_result(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        db.log_tool_event(
            client_id="c1",
            session_id="s1",
            agent="Research",
            tool_name="fetch_uniprot_fasta",
            event_type="call",
        )
        db.log_tool_event(
            client_id="c1",
            session_id="s1",
            agent="Research",
            tool_name="fetch_uniprot_fasta",
            event_type="result",
            result=">sp|P04637|P53_HUMAN...",
            duration_ms=150.5,
        )
        events = db.list_tool_events(session_id="s1")
        assert events["total"] == 2

    def test_filter_by_tool_name(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        db.log_tool_event(
            client_id="c1", session_id="s1", agent="Research", tool_name="tool_a", event_type="call"
        )
        db.log_tool_event(
            client_id="c1", session_id="s1", agent="Research", tool_name="tool_b", event_type="call"
        )
        result = db.list_tool_events(tool_name="tool_a")
        assert result["total"] == 1


class TestAgentDecisions:
    def test_log_decision(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        row_id = db.log_agent_decision(
            client_id="c1",
            session_id="s1",
            agent="supervisor",
            decision="Research",
            reasoning="User is asking about proteins",
            step_index=0,
        )
        assert row_id > 0
        decisions = db.list_agent_decisions(session_id="s1")
        assert decisions["total"] == 1
        assert decisions["items"][0]["reasoning"] == "User is asking about proteins"


class TestEngagementEvents:
    def test_log_engagement_and_response(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        db.log_engagement_event(
            client_id="c1",
            session_id="s1",
            engagement_id="eng1",
            engagement_type="clarification",
            question="Search recent papers?",
            options='["Yes", "No"]',
        )
        db.update_engagement_event(
            engagement_id="eng1", response_content="Yes", selected_option="Yes"
        )
        events = db.list_engagement_events(session_id="s1")
        assert events["total"] == 1
        assert events["items"][0]["response_content"] == "Yes"

    def test_engagement_timeout(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        db.log_engagement_event(client_id="c1", session_id="s1", engagement_id="eng2")
        db.update_engagement_event(engagement_id="eng2", timed_out=True)
        events = db.list_engagement_events(session_id="s1")
        assert events["items"][0]["timed_out"] == 1


class TestArtifactEvents:
    def test_log_artifact(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        row_id = db.log_artifact_event(
            client_id="c1",
            session_id="s1",
            artifact_name="report.md",
            artifact_type="markdown",
            artifact_size=4200,
            source_agent="Report",
        )
        assert row_id > 0
        artifacts = db.list_artifact_events(session_id="s1")
        assert artifacts["total"] == 1
        assert artifacts["items"][0]["artifact_name"] == "report.md"


class TestToolApprovalEvents:
    def test_log_approval_and_resolve(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        db.log_tool_approval_event(
            client_id="c1",
            session_id="s1",
            request_id="req1",
            tool_name="shell_exec",
            agent="CodeGen",
            outcome="pending",
            reason="External API tool",
            risk_level="high",
        )
        db.update_tool_approval_event(request_id="req1", outcome="approved")
        approvals = db.list_tool_approval_events(session_id="s1")
        assert approvals["total"] == 1
        assert approvals["items"][0]["outcome"] == "approved"
        assert approvals["items"][0]["resolved_at"] is not None

    def test_log_policy_block(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        db.log_tool_approval_event(
            client_id="c1",
            session_id="s1",
            request_id="req2",
            tool_name="shell_exec",
            outcome="blocked",
        )
        approvals = db.list_tool_approval_events(session_id="s1")
        assert approvals["items"][0]["outcome"] == "blocked"


class TestSessionTimeline:
    def test_empty_timeline(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        timeline = db.get_session_timeline("s1")
        assert timeline["total"] == 0
        assert timeline["items"] == []

    def test_merged_timeline(self, db: AdminDatabase) -> None:
        db.upsert_client("c1", "ip", "ua")
        db.upsert_session("s1", "c1")
        db.log_chat_message(client_id="c1", session_id="s1", role="user", content="Hello")
        db.log_agent_decision(
            client_id="c1", session_id="s1", agent="supervisor", decision="Research"
        )
        db.log_tool_event(
            client_id="c1", session_id="s1", agent="Research", tool_name="search", event_type="call"
        )
        db.log_chat_message(
            client_id="c1", session_id="s1", role="assistant", agent="Research", content="Found it"
        )
        db.log_artifact_event(client_id="c1", session_id="s1", artifact_name="report.md")
        timeline = db.get_session_timeline("s1")
        assert timeline["total"] == 5
        types = [e["event_type"] for e in timeline["items"]]
        assert "message" in types
        assert "tool" in types
        assert "decision" in types
        assert "artifact" in types
