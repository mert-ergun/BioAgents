"""SQLite database layer for admin dashboard logging.

Manages client tracking, session data, activity logs, chat messages,
experiment logs, and workflow logs. Uses thread-safe writes with
WAL mode for concurrent read access.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/bioagents_admin.db"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS clients (
    client_id TEXT PRIMARY KEY,
    ip_hash TEXT NOT NULL,
    user_agent_hash TEXT NOT NULL,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    total_requests INTEGER DEFAULT 1,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS client_sessions (
    session_id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    last_activity TEXT NOT NULL,
    ended_at TEXT,
    total_queries INTEGER DEFAULT 0,
    total_experiments INTEGER DEFAULT 0,
    total_workflows INTEGER DEFAULT 0,
    provider TEXT,
    model TEXT,
    FOREIGN KEY (client_id) REFERENCES clients(client_id)
);

CREATE TABLE IF NOT EXISTS activity_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id TEXT NOT NULL,
    session_id TEXT,
    action TEXT NOT NULL,
    details TEXT DEFAULT '{}',
    agent_used TEXT,
    duration_ms REAL,
    status TEXT DEFAULT 'success',
    error_message TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (client_id) REFERENCES clients(client_id)
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    agent TEXT,
    content TEXT,
    tool_calls TEXT,
    artifacts TEXT,
    references_count INTEGER DEFAULT 0,
    tokens_used INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (client_id) REFERENCES clients(client_id)
);

CREATE TABLE IF NOT EXISTS experiment_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id TEXT NOT NULL,
    session_id TEXT,
    run_id TEXT NOT NULL,
    use_case_ids TEXT,
    config TEXT,
    status TEXT DEFAULT 'started',
    results_summary TEXT,
    duration_ms REAL,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    FOREIGN KEY (client_id) REFERENCES clients(client_id)
);

CREATE TABLE IF NOT EXISTS workflow_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id TEXT NOT NULL,
    session_id TEXT,
    workflow_type TEXT NOT NULL,
    preset_id TEXT,
    definition TEXT,
    inputs TEXT,
    outputs TEXT,
    status TEXT DEFAULT 'started',
    duration_ms REAL,
    error_message TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    FOREIGN KEY (client_id) REFERENCES clients(client_id)
);

CREATE INDEX IF NOT EXISTS idx_activity_client ON activity_log(client_id);
CREATE INDEX IF NOT EXISTS idx_activity_session ON activity_log(session_id);
CREATE INDEX IF NOT EXISTS idx_activity_action ON activity_log(action);
CREATE INDEX IF NOT EXISTS idx_activity_created ON activity_log(created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_client ON client_sessions(client_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON client_sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_client ON chat_messages(client_id);
CREATE INDEX IF NOT EXISTS idx_chat_created ON chat_messages(created_at);
CREATE INDEX IF NOT EXISTS idx_exp_run ON experiment_logs(run_id);
CREATE INDEX IF NOT EXISTS idx_exp_client ON experiment_logs(client_id);
CREATE INDEX IF NOT EXISTS idx_wf_client ON workflow_logs(client_id);
CREATE INDEX IF NOT EXISTS idx_wf_created ON workflow_logs(created_at);

CREATE TABLE IF NOT EXISTS tool_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    chat_message_id INTEGER,
    agent TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    event_type TEXT NOT NULL DEFAULT 'call',
    arguments TEXT,
    result TEXT,
    result_truncated INTEGER DEFAULT 0,
    duration_ms REAL,
    status TEXT DEFAULT 'success',
    created_at TEXT NOT NULL,
    FOREIGN KEY (client_id) REFERENCES clients(client_id),
    FOREIGN KEY (chat_message_id) REFERENCES chat_messages(id)
);

CREATE TABLE IF NOT EXISTS agent_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    agent TEXT NOT NULL,
    decision TEXT,
    reasoning TEXT,
    step_messages TEXT,
    step_index INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (client_id) REFERENCES clients(client_id)
);

CREATE TABLE IF NOT EXISTS engagement_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    engagement_id TEXT NOT NULL,
    engagement_type TEXT,
    question TEXT,
    options TEXT,
    context TEXT,
    agent TEXT,
    response_content TEXT,
    selected_option TEXT,
    timed_out INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    responded_at TEXT,
    FOREIGN KEY (client_id) REFERENCES clients(client_id)
);

CREATE TABLE IF NOT EXISTS artifact_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    artifact_name TEXT NOT NULL,
    artifact_path TEXT,
    artifact_type TEXT,
    artifact_size INTEGER,
    source_agent TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (client_id) REFERENCES clients(client_id)
);

CREATE TABLE IF NOT EXISTS tool_approval_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    agent TEXT,
    reason TEXT,
    risk_level TEXT,
    outcome TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    FOREIGN KEY (client_id) REFERENCES clients(client_id)
);

CREATE INDEX IF NOT EXISTS idx_tool_session ON tool_events(session_id);
CREATE INDEX IF NOT EXISTS idx_tool_agent ON tool_events(agent);
CREATE INDEX IF NOT EXISTS idx_tool_name ON tool_events(tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_created ON tool_events(created_at);
CREATE INDEX IF NOT EXISTS idx_decisions_session ON agent_decisions(session_id);
CREATE INDEX IF NOT EXISTS idx_decisions_agent ON agent_decisions(agent);
CREATE INDEX IF NOT EXISTS idx_decisions_created ON agent_decisions(created_at);
CREATE INDEX IF NOT EXISTS idx_engagement_session ON engagement_events(session_id);
CREATE INDEX IF NOT EXISTS idx_engagement_created ON engagement_events(created_at);
CREATE INDEX IF NOT EXISTS idx_artifact_session ON artifact_events(session_id);
CREATE INDEX IF NOT EXISTS idx_artifact_created ON artifact_events(created_at);
CREATE INDEX IF NOT EXISTS idx_approval_session ON tool_approval_events(session_id);
CREATE INDEX IF NOT EXISTS idx_approval_created ON tool_approval_events(created_at);
"""


class AdminDatabase:
    """Thread-safe SQLite database for admin logging."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._is_memory = db_path == ":memory:"

        if not self._is_memory:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # For :memory:, share a single connection (each connect() creates a
        # separate DB).  For file-based DBs, use thread-local connections.
        if self._is_memory:
            self._shared_conn: sqlite3.Connection | None = sqlite3.connect(
                db_path, check_same_thread=False
            )
            self._shared_conn.row_factory = sqlite3.Row
            self._shared_conn.execute("PRAGMA foreign_keys=ON")
            self._local: threading.local | None = None
        else:
            self._shared_conn = None
            self._local = threading.local()

        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get the appropriate connection (shared for :memory:, thread-local for files)."""
        if self._is_memory:
            return self._shared_conn  # type: ignore[return-value]

        if not hasattr(self._local, "conn") or self._local.conn is None:  # type: ignore[attr-defined]
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn  # type: ignore[attr-defined]
        return self._local.conn  # type: ignore[return-value]

    def _init_db(self) -> None:
        """Create tables and indexes if they don't exist."""
        conn = self._get_conn()
        conn.executescript(_SCHEMA_SQL)
        conn.commit()

    # ------------------------------------------------------------------
    # Client operations
    # ------------------------------------------------------------------

    def upsert_client(
        self,
        client_id: str,
        ip_hash: str,
        user_agent_hash: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert a new client or update last_seen / total_requests."""
        now = datetime.now().isoformat()
        meta_json = json.dumps(metadata or {})
        with self._lock:
            conn = self._get_conn()
            existing = conn.execute(
                "SELECT client_id FROM clients WHERE client_id = ?",
                (client_id,),
            ).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE clients
                    SET last_seen = ?, total_requests = total_requests + 1,
                        metadata = ?
                    WHERE client_id = ?
                    """,
                    (now, meta_json, client_id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO clients (client_id, ip_hash, user_agent_hash,
                                         first_seen, last_seen, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (client_id, ip_hash, user_agent_hash, now, now, meta_json),
                )
            conn.commit()

    def get_client(self, client_id: str) -> dict[str, Any] | None:
        """Return a client record as a dict, or None."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM clients WHERE client_id = ?", (client_id,)).fetchone()
        return dict(row) if row else None

    def list_clients(
        self,
        page: int = 1,
        limit: int = 20,
        sort: str = "last_seen",
        order: str = "desc",
    ) -> dict[str, Any]:
        """List clients with pagination."""
        conn = self._get_conn()
        allowed_sorts = {"last_seen", "first_seen", "total_requests", "client_id"}
        sort = sort if sort in allowed_sorts else "last_seen"
        order = "DESC" if order.lower() == "desc" else "ASC"
        offset = (page - 1) * limit

        total = conn.execute("SELECT COUNT(*) FROM clients").fetchone()[0]
        rows = conn.execute(
            f"SELECT * FROM clients ORDER BY {sort} {order} LIMIT ? OFFSET ?",  # nosec B608 - sort/order whitelisted
            (limit, offset),
        ).fetchall()
        return {
            "items": [dict(r) for r in rows],
            "total": total,
            "page": page,
            "limit": limit,
        }

    # ------------------------------------------------------------------
    # Session operations
    # ------------------------------------------------------------------

    def upsert_session(
        self,
        session_id: str,
        client_id: str,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        """Insert or update a client session."""
        now = datetime.now().isoformat()
        with self._lock:
            conn = self._get_conn()
            existing = conn.execute(
                "SELECT session_id FROM client_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE client_sessions
                    SET last_activity = ?, provider = COALESCE(?, provider),
                        model = COALESCE(?, model)
                    WHERE session_id = ?
                    """,
                    (now, provider, model, session_id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO client_sessions
                        (session_id, client_id, started_at, last_activity,
                         provider, model)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, client_id, now, now, provider, model),
                )
            conn.commit()

    def increment_session_counter(self, session_id: str, counter: str) -> None:
        """Increment a session counter (total_queries, total_experiments, total_workflows)."""
        allowed = {"total_queries", "total_experiments", "total_workflows"}
        if counter not in allowed:
            return
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                f"UPDATE client_sessions SET {counter} = {counter} + 1, "  # nosec B608 - counter whitelisted
                "last_activity = ? WHERE session_id = ?",
                (datetime.now().isoformat(), session_id),
            )
            conn.commit()

    def end_session(self, session_id: str) -> None:
        """Mark a session as ended."""
        now = datetime.now().isoformat()
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE client_sessions SET ended_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            conn.commit()

    def list_sessions(
        self,
        page: int = 1,
        limit: int = 20,
        client_id: str | None = None,
        sort: str = "started_at",
        order: str = "desc",
    ) -> dict[str, Any]:
        """List sessions with optional client filter and pagination."""
        conn = self._get_conn()
        allowed_sorts = {"started_at", "last_activity", "total_queries"}
        sort = sort if sort in allowed_sorts else "started_at"
        order = "DESC" if order.lower() == "desc" else "ASC"
        offset = (page - 1) * limit

        where = ""
        params: list[Any] = []
        if client_id:
            where = "WHERE client_id = ?"
            params.append(client_id)

        total = conn.execute(f"SELECT COUNT(*) FROM client_sessions {where}", params).fetchone()[0]  # nosec B608 - where built from parameterized conditions
        rows = conn.execute(
            f"SELECT * FROM client_sessions {where} ORDER BY {sort} {order} LIMIT ? OFFSET ?",  # nosec B608
            (*params, limit, offset),
        ).fetchall()
        return {
            "items": [dict(r) for r in rows],
            "total": total,
            "page": page,
            "limit": limit,
        }

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Return a session record as a dict, or None."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM client_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Activity log
    # ------------------------------------------------------------------

    def log_activity(
        self,
        client_id: str,
        action: str,
        session_id: str | None = None,
        details: dict[str, Any] | None = None,
        agent_used: str | None = None,
        duration_ms: float | None = None,
        status: str = "success",
        error_message: str | None = None,
    ) -> int:
        """Log an activity entry. Returns the inserted row id."""
        now = datetime.now().isoformat()
        details_json = json.dumps(details or {})
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                """
                INSERT INTO activity_log
                    (client_id, session_id, action, details, agent_used,
                     duration_ms, status, error_message, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    client_id,
                    session_id,
                    action,
                    details_json,
                    agent_used,
                    duration_ms,
                    status,
                    error_message,
                    now,
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def get_activity_feed(
        self,
        page: int = 1,
        limit: int = 50,
        action: str | None = None,
        client_id: str | None = None,
        session_id: str | None = None,
        cursor_id: int | None = None,
    ) -> dict[str, Any]:
        """Get activity log entries with pagination and optional filters."""
        conn = self._get_conn()
        conditions: list[str] = []
        params: list[Any] = []

        if action:
            conditions.append("action = ?")
            params.append(action)
        if client_id:
            conditions.append("client_id = ?")
            params.append(client_id)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if cursor_id:
            conditions.append("id < ?")
            params.append(cursor_id)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        total = conn.execute(f"SELECT COUNT(*) FROM activity_log {where}", params).fetchone()[0]  # nosec B608

        offset = (page - 1) * limit
        rows = conn.execute(
            f"SELECT * FROM activity_log {where} ORDER BY id DESC LIMIT ? OFFSET ?",  # nosec B608
            (*params, limit, offset),
        ).fetchall()
        return {
            "items": [dict(r) for r in rows],
            "total": total,
            "page": page,
            "limit": limit,
        }

    # ------------------------------------------------------------------
    # Chat messages
    # ------------------------------------------------------------------

    def log_chat_message(
        self,
        client_id: str,
        session_id: str,
        role: str,
        content: str | None = None,
        agent: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        references_count: int = 0,
        tokens_used: int | None = None,
    ) -> int:
        """Log a chat message. Returns the inserted row id."""
        now = datetime.now().isoformat()
        truncated_content = content or ""
        tc_json = json.dumps(tool_calls) if tool_calls else None
        art_json = json.dumps(artifacts) if artifacts else None
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                """
                INSERT INTO chat_messages
                    (client_id, session_id, role, agent, content,
                     tool_calls, artifacts, references_count, tokens_used,
                     created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    client_id,
                    session_id,
                    role,
                    agent,
                    truncated_content,
                    tc_json,
                    art_json,
                    references_count,
                    tokens_used,
                    now,
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def get_chat_history(
        self,
        session_id: str | None = None,
        client_id: str | None = None,
        page: int = 1,
        limit: int = 50,
        search: str | None = None,
        sort: str = "created_at",
        order: str = "desc",
    ) -> dict[str, Any]:
        """Get chat messages with filters and pagination."""
        conn = self._get_conn()
        conditions: list[str] = []
        params: list[Any] = []

        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if client_id:
            conditions.append("client_id = ?")
            params.append(client_id)
        if search:
            conditions.append("content LIKE ?")
            params.append(f"%{search}%")

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        total = conn.execute(f"SELECT COUNT(*) FROM chat_messages {where}", params).fetchone()[0]  # nosec B608

        allowed_sorts = {"created_at", "role", "agent"}
        sort = sort if sort in allowed_sorts else "created_at"
        order_sql = "DESC" if order.lower() == "desc" else "ASC"
        offset = (page - 1) * limit

        rows = conn.execute(
            f"SELECT * FROM chat_messages {where} ORDER BY {sort} {order_sql} LIMIT ? OFFSET ?",  # nosec B608
            (*params, limit, offset),
        ).fetchall()
        return {
            "items": [dict(r) for r in rows],
            "total": total,
            "page": page,
            "limit": limit,
        }

    # ------------------------------------------------------------------
    # Experiment logs
    # ------------------------------------------------------------------

    def log_experiment(
        self,
        client_id: str,
        run_id: str,
        session_id: str | None = None,
        use_case_ids: list[str] | None = None,
        config: dict[str, Any] | None = None,
        status: str = "started",
        results_summary: dict[str, Any] | None = None,
        duration_ms: float | None = None,
    ) -> int:
        """Log an experiment run. Returns the inserted row id."""
        now = datetime.now().isoformat()
        uc_json = json.dumps(use_case_ids) if use_case_ids else None
        cfg_json = json.dumps(config) if config else None
        rs_json = json.dumps(results_summary) if results_summary else None
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                """
                INSERT INTO experiment_logs
                    (client_id, session_id, run_id, use_case_ids, config,
                     status, results_summary, duration_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    client_id,
                    session_id,
                    run_id,
                    uc_json,
                    cfg_json,
                    status,
                    rs_json,
                    duration_ms,
                    now,
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def update_experiment(
        self,
        run_id: str,
        status: str | None = None,
        results_summary: dict[str, Any] | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Update an experiment run record."""
        now = datetime.now().isoformat()
        with self._lock:
            conn = self._get_conn()
            sets: list[str] = []
            params: list[Any] = []
            if status:
                sets.append("status = ?")
                params.append(status)
            if results_summary is not None:
                sets.append("results_summary = ?")
                params.append(json.dumps(results_summary))
            if duration_ms is not None:
                sets.append("duration_ms = ?")
                params.append(duration_ms)
            if status in ("completed", "failed"):
                sets.append("completed_at = ?")
                params.append(now)
            if not sets:
                return
            params.append(run_id)
            conn.execute(
                f"UPDATE experiment_logs SET {', '.join(sets)} WHERE run_id = ?",  # nosec B608 - sets built from safe literals
                params,
            )
            conn.commit()

    def list_experiments(
        self,
        page: int = 1,
        limit: int = 20,
        client_id: str | None = None,
        status: str | None = None,
        sort: str = "created_at",
        order: str = "desc",
    ) -> dict[str, Any]:
        """List experiment runs with optional filters."""
        conn = self._get_conn()
        conditions: list[str] = []
        params: list[Any] = []

        if client_id:
            conditions.append("client_id = ?")
            params.append(client_id)
        if status:
            conditions.append("status = ?")
            params.append(status)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        total = conn.execute(f"SELECT COUNT(*) FROM experiment_logs {where}", params).fetchone()[0]  # nosec B608

        allowed_sorts = {"created_at", "duration_ms", "status"}
        sort = sort if sort in allowed_sorts else "created_at"
        order_sql = "DESC" if order.lower() == "desc" else "ASC"
        offset = (page - 1) * limit

        rows = conn.execute(
            f"SELECT * FROM experiment_logs {where} ORDER BY {sort} {order_sql} LIMIT ? OFFSET ?",  # nosec B608
            (*params, limit, offset),
        ).fetchall()
        return {
            "items": [dict(r) for r in rows],
            "total": total,
            "page": page,
            "limit": limit,
        }

    # ------------------------------------------------------------------
    # Workflow logs
    # ------------------------------------------------------------------

    def log_workflow(
        self,
        client_id: str,
        workflow_type: str,
        session_id: str | None = None,
        preset_id: str | None = None,
        definition: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        status: str = "started",
        duration_ms: float | None = None,
        error_message: str | None = None,
    ) -> int:
        """Log a workflow run. Returns the inserted row id."""
        now = datetime.now().isoformat()
        def_json = json.dumps(definition) if definition else None
        inp_json = json.dumps(inputs) if inputs else None
        out_json = json.dumps(outputs) if outputs else None
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                """
                INSERT INTO workflow_logs
                    (client_id, session_id, workflow_type, preset_id,
                     definition, inputs, outputs, status, duration_ms,
                     error_message, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    client_id,
                    session_id,
                    workflow_type,
                    preset_id,
                    def_json,
                    inp_json,
                    out_json,
                    status,
                    duration_ms,
                    error_message,
                    now,
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def update_workflow(
        self,
        workflow_id: int,
        status: str | None = None,
        outputs: dict[str, Any] | None = None,
        duration_ms: float | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update a workflow run record."""
        now = datetime.now().isoformat()
        with self._lock:
            conn = self._get_conn()
            sets: list[str] = []
            params: list[Any] = []
            if status:
                sets.append("status = ?")
                params.append(status)
            if outputs is not None:
                sets.append("outputs = ?")
                params.append(json.dumps(outputs))
            if duration_ms is not None:
                sets.append("duration_ms = ?")
                params.append(duration_ms)
            if error_message is not None:
                sets.append("error_message = ?")
                params.append(error_message)
            if status in ("completed", "failed"):
                sets.append("completed_at = ?")
                params.append(now)
            if not sets:
                return
            params.append(workflow_id)
            conn.execute(
                f"UPDATE workflow_logs SET {', '.join(sets)} WHERE id = ?",  # nosec B608 - sets built from safe literals
                params,
            )
            conn.commit()

    def list_workflows(
        self,
        page: int = 1,
        limit: int = 20,
        client_id: str | None = None,
        status: str | None = None,
        workflow_type: str | None = None,
        sort: str = "created_at",
        order: str = "desc",
    ) -> dict[str, Any]:
        """List workflow runs with optional filters."""
        conn = self._get_conn()
        conditions: list[str] = []
        params: list[Any] = []

        if client_id:
            conditions.append("client_id = ?")
            params.append(client_id)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if workflow_type:
            conditions.append("workflow_type = ?")
            params.append(workflow_type)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        total = conn.execute(f"SELECT COUNT(*) FROM workflow_logs {where}", params).fetchone()[0]  # nosec B608

        allowed_sorts = {"created_at", "duration_ms", "status"}
        sort = sort if sort in allowed_sorts else "created_at"
        order_sql = "DESC" if order.lower() == "desc" else "ASC"
        offset = (page - 1) * limit

        rows = conn.execute(
            f"SELECT * FROM workflow_logs {where} ORDER BY {sort} {order_sql} LIMIT ? OFFSET ?",  # nosec B608
            (*params, limit, offset),
        ).fetchall()
        return {
            "items": [dict(r) for r in rows],
            "total": total,
            "page": page,
            "limit": limit,
        }

    # ------------------------------------------------------------------
    # Dashboard stats
    # ------------------------------------------------------------------

    def get_dashboard_stats(self) -> dict[str, Any]:
        """Get aggregated stats for the admin dashboard overview."""
        conn = self._get_conn()
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")

        total_clients = conn.execute("SELECT COUNT(*) FROM clients").fetchone()[0]
        total_sessions = conn.execute("SELECT COUNT(*) FROM client_sessions").fetchone()[0]
        active_sessions = conn.execute(
            "SELECT COUNT(*) FROM client_sessions WHERE ended_at IS NULL"
        ).fetchone()[0]

        total_queries = conn.execute(
            "SELECT COUNT(*) FROM activity_log WHERE action = 'query'"
        ).fetchone()[0]
        queries_today = conn.execute(
            "SELECT COUNT(*) FROM activity_log WHERE action = 'query' AND created_at >= ?",
            (today,),
        ).fetchone()[0]

        total_experiments = conn.execute("SELECT COUNT(*) FROM experiment_logs").fetchone()[0]
        total_workflows = conn.execute("SELECT COUNT(*) FROM workflow_logs").fetchone()[0]
        total_activities = conn.execute("SELECT COUNT(*) FROM activity_log").fetchone()[0]
        activities_today = conn.execute(
            "SELECT COUNT(*) FROM activity_log WHERE created_at >= ?",
            (today,),
        ).fetchone()[0]

        # Most active clients (top 5)
        top_clients_rows = conn.execute(
            "SELECT client_id, total_requests FROM clients ORDER BY total_requests DESC LIMIT 5"
        ).fetchall()

        return {
            "total_clients": total_clients,
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "total_queries": total_queries,
            "queries_today": queries_today,
            "total_experiments": total_experiments,
            "total_workflows": total_workflows,
            "total_activities": total_activities,
            "activities_today": activities_today,
            "top_clients": [dict(r) for r in top_clients_rows],
        }

    def get_hourly_stats(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get hourly activity counts for the last N hours."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT
                strftime('%Y-%m-%d %H:00', created_at) AS hour,
                COUNT(*) AS count,
                action
            FROM activity_log
            WHERE created_at >= datetime('now', ?)
            GROUP BY hour, action
            ORDER BY hour DESC
            """,
            (f"-{hours} hours",),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_agent_stats(self) -> list[dict[str, Any]]:
        """Get usage statistics per agent."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT agent, COUNT(*) AS count
            FROM chat_messages
            WHERE role = 'assistant' AND agent IS NOT NULL
            GROUP BY agent
            ORDER BY count DESC
            """,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_provider_stats(self) -> list[dict[str, Any]]:
        """Get usage breakdown by LLM provider."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT provider, COUNT(*) AS count
            FROM client_sessions
            WHERE provider IS NOT NULL
            GROUP BY provider
            ORDER BY count DESC
            """,
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Full log search
    # ------------------------------------------------------------------

    def search_logs(
        self,
        query: str | None = None,
        action: str | None = None,
        client_id: str | None = None,
        status: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        page: int = 1,
        limit: int = 50,
        sort: str = "created_at",
        order: str = "desc",
    ) -> dict[str, Any]:
        """Full-text search across activity logs with multiple filters."""
        conn = self._get_conn()
        conditions: list[str] = []
        params: list[Any] = []

        if query:
            conditions.append("details LIKE ?")
            params.append(f"%{query}%")
        if action:
            conditions.append("action = ?")
            params.append(action)
        if client_id:
            conditions.append("client_id = ?")
            params.append(client_id)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if date_from:
            conditions.append("created_at >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("created_at <= ?")
            params.append(date_to)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        total = conn.execute(f"SELECT COUNT(*) FROM activity_log {where}", params).fetchone()[0]  # nosec B608

        allowed_sorts = {"created_at", "action", "duration_ms", "status"}
        sort = sort if sort in allowed_sorts else "created_at"
        order_sql = "DESC" if order.lower() == "desc" else "ASC"
        offset = (page - 1) * limit

        rows = conn.execute(
            f"SELECT * FROM activity_log {where} ORDER BY {sort} {order_sql} LIMIT ? OFFSET ?",  # nosec B608
            (*params, limit, offset),
        ).fetchall()
        return {
            "items": [dict(r) for r in rows],
            "total": total,
            "page": page,
            "limit": limit,
        }

    # ------------------------------------------------------------------
    # Tool events
    # ------------------------------------------------------------------

    def log_tool_event(
        self,
        client_id: str,
        session_id: str,
        agent: str,
        tool_name: str,
        event_type: str = "call",
        arguments: str | None = None,
        result: str | None = None,
        chat_message_id: int | None = None,
        duration_ms: float | None = None,
        result_truncated: int = 0,
        status: str = "success",
    ) -> int:
        """Log a tool call or result event. Returns the inserted row id."""
        now = datetime.now().isoformat()
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                """
                INSERT INTO tool_events
                    (client_id, session_id, chat_message_id, agent, tool_name,
                     event_type, arguments, result, result_truncated,
                     duration_ms, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    client_id,
                    session_id,
                    chat_message_id,
                    agent,
                    tool_name,
                    event_type,
                    arguments,
                    result,
                    result_truncated,
                    duration_ms,
                    status,
                    now,
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def list_tool_events(
        self,
        session_id: str | None = None,
        client_id: str | None = None,
        agent: str | None = None,
        tool_name: str | None = None,
        page: int = 1,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List tool events with optional filters."""
        conn = self._get_conn()
        conditions: list[str] = []
        params: list[Any] = []
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if client_id:
            conditions.append("client_id = ?")
            params.append(client_id)
        if agent:
            conditions.append("agent = ?")
            params.append(agent)
        if tool_name:
            conditions.append("tool_name = ?")
            params.append(tool_name)
        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)
        total = conn.execute(f"SELECT COUNT(*) FROM tool_events {where}", params).fetchone()[0]  # nosec B608
        offset = (page - 1) * limit
        rows = conn.execute(
            f"SELECT * FROM tool_events {where} ORDER BY id DESC LIMIT ? OFFSET ?",  # nosec B608
            (*params, limit, offset),
        ).fetchall()
        return {"items": [dict(r) for r in rows], "total": total, "page": page, "limit": limit}

    # ------------------------------------------------------------------
    # Agent decisions
    # ------------------------------------------------------------------

    def log_agent_decision(
        self,
        client_id: str,
        session_id: str,
        agent: str,
        decision: str | None = None,
        reasoning: str | None = None,
        step_messages: str | None = None,
        step_index: int = 0,
    ) -> int:
        """Log an agent routing decision. Returns the inserted row id."""
        now = datetime.now().isoformat()
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                """
                INSERT INTO agent_decisions
                    (client_id, session_id, agent, decision, reasoning,
                     step_messages, step_index, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (client_id, session_id, agent, decision, reasoning, step_messages, step_index, now),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def list_agent_decisions(
        self,
        session_id: str | None = None,
        agent: str | None = None,
        page: int = 1,
        limit: int = 100,
    ) -> dict[str, Any]:
        """List agent decisions with optional filters."""
        conn = self._get_conn()
        conditions: list[str] = []
        params: list[Any] = []
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if agent:
            conditions.append("agent = ?")
            params.append(agent)
        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)
        total = conn.execute(f"SELECT COUNT(*) FROM agent_decisions {where}", params).fetchone()[0]  # nosec B608
        offset = (page - 1) * limit
        rows = conn.execute(
            f"SELECT * FROM agent_decisions {where} ORDER BY id DESC LIMIT ? OFFSET ?",  # nosec B608
            (*params, limit, offset),
        ).fetchall()
        return {"items": [dict(r) for r in rows], "total": total, "page": page, "limit": limit}

    # ------------------------------------------------------------------
    # Engagement events
    # ------------------------------------------------------------------

    def log_engagement_event(
        self,
        client_id: str,
        session_id: str,
        engagement_id: str,
        engagement_type: str | None = None,
        question: str | None = None,
        options: str | None = None,
        context: str | None = None,
        agent: str | None = None,
    ) -> int:
        """Log an engagement request. Returns the inserted row id."""
        now = datetime.now().isoformat()
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                """
                INSERT INTO engagement_events
                    (client_id, session_id, engagement_id, engagement_type,
                     question, options, context, agent, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    client_id,
                    session_id,
                    engagement_id,
                    engagement_type,
                    question,
                    options,
                    context,
                    agent,
                    now,
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def update_engagement_event(
        self,
        engagement_id: str,
        response_content: str | None = None,
        selected_option: str | None = None,
        timed_out: bool = False,
    ) -> None:
        """Update an engagement event with the user's response."""
        now = datetime.now().isoformat()
        with self._lock:
            conn = self._get_conn()
            sets: list[str] = []
            params: list[Any] = []
            if response_content is not None:
                sets.append("response_content = ?")
                params.append(response_content)
            if selected_option is not None:
                sets.append("selected_option = ?")
                params.append(selected_option)
            if timed_out:
                sets.append("timed_out = 1")
            sets.append("responded_at = ?")
            params.append(now)
            params.append(engagement_id)
            conn.execute(
                f"UPDATE engagement_events SET {', '.join(sets)} WHERE engagement_id = ?",  # nosec B608 - sets built from safe literals
                params,
            )
            conn.commit()

    def list_engagement_events(
        self,
        session_id: str | None = None,
        page: int = 1,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List engagement events with optional filters."""
        conn = self._get_conn()
        conditions: list[str] = []
        params: list[Any] = []
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)
        total = conn.execute(f"SELECT COUNT(*) FROM engagement_events {where}", params).fetchone()[  # nosec B608
            0
        ]
        offset = (page - 1) * limit
        rows = conn.execute(
            f"SELECT * FROM engagement_events {where} ORDER BY id DESC LIMIT ? OFFSET ?",  # nosec B608
            (*params, limit, offset),
        ).fetchall()
        return {"items": [dict(r) for r in rows], "total": total, "page": page, "limit": limit}

    # ------------------------------------------------------------------
    # Artifact events
    # ------------------------------------------------------------------

    def log_artifact_event(
        self,
        client_id: str,
        session_id: str,
        artifact_name: str,
        artifact_path: str | None = None,
        artifact_type: str | None = None,
        artifact_size: int | None = None,
        source_agent: str | None = None,
    ) -> int:
        """Log an artifact generation/download event. Returns the inserted row id."""
        now = datetime.now().isoformat()
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                """
                INSERT INTO artifact_events
                    (client_id, session_id, artifact_name, artifact_path,
                     artifact_type, artifact_size, source_agent, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    client_id,
                    session_id,
                    artifact_name,
                    artifact_path,
                    artifact_type,
                    artifact_size,
                    source_agent,
                    now,
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def list_artifact_events(
        self,
        session_id: str | None = None,
        page: int = 1,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List artifact events with optional filters."""
        conn = self._get_conn()
        conditions: list[str] = []
        params: list[Any] = []
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)
        total = conn.execute(f"SELECT COUNT(*) FROM artifact_events {where}", params).fetchone()[0]  # nosec B608
        offset = (page - 1) * limit
        rows = conn.execute(
            f"SELECT * FROM artifact_events {where} ORDER BY id DESC LIMIT ? OFFSET ?",  # nosec B608
            (*params, limit, offset),
        ).fetchall()
        return {"items": [dict(r) for r in rows], "total": total, "page": page, "limit": limit}

    # ------------------------------------------------------------------
    # Tool approval events
    # ------------------------------------------------------------------

    def log_tool_approval_event(
        self,
        client_id: str,
        session_id: str,
        request_id: str,
        tool_name: str,
        agent: str | None = None,
        reason: str | None = None,
        risk_level: str | None = None,
        outcome: str = "pending",
    ) -> int:
        """Log a tool approval request or block. Returns the inserted row id."""
        now = datetime.now().isoformat()
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                """
                INSERT INTO tool_approval_events
                    (client_id, session_id, request_id, tool_name,
                     agent, reason, risk_level, outcome, created_at, resolved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    client_id,
                    session_id,
                    request_id,
                    tool_name,
                    agent,
                    reason,
                    risk_level,
                    outcome,
                    now,
                    now if outcome != "pending" else None,
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def update_tool_approval_event(
        self,
        request_id: str,
        outcome: str,
    ) -> None:
        """Update a tool approval event with the resolution."""
        now = datetime.now().isoformat()
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE tool_approval_events SET outcome = ?, resolved_at = ? WHERE request_id = ?",
                (outcome, now, request_id),
            )
            conn.commit()

    def list_tool_approval_events(
        self,
        session_id: str | None = None,
        page: int = 1,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List tool approval events with optional filters."""
        conn = self._get_conn()
        conditions: list[str] = []
        params: list[Any] = []
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)
        total = conn.execute(
            f"SELECT COUNT(*) FROM tool_approval_events {where}",  # nosec B608
            params,
        ).fetchone()[0]
        offset = (page - 1) * limit
        rows = conn.execute(
            f"SELECT * FROM tool_approval_events {where} ORDER BY id DESC LIMIT ? OFFSET ?",  # nosec B608
            (*params, limit, offset),
        ).fetchall()
        return {"items": [dict(r) for r in rows], "total": total, "page": page, "limit": limit}

    # ------------------------------------------------------------------
    # Session timeline (merged view)
    # ------------------------------------------------------------------

    def get_session_timeline(self, session_id: str, limit: int = 500) -> dict[str, Any]:
        """Get a merged chronological timeline for a session.

        Queries all event tables and merges them into a single list
        sorted by created_at, tagged with event_type.
        """
        conn = self._get_conn()
        events: list[dict[str, Any]] = []

        # Chat messages
        for row in conn.execute(
            "SELECT id, role, agent, content, tool_calls, artifacts, "
            "references_count, tokens_used, created_at "
            "FROM chat_messages WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall():
            events.append({**dict(row), "event_type": "message"})

        # Tool events
        for row in conn.execute(
            "SELECT id, agent, tool_name, event_type AS tool_event_type, "
            "arguments, result, result_truncated, duration_ms, status, created_at "
            "FROM tool_events WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall():
            events.append({**dict(row), "event_type": "tool"})

        # Agent decisions
        for row in conn.execute(
            "SELECT id, agent, decision, reasoning, step_messages, "
            "step_index, created_at "
            "FROM agent_decisions WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall():
            events.append({**dict(row), "event_type": "decision"})

        # Engagement events
        for row in conn.execute(
            "SELECT id, engagement_id, engagement_type, question, options, "
            "context, agent, response_content, selected_option, timed_out, "
            "created_at, responded_at "
            "FROM engagement_events WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall():
            events.append({**dict(row), "event_type": "engagement"})

        # Artifact events
        for row in conn.execute(
            "SELECT id, artifact_name, artifact_path, artifact_type, "
            "artifact_size, source_agent, created_at "
            "FROM artifact_events WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall():
            events.append({**dict(row), "event_type": "artifact"})

        # Tool approval events
        for row in conn.execute(
            "SELECT id, request_id, tool_name, agent, reason, risk_level, "
            "outcome, created_at, resolved_at "
            "FROM tool_approval_events WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall():
            events.append({**dict(row), "event_type": "approval"})

        # Sort all events by created_at
        events.sort(key=lambda e: e.get("created_at", ""))
        return {"items": events[:limit], "total": len(events)}
