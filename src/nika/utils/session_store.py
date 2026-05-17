import json
import os
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from nika.config import BASE_DIR

SESSION_DB_PATH = os.path.join(BASE_DIR, "runtime", "sessions.db")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    def __init__(self, db_path: str = SESSION_DB_PATH) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    lab_name TEXT NOT NULL,
                    scenario_name TEXT NOT NULL,
                    scenario_topo_size TEXT,
                    status TEXT NOT NULL,
                    problem_names_json TEXT,
                    root_cause_name TEXT,
                    task_description TEXT,
                    scenario_params_json TEXT,
                    agent_type TEXT,
                    llm_backend TEXT,
                    model TEXT,
                    start_time REAL,
                    end_time REAL,
                    eval_metrics_json TEXT,
                    llm_judge_json TEXT,
                    eval_summary_json TEXT,
                    session_dir TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_scenario_name ON sessions(scenario_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS failure_injections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    problem_name TEXT NOT NULL,
                    root_cause_category TEXT,
                    scenario_name TEXT,
                    lab_name TEXT,
                    injection_params_json TEXT,
                    status TEXT NOT NULL,
                    start_time REAL,
                    end_time REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_failure_injections_session_id ON failure_injections(session_id)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_failure_injections_status ON failure_injections(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_failure_injections_problem_name ON failure_injections(problem_name)")

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return value

    @staticmethod
    def _deserialize_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        payload = dict(row)
        for key in ("problem_names_json", "scenario_params_json", "eval_metrics_json", "llm_judge_json", "eval_summary_json"):
            raw = payload.get(key)
            if raw:
                payload[key] = json.loads(raw)
            else:
                payload[key] = None
        if payload.get("problem_names_json") is not None:
            payload["problem_names"] = payload["problem_names_json"]
        if payload.get("scenario_params_json") is not None:
            payload["scenario_params"] = payload["scenario_params_json"]
        return payload

    @staticmethod
    def _deserialize_failure_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        payload = dict(row)
        raw = payload.get("injection_params_json")
        if raw:
            payload["injection_params_json"] = json.loads(raw)
            payload["injection_params"] = payload["injection_params_json"]
        else:
            payload["injection_params_json"] = None
            payload["injection_params"] = None
        return payload

    def create_session(self, values: Mapping[str, Any]) -> None:
        data = dict(values)
        data.setdefault("status", "running")
        now = _now_iso()
        data.setdefault("created_at", now)
        data.setdefault("updated_at", now)
        columns = list(data.keys())
        serialized = [self._serialize_value(data[c]) for c in columns]
        placeholders = ", ".join("?" for _ in columns)
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO sessions ({', '.join(columns)}) VALUES ({placeholders})",
                serialized,
            )

    def update_session(self, session_id: str, values: Mapping[str, Any]) -> None:
        data = dict(values)
        data["updated_at"] = _now_iso()
        assignments = ", ".join(f"{key} = ?" for key in data.keys())
        serialized = [self._serialize_value(v) for v in data.values()]
        with self._connect() as conn:
            cur = conn.execute(
                f"UPDATE sessions SET {assignments} WHERE session_id = ?",
                [*serialized, session_id],
            )
            if cur.rowcount == 0:
                raise FileNotFoundError(f"Session '{session_id}' not found.")

    def get_session(self, session_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        payload = self._deserialize_row(row)
        if payload is None:
            raise FileNotFoundError(f"Session '{session_id}' not found.")
        return payload

    def list_running_sessions(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE status = 'running' ORDER BY created_at DESC"
            ).fetchall()
        return [self._deserialize_row(row) for row in rows if row is not None]

    def get_unique_running_session(self) -> dict[str, Any]:
        rows = self.list_running_sessions()
        if not rows:
            raise FileNotFoundError(
                "No running session found. Run `nika env run <scenario>` first."
            )
        if len(rows) > 1:
            raise ValueError(
                "Multiple running sessions found. Please pass --session-id to select one."
            )
        return rows[0]

    def create_failure_injection(self, values: Mapping[str, Any]) -> int:
        data = dict(values)
        data.setdefault("status", "pending")
        now = _now_iso()
        data.setdefault("created_at", now)
        data.setdefault("updated_at", now)
        columns = list(data.keys())
        serialized = [self._serialize_value(data[c]) for c in columns]
        placeholders = ", ".join("?" for _ in columns)
        with self._connect() as conn:
            cur = conn.execute(
                f"INSERT INTO failure_injections ({', '.join(columns)}) VALUES ({placeholders})",
                serialized,
            )
            return int(cur.lastrowid)

    def update_failure_injection(self, failure_id: int, values: Mapping[str, Any]) -> None:
        data = dict(values)
        data["updated_at"] = _now_iso()
        assignments = ", ".join(f"{key} = ?" for key in data.keys())
        serialized = [self._serialize_value(v) for v in data.values()]
        with self._connect() as conn:
            cur = conn.execute(
                f"UPDATE failure_injections SET {assignments} WHERE id = ?",
                [*serialized, failure_id],
            )
            if cur.rowcount == 0:
                raise FileNotFoundError(f"Failure injection '{failure_id}' not found.")

    def list_failure_injections(
        self,
        *,
        session_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM failure_injections"
        where: list[str] = []
        args: list[Any] = []
        if session_id is not None:
            where.append("session_id = ?")
            args.append(session_id)
        if status is not None:
            where.append("status = ?")
            args.append(status)
        if where:
            query += f" WHERE {' AND '.join(where)}"
        query += " ORDER BY created_at DESC, id DESC"
        with self._connect() as conn:
            rows = conn.execute(query, args).fetchall()
        return [self._deserialize_failure_row(row) for row in rows if row is not None]

    def count_failure_statuses(self, *, session_id: str) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT status, COUNT(*) AS cnt
                FROM failure_injections
                WHERE session_id = ?
                GROUP BY status
                """,
                (session_id,),
            ).fetchall()
        counts: dict[str, int] = {}
        for row in rows:
            counts[str(row["status"])] = int(row["cnt"])
        return counts

    def mark_session_failures_ended(self, session_id: str, *, end_time: float | None = None) -> int:
        ended_at = end_time if end_time is not None else datetime.now().timestamp()
        now_iso = _now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE failure_injections
                SET status = ?, end_time = ?, updated_at = ?
                WHERE session_id = ? AND status != ?
                """,
                ("ended", ended_at, now_iso, session_id, "ended"),
            )
        return int(cur.rowcount)
