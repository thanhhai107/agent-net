"""Canonical stores for atomic procedural memories."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from agent.memory.models import (
    MemoryCandidate,
    MemoryLinkType,
    MemoryStatus,
    StoredMemory,
)

DEFAULT_MEMORY_DATABASE_URL = "postgresql://nika:nika@localhost:5432/nika_memory"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _content_hash(candidate: MemoryCandidate) -> str:
    normalized = " ".join(candidate.content.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class SQLiteMemoryStore:
    """Explicit compatibility/test store; runtime defaults to PostgreSQL."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                INSERT OR IGNORE INTO schema_meta(key, value)
                VALUES ('schema_version', '1');

                CREATE TABLE IF NOT EXISTS episodes (
                    bank_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    evaluated_at TEXT,
                    snapshot_path TEXT,
                    metrics_json TEXT,
                    PRIMARY KEY (bank_id, session_id)
                );

                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    bank_id TEXT NOT NULL,
                    -- Legacy/internal compatibility column. Public memory no
                    -- longer exposes a taxonomy such as observation/learning.
                    memory_type TEXT NOT NULL DEFAULT 'atomic',
                    content TEXT NOT NULL,
                    applicability_json TEXT NOT NULL,
                    evidence_required_json TEXT NOT NULL,
                    avoid_json TEXT NOT NULL,
                    attributes_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    source_session_id TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    validation_count INTEGER NOT NULL DEFAULT 0,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    superseded_at TEXT,
                    superseded_by TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_memories_bank_status
                    ON memories(bank_id, status);
                CREATE INDEX IF NOT EXISTS idx_memories_hash
                    ON memories(bank_id, content_hash);

                CREATE TABLE IF NOT EXISTS memory_links (
                    bank_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(source_id, target_id, relation),
                    FOREIGN KEY(source_id) REFERENCES memories(memory_id),
                    FOREIGN KEY(target_id) REFERENCES memories(memory_id)
                );

                CREATE TABLE IF NOT EXISTS retrieval_events (
                    event_id TEXT PRIMARY KEY,
                    bank_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    memory_ids_json TEXT NOT NULL,
                    scores_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    memory_id UNINDEXED,
                    bank_id UNINDEXED,
                    content,
                    attributes
                );
                """
            )

    def record_episode_start(self, bank_id: str, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes(bank_id, session_id, started_at)
                VALUES (?, ?, ?)
                ON CONFLICT(bank_id, session_id) DO NOTHING
                """,
                (bank_id, session_id, _now_iso()),
            )

    def record_episode_evaluation(
        self,
        bank_id: str,
        session_id: str,
        metrics: dict[str, Any],
        snapshot_path: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes(
                    bank_id, session_id, started_at, evaluated_at,
                    snapshot_path, metrics_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(bank_id, session_id) DO UPDATE SET
                    evaluated_at=excluded.evaluated_at,
                    snapshot_path=excluded.snapshot_path,
                    metrics_json=excluded.metrics_json
                """,
                (
                    bank_id,
                    session_id,
                    _now_iso(),
                    _now_iso(),
                    snapshot_path,
                    _json(metrics),
                ),
            )

    def episode_is_evaluated(self, bank_id: str, session_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT evaluated_at FROM episodes
                WHERE bank_id=? AND session_id=?
                """,
                (bank_id, session_id),
            ).fetchone()
        return bool(row and row["evaluated_at"])

    def add_or_corroborate(
        self,
        *,
        bank_id: str,
        candidate: MemoryCandidate,
        status: MemoryStatus,
        confidence: float,
        source_session_id: str,
        successful_episode: bool,
    ) -> tuple[StoredMemory, bool]:
        digest = _content_hash(candidate)
        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT * FROM memories
                WHERE bank_id=? AND content_hash=? AND status != ?
                ORDER BY version DESC LIMIT 1
                """,
                (bank_id, digest, MemoryStatus.SUPERSEDED.value),
            ).fetchone()
            if existing is not None:
                validation_delta = 1 if successful_episode else 0
                failure_delta = 0 if successful_episode else 1
                new_confidence = max(
                    0.0,
                    min(
                        1.0,
                        float(existing["confidence"])
                        + (0.08 if successful_episode else -0.08),
                    ),
                )
                new_status = existing["status"]
                validations = int(existing["validation_count"]) + validation_delta
                if successful_episode and new_status == MemoryStatus.STAGED.value:
                    new_status = MemoryStatus.VALIDATED.value
                conn.execute(
                    """
                    UPDATE memories SET
                        confidence=?,
                        status=?,
                        validation_count=?,
                        failure_count=failure_count+?
                    WHERE memory_id=?
                    """,
                    (
                        new_confidence,
                        new_status,
                        validations,
                        failure_delta,
                        existing["memory_id"],
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM memories WHERE memory_id=?",
                    (existing["memory_id"],),
                ).fetchone()
                self._refresh_fts(conn, row)
                return self._row_to_memory(row), False

            memory_id = str(uuid.uuid4())
            created_at = _now_iso()
            attrs_json = _json(candidate.attributes.normalized().model_dump())
            conn.execute(
                """
                INSERT INTO memories(
                    memory_id, bank_id, memory_type, content,
                    applicability_json, evidence_required_json, avoid_json,
                    attributes_json, confidence, status, source_session_id,
                    validation_count, failure_count, content_hash, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    bank_id,
                    "atomic",
                    candidate.content.strip(),
                    _json(candidate.applicability),
                    _json(candidate.evidence_required),
                    _json(candidate.avoid),
                    attrs_json,
                    confidence,
                    status.value,
                    source_session_id,
                    1 if successful_episode else 0,
                    0 if successful_episode else 1,
                    digest,
                    created_at,
                ),
            )
            row = conn.execute(
                "SELECT * FROM memories WHERE memory_id=?", (memory_id,)
            ).fetchone()
            self._refresh_fts(conn, row)
            return self._row_to_memory(row), True

    def _refresh_fts(self, conn: sqlite3.Connection, row: sqlite3.Row) -> None:
        conn.execute("DELETE FROM memories_fts WHERE memory_id=?", (row["memory_id"],))
        attrs = json.loads(row["attributes_json"])
        attr_text = " ".join(
            value
            for values in attrs.values()
            for value in values
            if isinstance(value, str)
        )
        supporting = " ".join(
            json.loads(row["applicability_json"])
            + json.loads(row["evidence_required_json"])
            + json.loads(row["avoid_json"])
        )
        conn.execute(
            """
            INSERT INTO memories_fts(memory_id, bank_id, content, attributes)
            VALUES (?, ?, ?, ?)
            """,
            (
                row["memory_id"],
                row["bank_id"],
                f"{row['content']} {supporting}",
                attr_text,
            ),
        )

    def get(self, memory_id: str) -> StoredMemory | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE memory_id=?", (memory_id,)
            ).fetchone()
        return self._row_to_memory(row) if row else None

    def get_many(self, memory_ids: list[str]) -> list[StoredMemory]:
        if not memory_ids:
            return []
        placeholders = ",".join("?" for _ in memory_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM memories WHERE memory_id IN ({placeholders})",
                memory_ids,
            ).fetchall()
        by_id = {row["memory_id"]: self._row_to_memory(row) for row in rows}
        return [by_id[memory_id] for memory_id in memory_ids if memory_id in by_id]

    def search_fts(
        self,
        *,
        bank_id: str,
        query: str,
        limit: int,
        statuses: tuple[MemoryStatus, ...] = (MemoryStatus.VALIDATED,),
        exclude_id: str | None = None,
        fallback: bool = True,
    ) -> list[tuple[StoredMemory, float]]:
        terms = [
            term.lower() for term in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", query)
        ]
        seen: set[str] = set()
        terms = [term for term in terms if not (term in seen or seen.add(term))]
        status_values = [status.value for status in statuses]
        status_placeholders = ",".join("?" for _ in status_values)

        with self._connect() as conn:
            if terms:
                fts_query = " OR ".join(
                    f'"{term.replace(chr(34), "")}"' for term in terms[:24]
                )
                sql = f"""
                    SELECT m.*, bm25(memories_fts) AS rank
                    FROM memories_fts
                    JOIN memories m ON m.memory_id = memories_fts.memory_id
                    WHERE memories_fts MATCH ?
                      AND m.bank_id=?
                      AND m.status IN ({status_placeholders})
                """
                params: list[Any] = [fts_query, bank_id, *status_values]
                if exclude_id:
                    sql += " AND m.memory_id != ?"
                    params.append(exclude_id)
                sql += " ORDER BY rank ASC LIMIT ?"
                params.append(limit)
                rows = conn.execute(sql, params).fetchall()
                if rows:
                    ranks = [abs(float(row["rank"])) for row in rows]
                    max_rank = max(ranks) or 1.0
                    return [
                        (self._row_to_memory(row), abs(float(row["rank"])) / max_rank)
                        for row in rows
                    ]

            if not fallback:
                return []

            sql = f"""
                SELECT * FROM memories
                WHERE bank_id=? AND status IN ({status_placeholders})
            """
            params = [bank_id, *status_values]
            if exclude_id:
                sql += " AND memory_id != ?"
                params.append(exclude_id)
            sql += " ORDER BY confidence DESC, created_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            return [(self._row_to_memory(row), 0.0) for row in rows]

    def add_link(
        self,
        *,
        bank_id: str,
        source_id: str,
        target_id: str,
        relation: MemoryLinkType,
        reason: str = "",
    ) -> None:
        if source_id == target_id:
            return
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO memory_links(
                    bank_id, source_id, target_id, relation, reason, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    bank_id,
                    source_id,
                    target_id,
                    relation.value,
                    reason,
                    _now_iso(),
                ),
            )

    def supersede(self, memory_id: str, replacement_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE memories
                SET version=MAX(
                    version,
                    COALESCE(
                        (SELECT version + 1 FROM memories WHERE memory_id=?),
                        1
                    )
                )
                WHERE memory_id=?
                """,
                (memory_id, replacement_id),
            )
            conn.execute(
                """
                UPDATE memories SET status=?, superseded_at=?, superseded_by=?
                WHERE memory_id=?
                """,
                (
                    MemoryStatus.SUPERSEDED.value,
                    _now_iso(),
                    replacement_id,
                    memory_id,
                ),
            )

    def record_retrieval(
        self,
        *,
        bank_id: str,
        session_id: str,
        query_text: str,
        memory_ids: list[str],
        scores: list[float],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO retrieval_events(
                    event_id, bank_id, session_id, query_text,
                    memory_ids_json, scores_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    bank_id,
                    session_id,
                    query_text,
                    _json(memory_ids),
                    _json(scores),
                    _now_iso(),
                ),
            )

    def link_counts(self, bank_id: str, memory_ids: list[str]) -> dict[str, int]:
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT memory_id, COUNT(*) AS count
                FROM (
                    SELECT source_id AS memory_id
                    FROM memory_links
                    WHERE bank_id=? AND source_id IN ({placeholders})
                    UNION ALL
                    SELECT target_id AS memory_id
                    FROM memory_links
                    WHERE bank_id=? AND target_id IN ({placeholders})
                )
                GROUP BY memory_id
                """,
                [bank_id, *memory_ids, bank_id, *memory_ids],
            ).fetchall()
        return {str(row["memory_id"]): int(row["count"]) for row in rows}

    def export_bank(self, bank_id: str) -> tuple[list[dict], list[dict]]:
        with self._connect() as conn:
            memories = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM memories WHERE bank_id=? ORDER BY created_at",
                    (bank_id,),
                ).fetchall()
            ]
            links = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM memory_links WHERE bank_id=? ORDER BY created_at",
                    (bank_id,),
                ).fetchall()
            ]
        for memory in memories:
            memory.pop("memory_type", None)
            for field in (
                "applicability_json",
                "evidence_required_json",
                "avoid_json",
                "attributes_json",
            ):
                memory[field.removesuffix("_json")] = json.loads(memory.pop(field))
        return memories, links

    def bank_stats(self, bank_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            status_rows = conn.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM memories WHERE bank_id=? GROUP BY status
                """,
                (bank_id,),
            ).fetchall()
            episode_count = conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE bank_id=?",
                (bank_id,),
            ).fetchone()[0]
            retrieval_count = conn.execute(
                "SELECT COUNT(*) FROM retrieval_events WHERE bank_id=?",
                (bank_id,),
            ).fetchone()[0]
        return {
            "bank_id": bank_id,
            "memories_by_status": {
                row["status"]: int(row["count"]) for row in status_rows
            },
            "episodes": int(episode_count),
            "retrievals": int(retrieval_count),
        }

    def clear_bank(self, bank_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM memories_fts WHERE bank_id=?", (bank_id,))
            conn.execute("DELETE FROM memory_links WHERE bank_id=?", (bank_id,))
            conn.execute("DELETE FROM retrieval_events WHERE bank_id=?", (bank_id,))
            conn.execute("DELETE FROM episodes WHERE bank_id=?", (bank_id,))
            conn.execute("DELETE FROM memories WHERE bank_id=?", (bank_id,))

    @staticmethod
    def _row_to_memory(row: Mapping[str, Any]) -> StoredMemory:
        return StoredMemory(
            memory_id=row["memory_id"],
            bank_id=row["bank_id"],
            content=row["content"],
            applicability=json.loads(row["applicability_json"]),
            evidence_required=json.loads(row["evidence_required_json"]),
            avoid=json.loads(row["avoid_json"]),
            attributes=json.loads(row["attributes_json"]),
            confidence=float(row["confidence"]),
            status=row["status"],
            source_session_id=row["source_session_id"],
            version=int(row["version"]),
            validation_count=int(row["validation_count"]),
            failure_count=int(row["failure_count"]),
            created_at=row["created_at"],
            superseded_at=row["superseded_at"],
            superseded_by=row["superseded_by"],
        )


class PostgreSQLMemoryStore:
    """Canonical PostgreSQL store for procedural-memory banks."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._initialize()

    def _connect(self):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL memory store requires the 'psycopg[binary]' dependency."
            ) from exc
        return psycopg.connect(
            self.database_url,
            autocommit=False,
            row_factory=dict_row,
        )

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT INTO memory_schema_meta(key, value)
                VALUES ('schema_version', '1')
                ON CONFLICT (key) DO NOTHING
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_episodes (
                    bank_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    evaluated_at TEXT,
                    snapshot_path TEXT,
                    metrics_json TEXT,
                    PRIMARY KEY (bank_id, session_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_memories (
                    memory_id TEXT PRIMARY KEY,
                    bank_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL DEFAULT 'atomic',
                    content TEXT NOT NULL,
                    applicability_json TEXT NOT NULL,
                    evidence_required_json TEXT NOT NULL,
                    avoid_json TEXT NOT NULL,
                    attributes_json TEXT NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    status TEXT NOT NULL,
                    source_session_id TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    validation_count INTEGER NOT NULL DEFAULT 0,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    superseded_at TEXT,
                    superseded_by TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_memories_bank_status
                ON memory_memories(bank_id, status)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_memories_hash
                ON memory_memories(bank_id, content_hash)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_links (
                    bank_id TEXT NOT NULL,
                    source_id TEXT NOT NULL REFERENCES memory_memories(memory_id),
                    target_id TEXT NOT NULL REFERENCES memory_memories(memory_id),
                    relation TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(source_id, target_id, relation)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_retrieval_events (
                    event_id TEXT PRIMARY KEY,
                    bank_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    memory_ids_json TEXT NOT NULL,
                    scores_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def record_episode_start(self, bank_id: str, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_episodes(bank_id, session_id, started_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (bank_id, session_id) DO NOTHING
                """,
                (bank_id, session_id, _now_iso()),
            )

    def record_episode_evaluation(
        self,
        bank_id: str,
        session_id: str,
        metrics: dict[str, Any],
        snapshot_path: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_episodes(
                    bank_id, session_id, started_at, evaluated_at,
                    snapshot_path, metrics_json
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (bank_id, session_id) DO UPDATE SET
                    evaluated_at=EXCLUDED.evaluated_at,
                    snapshot_path=EXCLUDED.snapshot_path,
                    metrics_json=EXCLUDED.metrics_json
                """,
                (
                    bank_id,
                    session_id,
                    _now_iso(),
                    _now_iso(),
                    snapshot_path,
                    _json(metrics),
                ),
            )

    def episode_is_evaluated(self, bank_id: str, session_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT evaluated_at FROM memory_episodes
                WHERE bank_id=%s AND session_id=%s
                """,
                (bank_id, session_id),
            ).fetchone()
        return bool(row and row["evaluated_at"])

    def add_or_corroborate(
        self,
        *,
        bank_id: str,
        candidate: MemoryCandidate,
        status: MemoryStatus,
        confidence: float,
        source_session_id: str,
        successful_episode: bool,
    ) -> tuple[StoredMemory, bool]:
        digest = _content_hash(candidate)
        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT * FROM memory_memories
                WHERE bank_id=%s AND content_hash=%s AND status != %s
                ORDER BY version DESC LIMIT 1
                """,
                (bank_id, digest, MemoryStatus.SUPERSEDED.value),
            ).fetchone()
            if existing is not None:
                validation_delta = 1 if successful_episode else 0
                failure_delta = 0 if successful_episode else 1
                new_confidence = max(
                    0.0,
                    min(
                        1.0,
                        float(existing["confidence"])
                        + (0.08 if successful_episode else -0.08),
                    ),
                )
                new_status = existing["status"]
                validations = int(existing["validation_count"]) + validation_delta
                if successful_episode and new_status == MemoryStatus.STAGED.value:
                    new_status = MemoryStatus.VALIDATED.value
                conn.execute(
                    """
                    UPDATE memory_memories SET
                        confidence=%s,
                        status=%s,
                        validation_count=%s,
                        failure_count=failure_count+%s
                    WHERE memory_id=%s
                    """,
                    (
                        new_confidence,
                        new_status,
                        validations,
                        failure_delta,
                        existing["memory_id"],
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM memory_memories WHERE memory_id=%s",
                    (existing["memory_id"],),
                ).fetchone()
                return self._row_to_memory(row), False

            memory_id = str(uuid.uuid4())
            created_at = _now_iso()
            conn.execute(
                """
                INSERT INTO memory_memories(
                    memory_id, bank_id, memory_type, content,
                    applicability_json, evidence_required_json, avoid_json,
                    attributes_json, confidence, status, source_session_id,
                    validation_count, failure_count, content_hash, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    memory_id,
                    bank_id,
                    "atomic",
                    candidate.content.strip(),
                    _json(candidate.applicability),
                    _json(candidate.evidence_required),
                    _json(candidate.avoid),
                    _json(candidate.attributes.normalized().model_dump()),
                    confidence,
                    status.value,
                    source_session_id,
                    1 if successful_episode else 0,
                    0 if successful_episode else 1,
                    digest,
                    created_at,
                ),
            )
            row = conn.execute(
                "SELECT * FROM memory_memories WHERE memory_id=%s",
                (memory_id,),
            ).fetchone()
            return self._row_to_memory(row), True

    def get(self, memory_id: str) -> StoredMemory | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_memories WHERE memory_id=%s",
                (memory_id,),
            ).fetchone()
        return self._row_to_memory(row) if row else None

    def get_many(self, memory_ids: list[str]) -> list[StoredMemory]:
        if not memory_ids:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memory_memories
                WHERE memory_id = ANY(%s)
                """,
                (memory_ids,),
            ).fetchall()
        by_id = {row["memory_id"]: self._row_to_memory(row) for row in rows}
        return [by_id[memory_id] for memory_id in memory_ids if memory_id in by_id]

    def search_fts(
        self,
        *,
        bank_id: str,
        query: str,
        limit: int,
        statuses: tuple[MemoryStatus, ...] = (MemoryStatus.VALIDATED,),
        exclude_id: str | None = None,
        fallback: bool = True,
    ) -> list[tuple[StoredMemory, float]]:
        terms = " ".join(
            term.lower() for term in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", query)
        )
        status_values = [status.value for status in statuses]
        rows: list[Mapping[str, Any]] = []
        with self._connect() as conn:
            if terms:
                sql = """
                    WITH q AS (SELECT websearch_to_tsquery('simple', %s) AS query)
                    SELECT m.*,
                           ts_rank_cd(
                               to_tsvector(
                                   'simple',
                                   m.content || ' ' ||
                                   m.applicability_json || ' ' ||
                                   m.evidence_required_json || ' ' ||
                                   m.avoid_json || ' ' ||
                                   m.attributes_json
                               ),
                               q.query
                           ) AS rank
                    FROM memory_memories m, q
                    WHERE q.query @@ to_tsvector(
                            'simple',
                            m.content || ' ' ||
                            m.applicability_json || ' ' ||
                            m.evidence_required_json || ' ' ||
                            m.avoid_json || ' ' ||
                            m.attributes_json
                          )
                      AND m.bank_id=%s
                      AND m.status = ANY(%s)
                """
                params: list[Any] = [terms, bank_id, status_values]
                if exclude_id:
                    sql += " AND m.memory_id != %s"
                    params.append(exclude_id)
                sql += " ORDER BY rank DESC LIMIT %s"
                params.append(limit)
                rows = conn.execute(sql, params).fetchall()
                if rows:
                    max_rank = max(float(row["rank"]) for row in rows) or 1.0
                    return [
                        (self._row_to_memory(row), float(row["rank"]) / max_rank)
                        for row in rows
                    ]

            if not fallback:
                return []

            sql = """
                SELECT * FROM memory_memories
                WHERE bank_id=%s AND status = ANY(%s)
            """
            params = [bank_id, status_values]
            if exclude_id:
                sql += " AND memory_id != %s"
                params.append(exclude_id)
            sql += " ORDER BY confidence DESC, created_at DESC LIMIT %s"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            return [(self._row_to_memory(row), 0.0) for row in rows]

    def add_link(
        self,
        *,
        bank_id: str,
        source_id: str,
        target_id: str,
        relation: MemoryLinkType,
        reason: str = "",
    ) -> None:
        if source_id == target_id:
            return
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_links(
                    bank_id, source_id, target_id, relation, reason, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (source_id, target_id, relation) DO NOTHING
                """,
                (
                    bank_id,
                    source_id,
                    target_id,
                    relation.value,
                    reason,
                    _now_iso(),
                ),
            )

    def supersede(self, memory_id: str, replacement_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE memory_memories
                SET version=GREATEST(
                    version,
                    COALESCE(
                        (SELECT version + 1 FROM memory_memories WHERE memory_id=%s),
                        1
                    )
                )
                WHERE memory_id=%s
                """,
                (memory_id, replacement_id),
            )
            conn.execute(
                """
                UPDATE memory_memories
                SET status=%s, superseded_at=%s, superseded_by=%s
                WHERE memory_id=%s
                """,
                (
                    MemoryStatus.SUPERSEDED.value,
                    _now_iso(),
                    replacement_id,
                    memory_id,
                ),
            )

    def record_retrieval(
        self,
        *,
        bank_id: str,
        session_id: str,
        query_text: str,
        memory_ids: list[str],
        scores: list[float],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_retrieval_events(
                    event_id, bank_id, session_id, query_text,
                    memory_ids_json, scores_json, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(uuid.uuid4()),
                    bank_id,
                    session_id,
                    query_text,
                    _json(memory_ids),
                    _json(scores),
                    _now_iso(),
                ),
            )

    def link_counts(self, bank_id: str, memory_ids: list[str]) -> dict[str, int]:
        if not memory_ids:
            return {}
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT memory_id, COUNT(*) AS count
                FROM (
                    SELECT source_id AS memory_id
                    FROM memory_links
                    WHERE bank_id=%s AND source_id = ANY(%s)
                    UNION ALL
                    SELECT target_id AS memory_id
                    FROM memory_links
                    WHERE bank_id=%s AND target_id = ANY(%s)
                ) linked
                GROUP BY memory_id
                """,
                (bank_id, memory_ids, bank_id, memory_ids),
            ).fetchall()
        return {str(row["memory_id"]): int(row["count"]) for row in rows}

    def export_bank(self, bank_id: str) -> tuple[list[dict], list[dict]]:
        with self._connect() as conn:
            memories = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT * FROM memory_memories
                    WHERE bank_id=%s ORDER BY created_at
                    """,
                    (bank_id,),
                ).fetchall()
            ]
            links = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT * FROM memory_links
                    WHERE bank_id=%s ORDER BY created_at
                    """,
                    (bank_id,),
                ).fetchall()
            ]
        for memory in memories:
            memory.pop("memory_type", None)
            for field in (
                "applicability_json",
                "evidence_required_json",
                "avoid_json",
                "attributes_json",
            ):
                memory[field.removesuffix("_json")] = json.loads(memory.pop(field))
        return memories, links

    def bank_stats(self, bank_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            status_rows = conn.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM memory_memories
                WHERE bank_id=%s GROUP BY status
                """,
                (bank_id,),
            ).fetchall()
            episode_count = conn.execute(
                "SELECT COUNT(*) AS count FROM memory_episodes WHERE bank_id=%s",
                (bank_id,),
            ).fetchone()["count"]
            retrieval_count = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM memory_retrieval_events WHERE bank_id=%s
                """,
                (bank_id,),
            ).fetchone()["count"]
        return {
            "bank_id": bank_id,
            "memories_by_status": {
                row["status"]: int(row["count"]) for row in status_rows
            },
            "episodes": int(episode_count),
            "retrievals": int(retrieval_count),
        }

    def clear_bank(self, bank_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM memory_links WHERE bank_id=%s", (bank_id,))
            conn.execute(
                "DELETE FROM memory_retrieval_events WHERE bank_id=%s",
                (bank_id,),
            )
            conn.execute("DELETE FROM memory_episodes WHERE bank_id=%s", (bank_id,))
            conn.execute("DELETE FROM memory_memories WHERE bank_id=%s", (bank_id,))

    @staticmethod
    def _row_to_memory(row: Mapping[str, Any]) -> StoredMemory:
        return SQLiteMemoryStore._row_to_memory(row)


def create_memory_store(
    *,
    sqlite_path: str | Path,
    database_url: str | None = None,
    force_sqlite: bool = False,
) -> SQLiteMemoryStore | PostgreSQLMemoryStore:
    if force_sqlite:
        return SQLiteMemoryStore(sqlite_path)
    return PostgreSQLMemoryStore(database_url or DEFAULT_MEMORY_DATABASE_URL)
