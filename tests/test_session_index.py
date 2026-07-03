"""Unit tests for SQLite session index behavior not covered by test_pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nika.utils.session import Session
from nika.utils.session_index import SessionIndex
from nika.utils.session_store import SessionStore

RUN_FILENAME = "run.json"
GROUND_TRUTH_FILENAME = "ground_truth.json"
EVAL_METRICS_FILENAME = "eval_metrics.json"


class SessionIndexTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.sessions_dir = self.root / "sessions"
        self.sessions_dir.mkdir()
        self.db_path = self.root / "sessions.db"
        self.results_dir = self.root / "results"
        self.results_dir.mkdir()
        self.index = SessionIndex(self.db_path)
        self.store = SessionStore(self.sessions_dir, self.db_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _create_session(self, session_id: str = "20260101-120000-abc123") -> None:
        self.store.create_session(
            {
                "session_id": session_id,
                "lab_name": f"bgp__{session_id[-6:]}",
                "scenario_name": "bgp",
                "scenario_topo_size": "s",
                "session_dir": str(self.results_dir / session_id),
                "status": "running",
            }
        )

    def test_failure_injection_increments_count(self) -> None:
        self._create_session()
        self.store.create_failure_injection(
            {
                "session_id": "20260101-120000-abc123",
                "problem_name": "link_down",
                "scenario_name": "bgp",
                "lab_name": "bgp__abc123",
            }
        )
        row = self.index.get_row("20260101-120000-abc123")
        assert row is not None
        self.assertEqual(row["failure_count"], 1)

    def test_purge_and_truncate(self) -> None:
        self._create_session()
        self.index.purge("20260101-120000-abc123")
        self.assertIsNone(self.index.get_row("20260101-120000-abc123"))
        self._create_session("20260102-120000-def456")
        self.index.truncate()
        self.assertEqual(len(self.index.list_sessions(running_only=False)), 0)

    def test_rebuild_from_results(self) -> None:
        session_id = "20260103-120000-fff999"
        session_dir = self.results_dir / session_id
        session_dir.mkdir()
        run_meta = {
            "session_id": session_id,
            "status": "finished",
            "scenario_name": "bgp",
            "lab_name": "bgp__fff999",
            "agent_type": "mock",
            "problem_names": ["link_down"],
            "root_cause_name": "link_down",
        }
        (session_dir / RUN_FILENAME).write_text(json.dumps(run_meta), encoding="utf-8")
        (session_dir / GROUND_TRUTH_FILENAME).write_text(
            json.dumps({"faulty_devices": ["pc1"], "root_cause_name": ["link_down"]}),
            encoding="utf-8",
        )
        (session_dir / EVAL_METRICS_FILENAME).write_text(
            json.dumps({"detection_score": 1.0, "localization_f1": 1.0, "rca_f1": 1.0}),
            encoding="utf-8",
        )

        count = self.index.rebuild_from_results(self.results_dir)
        self.assertEqual(count, 1)
        row = self.index.get_row(session_id)
        assert row is not None
        self.assertEqual(row["status"], "finished")
        self.assertEqual(row["agent_type"], "mock")
        self.assertEqual(row["faulty_devices"], ["pc1"])
        self.assertEqual(row["detection_score"], 1.0)
        self.assertEqual(row["failure_count"], 1)

    def test_load_closed_session_from_custom_results_dir(self) -> None:
        session_id = "20260104-120000-custom"
        custom_root = self.root / "custom-results"
        session_dir = custom_root / session_id
        session_dir.mkdir(parents=True)
        run_meta = {
            "session_id": session_id,
            "status": "finished",
            "scenario_name": "bgp",
            "lab_name": "bgp__custom",
            "session_dir": str(session_dir),
        }
        (session_dir / RUN_FILENAME).write_text(json.dumps(run_meta), encoding="utf-8")

        session = Session().load_closed_session(
            session_id=session_id,
            results_dir=custom_root,
        )

        self.assertEqual(session.session_id, session_id)
        self.assertEqual(Path(session.session_dir), session_dir)


if __name__ == "__main__":
    unittest.main()
