"""Unit tests for backend resolution and session persistence."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nika.runtime.factory import resolve_backend, runtime_for_session
from nika.utils.session import Session
from nika.utils.session_store import SessionStore


class BackendResolveTest(unittest.TestCase):
    def test_legacy_session_defaults_to_kathara(self) -> None:
        meta = {
            "session_id": "legacy-1",
            "lab_name": "simple_bgp__abc123",
            "scenario_name": "simple_bgp",
            "scenario_params": {"lab_name": "simple_bgp__abc123"},
        }
        self.assertEqual(resolve_backend(meta), "kathara")

    def test_session_backend_field(self) -> None:
        meta = {"backend": "containerlab", "scenario_params": {}}
        self.assertEqual(resolve_backend(meta), "containerlab")

    def test_scenario_params_backend_fallback(self) -> None:
        meta = {"scenario_params": {"backend": "containerlab"}}
        self.assertEqual(resolve_backend(meta), "containerlab")

    def test_containerlab_only_scenario_infers_backend(self) -> None:
        meta = {
            "session_id": "clab-1",
            "lab_name": "min5clos__abc123",
            "scenario_name": "min5clos",
            "scenario_params": {"lab_name": "min5clos__abc123"},
        }
        self.assertEqual(resolve_backend(meta), "containerlab")

    def test_init_session_persists_backend_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_root = Path(tmp) / "results"
            sessions_dir = Path(tmp) / "runtime" / "sessions"
            db_path = Path(tmp) / "runtime" / "sessions.db"
            store = SessionStore(sessions_dir=sessions_dir, db_path=db_path)
            session = Session()
            session.store = store
            session.init_session(
                session_id="20260706-120000-abc123",
                scenario_name="srlceos01",
                lab_name="srlceos01__tag",
                scenario_topo_size=None,
                scenario_params={"lab_name": "srlceos01__tag", "backend": "containerlab"},
                result_dir=results_root,
                backend="containerlab",
                topology_file="/tmp/topo.clab.yml",
                runtime_workdir="/tmp/runtime/clab",
            )
            stored = store.get_session("20260706-120000-abc123")
            self.assertEqual(stored["backend"], "containerlab")
            self.assertEqual(stored["topology_file"], "/tmp/topo.clab.yml")
            self.assertEqual(stored["runtime_workdir"], "/tmp/runtime/clab")
            run_path = results_root / "20260706-120000-abc123" / "run.json"
            run_meta = json.loads(run_path.read_text(encoding="utf-8"))
            self.assertEqual(run_meta["backend"], "containerlab")

    def test_runtime_for_session_containerlab_requires_topology(self) -> None:
        with self.assertRaises(ValueError):
            runtime_for_session(
                {
                    "backend": "containerlab",
                    "lab_name": "srlceos01__x",
                    "scenario_name": "srlceos01",
                }
            )


if __name__ == "__main__":
    unittest.main()
