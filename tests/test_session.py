"""Session persistence: SQLite store and Session facade."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nika.utils.session import Session
from nika.utils.session_store import SessionStore


class _SessionDbTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "sessions.db")
        self.store = SessionStore(db_path=self.db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()


class SessionStoreTest(_SessionDbTestCase):
    def test_create_and_load_unique_running_session(self) -> None:
        self.store.create_session(
            {
                "session_id": "sid-1",
                "lab_name": "dc_clos_bgp__a",
                "scenario_name": "dc_clos_bgp",
                "scenario_topo_size": "s",
                "status": "running",
                "scenario_params_json": {"lab_name": "dc_clos_bgp__a", "topo_size": "s"},
            }
        )

        loaded = self.store.get_unique_running_session()
        self.assertEqual(loaded["session_id"], "sid-1")
        self.assertEqual(loaded["scenario_params"]["lab_name"], "dc_clos_bgp__a")

    def test_multiple_running_sessions_raise(self) -> None:
        for suffix in ("a", "b"):
            self.store.create_session(
                {
                    "session_id": f"sid-{suffix}",
                    "lab_name": f"dc_clos_bgp__{suffix}",
                    "scenario_name": "dc_clos_bgp",
                    "status": "running",
                }
            )

        with self.assertRaises(ValueError):
            self.store.get_unique_running_session()

    def test_json_columns_roundtrip(self) -> None:
        self.store.create_session(
            {
                "session_id": "sid-json",
                "lab_name": "dc_clos_bgp__json",
                "scenario_name": "dc_clos_bgp",
                "status": "running",
                "problem_names_json": ["link_down", "dhcp_service_down"],
                "eval_metrics_json": {"detection_score": 1.0},
            }
        )

        row = self.store.get_session("sid-json")
        self.assertEqual(row["problem_names"], ["link_down", "dhcp_service_down"])
        self.assertEqual(row["eval_metrics_json"]["detection_score"], 1.0)

    def test_failure_injection_create_list_and_mark_ended(self) -> None:
        self.store.create_session(
            {
                "session_id": "sid-failure",
                "lab_name": "dc_clos_bgp__failure",
                "scenario_name": "dc_clos_bgp",
                "status": "running",
            }
        )
        failure_id = self.store.create_failure_injection(
            {
                "session_id": "sid-failure",
                "problem_name": "link_down",
                "root_cause_category": "link_failure",
                "scenario_name": "dc_clos_bgp",
                "lab_name": "dc_clos_bgp__failure",
                "injection_params_json": {"faulty_devices": ["pc1"], "faulty_intf": "eth0"},
                "status": "pending",
                "start_time": 123.0,
            }
        )
        self.store.update_failure_injection(failure_id, {"status": "injected"})

        rows = self.store.list_failure_injections(session_id="sid-failure")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["problem_name"], "link_down")
        self.assertEqual(rows[0]["status"], "injected")
        self.assertEqual(rows[0]["injection_params"]["faulty_intf"], "eth0")

        updated = self.store.mark_session_failures_ended("sid-failure", end_time=456.0)
        self.assertEqual(updated, 1)
        rows_after = self.store.list_failure_injections(session_id="sid-failure")
        self.assertEqual(rows_after[0]["status"], "ended")
        self.assertEqual(rows_after[0]["end_time"], 456.0)

    def test_count_failure_statuses(self) -> None:
        self.store.create_session(
            {
                "session_id": "sid-counts",
                "lab_name": "dc_clos_bgp__counts",
                "scenario_name": "dc_clos_bgp",
                "status": "running",
            }
        )
        self.store.create_failure_injection(
            {
                "session_id": "sid-counts",
                "problem_name": "link_down",
                "status": "injected",
            }
        )
        self.store.create_failure_injection(
            {
                "session_id": "sid-counts",
                "problem_name": "host_crash",
                "status": "pending",
            }
        )
        counts = self.store.count_failure_statuses(session_id="sid-counts")
        self.assertEqual(counts["injected"], 1)
        self.assertEqual(counts["pending"], 1)


class SessionTest(_SessionDbTestCase):
    def _new_session(self) -> Session:
        with patch("nika.utils.session.SessionStore", return_value=self.store):
            return Session()

    def test_load_running_session_by_id(self) -> None:
        self.store.create_session(
            {
                "session_id": "sid-1",
                "lab_name": "dc_clos_bgp__a",
                "scenario_name": "dc_clos_bgp",
                "status": "running",
            }
        )
        session = self._new_session()
        session.load_running_session(session_id="sid-1")
        self.assertEqual(session.lab_name, "dc_clos_bgp__a")

    def test_load_running_rejects_non_running_status(self) -> None:
        self.store.create_session(
            {
                "session_id": "sid-finished",
                "lab_name": "dc_clos_bgp__done",
                "scenario_name": "dc_clos_bgp",
                "status": "finished",
            }
        )
        session = self._new_session()
        with self.assertRaises(ValueError):
            session.load_running_session(session_id="sid-finished")

    def test_update_session_sets_root_cause_and_session_dir(self) -> None:
        self.store.create_session(
            {
                "session_id": "sid-rca",
                "lab_name": "dc_clos_bgp__rca",
                "scenario_name": "dc_clos_bgp",
                "status": "running",
            }
        )
        session = self._new_session()
        session.load_running_session(session_id="sid-rca")
        session.update_session("problem_names", ["link_down"])
        self.assertEqual(session.root_cause_name, "link_down")
        self.assertTrue(session.session_dir.endswith("/link_down/sid-rca"))
