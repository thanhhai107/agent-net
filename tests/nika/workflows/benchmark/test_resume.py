"""Unit tests for benchmark resume logic.

Run via: uv run python -m unittest tests.nika.workflows.benchmark.test_resume -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from nika.workflows.benchmark import run as benchmark_run
from nika.utils.session_artifacts import RUN_FILENAME
from nika.workflows.benchmark.resume import (
    benchmark_row_fingerprint,
    cleanup_benchmark_session,
    is_benchmark_case_complete,
    scan_benchmark_cases,
    session_matches_row,
)

ROW_A = {
    "scenario": "simple_bgp",
    "problem": "link_down",
    "topo_size": "",
    "inject": {"host_name": "pc1", "intf_name": "eth0"},
}
ROW_B = {
    "scenario": "simple_bgp",
    "problem": "link_flap",
    "topo_size": "",
    "inject": {"host_name": "pc1", "intf_name": "eth0"},
}


def _write_finished_session(session_dir: Path, row: dict, session_id: str) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "session_id": session_id,
        "status": "finished",
        "benchmark_fingerprint": benchmark_row_fingerprint(row),
    }
    (session_dir / RUN_FILENAME).write_text(json.dumps(run_meta), encoding="utf-8")


class BenchmarkResumeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.results_root = Path(self.tmp.name) / "results" / "list1"
        self.results_root.mkdir(parents=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_is_complete_requires_finished_status_not_eval_artifacts(self) -> None:
        session_dir = self.results_root / "20260101-120000-abc123"
        _write_finished_session(session_dir, ROW_A, "20260101-120000-abc123")
        self.assertTrue(is_benchmark_case_complete(session_dir, ROW_A))
        self.assertFalse((session_dir / "eval_metrics.json").exists())
        self.assertFalse((session_dir / "llm_judge.json").exists())

    def test_fingerprint_distinguishes_same_scenario_problem(self) -> None:
        row_a2 = {**ROW_A, "inject": {"host_name": "pc2", "intf_name": "eth0"}}
        meta = {"benchmark_fingerprint": benchmark_row_fingerprint(ROW_A)}
        self.assertTrue(session_matches_row(meta, ROW_A))
        self.assertFalse(session_matches_row(meta, row_a2))

    def test_incomplete_running_session_is_not_complete(self) -> None:
        session_dir = self.results_root / "20260101-120000-run111"
        session_dir.mkdir(parents=True)
        run_meta = {
            "session_id": "20260101-120000-run111",
            "status": "running",
            "benchmark_fingerprint": benchmark_row_fingerprint(ROW_A),
            "end_time": "2026-01-01T12:05:00",
        }
        (session_dir / RUN_FILENAME).write_text(json.dumps(run_meta), encoding="utf-8")
        self.assertFalse(is_benchmark_case_complete(session_dir, ROW_A))

    def test_failed_session_with_agent_end_time_is_not_complete(self) -> None:
        session_dir = self.results_root / "20260101-120000-failed"
        session_dir.mkdir(parents=True)
        run_meta = {
            "session_id": "20260101-120000-failed",
            "status": "failed",
            "benchmark_fingerprint": benchmark_row_fingerprint(ROW_A),
            "end_time": "2026-01-01T12:05:00",
        }
        (session_dir / RUN_FILENAME).write_text(json.dumps(run_meta), encoding="utf-8")
        self.assertFalse(is_benchmark_case_complete(session_dir, ROW_A))

    def test_scan_benchmark_cases_skips_completed_and_returns_pending(self) -> None:
        completed_dir = self.results_root / "20260101-120000-done111"
        _write_finished_session(completed_dir, ROW_A, "20260101-120000-done111")

        _results_root, pending = scan_benchmark_cases(
            rows=[ROW_A, ROW_B],
            result_dir=str(self.results_root),
            resume=True,
        )
        self.assertEqual(pending, [1])

    def test_scan_benchmark_cases_all_complete(self) -> None:
        for index, row in enumerate([ROW_A, ROW_B]):
            session_id = f"2026010{index}-120000-done"
            _write_finished_session(self.results_root / session_id, row, session_id)

        _results_root, pending = scan_benchmark_cases(
            rows=[ROW_A, ROW_B],
            result_dir=str(self.results_root),
            resume=True,
        )
        self.assertEqual(pending, [])

    def test_session_without_fingerprint_does_not_match(self) -> None:
        session_dir = self.results_root / "20260101-120000-legacy"
        session_dir.mkdir(parents=True)
        (session_dir / RUN_FILENAME).write_text(
            json.dumps({"session_id": "20260101-120000-legacy", "status": "finished"}),
            encoding="utf-8",
        )
        self.assertFalse(is_benchmark_case_complete(session_dir, ROW_A))

    def test_cleanup_removes_session_dir(self) -> None:
        session_dir = self.results_root / "20260101-120000-bad111"
        session_dir.mkdir(parents=True)
        (session_dir / RUN_FILENAME).write_text("{}", encoding="utf-8")
        cleanup_benchmark_session("20260101-120000-bad111", session_dir)
        self.assertFalse(session_dir.exists())

    def test_cleanup_undeploys_lab_from_persisted_failed_session(self) -> None:
        session_id = "20260101-120000-failed"
        session_dir = self.results_root / session_id
        session_dir.mkdir(parents=True)
        run_meta = {
            "session_id": session_id,
            "status": "failed",
            "scenario_name": "simple_bgp",
            "lab_name": "simple_bgp__old-case",
            "backend": "kathara",
        }
        (session_dir / RUN_FILENAME).write_text(json.dumps(run_meta), encoding="utf-8")
        runtime = Mock()
        store = Mock()
        store.get_session.side_effect = FileNotFoundError

        with (
            patch("nika.workflows.benchmark.resume.SessionStore", return_value=store),
            patch("nika.workflows.benchmark.resume.SessionIndex") as index,
            patch(
                "nika.workflows.benchmark.resume.runtime_for_session",
                return_value=runtime,
            ) as runtime_factory,
        ):
            cleanup_benchmark_session(session_id, session_dir)

        runtime_factory.assert_called_once_with(run_meta)
        runtime.destroy.assert_called_once_with()
        index.return_value.purge.assert_called_once_with(session_id)
        self.assertFalse(session_dir.exists())

    def test_case_uses_session_scoped_cleanup_after_failure(self) -> None:
        close_failed_case = Mock(return_value=None)
        store = Mock()
        store.get_session.return_value = {"session_dir": str(self.results_root)}

        with (
            patch.object(
                benchmark_run,
                "scenario_requires_topo_size",
                return_value=False,
            ),
            patch.object(benchmark_run, "validate_inject_params"),
            patch.object(benchmark_run, "start_net_env", return_value="session-1"),
            patch.object(benchmark_run, "SessionStore", return_value=store),
            patch.object(
                benchmark_run,
                "inject_failure",
                side_effect=RuntimeError("inject failed"),
            ),
            patch.object(
                benchmark_run,
                "close_session_after_failure",
                close_failed_case,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "inject failed"):
                benchmark_run.run_single_case(
                    problem="link_down",
                    scenario="simple_bgp",
                    topo_size="",
                    agent_type="react",
                    llm_provider="custom",
                    model="model",
                    max_steps=1,
                    inject_params={"host_name": "pc1", "intf_name": "eth0"},
                )

        close_failed_case.assert_called_once()
        self.assertEqual(close_failed_case.call_args.args[0], "session-1")


if __name__ == "__main__":
    unittest.main()
