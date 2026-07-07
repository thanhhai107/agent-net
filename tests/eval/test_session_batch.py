"""Unit tests for batch eval under --result_dir."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nika.evaluator.result_log import EVAL_METRICS_FILENAME, MESSAGES_FILENAME
from nika.utils.session_artifacts import RUN_FILENAME
from nika.workflows.eval.session import (
    _iter_eval_session_ids,
    run_eval_metrics,
    run_llm_judge,
)


def _write_closed_session(
    results_root: Path,
    session_id: str,
    *,
    with_submission: bool = True,
) -> Path:
    session_dir = results_root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "session_id": session_id,
        "status": "finished",
        "scenario_name": "simple_bgp",
        "end_time": "2026-01-01T12:00:00",
    }
    (session_dir / RUN_FILENAME).write_text(json.dumps(run_meta), encoding="utf-8")
    (session_dir / "ground_truth.json").write_text(
        json.dumps(
            {
                "is_anomaly": True,
                "root_cause_name": ["link_down"],
                "faulty_devices": ["pc1"],
            }
        ),
        encoding="utf-8",
    )
    if with_submission:
        (session_dir / "submission.json").write_text(
            json.dumps(
                {
                    "is_anomaly": True,
                    "root_cause_name": ["link_down"],
                    "faulty_devices": ["pc1"],
                }
            ),
            encoding="utf-8",
        )
    (session_dir / MESSAGES_FILENAME).write_text("", encoding="utf-8")
    return session_dir


class EvalSessionBatchTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.results_root = self.root / "results" / "folder"
        self.results_root.mkdir(parents=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_iter_eval_session_ids_returns_all_under_result_dir(self) -> None:
        _write_closed_session(self.results_root, "20260101-120000-aaa111")
        _write_closed_session(self.results_root, "20260101-120000-bbb222")

        session_ids = _iter_eval_session_ids(result_dir=str(self.results_root))

        self.assertEqual(
            sorted(session_ids),
            ["20260101-120000-aaa111", "20260101-120000-bbb222"],
        )

    def test_iter_eval_session_ids_errors_when_default_dir_has_multiple(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            default_root = Path(tmp_name) / "results"
            default_root.mkdir()
            _write_closed_session(default_root, "20260101-120000-aaa111")
            _write_closed_session(default_root, "20260101-120000-bbb222")

            with patch(
                "nika.workflows.eval.session.resolve_results_root",
                return_value=default_root,
            ):
                with self.assertRaisesRegex(ValueError, "Multiple closed sessions"):
                    _iter_eval_session_ids()

    def test_run_eval_metrics_batch_under_result_dir(self) -> None:
        sid_one = "20260101-120000-aaa111"
        sid_two = "20260101-120000-bbb222"
        _write_closed_session(self.results_root, sid_one)
        _write_closed_session(self.results_root, sid_two)

        run_eval_metrics(result_dir=str(self.results_root))

        for sid in (sid_one, sid_two):
            metrics_path = self.results_root / sid / EVAL_METRICS_FILENAME
            self.assertTrue(metrics_path.exists(), f"missing metrics for {sid}")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["detection_score"], 1.0)

    def test_run_llm_judge_batch_under_result_dir(self) -> None:
        sid_one = "20260101-120000-aaa111"
        sid_two = "20260101-120000-bbb222"
        _write_closed_session(self.results_root, sid_one)
        _write_closed_session(self.results_root, sid_two)

        with patch("nika.workflows.eval.session.LLMJudge") as judge_cls:
            judge = judge_cls.return_value
            run_llm_judge("openai", "gpt-test", result_dir=str(self.results_root))

        self.assertEqual(judge.evaluate_agent.call_count, 2)
        save_paths = {
            call.kwargs["save_path"] for call in judge.evaluate_agent.call_args_list
        }
        self.assertEqual(
            save_paths,
            {
                str(self.results_root / sid_one / "llm_judge.json"),
                str(self.results_root / sid_two / "llm_judge.json"),
            },
        )


if __name__ == "__main__":
    unittest.main()
