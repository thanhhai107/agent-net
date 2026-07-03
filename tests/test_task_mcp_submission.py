"""Unit tests for task MCP submission validation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nika.service.mcp_server.task_mcp_server import _validate_submission, submit
from nika.workflows.eval.session import detection_confusion_metrics


class TaskMcpSubmissionValidationTest(unittest.TestCase):
    def test_accepts_and_normalizes_exact_problem_names(self) -> None:
        parsed = _validate_submission(
            is_anomaly=True,
            faulty_devices=[" pc1 ", "pc1"],
            root_cause_name=[" link_down ", "link_down"],
        )

        self.assertTrue(parsed.is_anomaly)
        self.assertEqual(parsed.faulty_devices, ["pc1"])
        self.assertEqual(parsed.root_cause_name, ["link_down"])

    def test_accepts_partial_anomaly_submission_for_scoring(self) -> None:
        missing_localization = _validate_submission(
            is_anomaly=True,
            faulty_devices=[],
            root_cause_name=["link_down"],
        )
        missing_rca = _validate_submission(
            is_anomaly=True,
            faulty_devices=["pc1"],
            root_cause_name=[],
        )

        self.assertEqual(missing_localization.root_cause_name, ["link_down"])
        self.assertEqual(missing_localization.faulty_devices, [])
        self.assertEqual(missing_rca.faulty_devices, ["pc1"])
        self.assertEqual(missing_rca.root_cause_name, [])

    def test_rejects_invalid_problem_names_with_suggestion(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "bgp_blackhole_route_leak",
        ):
            _validate_submission(
                is_anomaly=True,
                faulty_devices=["router1"],
                root_cause_name=["bgp_blackhole_route"],
            )

    def test_no_anomaly_submission_must_be_empty(self) -> None:
        with self.assertRaisesRegex(ValueError, "is_anomaly=False"):
            _validate_submission(
                is_anomaly=False,
                faulty_devices=["pc1"],
                root_cause_name=["link_down"],
            )

    def test_accepts_anomaly_only_submission_for_inconclusive_localization(
        self,
    ) -> None:
        parsed = _validate_submission(
            is_anomaly=True,
            faulty_devices=[],
            root_cause_name=[],
        )

        self.assertTrue(parsed.is_anomaly)
        self.assertEqual(parsed.faulty_devices, [])
        self.assertEqual(parsed.root_cause_name, [])

    def test_submit_does_not_write_invalid_problem_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "nika.service.mcp_server.task_mcp_server.get_session_dir",
                return_value=tmp,
            ):
                result = submit(
                    is_anomaly=True,
                    faulty_devices=["router1"],
                    root_cause_name=["bgp_blackhole_route"],
                )

            self.assertIsInstance(result, dict)
            self.assertIn("bgp_blackhole_route_leak", str(result))
            self.assertFalse((Path(tmp) / "submission.json").exists())

    def test_submit_writes_normalized_submission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "nika.service.mcp_server.task_mcp_server.get_session_dir",
                return_value=tmp,
            ):
                result = submit(
                    is_anomaly=True,
                    faulty_devices=[" pc1 ", "pc1"],
                    root_cause_name=[" link_down ", "link_down"],
                )

            submission = json.loads((Path(tmp) / "submission.json").read_text())

        self.assertEqual(result, ["Submission success."])
        self.assertEqual(submission["faulty_devices"], ["pc1"])
        self.assertEqual(submission["root_cause_name"], ["link_down"])


class DetectionConfusionMetricsTest(unittest.TestCase):
    def test_false_positive_anomaly_submission_is_visible(self) -> None:
        metrics = detection_confusion_metrics(
            {"is_anomaly": False},
            {"is_anomaly": True},
        )

        self.assertTrue(metrics["detection_valid"])
        self.assertEqual(metrics["detection_tp"], 0)
        self.assertEqual(metrics["detection_tn"], 0)
        self.assertEqual(metrics["detection_fp"], 1)
        self.assertEqual(metrics["detection_fn"], 0)
        self.assertEqual(metrics["detection_precision"], 0.0)
        self.assertEqual(metrics["detection_false_positive_rate"], 1.0)
        self.assertIsNone(metrics["detection_recall"])

    def test_true_positive_anomaly_submission_has_full_detection_breakdown(
        self,
    ) -> None:
        metrics = detection_confusion_metrics(
            {"is_anomaly": True},
            {"is_anomaly": True},
        )

        self.assertTrue(metrics["detection_valid"])
        self.assertEqual(metrics["detection_tp"], 1)
        self.assertEqual(metrics["detection_tn"], 0)
        self.assertEqual(metrics["detection_fp"], 0)
        self.assertEqual(metrics["detection_fn"], 0)
        self.assertEqual(metrics["detection_precision"], 1.0)
        self.assertEqual(metrics["detection_recall"], 1.0)
        self.assertEqual(metrics["detection_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
