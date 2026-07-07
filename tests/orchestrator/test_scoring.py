"""Tests for rule-based eval scoring."""

from __future__ import annotations

import unittest

from nika.orchestrator.eval.scoring import (
    score_detection,
    score_localization,
    score_rca,
)


class ScoringTest(unittest.TestCase):
    def test_score_detection_match(self) -> None:
        self.assertEqual(
            score_detection({"is_anomaly": True}, {"is_anomaly": True}), 1.0
        )

    def test_score_detection_mismatch(self) -> None:
        self.assertEqual(
            score_detection({"is_anomaly": False}, {"is_anomaly": True}), 0.0
        )

    def test_score_localization_perfect(self) -> None:
        acc, prec, rec, f1 = score_localization(
            {"faulty_devices": ["pc1", "router1"]},
            {"faulty_devices": ["pc1", "router1"]},
        )
        self.assertEqual((acc, prec, rec, f1), (1.0, 1.0, 1.0, 1.0))

    def test_score_rca_partial(self) -> None:
        acc, prec, rec, f1 = score_rca(
            {"root_cause_name": ["link_down", "extra"]},
            {"root_cause_name": ["link_down"]},
        )
        self.assertEqual(acc, 1.0)
        self.assertEqual(prec, 0.5)
        self.assertEqual(rec, 1.0)
        self.assertEqual(f1, 0.6667)


if __name__ == "__main__":
    unittest.main()
