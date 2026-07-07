"""Tests for flat problem registration."""

from __future__ import annotations

import unittest

from nika.orchestrator.problems.prob_pool import (
    _PROBLEMS,
    get_problem_class,
    list_avail_problem_names,
)
from nika.orchestrator.problems.problem_base import ProblemBase, ProblemGroundTruth


class ProbPoolTest(unittest.TestCase):
    def test_single_registration_per_root_cause(self) -> None:
        names = list_avail_problem_names()
        self.assertGreater(len(names), 50)
        self.assertEqual(len(names), len(set(names)))
        for name in names:
            cls = get_problem_class(name)
            self.assertIsNotNone(cls)
            assert cls is not None
            self.assertTrue(issubclass(cls, ProblemBase))
            self.assertIsNotNone(cls.META)

    def test_flat_registry_structure(self) -> None:
        for cls in _PROBLEMS.values():
            self.assertTrue(issubclass(cls, ProblemBase))
            self.assertIsNotNone(cls.META.root_cause_name)

    def test_bgp_hijacking_registered_once_under_attack(self) -> None:
        cls = get_problem_class("bgp_hijacking")
        self.assertIsNotNone(cls)
        assert cls is not None
        self.assertEqual(str(cls.root_cause_category), "network_under_attack")

    def test_get_ground_truth_fields(self) -> None:
        cls = get_problem_class("link_down")
        self.assertIsNotNone(cls)
        assert cls is not None
        problem = cls(scenario_name=None)
        problem.set_faulty_devices(["pc1"])
        gt = problem.get_ground_truth()
        self.assertIsInstance(gt, ProblemGroundTruth)
        self.assertTrue(gt.is_anomaly)
        self.assertEqual(gt.faulty_devices, ["pc1"])
        self.assertEqual(gt.root_cause_name, ["link_down"])
        self.assertEqual(gt.root_cause_category, "link_failure")


if __name__ == "__main__":
    unittest.main()
