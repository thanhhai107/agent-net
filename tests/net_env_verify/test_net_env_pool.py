"""Shared unit tests for all net_env scenarios (no Docker required).

Covers registration, CLI discovery, and basic instantiation for every
scenario in ``net_env_pool``.  Scenario-specific structure checks live in
``test_<scenario>_verify.py`` under this directory.

Prerequisites:
  - Run via: uv run python -m unittest tests/net_env_verify/test_net_env_pool.py -v
"""

from __future__ import annotations

import unittest

from typer.testing import CliRunner

from nika.cli.main import app
from nika.net_env.net_env_pool import get_net_env_instance, list_all_net_envs


class NetEnvPoolTest(unittest.TestCase):
    """Verify every registered scenario is discoverable and instantiable."""

    def test_pool_is_non_empty(self) -> None:
        self.assertGreater(len(list_all_net_envs()), 0)

    def test_get_instance_for_each_scenario(self) -> None:
        """get_net_env_instance must succeed for every registered scenario."""
        for name, cls in list_all_net_envs().items():
            with self.subTest(scenario=name):
                inst = get_net_env_instance(name)
                self.assertIsNotNone(inst)
                self.assertEqual(inst.name, name)
                self.assertIsInstance(inst, cls)

    def test_env_list_cli_includes_all_scenarios(self) -> None:
        """``nika env list`` output must include every registered scenario."""
        runner = CliRunner()
        result = runner.invoke(app, ["env", "list"])
        self.assertEqual(result.exit_code, 0, result.output)
        for name in list_all_net_envs():
            with self.subTest(scenario=name):
                self.assertIn(name, result.output)


if __name__ == "__main__":
    unittest.main()
