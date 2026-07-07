"""Integration tests for Containerlab CLI failure injection.

Prerequisites:
  - Docker running
  - containerlab CLI on PATH
  - Run via: uv run python -m unittest tests.failure_inject_verify.test_clab_failure_inject -v
"""

from __future__ import annotations

import shutil
import unittest

from tests.integration_base import PerTestEnvTestCase

HOST = "leaf1"
INTF = "e1-1"
LINK_PARAMS = {"host_name": HOST, "intf_name": INTF}


@unittest.skipUnless(shutil.which("clab"), "containerlab not installed")
class ClabLinkFailureVerifyTest(PerTestEnvTestCase):
    SCENARIO = "min3clos"
    ENV_RUN_ARGS = ["--backend", "containerlab"]

    def test_link_down(self) -> None:
        self._inject_failure("link_down", LINK_PARAMS)
        self._assert_failure_injected("link_down")


if __name__ == "__main__":
    unittest.main()
