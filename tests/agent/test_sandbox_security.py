"""Sandbox container security probes (requires Docker + sandbox image)."""

from __future__ import annotations

import unittest

from tests.agent.sandbox_support import (
    docker_available,
    run_security_probe_with_gateway,
)


@unittest.skipUnless(docker_available(), "docker not available")
class SandboxSecurityIntegrationTest(unittest.TestCase):
    def test_security_probe_with_gateway(self) -> None:
        run_security_probe_with_gateway()


if __name__ == "__main__":
    unittest.main()
