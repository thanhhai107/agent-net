"""Unit tests for KatharaRuntime and factory defaults."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from nika.runtime.factory import resolve_backend, runtime_for_net_env
from nika.runtime.kathara import KatharaRuntime
from nika.net_env.kathara.interdomain_routing.simple_bgp.lab import SimpleBGP
from nika.workflows.session import close as session_close


class KatharaRuntimeCompatTest(unittest.TestCase):
    def test_factory_default_backend(self) -> None:
        self.assertEqual(resolve_backend({}), "kathara")

    def test_runtime_for_kathara_net_env(self) -> None:
        env = SimpleBGP()
        runtime = runtime_for_net_env(env)
        self.assertIsInstance(runtime, KatharaRuntime)
        self.assertEqual(runtime.lab_name, env.name)

    def test_kathara_runtime_list_nodes_before_deploy(self) -> None:
        env = SimpleBGP()
        runtime = KatharaRuntime(env)
        nodes = runtime.list_nodes()
        self.assertIn("pc1", nodes)
        self.assertIn("router1", nodes)

    def test_global_cleanup_uses_official_backend_commands(self) -> None:
        kathara = Mock()
        clab_result = Mock(returncode=0, stdout="", stderr="")

        with (
            patch.object(session_close.Kathara, "get_instance", return_value=kathara),
            patch.object(session_close.shutil, "which", return_value="/usr/bin/clab"),
            patch.object(
                session_close.subprocess,
                "run",
                return_value=clab_result,
            ) as run,
        ):
            session_close.clean_emulation_environment()

        kathara.wipe.assert_called_once_with(all_users=False)
        run.assert_called_once_with(
            [
                "clab",
                "destroy",
                "--all",
                "--cleanup",
                "--yes",
                "--log-level",
                "error",
            ],
            check=False,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
