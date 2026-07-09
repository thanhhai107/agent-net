"""Unit tests for SRL API."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from nika.service.containerlab.adapters import LabRuntimeContainerlabAPI
from nika.service.containerlab.srl_api import NIKA_BGP_WITHDRAW, SRLAPIMixin


class _SRLStub(SRLAPIMixin):
    backend = "containerlab"

    def __init__(self) -> None:
        self.exec_calls: list[tuple[str, str]] = []

    def exec_cmd(self, host_name: str, command: str, timeout: float = 10) -> str:
        self.exec_calls.append((host_name, command))
        return ""


class SrlApiTest(unittest.TestCase):
    def test_uses_srl_router_true_when_sr_cli_present(self) -> None:
        api = _SRLStub()
        api.exec_cmd = MagicMock(return_value="/usr/bin/sr_cli")  # type: ignore[method-assign]
        self.assertTrue(api.uses_srl_router("leaf1"))

    def test_uses_srl_router_false_on_kathara(self) -> None:
        api = _SRLStub()
        api.backend = "kathara"  # type: ignore[misc]
        self.assertFalse(api.uses_srl_router("router1"))

    def test_srl_get_bgp_as_parses_output(self) -> None:
        runtime = MagicMock()
        runtime.backend = "containerlab"
        runtime.exec.return_value = "Global AS number  : 65001"
        adapter = LabRuntimeContainerlabAPI(runtime)
        self.assertEqual(adapter.srl_get_bgp_as("leaf1"), 65001)

    def test_srl_exec_cli_uses_quoted_command(self) -> None:
        runtime = MagicMock()
        runtime.backend = "containerlab"
        runtime.exec.return_value = "ok"
        adapter = LabRuntimeContainerlabAPI(runtime)
        adapter.srl_exec_cli("leaf1", "show version")
        runtime.exec.assert_called_once()
        cmd = runtime.exec.call_args[0][1]
        self.assertIn('sr_cli "show version"', cmd)

    def test_srl_bgp_acl_present(self) -> None:
        runtime = MagicMock()
        runtime.backend = "containerlab"
        runtime.exec.return_value = (
            "Chain INPUT (policy ACCEPT)\n"
            "DROP       tcp  --  0.0.0.0/0  0.0.0.0/0  tcp dpt:179"
        )
        adapter = LabRuntimeContainerlabAPI(runtime)
        self.assertTrue(adapter.srl_bgp_acl_drop_179_present("leaf1"))

    def test_srl_withdraw_bgp_prefix_uses_candidate(self) -> None:
        api = _SRLStub()
        api.srl_withdraw_bgp_prefix("leaf1", "10.0.0.24/31")
        self.assertEqual(len(api.exec_calls), 1)
        script = api.exec_calls[0][1]
        self.assertIn("enter candidate", script)
        self.assertIn(NIKA_BGP_WITHDRAW, script)
        self.assertIn("10.0.0.24/31", script)
        self.assertIn("default-action policy-result accept", script)
        self.assertIn("match prefix-set", script)
        self.assertIn("export-policy", script)

    def test_srl_bgp_prefix_withdrawn(self) -> None:
        runtime = MagicMock()
        runtime.backend = "containerlab"
        runtime.exec.side_effect = [
            f"policy {NIKA_BGP_WITHDRAW} prefix 10.0.0.24/31",
            f"group clos01 export-policy {NIKA_BGP_WITHDRAW}",
        ]
        adapter = LabRuntimeContainerlabAPI(runtime)
        self.assertTrue(adapter.srl_bgp_prefix_withdrawn("leaf1", "10.0.0.24/31"))


if __name__ == "__main__":
    unittest.main()
