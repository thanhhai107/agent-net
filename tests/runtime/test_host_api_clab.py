"""Unit tests for ContainerlabBaseAPI on Containerlab sessions."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import MagicMock

from nika.service.containerlab import ContainerlabBaseAPI


class RuntimeHostApiClabTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = MagicMock()
        self.runtime.lab_name = "min3clos__x"
        self.runtime.list_nodes.return_value = [
            "leaf1",
            "leaf2",
            "spine",
            "client1",
            "client2",
        ]
        self.runtime.get_host_ip.side_effect = lambda host, iface="eth0", **_: {
            "client1": "10.0.0.25",
            "client2": "10.0.0.27",
        }.get(host)
        self.runtime.exec.return_value = "ok"
        self.api = ContainerlabBaseAPI(self.runtime)

    def test_get_host_net_config(self) -> None:
        cfg = self.api.get_host_net_config("client1")
        self.assertEqual(cfg["host_name"], "client1")
        self.assertIn("ip_addr", cfg)

    def test_ping_pair_resolves_host_name(self) -> None:
        self.api.exec_cmd = MagicMock(return_value="1 received")
        out = self.api.ping_pair("client1", "client2", count=1)
        self.assertIn("received", out)
        self.api.exec_cmd.assert_called_once()
        self.assertIn("10.0.0.27", self.api.exec_cmd.call_args[0][1])

    def test_get_reachability_json(self) -> None:
        async def _ping(host: str, dst_ip: str) -> dict:
            return {"tx": 2, "rx": 2, "loss_percent": 0.0, "status": "ok"}

        self.api._check_ping_success_async = _ping  # type: ignore[method-assign]
        payload = asyncio.run(self.api.get_reachability())
        self.assertIn("client1", payload)
        self.assertIn("results", payload)


if __name__ == "__main__":
    unittest.main()
