"""Unit tests for ping output parsing."""

from __future__ import annotations

import unittest

from nika.service.pingmesh.parser import parse_ping_output

OK_OUTPUT = """
PING 10.0.0.2 (10.0.0.2) 56(84) bytes of data.

--- 10.0.0.2 ping statistics ---
4 packets transmitted, 4 received, 0% packet loss, time 3005ms
rtt min/avg/max/mdev = 0.045/0.062/0.089/0.018 ms
"""

LOSS_OUTPUT = """
--- 10.0.0.2 ping statistics ---
4 packets transmitted, 2 received, 50% packet loss, time 3005ms
rtt min/avg/max/mdev = 0.045/0.062/0.089/0.018 ms
"""

DOWN_OUTPUT = """
--- 10.0.0.2 ping statistics ---
4 packets transmitted, 0 received, +1 errors, 100% packet loss, time 3050ms
"""

NO_RTT_OUTPUT = """
--- 10.0.0.2 ping statistics ---
4 packets transmitted, 4 received, 0% packet loss, time 3005ms
"""


class ParsePingOutputTest(unittest.TestCase):
    def test_ok_output(self) -> None:
        stats = parse_ping_output(OK_OUTPUT)
        self.assertEqual(stats["tx"], 4)
        self.assertEqual(stats["rx"], 4)
        self.assertEqual(stats["loss_percent"], 0.0)
        self.assertEqual(stats["status"], "ok")
        self.assertAlmostEqual(stats["rtt_avg_ms"], 0.062)

    def test_partial_loss(self) -> None:
        stats = parse_ping_output(LOSS_OUTPUT)
        self.assertEqual(stats["status"], "ok")
        self.assertEqual(stats["loss_percent"], 50.0)

    def test_unreachable(self) -> None:
        stats = parse_ping_output(DOWN_OUTPUT)
        self.assertEqual(stats["status"], "down")
        self.assertEqual(stats["loss_percent"], 100.0)
        self.assertEqual(stats["rx"], 0)

    def test_missing_rtt(self) -> None:
        stats = parse_ping_output(NO_RTT_OUTPUT)
        self.assertEqual(stats["status"], "ok")
        self.assertIsNone(stats["rtt_avg_ms"])

    def test_network_unreachable(self) -> None:
        stats = parse_ping_output("ping: connect: Network is unreachable")
        self.assertEqual(stats["status"], "down")
        self.assertEqual(stats["loss_percent"], 100.0)
        stats = parse_ping_output("command not found")
        self.assertEqual(stats["status"], "unknown")
        self.assertIsNone(stats["tx"])


if __name__ == "__main__":
    unittest.main()
