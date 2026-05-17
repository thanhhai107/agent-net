import unittest
from unittest.mock import MagicMock, patch

import typer

from nika.cli.commands.env import env_ps
from nika.cli.commands.failure import _parse_set_options, failure_describe, failure_ps


class FailureCliTest(unittest.TestCase):
    def test_parse_set_options(self) -> None:
        parsed = _parse_set_options(["host_name=pc1", "intf_name=eth0"])
        self.assertEqual(parsed, {"host_name": "pc1", "intf_name": "eth0"})

    def test_parse_set_options_rejects_invalid_item(self) -> None:
        with self.assertRaises(typer.BadParameter):
            _parse_set_options(["host_name"])

    def test_failure_ps_prints_rows(self) -> None:
        fake_store = MagicMock()
        fake_store.list_failure_injections.return_value = [
            {
                "id": 1,
                "session_id": "sid-1",
                "problem_name": "link_down",
                "status": "injected",
                "start_time": 123.0,
                "end_time": None,
                "injection_params_json": {"faulty_devices": ["pc1"], "faulty_intf": "eth0"},
            }
        ]
        with (
            patch("nika.utils.session_store.SessionStore", return_value=fake_store),
            patch("nika.cli.commands.failure.typer.echo") as echo,
        ):
            failure_ps(session_id="sid-1")

        fake_store.list_failure_injections.assert_called_once_with(session_id="sid-1")
        self.assertTrue(echo.called)
        self.assertIn("problem=link_down", echo.call_args[0][0])
        self.assertIn("status=injected", echo.call_args[0][0])

    def test_failure_describe_prints_schema(self) -> None:
        with patch("nika.cli.commands.failure.typer.echo") as echo:
            failure_describe("link_down")
        self.assertTrue(echo.called)
        out = "\n".join(call.args[0] for call in echo.call_args_list if call.args)
        self.assertIn("link_down", out)
        self.assertIn("Parameters:", out)
        self.assertIn("host_name", out)

    def test_env_ps_appends_failure_summary(self) -> None:
        fake_store = MagicMock()
        fake_store.list_running_sessions.return_value = [
            {
                "session_id": "sid-1",
                "lab_name": "lab-a",
                "scenario_name": "simple_bgp",
                "scenario_topo_size": None,
                "created_at": "2026-05-16T00:00:00+00:00",
            }
        ]
        fake_store.count_failure_statuses.return_value = {"injected": 1, "ended": 2}
        with (
            patch("nika.utils.session_store.SessionStore", return_value=fake_store),
            patch("nika.cli.commands.env.typer.echo") as echo,
        ):
            env_ps()

        self.assertTrue(echo.called)
        printed = echo.call_args[0][0]
        self.assertIn("session_id=sid-1", printed)
        self.assertIn("failures=ended:2,injected:1", printed)


if __name__ == "__main__":
    unittest.main()
