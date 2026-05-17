import unittest
from unittest.mock import patch

import typer
from typer.testing import CliRunner

from nika.cli.commands.exec import exec_command


class ExecCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()
        self.app = typer.Typer()
        self.app.command("exec", context_settings={"allow_interspersed_args": False})(exec_command)

    def test_exec_uses_default_timeout(self) -> None:
        with patch("nika.cli.commands.exec._exec_in_host", return_value="ok") as exec_in_host:
            result = self.runner.invoke(self.app, ["pc1", "ip a"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("ok", result.output)
        exec_in_host.assert_called_once_with(host="pc1", command="ip a", session_id=None, timeout=10.0)

    def test_exec_accepts_unquoted_multi_token_command(self) -> None:
        with patch("nika.cli.commands.exec._exec_in_host", return_value="pong") as exec_in_host:
            result = self.runner.invoke(self.app, ["pc1", "ping", "10.0.0.2"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("pong", result.output)
        exec_in_host.assert_called_once_with(host="pc1", command="ping 10.0.0.2", session_id=None, timeout=10.0)

    def test_exec_passes_session_and_timeout(self) -> None:
        with patch("nika.cli.commands.exec._exec_in_host", return_value="done") as exec_in_host:
            result = self.runner.invoke(
                self.app,
                ["--session-id", "sid-1", "--timeout", "3.5", "pc1", "ifconfig -a"],
            )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("done", result.output)
        exec_in_host.assert_called_once_with(host="pc1", command="ifconfig -a", session_id="sid-1", timeout=3.5)

    def test_exec_treats_post_host_options_as_inner_command_args(self) -> None:
        with patch("nika.cli.commands.exec._exec_in_host", return_value="ok") as exec_in_host:
            result = self.runner.invoke(self.app, ["pc1", "ping", "--timeout", "3"])

        self.assertEqual(result.exit_code, 0)
        exec_in_host.assert_called_once_with(host="pc1", command="ping --timeout 3", session_id=None, timeout=10.0)

    def test_exec_maps_workflow_errors(self) -> None:
        with patch("nika.cli.commands.exec._exec_in_host", side_effect=ValueError("missing session")):
            result = self.runner.invoke(self.app, ["pc1", "ip a"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("missing session", result.output)

    def test_exec_rejects_non_positive_timeout(self) -> None:
        result = self.runner.invoke(self.app, ["--timeout", "0", "pc1", "ip a"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--timeout must be greater than 0", result.output)
