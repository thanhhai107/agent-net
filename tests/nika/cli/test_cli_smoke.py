"""CLI smoke tests: command modules, handler workflow imports, and safe invocations.

These tests catch adapter drift between CLI handlers and workflow/runtime modules.
``--help`` alone is insufficient because lazy loading and deferred imports can hide
ImportError until a command actually runs.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import unittest
from pathlib import Path

from typer.testing import CliRunner

from nika.cli.main import app

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER = CliRunner()

# Modules imported at load time by ``nika.cli.commands.*``.
CLI_COMMAND_MODULES = [
    "nika.cli.commands.agent",
    "nika.cli.commands.benchmark",
    "nika.cli.commands.env",
    "nika.cli.commands.evaluation",
    "nika.cli.commands.exec",
    "nika.cli.commands.failure",
    "nika.cli.commands.session",
    "nika.cli.commands.traffic",
]

# Workflow modules imported inside CLI command handlers.
CLI_HANDLER_WORKFLOWS = [
    "nika.workflows.agent.run",
    "nika.workflows.benchmark.run",
    "nika.workflows.env.start",
    "nika.workflows.eval.clean",
    "nika.workflows.eval.session",
    "nika.workflows.eval.summary",
    "nika.workflows.exec.command",
    "nika.workflows.failure.inject",
    "nika.workflows.session.close",
    "nika.workflows.session.containers",
    "nika.workflows.session.inspect",
    "nika.workflows.session.list",
]

# Every registered subcommand should expose help without ImportError.
CLI_HELP_ARGS = [
    ["--help"],
    ["agent", "--help"],
    ["agent", "list", "--help"],
    ["agent", "run", "--help"],
    ["benchmark", "--help"],
    ["benchmark", "run", "--help"],
    ["env", "--help"],
    ["env", "list", "--help"],
    ["env", "run", "--help"],
    ["env", "ps", "--help"],
    ["eval", "--help"],
    ["eval", "metrics", "--help"],
    ["eval", "judge", "--help"],
    ["eval", "summary", "--help"],
    ["eval", "clean", "--help"],
    ["exec", "--help"],
    ["failure", "--help"],
    ["failure", "list", "--help"],
    ["failure", "inject", "--help"],
    ["failure", "describe", "--help"],
    ["failure", "ps", "--help"],
    ["session", "--help"],
    ["session", "ps", "--help"],
    ["session", "inspect", "--help"],
    ["session", "containers", "--help"],
    ["session", "close", "--help"],
    ["session", "wipe", "--help"],
    ["traffic", "--help"],
    ["traffic", "list", "--help"],
    ["traffic", "run", "--help"],
]

# Read-only commands that should succeed in any environment.
CLI_READ_ONLY_ARGS = [
    ["agent", "list"],
    ["env", "list"],
    ["failure", "list"],
    ["traffic", "list"],
    ["session", "ps"],
]


class CliSmokeTest(unittest.TestCase):
    def test_cli_command_modules_import(self) -> None:
        for module_name in CLI_COMMAND_MODULES:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

    def test_cli_handler_workflow_modules_import(self) -> None:
        for module_name in CLI_HANDLER_WORKFLOWS:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

    def test_eval_clean_import_regression(self) -> None:
        """``nika eval clean`` imports ``eval.clean`` via the package namespace."""
        importlib.invalidate_caches()
        clean = importlib.import_module("nika.workflows.eval.clean")
        self.assertTrue(callable(clean.run_eval_clean))

    def test_cli_help_invocations(self) -> None:
        for args in CLI_HELP_ARGS:
            with self.subTest(args=args):
                result = _RUNNER.invoke(app, args)
                self.assertEqual(
                    result.exit_code,
                    0,
                    msg=result.stdout + result.stderr,
                )

    def test_cli_read_only_invocations(self) -> None:
        for args in CLI_READ_ONLY_ARGS:
            with self.subTest(args=args):
                result = _RUNNER.invoke(app, args)
                self.assertEqual(
                    result.exit_code,
                    0,
                    msg=result.stdout + result.stderr,
                )

    def test_console_script_help_invocations(self) -> None:
        for args in CLI_HELP_ARGS:
            with self.subTest(args=args):
                completed = subprocess.run(
                    [sys.executable, "-m", "nika.cli.main", *args],
                    cwd=_REPO_ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    msg=completed.stdout + completed.stderr,
                )


if __name__ == "__main__":
    unittest.main()
