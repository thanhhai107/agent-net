"""Shared bases for Kathara integration tests."""

from __future__ import annotations

import json
import os
import re
import unittest
from pathlib import Path
from typing import ClassVar

from typer.testing import CliRunner

from nika.cli.main import app
from nika.service.mcp_server.mcp_session_context import SESSION_ID_ENV, get_lab_name
from nika.utils.session_store import SessionStore
from nika.workflows.eval.clean import remove_session_results


class CliIntegrationTestCase(unittest.TestCase):
    """Common CLI runner and env lifecycle helpers."""

    runner: CliRunner

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = CliRunner()

    def _invoke_ok(self, args: list[str]) -> str:
        result = self.runner.invoke(app, args)
        self.assertEqual(result.exit_code, 0, result.output)
        return result.output

    @classmethod
    def _invoke_ok_class(cls, runner: CliRunner, args: list[str]) -> str:
        result = runner.invoke(app, args)
        if result.exit_code != 0:
            raise RuntimeError(f"`nika {' '.join(args)}` exited {result.exit_code}:\n{result.output}")
        return result.output

    def _start_env(self, scenario: str, extra_args: list[str] | None = None) -> str:
        args = ["env", "run", scenario, *(extra_args or [])]
        result = self.runner.invoke(app, args)
        if result.exit_code != 0:
            raise RuntimeError(f"nika env run failed:\n{result.output}")
        match = re.search(r"session_id=(\S+)", result.output.strip())
        if match is None:
            raise RuntimeError(f"session_id not found in env run output:\n{result.output}")
        return match.group(1)

    @classmethod
    def _start_env_class(cls, scenario: str, extra_args: list[str] | None = None) -> str:
        output = cls._invoke_ok_class(cls.runner, ["env", "run", scenario, *(extra_args or [])])
        return cls._parse_session_id(output)

    @staticmethod
    def _parse_session_id(output: str) -> str:
        match = re.search(r"session_id=(\S+)", output.strip())
        if match is None:
            raise RuntimeError(f"session_id not found in env run output:\n{output}")
        return match.group(1)

    def _close_session(self, session_id: str) -> None:
        self.runner.invoke(app, ["session", "close", "--session_id", session_id, "-y"])
        self._remove_session_results(session_id)

    @classmethod
    def _close_session_class(cls, session_id: str) -> None:
        cls.runner.invoke(app, ["session", "close", "--session_id", session_id, "-y"])
        cls._remove_session_results(session_id)

    @staticmethod
    def _remove_session_results(session_id: str) -> None:
        remove_session_results(session_id)

    def _session_row(self, session_id: str | None = None) -> dict:
        sid = session_id or getattr(self, "session_id", None)
        if sid is None:
            raise ValueError("session_id is required")
        return SessionStore().get_session(sid)

    def _assert_session_ready(self, session_id: str, scenario: str) -> dict:
        row = self._session_row(session_id)
        self.assertEqual(row["session_id"], session_id)
        self.assertEqual(row["status"], "running")
        self.assertEqual(row["scenario_name"], scenario)
        self.assertIsNotNone(row.get("lab_name"), "lab_name must be set after env run")
        self.assertIn(scenario, row["lab_name"])
        self.assertRegex(
            session_id,
            r"^\d{8}-\d{6}-[0-9a-f]{6}$",
            "session_id does not match expected YYYYMMDD-HHMMSS-{6hex} format",
        )
        if os.environ.get(SESSION_ID_ENV) == session_id:
            self.assertEqual(get_lab_name(), row["lab_name"])
        return row

    def _scenario_kwargs(self, session_id: str | None = None) -> dict:
        row = self._session_row(session_id)
        kwargs = dict(row.get("scenario_params") or {})
        if row.get("lab_name"):
            kwargs["lab_name"] = row["lab_name"]
        if row.get("scenario_topo_size") is not None:
            kwargs["topo_size"] = row["scenario_topo_size"]
        return kwargs

    def _problem(self, cls_, session_id: str | None = None):
        scenario = getattr(self, "SCENARIO", None) or self._session_row(session_id)["scenario_name"]
        return cls_(scenario_name=scenario, **self._scenario_kwargs(session_id))

    def _topo_size_from_env_args(self) -> str:
        args = getattr(self, "ENV_RUN_ARGS", [])
        if "-s" in args:
            return args[args.index("-s") + 1]
        return ""

    @staticmethod
    def _benchmark_inject_from_yaml(
        scenario: str,
        problem: str,
        topo_size: str = "",
    ) -> dict[str, str]:
        from nika.config import BENCHMARK_DIR
        from nika.workflows.benchmark.load_config import load_benchmark_yaml

        normalized_topo = topo_size or ""
        for yaml_name in ("benchmark_full.yaml", "benchmark_selected.yaml"):
            path = BENCHMARK_DIR / yaml_name
            if not path.is_file():
                continue
            for row in load_benchmark_yaml(path):
                if (
                    row["scenario"] == scenario
                    and row["problem"] == problem
                    and (row.get("topo_size") or "") == normalized_topo
                ):
                    return dict(row["inject"])
        raise ValueError(
            f"No benchmark inject entry for scenario={scenario!r}, problem={problem!r}, "
            f"topo_size={topo_size!r}; pass explicit inject parameters."
        )

    def _inject_via_cli(self, problem: str, params: dict[str, str] | None = None) -> None:
        inject_params = dict(
            params
            if params is not None
            else self._benchmark_inject_from_yaml(
                self.SCENARIO,
                problem,
                self._topo_size_from_env_args(),
            )
        )
        args = ["failure", "inject", problem, "--session_id", self.session_id]
        for key, value in inject_params.items():
            args += ["--set", f"{key}={value}"]
        self._invoke_ok(args)

    def _assert_failure_injected(self, problem: str) -> None:
        ps_output = self._invoke_ok(["failure", "ps", "--session_id", self.session_id])
        self.assertIn(f"problem={problem}", ps_output)
        self.assertIn("status=injected", ps_output)
        failures = SessionStore().list_failure_injections(session_id=self.session_id)
        matching = [row for row in failures if row.get("problem_name") == problem]
        self.assertTrue(matching, f"No failure record for {problem}")
        self.assertEqual(matching[-1].get("status"), "injected")


class PerTestEnvTestCase(CliIntegrationTestCase):
    """Start a fresh Kathara lab per test; bind operations to NIKA_SESSION_ID."""

    SCENARIO: ClassVar[str]
    ENV_RUN_ARGS: ClassVar[list[str]] = []

    session_id: str
    _prev_nika_session_id: str | None

    def setUp(self) -> None:
        self.session_id = self._start_env(self.SCENARIO, self.ENV_RUN_ARGS)
        self._prev_nika_session_id = os.environ.get(SESSION_ID_ENV)
        os.environ[SESSION_ID_ENV] = self.session_id
        self._assert_session_ready(self.session_id, self.SCENARIO)

    def tearDown(self) -> None:
        if getattr(self, "session_id", None):
            self._close_session(self.session_id)
        if getattr(self, "_prev_nika_session_id", None) is None:
            os.environ.pop(SESSION_ID_ENV, None)
        else:
            os.environ[SESSION_ID_ENV] = self._prev_nika_session_id

    def _scenario_kwargs(self, session_id: str | None = None) -> dict:
        return super()._scenario_kwargs(session_id or self.session_id)

    def _problem(self, cls_):
        return super()._problem(cls_, self.session_id)

    @property
    def lab_name(self) -> str:
        return self._session_row(self.session_id)["lab_name"]


class SharedSessionTestCase(CliIntegrationTestCase):
    """Start one lab for the whole test class; optionally inject a failure up front."""

    SCENARIO: ClassVar[str]
    ENV_RUN_ARGS: ClassVar[list[str]] = []
    INJECT_PROBLEM: ClassVar[str | None] = None
    INJECT_ARGS: ClassVar[list[str]] = []

    session_id: str

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.session_id = cls._start_env_class(cls.SCENARIO, cls.ENV_RUN_ARGS)
        if cls.INJECT_PROBLEM is not None:
            inject_args = [
                "failure",
                "inject",
                cls.INJECT_PROBLEM,
                "--session_id",
                cls.session_id,
                *cls.INJECT_ARGS,
            ]
            try:
                cls._invoke_ok_class(cls.runner, inject_args)
            except RuntimeError as exc:
                cls._close_session_class(cls.session_id)
                raise exc

    @classmethod
    def tearDownClass(cls) -> None:
        cls._close_session_class(cls.session_id)


class OrderedPipelineTestCase(CliIntegrationTestCase):
    """Ordered step tests that share session state across methods in one class."""

    session_id: str | None = None
    session_dir: Path | None = None
    env_destroyed: bool = False

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.session_id and not cls.env_destroyed:
            cls._close_session_class(cls.session_id)
        elif cls.session_id:
            cls._remove_session_results(cls.session_id)

    def _load_json(self, filename: str) -> dict:
        assert self.session_dir is not None
        return json.loads((self.session_dir / filename).read_text(encoding="utf-8"))

    def _load_jsonl(self, filename: str) -> list[dict]:
        assert self.session_dir is not None
        return [
            json.loads(line)
            for line in (self.session_dir / filename).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]