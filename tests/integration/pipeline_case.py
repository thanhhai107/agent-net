"""Shared ordered pipeline integration steps for Kathara and Containerlab labs."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import ClassVar

from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.utils.mcp_servers import MCPServerConfig
from agent.utils.phases import DIAGNOSIS, SUBMISSION
from nika.cli.utils import env_id_from_lab
from nika.utils.session_store import SessionStore
from nika.workflows.eval.session import run_eval_metrics
from nika.workflows.failure.inject import inject_failure
from nika.workflows.session.close import close_session
from nika.workflows.session.containers import list_session_containers
from tests.integration_base import CliIntegrationTestCase, OrderedPipelineTestCase


def tool_text_list(result: object) -> list[str]:
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            return [result]
    if not isinstance(result, list):
        return [str(result)]
    return [
        str(item["text"]) if isinstance(item, dict) and "text" in item else str(item)
        for item in result
    ]


class PipelineCaseBase(CliIntegrationTestCase, OrderedPipelineTestCase):
    """Parameterized end-to-end pipeline: env → inject → MCP → mock agent → eval."""

    __test__ = False

    SCENARIO: ClassVar[str]
    BACKEND: ClassVar[str] = "kathara"
    ENV_RUN_ARGS: ClassVar[list[str]] = []
    PROBLEM: ClassVar[str] = "link_down"
    INJECT_PARAMS: ClassVar[dict[str, str]]
    EXPECTED_NODES: ClassVar[frozenset[str]]
    EXEC_PROBE_HOST: ClassVar[str]
    EXEC_PROBE_CMD: ClassVar[str] = "hostname"
    SUBMIT_FAULTY_DEVICES: ClassVar[list[str]]
    ROOT_CAUSE_CATEGORY: ClassVar[str] = "link_failure"
    IMAGE_SUBSTRING: ClassVar[str | None] = "kathara"
    DIAGNOSIS_MCP_SERVERS: ClassVar[list[str]] = ["kathara_base_mcp_server"]

    def test_step_01_start_env(self) -> None:
        list_output = self._invoke_ok(["env", "list"])
        self.assertIn(self.SCENARIO, list_output)
        type(self).session_id = self._start_env(self.SCENARIO, self.ENV_RUN_ARGS)
        row = self._assert_session_ready(self.session_id, self.SCENARIO)
        if self.BACKEND != "kathara":
            self.assertEqual(row.get("backend"), self.BACKEND)

    def test_step_02_verify_session_and_cli(self) -> None:
        self.assertIsNotNone(self.session_id)
        row = SessionStore().get_session(self.session_id)
        self.assertEqual(row["status"], "running")
        self.assertEqual(row["scenario_name"], self.SCENARIO)
        lab_name = row.get("lab_name")
        self.assertIsNotNone(lab_name)

        ps_output = self._invoke_ok(["env", "ps"])
        self.assertIn(env_id_from_lab(lab_name), ps_output)
        self.assertIn(self.SCENARIO, ps_output)
        self.assertIn("1 active", ps_output)

        resolved_id, resolved_lab, container_rows = list_session_containers(
            self.session_id
        )
        self.assertEqual(resolved_id, self.session_id)
        self.assertEqual(resolved_lab, lab_name)
        self.assertEqual(len(container_rows), len(self.EXPECTED_NODES))
        self.assertEqual({r["name"] for r in container_rows}, self.EXPECTED_NODES)
        for container_row in container_rows:
            self.assertEqual(container_row["status"], "running")
            self.assertRegex(container_row["container_id"], r"^[0-9a-f]{12}$")
            if self.IMAGE_SUBSTRING:
                self.assertIn(self.IMAGE_SUBSTRING, container_row["image"].lower())

        exec_output = self._invoke_ok(
            [
                "exec",
                "--session_id",
                self.session_id,
                self.EXEC_PROBE_HOST,
                self.EXEC_PROBE_CMD,
            ],
        )
        self.assertTrue(exec_output.strip())

        describe_output = self._invoke_ok(["failure", "describe", self.PROBLEM])
        self.assertIn(self.PROBLEM, describe_output)

    def test_step_03_inject_failure(self) -> None:
        self.assertIsNotNone(self.session_id)
        inject_failure(
            [self.PROBLEM],
            session_id=self.session_id,
            param_overrides=dict(self.INJECT_PARAMS),
        )
        self._assert_failure_injected(self.PROBLEM)

        row = SessionStore().get_session(self.session_id)
        self.assertIn(self.PROBLEM, row.get("problem_names", []))
        type(self).session_dir = Path(row["session_dir"])
        ground_truth = self._load_json("ground_truth.json")
        self.assertTrue(ground_truth["is_anomaly"])
        self.assertIn(self.PROBLEM, ground_truth["root_cause_name"])
        self.assertEqual(ground_truth["root_cause_category"], self.ROOT_CAUSE_CATEGORY)
        for device in self.SUBMIT_FAULTY_DEVICES:
            self.assertIn(device, ground_truth["faulty_devices"])

    def test_step_04_mcp_session_context(self) -> None:
        self.assertIsNotNone(self.session_id)
        row = SessionStore().get_session(self.session_id)
        prev = os.environ.get("NIKA_SESSION_ID")
        try:
            os.environ["NIKA_SESSION_ID"] = self.session_id
            from nika.service.mcp_server.mcp_session_context import (
                get_lab_name,
                get_session_dir,
                require_session_id,
            )

            self.assertEqual(require_session_id(), self.session_id)
            self.assertEqual(get_lab_name(), row["lab_name"])
            self.assertEqual(get_session_dir(), row["session_dir"])
        finally:
            if prev is None:
                os.environ.pop("NIKA_SESSION_ID", None)
            else:
                os.environ["NIKA_SESSION_ID"] = prev

    def test_step_05_diagnosis_mcp_tools(self) -> None:
        self.assertIsNotNone(self.session_id)
        mcp_config = MCPServerConfig(session_id=self.session_id)
        full = mcp_config.load_config(if_submit=False)
        diagnosis_config = {
            k: v for k, v in full.items() if k in self.DIAGNOSIS_MCP_SERVERS
        }

        async def _run() -> dict:
            client = MultiServerMCPClient(connections=diagnosis_config)
            tools = {t.name: t for t in await client.get_tools()}
            reach = await tools["get_reachability"].ainvoke({})
            host_cfg = await tools["get_host_net_config"].ainvoke(
                {"host_name": self.EXEC_PROBE_HOST}
            )
            exec_out = await tools["exec_shell"].ainvoke(
                {"host_name": self.EXEC_PROBE_HOST, "command": self.EXEC_PROBE_CMD}
            )
            return {
                "reachability": str(reach),
                "host_net_config": str(host_cfg),
                "exec_shell": str(exec_out),
            }

        results = asyncio.run(_run())
        for key, output in results.items():
            self.assertTrue(len(output) > 0, f"{key} must return non-empty output")
            self.assertNotIn("NIKA_SESSION_ID is not set", output)

    def test_step_06_submit_via_mcp(self) -> None:
        self.assertIsNotNone(self.session_id)
        self.assertIsNotNone(self.session_dir)
        config = MCPServerConfig(session_id=self.session_id).load_config(if_submit=True)

        async def _run() -> str:
            client = MultiServerMCPClient(connections=config)
            tools = {t.name: t for t in await client.get_tools()}
            submit_result = await tools["submit"].ainvoke(
                {
                    "is_anomaly": True,
                    "faulty_devices": self.SUBMIT_FAULTY_DEVICES,
                    "root_cause_name": [self.PROBLEM],
                }
            )
            return str(submit_result)

        result_str = asyncio.run(_run())
        self.assertIn("success", result_str.lower())
        submission = self._load_json("submission.json")
        self.assertTrue(submission["is_anomaly"])
        for device in self.SUBMIT_FAULTY_DEVICES:
            self.assertIn(device, submission["faulty_devices"])

    def test_step_07_run_mock_agent(self) -> None:
        self.assertIsNotNone(self.session_id)
        self._run_agent(agent_type="mock", model="mock-v1", max_steps=20)
        for name in (
            "ground_truth.json",
            "messages.jsonl",
            "submission.json",
            "run.json",
        ):
            self.assertTrue((self.session_dir / name).exists(), f"missing {name}")
        messages = self._load_jsonl("messages.jsonl")
        agents = {entry["agent"] for entry in messages}
        self.assertIn(DIAGNOSIS, agents)
        self.assertIn(SUBMISSION, agents)

    def test_step_08_session_close(self) -> None:
        self.assertIsNotNone(self.session_id)
        close_session(session_id=self.session_id)
        type(self).env_destroyed = True
        run = self._load_json("run.json")
        self.assertEqual(run["status"], "finished")
        with self.assertRaises(FileNotFoundError):
            SessionStore().get_session(self.session_id)

    def test_step_09_eval_metrics(self) -> None:
        self.assertIsNotNone(self.session_id)
        run_eval_metrics(session_id=self.session_id)
        metrics = self._load_json("eval_metrics.json")
        self.assertEqual(metrics["detection_score"], 1.0)
        self.assertEqual(metrics["rca_accuracy"], 1.0)
        self.assertGreater(metrics["tool_calls"], 0)
