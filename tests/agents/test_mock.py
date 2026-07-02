"""Mock agent tests: unit checks + ``simple_bgp`` / ``link_down`` pipeline."""

from __future__ import annotations

import asyncio
import json
import os
import unittest
import unittest.mock

from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.utils.mcp_servers import MCPServerConfig
from agent.utils.phases import DIAGNOSIS, SUBMISSION
from nika.cli.main import app
from nika.utils.agent_config import (
    ENV_AGENT_TYPE,
    ENV_JUDGE_MODEL,
    ENV_JUDGE_PROVIDER,
    ENV_MAX_STEPS,
    ENV_MODEL,
    resolve_agent_model,
    resolve_agent_type,
    resolve_judge_model,
    resolve_judge_provider,
    resolve_llm_provider,
    resolve_max_steps,
)
from nika.utils.session_store import SessionStore
from tests.agents._assertions import assert_phase_messages, assert_submission_fields
from tests.integration_base import OrderedPipelineTestCase
from tests.integration_pipeline import (
    PROBLEM,
    SCENARIO,
    CommonPipelineSteps,
    load_test_env,
    tool_text_list,
)

load_test_env()


# ---------------------------------------------------------------------------
# Unit tests (no Docker)
# ---------------------------------------------------------------------------


class MockAgentConfigTest(unittest.TestCase):
    """CLI env resolution for the mock agent and shared agent CLI options."""

    def test_cli_overrides_env(self) -> None:
        with unittest.mock.patch.dict(
            os.environ,
            {ENV_AGENT_TYPE: "mock", ENV_MAX_STEPS: "99"},
            clear=True,
        ):
            self.assertEqual(resolve_agent_type("mock"), "mock")
            self.assertEqual(resolve_max_steps(20), 20)

    def test_missing_agent_type_raises(self) -> None:
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                resolve_agent_type(None)

    def test_missing_max_steps_raises(self) -> None:
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                resolve_max_steps(None)

    def test_llm_provider_not_required(self) -> None:
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(resolve_llm_provider(None, agent_type="mock"))

    def test_model_from_env(self) -> None:
        with unittest.mock.patch.dict(os.environ, {ENV_MODEL: "mock-v1"}, clear=True):
            self.assertEqual(resolve_agent_model("mock", None), "mock-v1")
            self.assertEqual(resolve_agent_model("mock", "override"), "override")

    def test_judge_from_env(self) -> None:
        with unittest.mock.patch.dict(
            os.environ,
            {ENV_JUDGE_PROVIDER: "deepseek", ENV_JUDGE_MODEL: "deepseek-chat"},
            clear=True,
        ):
            self.assertEqual(resolve_judge_provider(None), "deepseek")
            self.assertEqual(resolve_judge_model(None), "deepseek-chat")

    def test_judge_missing_raises(self) -> None:
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                resolve_judge_provider(None)


# ---------------------------------------------------------------------------
# Integration pipeline (Docker)
# ---------------------------------------------------------------------------


class MockAgentPipelineTest(CommonPipelineSteps, OrderedPipelineTestCase):
    """Full pipeline with the mock agent — also validates MCP infrastructure."""

    def test_step_01_start_env(self) -> None:
        self._step_start_env()

    def test_step_02_verify_session_and_cli(self) -> None:
        self.assertIsNotNone(self.session_id)

        row = SessionStore().get_session(self.session_id)
        self.assertEqual(row["status"], "running")
        self.assertEqual(row["scenario_name"], SCENARIO)
        self.assertIsNotNone(row.get("lab_name"))

        self._invoke_ok(["env", "ps"])
        self._invoke_ok(["session", "ps"])
        self._invoke_ok(["session", "inspect", "--session_id", self.session_id])
        self._invoke_ok(["failure", "ps"])
        self._invoke_ok(["exec", "pc1", "hostname"])
        desc = self._invoke_ok(["failure", "describe", PROBLEM])
        self.assertIn(PROBLEM, desc)

    def test_step_03_inject_failure(self) -> None:
        self._step_inject_failure()

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
        self.assertEqual(mcp_config._server_env()["NIKA_SESSION_ID"], self.session_id)

        config = {k: v for k, v in mcp_config.load_config(if_submit=False).items() if k == "kathara_base_mcp_server"}

        async def _run() -> None:
            client = MultiServerMCPClient(connections=config)
            tools = {t.name: t for t in await client.get_tools()}
            self.assertIn("get_reachability", tools)
            result = str(await tools["get_reachability"].ainvoke({}))
            self.assertGreater(len(result), 0)
            self.assertNotIn("NIKA_SESSION_ID is not set", result)

            self.assertIn("exec_shell", tools)
            exec_result = str(await tools["exec_shell"].ainvoke({"host_name": "pc1", "command": "hostname"}))
            self.assertGreater(len(exec_result), 0)

        asyncio.run(_run())

    def test_step_06_task_mcp_submit(self) -> None:
        self.assertIsNotNone(self.session_id)
        self.assertIsNotNone(self.session_dir)

        config = MCPServerConfig(session_id=self.session_id).load_config(if_submit=True)

        async def _run() -> str:
            client = MultiServerMCPClient(connections=config)
            tools = {t.name: t for t in await client.get_tools()}
            self.assertIn("list_avail_problems", tools)
            self.assertIn("submit", tools)

            avail = tool_text_list(await tools["list_avail_problems"].ainvoke({}))
            self.assertIn(PROBLEM, avail)

            result = await tools["submit"].ainvoke(
                {"is_anomaly": True, "faulty_devices": ["pc1"], "root_cause_name": [PROBLEM]}
            )
            return str(result)

        result_str = asyncio.run(_run())
        self.assertIn("success", result_str.lower())

        submission = json.loads((self.session_dir / "submission.json").read_text())
        self.assertTrue(submission["is_anomaly"])
        self.assertIn("pc1", submission["faulty_devices"])
        self.assertIn(PROBLEM, submission["root_cause_name"])

    def test_step_07_run_mock_agent(self) -> None:
        self.assertIsNotNone(self.session_id)
        self._invoke_ok(
            [
                "agent",
                "run",
                "--agent",
                "mock",
                "--model",
                "mock-v1",
                "--max-steps",
                "20",
                "--session_id",
                self.session_id,
            ]
        )

        messages = self._load_jsonl("messages.jsonl")
        assert_phase_messages(self, messages)

        diag_events = [e["event"] for e in messages if e["agent"] == DIAGNOSIS]
        self.assertIn("tool_start", diag_events)
        self.assertIn("llm_end", diag_events)

        assert_submission_fields(self, self.session_dir)
        run = self._load_json("run.json")
        self.assertEqual(run["agent_type"], "mock")

    def test_step_08_session_close(self) -> None:
        self._step_close_and_verify("mock")

    def test_step_09_eval_metrics(self) -> None:
        self._step_eval_metrics(min_tool_calls=1)
        metrics = self._load_json("eval_metrics.json")
        self.assertEqual(metrics["detection_score"], 1.0)
        self.assertEqual(metrics["rca_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
