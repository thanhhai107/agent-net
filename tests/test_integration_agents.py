"""Integrated agent pipeline tests for supported NIKA agents.

The supported agent surface is LangGraph workflows plus the mock agent:
``react``, ``plan-execute``, ``reflexion``, and ``mock``.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import unittest
from pathlib import Path

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.utils.mcp_servers import MCPServerConfig, select_diagnosis_servers
from agent.utils.phases import DIAGNOSIS, SUBMISSION
from nika.cli.main import app
from nika.utils.agent_config import (
    ENV_AGENT_TYPE,
    ENV_JUDGE_MODEL,
    ENV_JUDGE_PROVIDER,
    ENV_LLM_PROVIDER,
    ENV_MAX_STEPS,
    ENV_MOCK_MODEL,
    ENV_REACT_MODEL,
    resolve_agent_model,
    resolve_agent_type,
    resolve_judge_model,
    resolve_judge_provider,
    resolve_llm_provider,
    resolve_max_steps,
)
from nika.utils.session_index import SessionIndex
from nika.utils.session_store import SessionStore
from tests.integration_base import OrderedPipelineTestCase

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

SCENARIO = "simple_bgp"
PROBLEM = "link_down"
REACT_PROVIDER = "deepseek"
REACT_MODEL = "deepseek-chat"


def _deepseek_api_key_available() -> bool:
    return bool(os.environ.get("DEEPSEEK_API_KEY"))


def _tool_text_list(result: object) -> list[str]:
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


class AgentConfigTest(unittest.TestCase):
    """Unit tests for agent CLI env resolution."""

    def test_cli_overrides_env(self) -> None:
        with unittest.mock.patch.dict(
            os.environ,
            {ENV_AGENT_TYPE: "mock", ENV_LLM_PROVIDER: "deepseek", ENV_MAX_STEPS: "99"},
            clear=False,
        ):
            self.assertEqual(resolve_agent_type("react"), "react")
            self.assertEqual(resolve_llm_provider("openai", agent_type="react"), "openai")
            self.assertEqual(resolve_max_steps(20), 20)

    def test_env_fallback(self) -> None:
        with unittest.mock.patch.dict(
            os.environ,
            {
                ENV_AGENT_TYPE: "react",
                ENV_LLM_PROVIDER: "deepseek",
                ENV_MAX_STEPS: "30",
                ENV_REACT_MODEL: "deepseek-chat",
            },
            clear=False,
        ):
            self.assertEqual(resolve_agent_type(None), "react")
            self.assertEqual(resolve_llm_provider(None, agent_type="react"), "deepseek")
            self.assertEqual(resolve_max_steps(None), 30)
            self.assertEqual(resolve_agent_model("react", None), "deepseek-chat")

    def test_missing_config_raises(self) -> None:
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                resolve_agent_type(None)
            with self.assertRaises(ValueError):
                resolve_llm_provider(None, agent_type="react")
            self.assertEqual(resolve_max_steps(None), 100)

    def test_llm_provider_optional_for_non_react(self) -> None:
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(resolve_llm_provider(None, agent_type="mock"))
            self.assertIsNone(resolve_llm_provider(None, agent_type="plan-execute"))
            self.assertIsNone(resolve_llm_provider(None, agent_type="reflexion"))

    def test_agent_model_per_type(self) -> None:
        with unittest.mock.patch.dict(
            os.environ,
            {ENV_REACT_MODEL: "deepseek-chat", ENV_MOCK_MODEL: "mock-v1"},
            clear=False,
        ):
            self.assertEqual(resolve_agent_model("react", None), "deepseek-chat")
            self.assertEqual(resolve_agent_model("mock", None), "mock-v1")
            self.assertEqual(resolve_agent_model("react", "override"), "override")

    def test_judge_from_env(self) -> None:
        with unittest.mock.patch.dict(
            os.environ,
            {ENV_JUDGE_PROVIDER: "deepseek", ENV_JUDGE_MODEL: "deepseek-chat"},
            clear=False,
        ):
            self.assertEqual(resolve_judge_provider(None), "deepseek")
            self.assertEqual(resolve_judge_model(None), "deepseek-chat")


class DiagnosisServerSelectionTest(unittest.TestCase):
    def test_selector_uses_public_scenario_only(self) -> None:
        params = inspect.signature(select_diagnosis_servers).parameters
        self.assertEqual(list(params), ["scenario_name"])
        self.assertEqual(select_diagnosis_servers("simple_lan"), ["kathara_base_mcp_server"])
        self.assertEqual(
            select_diagnosis_servers("dc_clos_bgp"),
            ["kathara_base_mcp_server", "kathara_frr_mcp_server"],
        )
        self.assertIn(
            "kathara_frr_mcp_server",
            select_diagnosis_servers("dc_clos_service"),
        )
        self.assertNotIn(
            "kathara_frr_mcp_server",
            select_diagnosis_servers("sdn_clos"),
        )
        self.assertIn("kathara_bmv2_mcp_server", select_diagnosis_servers("p4_counter"))


class _CommonPipelineSteps:
    """Mixin with shared step helpers for ordered pipeline test cases."""

    def _step_start_env(self) -> None:
        out = self._invoke_ok(["env", "run", SCENARIO])  # type: ignore[attr-defined]
        match = re.search(r"session_id=(\S+)", out.strip())
        self.assertIsNotNone(match, f"session_id missing from env run output:\n{out}")  # type: ignore[attr-defined]
        type(self).session_id = match.group(1)
        self._assert_session_ready(self.session_id, SCENARIO)  # type: ignore[attr-defined]

    def _step_inject_failure(self) -> None:
        self.assertIsNotNone(self.session_id)  # type: ignore[attr-defined]
        self._invoke_ok(  # type: ignore[attr-defined]
            [
                "failure",
                "inject",
                PROBLEM,
                "--session-id",
                self.session_id,
                "--set",
                "host_name=pc1",
                "--set",
                "intf_name=eth0",
            ]
        )
        row = SessionStore().get_session(self.session_id)
        self.assertIn(PROBLEM, row.get("problem_names", []))  # type: ignore[attr-defined]
        type(self).session_dir = Path(row["session_dir"])
        gt = json.loads((type(self).session_dir / "ground_truth.json").read_text())
        self.assertTrue(gt["is_anomaly"])  # type: ignore[attr-defined]
        self.assertIn(PROBLEM, gt["root_cause_name"])  # type: ignore[attr-defined]

    def _step_close_and_verify(self, expected_agent_type: str) -> None:
        self.assertIsNotNone(self.session_id)  # type: ignore[attr-defined]
        self._invoke_ok(["session", "close", self.session_id, "-y"])  # type: ignore[attr-defined]
        type(self).env_destroyed = True
        run = self._load_json("run.json")  # type: ignore[attr-defined]
        self.assertEqual(run["status"], "finished")  # type: ignore[attr-defined]
        self.assertEqual(run["agent_type"], expected_agent_type)  # type: ignore[attr-defined]

    def _step_eval_metrics(self, min_tool_calls: int = 1) -> None:
        self.assertIsNotNone(self.session_id)  # type: ignore[attr-defined]
        self._invoke_ok(["eval", "metrics", "--session-id", self.session_id])  # type: ignore[attr-defined]
        metrics = self._load_json("eval_metrics.json")  # type: ignore[attr-defined]
        for field in ("detection_score", "localization_accuracy", "rca_accuracy", "tool_calls"):
            self.assertIn(field, metrics)  # type: ignore[attr-defined]
        self.assertGreaterEqual(metrics["tool_calls"], min_tool_calls)  # type: ignore[attr-defined]
        index_row = SessionIndex().get_row(self.session_id)  # type: ignore[attr-defined]
        self.assertIsNotNone(index_row)  # type: ignore[attr-defined]


class MockAgentPipelineTest(_CommonPipelineSteps, OrderedPipelineTestCase):
    """Full pipeline with the mock agent, including MCP infrastructure checks."""

    def test_step_01_start_env(self) -> None:
        self._step_start_env()

    def test_step_02_verify_session_and_cli(self) -> None:
        self.assertIsNotNone(self.session_id)
        row = SessionStore().get_session(self.session_id)
        self.assertEqual(row["status"], "running")
        self.assertEqual(row["scenario_name"], SCENARIO)
        self._invoke_ok(["env", "ps"])
        self._invoke_ok(["session", "ps"])
        self._invoke_ok(["failure", "ps"])
        self._invoke_ok(["exec", "pc1", "hostname"])

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
        config = {
            k: v
            for k, v in mcp_config.load_config(if_submit=False).items()
            if k == "kathara_base_mcp_server"
        }

        async def _run() -> None:
            client = MultiServerMCPClient(connections=config)
            tools = {t.name: t for t in await client.get_tools()}
            self.assertIn("get_reachability", tools)
            result = str(await tools["get_reachability"].ainvoke({}))
            self.assertGreater(len(result), 0)

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
            avail = _tool_text_list(await tools["list_avail_problems"].ainvoke({}))
            self.assertIn(PROBLEM, avail)
            result = await tools["submit"].ainvoke(
                {"is_anomaly": True, "faulty_devices": ["pc1"], "root_cause_name": [PROBLEM]}
            )
            return str(result)

        result_str = asyncio.run(_run())
        self.assertIn("success", result_str.lower())

    def test_step_07_run_mock_agent(self) -> None:
        self.assertIsNotNone(self.session_id)
        self._invoke_ok(
            ["agent", "run", "--agent", "mock", "--model", "mock-v1", "--session-id", self.session_id]
        )
        messages = self._load_jsonl("messages.jsonl")
        agents = {e["agent"] for e in messages}
        self.assertIn(DIAGNOSIS, agents)
        self.assertIn(SUBMISSION, agents)
        submission = self._load_json("submission.json")
        for field in ("is_anomaly", "faulty_devices", "root_cause_name"):
            self.assertIn(field, submission)
        run = self._load_json("run.json")
        self.assertEqual(run["agent_type"], "mock")

    def test_step_08_session_close(self) -> None:
        self._step_close_and_verify("mock")

    def test_step_09_eval_metrics(self) -> None:
        self._step_eval_metrics(min_tool_calls=1)
        metrics = self._load_json("eval_metrics.json")
        self.assertGreaterEqual(metrics["detection_score"], 0.0)
        self.assertGreaterEqual(metrics["rca_accuracy"], 0.0)


@unittest.skipUnless(_deepseek_api_key_available(), "DEEPSEEK_API_KEY required for react agent")
class ReactAgentPipelineTest(_CommonPipelineSteps, OrderedPipelineTestCase):
    """Full pipeline with the LangChain ReAct agent using the DeepSeek provider."""

    def test_step_01_start_env(self) -> None:
        self._step_start_env()

    def test_step_02_inject_failure(self) -> None:
        self._step_inject_failure()

    def test_step_03_run_react_agent(self) -> None:
        self.assertIsNotNone(self.session_id)
        result = self.runner.invoke(
            app,
            [
                "agent",
                "run",
                "--agent",
                "react",
                "--backend",
                REACT_PROVIDER,
                "--model",
                REACT_MODEL,
                "--max-steps",
                "20",
                "--session-id",
                self.session_id,
            ],
        )
        self.assertEqual(
            result.exit_code,
            0,
            f"agent run exited {result.exit_code}:\n{result.output}"
            + (f"\nException: {result.exception}" if result.exception else ""),
        )
        row = SessionStore().get_session(self.session_id)
        self.assertEqual(row.get("agent_type"), "react")

    def test_step_04_check_messages(self) -> None:
        self.assertIsNotNone(self.session_dir)
        messages = self._load_jsonl("messages.jsonl")
        agents = {e["agent"] for e in messages}
        self.assertIn(DIAGNOSIS, agents)
        self.assertIn(SUBMISSION, agents)
        diag_tools = [
            e["tool"]["name"]
            for e in messages
            if e["agent"] == DIAGNOSIS and e["event"] == "tool_start" and "tool" in e
        ]
        self.assertTrue(diag_tools, "diagnosis phase must call at least one MCP tool")

    def test_step_05_session_close(self) -> None:
        self._step_close_and_verify("react")

    def test_step_06_eval_metrics(self) -> None:
        self._step_eval_metrics()


if __name__ == "__main__":
    unittest.main()
