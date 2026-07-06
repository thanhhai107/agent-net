"""SADE community agent tests: unit checks + ``simple_bgp`` / ``link_down`` pipeline."""

from __future__ import annotations

import os
import sys
import unittest
import unittest.mock

from agent.sdk.mcp import to_sdk_mcp_servers
from agent.community.sade.config import prepare_sade_sdk_env, sade_credentials_available
from agent.utils.phases import DIAGNOSIS
from nika.utils.agent_config import resolve_agent_model
from nika.utils.session_store import SessionStore
from tests.agents._assertions import assert_submission_fields
from tests.integration_base import OrderedPipelineTestCase
from tests.integration_pipeline import CommonPipelineSteps, load_test_env, sade_available

load_test_env()


# ---------------------------------------------------------------------------
# Unit tests (no Docker)
# ---------------------------------------------------------------------------


class SadeConfigTest(unittest.TestCase):
    """Model and credential resolution for community.sade."""

    def test_model_from_anthropic_model_env(self) -> None:
        with unittest.mock.patch.dict(os.environ, {"ANTHROPIC_MODEL": "deepseek-v4-pro[1m]"}, clear=True):
            self.assertEqual(resolve_agent_model("community.sade", None), "deepseek-v4-pro[1m]")

    def test_model_from_nika_sade_model_env(self) -> None:
        with unittest.mock.patch.dict(
            os.environ,
            {"NIKA_SADE_MODEL": "deepseek-v4-flash", "ANTHROPIC_MODEL": "other-model"},
            clear=True,
        ):
            self.assertEqual(resolve_agent_model("community.sade", None), "deepseek-v4-flash")

    def test_prepare_env_maps_auth_token_to_api_key(self) -> None:
        with unittest.mock.patch.dict(
            os.environ,
            {"ANTHROPIC_AUTH_TOKEN": "tok", "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic"},
            clear=True,
        ):
            env = prepare_sade_sdk_env(session_id="sess-abc")
        self.assertEqual(env["ANTHROPIC_API_KEY"], "tok")
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "https://api.deepseek.com/anthropic")
        self.assertEqual(env["NIKA_SESSION_ID"], "sess-abc")

    def test_prepare_env_requires_credentials(self) -> None:
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                prepare_sade_sdk_env(session_id="sess-abc")

    def test_sade_credentials_available_with_auth_token(self) -> None:
        with unittest.mock.patch.dict(os.environ, {"ANTHROPIC_AUTH_TOKEN": "tok"}, clear=True):
            self.assertTrue(sade_credentials_available())


class SadeMcpAdapterTest(unittest.TestCase):
    """MCP config adaptation for claude-agent-sdk."""

    def test_converts_transport_to_stdio_type(self) -> None:
        servers = to_sdk_mcp_servers(
            {
                "kathara_base_mcp_server": {
                    "transport": "stdio",
                    "command": "python3",
                    "args": ["/path/base.py"],
                    "env": {"NIKA_SESSION_ID": "sess-abc"},
                }
            }
        )
        srv = servers["kathara_base_mcp_server"]
        self.assertEqual(srv["type"], "stdio")
        self.assertEqual(srv["command"], sys.executable)
        self.assertEqual(srv["args"], ["/path/base.py"])
        self.assertEqual(srv["env"]["NIKA_SESSION_ID"], "sess-abc")

    def test_multiple_servers_all_present(self) -> None:
        servers = to_sdk_mcp_servers(
            {
                "kathara_base_mcp_server": {"command": "python3", "args": ["/path/base.py"]},
                "task_mcp_server": {"command": "python3", "args": ["/path/task.py"]},
            }
        )
        self.assertIn("kathara_base_mcp_server", servers)
        self.assertIn("task_mcp_server", servers)


# ---------------------------------------------------------------------------
# Integration pipeline (Docker + claude-agent-sdk + DeepSeek-compatible creds)
# ---------------------------------------------------------------------------


@unittest.skipUnless(sade_available(), "claude-agent-sdk + ANTHROPIC credentials required")
class SadeAgentPipelineTest(CommonPipelineSteps, OrderedPipelineTestCase):
    """Full pipeline with the SADE community agent."""

    def test_step_01_start_env(self) -> None:
        self._step_start_env()

    def test_step_02_inject_failure(self) -> None:
        self._step_inject_failure()

    def test_step_03_run_sade_agent(self) -> None:
        self.assertIsNotNone(self.session_id)
        self._run_agent(agent_type="community.sade", max_steps=20)
        row = SessionStore().get_session(self.session_id)
        self.assertEqual(row.get("agent_type"), "community.sade")

    def test_step_04_check_messages(self) -> None:
        self.assertIsNotNone(self.session_dir)
        messages = self._load_jsonl("messages.jsonl")
        agents = {e["agent"] for e in messages}
        self.assertIn(DIAGNOSIS, agents)

        tool_starts = [e for e in messages if e.get("event") == "tool_start"]
        self.assertTrue(tool_starts, "SADE must emit tool_start events")
        tool_names = [e.get("tool", {}).get("name", "") for e in tool_starts]
        self.assertTrue(any("submit" in name for name in tool_names), "expected submit tool call")

        llm_ends = [e for e in messages if e.get("event") == "llm_end"]
        self.assertTrue(llm_ends, "SADE must emit llm_end events")

    def test_step_05_check_submission(self) -> None:
        self.assertIsNotNone(self.session_dir)
        self.assertTrue((self.session_dir / "submission.json").exists())
        assert_submission_fields(self, self.session_dir)

    def test_step_06_session_close(self) -> None:
        self._step_close_and_verify("community.sade")

    def test_step_07_eval_metrics(self) -> None:
        self._step_eval_metrics()


if __name__ == "__main__":
    unittest.main()
