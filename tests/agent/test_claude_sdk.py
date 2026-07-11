"""sdk.claude_sdk agent tests: unit checks + ``simple_bgp`` / ``link_down`` pipeline."""

from __future__ import annotations

import os
import sys
import unittest
import unittest.mock

from agent.sdk.claude_sdk.config import (
    claude_sdk_credentials_available,
    prepare_claude_sdk_env,
    resolve_claude_sdk_model,
)
from agent.sdk.mcp import to_sdk_mcp_servers
from nika.utils.session_store import SessionStore
from tests.agent._assertions import assert_phase_messages, assert_submission_fields
from tests.support.integration_base import OrderedPipelineTestCase
from tests.support.integration_pipeline import (
    ClabCommonPipelineSteps,
    CommonPipelineSteps,
    _min3clos_prerequisites,
    claude_sdk_available,
    load_test_env,
)

load_test_env()


# ---------------------------------------------------------------------------
# Unit tests (no Docker)
# ---------------------------------------------------------------------------


class ClaudeSdkConfigTest(unittest.TestCase):
    """Model and credential resolution for sdk.claude_sdk."""

    def test_prepare_env_maps_auth_token_to_api_key(self) -> None:
        with unittest.mock.patch.dict(
            os.environ,
            {
                "ANTHROPIC_AUTH_TOKEN": "tok",
                "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
            },
            clear=True,
        ):
            env = prepare_claude_sdk_env(session_id="sess-abc")
        self.assertEqual(env["ANTHROPIC_API_KEY"], "tok")
        self.assertEqual(
            env["ANTHROPIC_BASE_URL"], "https://api.deepseek.com/anthropic"
        )
        self.assertEqual(env["NIKA_SESSION_ID"], "sess-abc")

    def test_prepare_env_requires_credentials(self) -> None:
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                prepare_claude_sdk_env(session_id="sess-abc")

    def test_resolve_claude_sdk_model_explicit(self) -> None:
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_claude_sdk_model("custom-model"), "custom-model")


class ClaudeSdkMcpTest(unittest.TestCase):
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

    def test_credentials_available_with_auth_token(self) -> None:
        with unittest.mock.patch.dict(
            os.environ, {"ANTHROPIC_AUTH_TOKEN": "tok"}, clear=True
        ):
            self.assertTrue(claude_sdk_credentials_available())


# ---------------------------------------------------------------------------
# Integration pipeline (Docker + claude-agent-sdk + DeepSeek-compatible creds)
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    claude_sdk_available(), "claude-agent-sdk + ANTHROPIC credentials required"
)
class ClaudeSdkAgentPipelineTest(CommonPipelineSteps, OrderedPipelineTestCase):
    """Full pipeline with the sdk.claude_sdk agent."""

    def test_step_01_start_env(self) -> None:
        self._step_start_env()

    def test_step_02_inject_failure(self) -> None:
        self._step_inject_failure()

    def test_step_03_run_claude_sdk_agent(self) -> None:
        self.assertIsNotNone(self.session_id)
        self._run_agent(agent_type="sdk.claude_sdk", max_steps=20)
        row = SessionStore().get_session(self.session_id)
        self.assertEqual(row.get("agent_type"), "sdk.claude_sdk")

    def test_step_04_check_messages(self) -> None:
        self.assertIsNotNone(self.session_dir)
        assert_phase_messages(self, self._load_jsonl("messages.jsonl"))

    def test_step_05_check_submission(self) -> None:
        self.assertIsNotNone(self.session_dir)
        self.assertTrue((self.session_dir / "submission.json").exists())
        assert_submission_fields(self, self.session_dir)

    def test_step_06_session_close(self) -> None:
        self._step_close_and_verify("sdk.claude_sdk")

    def test_step_07_eval_metrics(self) -> None:
        self._step_eval_metrics()


@unittest.skipUnless(
    _min3clos_prerequisites() and claude_sdk_available(),
    "containerlab/gnmic/Docker or claude-agent-sdk credentials not available",
)
class ClaudeSdkClabPipelineTest(ClabCommonPipelineSteps, OrderedPipelineTestCase):
    """Full containerlab pipeline with the sdk.claude_sdk agent."""

    def test_step_01_start_env(self) -> None:
        self._step_start_env()

    def test_step_02_inject_failure(self) -> None:
        self._step_inject_failure()

    def test_step_03_run_claude_sdk_agent(self) -> None:
        self.assertIsNotNone(self.session_id)
        self._run_agent(agent_type="sdk.claude_sdk", max_steps=20)
        row = SessionStore().get_session(self.session_id)
        self.assertEqual(row.get("agent_type"), "sdk.claude_sdk")

    def test_step_04_check_messages(self) -> None:
        self.assertIsNotNone(self.session_dir)
        assert_phase_messages(self, self._load_jsonl("messages.jsonl"))

    def test_step_05_check_submission(self) -> None:
        self.assertIsNotNone(self.session_dir)
        self.assertTrue((self.session_dir / "submission.json").exists())
        assert_submission_fields(self, self.session_dir)

    def test_step_06_session_close(self) -> None:
        self._step_close_and_verify("sdk.claude_sdk")

    def test_step_07_eval_metrics(self) -> None:
        self._step_eval_metrics()


if __name__ == "__main__":
    unittest.main()
