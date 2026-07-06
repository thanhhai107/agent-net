"""sdk.codex_sdk agent tests: unit checks + ``simple_bgp`` / ``link_down`` pipeline."""

from __future__ import annotations

import os
import unittest
import unittest.mock

from agent.local_cli.codex_cli.codex_worker import _build_mcp_toml
from agent.sdk.codex_sdk.config import codex_sdk_local_auth_available, validate_reasoning_effort
from nika.utils.agent_config import ENV_CODEX_SDK_MODEL, ENV_CODEX_MODEL, resolve_agent_model
from nika.utils.session_store import SessionStore
from tests.agents._assertions import assert_phase_messages, assert_submission_fields
from tests.integration_base import OrderedPipelineTestCase
from tests.integration_pipeline import CommonPipelineSteps, codex_sdk_available, load_test_env

load_test_env()

CODEX_MODEL = "gpt-5.4-mini"


# ---------------------------------------------------------------------------
# Unit tests (no Docker)
# ---------------------------------------------------------------------------


class CodexSdkConfigTest(unittest.TestCase):
    """CLI env resolution for the sdk.codex_sdk agent."""

    def test_model_from_nika_codex_model_env(self) -> None:
        with unittest.mock.patch.dict(os.environ, {ENV_CODEX_MODEL: CODEX_MODEL}, clear=True):
            self.assertEqual(resolve_agent_model("sdk.codex_sdk", None), CODEX_MODEL)

    def test_model_from_nika_codex_sdk_model_env(self) -> None:
        with unittest.mock.patch.dict(
            os.environ,
            {ENV_CODEX_SDK_MODEL: "gpt-5.4-mini", ENV_CODEX_MODEL: "other-model"},
            clear=True,
        ):
            self.assertEqual(resolve_agent_model("sdk.codex_sdk", None), "gpt-5.4-mini")

    def test_validate_reasoning_effort_accepts_valid(self) -> None:
        self.assertEqual(validate_reasoning_effort("medium"), "medium")

    def test_validate_reasoning_effort_rejects_invalid(self) -> None:
        with self.assertRaises(ValueError):
            validate_reasoning_effort("invalid")

    def test_local_auth_detection(self) -> None:
        with unittest.mock.patch("agent.sdk.codex_sdk.config.Path") as mock_path:
            mock_home = mock_path.home.return_value
            mock_home.__truediv__.return_value.is_file.return_value = True
            self.assertTrue(codex_sdk_local_auth_available())


class CodexSdkMcpTest(unittest.TestCase):
    """MCP config TOML generation reused from codex_cli."""

    def test_includes_mcp_server_section(self) -> None:
        toml = _build_mcp_toml(
            {
                "kathara_base_mcp_server": {
                    "command": "python3",
                    "args": ["/path/base.py"],
                    "env": {"NIKA_SESSION_ID": "sess-abc"},
                }
            }
        )
        self.assertIn("[mcp_servers.kathara_base_mcp_server]", toml)
        self.assertIn('NIKA_SESSION_ID = "sess-abc"', toml)


# ---------------------------------------------------------------------------
# Integration pipeline (Docker + openai-codex + local ~/.codex/auth.json)
# ---------------------------------------------------------------------------


@unittest.skipUnless(codex_sdk_available(), "openai-codex + ~/.codex/auth.json required")
class CodexSdkAgentPipelineTest(CommonPipelineSteps, OrderedPipelineTestCase):
    """Full pipeline with the sdk.codex_sdk agent."""

    def test_step_01_start_env(self) -> None:
        self._step_start_env()

    def test_step_02_inject_failure(self) -> None:
        self._step_inject_failure()

    def test_step_03_run_codex_sdk_agent(self) -> None:
        self.assertIsNotNone(self.session_id)
        self._run_agent(agent_type="sdk.codex_sdk", model=CODEX_MODEL, max_steps=20)
        row = SessionStore().get_session(self.session_id)
        self.assertEqual(row.get("agent_type"), "sdk.codex_sdk")

    def test_step_04_check_messages(self) -> None:
        self.assertIsNotNone(self.session_dir)
        assert_phase_messages(self, self._load_jsonl("messages.jsonl"))

    def test_step_05_check_submission(self) -> None:
        self.assertIsNotNone(self.session_dir)
        self.assertTrue((self.session_dir / "submission.json").exists())
        assert_submission_fields(self, self.session_dir)

    def test_step_06_session_close(self) -> None:
        self._step_close_and_verify("sdk.codex_sdk")

    def test_step_07_eval_metrics(self) -> None:
        self._step_eval_metrics()


if __name__ == "__main__":
    unittest.main()
