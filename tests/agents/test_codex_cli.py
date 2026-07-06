"""Codex CLI agent tests: unit checks + ``simple_bgp`` / ``link_down`` pipeline."""

from __future__ import annotations

import json
import os
import unittest
import unittest.mock

from agent.local_cli.codex_cli.codex_display import format_codex_event
from agent.local_cli.codex_cli.codex_worker import CodexWorker, _build_mcp_toml
from agent.utils.phases import DIAGNOSIS, SUBMISSION
from nika.utils.agent_config import (
    ENV_AGENT_TYPE,
    ENV_CODEX_MODEL,
    ENV_LLM_PROVIDER,
    ENV_MAX_STEPS,
    resolve_agent_model,
    resolve_agent_type,
    resolve_llm_provider,
    resolve_max_steps,
    resolve_reasoning_effort,
)
from nika.utils.session_store import SessionStore
from tests.agents._assertions import assert_submission_fields
from tests.integration_base import OrderedPipelineTestCase
from tests.integration_pipeline import CommonPipelineSteps, codex_cli_available, load_test_env

load_test_env()

CODEX_MODEL = "gpt-5.4-mini"


# ---------------------------------------------------------------------------
# Unit tests (no Docker)
# ---------------------------------------------------------------------------


class CodexMcpTomlTest(unittest.TestCase):
    """MCP config TOML generation for Codex CLI."""

    def test_includes_noninteractive_approval_defaults(self) -> None:
        toml = _build_mcp_toml(
            {
                "kathara_base_mcp_server": {
                    "command": "python3",
                    "args": ["/path/kathara_base_mcp_server.py"],
                    "env": {"NIKA_SESSION_ID": "sess-123"},
                }
            }
        )
        self.assertIn('approval_policy = "never"', toml)
        self.assertIn('sandbox_mode = "workspace-write"', toml)
        self.assertIn('default_tools_approval_mode = "approve"', toml)
        self.assertIn("[mcp_servers.kathara_base_mcp_server]", toml)
        self.assertIn('NIKA_SESSION_ID = "sess-123"', toml)

    def test_approves_each_configured_server(self) -> None:
        toml = _build_mcp_toml(
            {
                "kathara_base_mcp_server": {"command": "python3", "args": ["/path/base.py"]},
                "task_mcp_server": {"command": "python3", "args": ["/path/task.py"]},
            }
        )
        self.assertEqual(toml.count('default_tools_approval_mode = "approve"'), 2)


class CodexAgentConfigTest(unittest.TestCase):
    """CLI env resolution for the Codex CLI agent."""

    def test_env_fallback(self) -> None:
        with unittest.mock.patch.dict(
            os.environ,
            {
                ENV_AGENT_TYPE: "local_cli.codex_cli",
                ENV_LLM_PROVIDER: "deepseek",
                ENV_MAX_STEPS: "30",
                ENV_CODEX_MODEL: "gpt-5.4-mini",
                "NIKA_CODEX_REASONING_EFFORT": "medium",
            },
            clear=True,
        ):
            self.assertEqual(resolve_agent_type(None), "local_cli.codex_cli")
            self.assertIsNone(resolve_llm_provider(None, agent_type="local_cli.codex_cli"))
            self.assertEqual(resolve_max_steps(None), 30)
            self.assertEqual(resolve_reasoning_effort(None), "medium")

    def test_model_from_env(self) -> None:
        with unittest.mock.patch.dict(os.environ, {ENV_CODEX_MODEL: "gpt-5.4-mini"}, clear=True):
            self.assertEqual(resolve_agent_model("local_cli.codex_cli", None), "gpt-5.4-mini")


class CodexWorkerConfigTest(unittest.TestCase):
    """CodexWorker constructor validation."""

    def test_rejects_invalid_reasoning_effort(self) -> None:
        with self.assertRaises(ValueError):
            CodexWorker(
                session_id="sess-123",
                session_dir="/tmp/sess-123",
                phase=DIAGNOSIS,
                reasoning_effort="turbo",
            )


class CodexDisplayTest(unittest.TestCase):
    """Codex JSONL terminal event formatting."""

    def test_agent_message(self) -> None:
        event = {
            "type": "item.completed",
            "item": {"id": "item_1", "type": "agent_message", "text": "BGP session is down."},
        }
        self.assertIn("BGP session is down.", format_codex_event(event) or "")

    def test_mcp_tool_call_lifecycle(self) -> None:
        started = {
            "type": "item.started",
            "item": {
                "type": "mcp_tool_call",
                "server": "kathara_frr_mcp_server",
                "tool": "show_bgp_summary",
                "arguments": {"device": "router1"},
                "status": "in_progress",
            },
        }
        completed = {
            "type": "item.completed",
            "item": {
                "type": "mcp_tool_call",
                "server": "kathara_frr_mcp_server",
                "tool": "show_bgp_summary",
                "status": "completed",
                "result": {"content": [{"type": "text", "text": "neighbor down"}]},
            },
        }
        self.assertIn("show_bgp_summary", format_codex_event(started) or "")
        self.assertIn("neighbor down", format_codex_event(completed) or "")

    def test_turn_completed_with_usage(self) -> None:
        event = {
            "type": "turn.completed",
            "usage": {"input_tokens": 100, "output_tokens": 20},
        }
        rendered = format_codex_event(event)
        self.assertIn("in=100", rendered or "")
        self.assertIn("out=20", rendered or "")

    def test_reconnecting_error_is_non_fatal(self) -> None:
        event = {"type": "error", "message": "Reconnecting... 1/5"}
        self.assertIn("Reconnecting", format_codex_event(event) or "")

    def test_unknown_event_returns_none(self) -> None:
        self.assertIsNone(format_codex_event({"type": "some_unknown_type"}))


# ---------------------------------------------------------------------------
# Integration pipeline (Docker + Codex CLI)
# ---------------------------------------------------------------------------


@unittest.skipUnless(codex_cli_available(), "Codex CLI and OpenAI credentials required")
class CodexCliAgentPipelineTest(CommonPipelineSteps, OrderedPipelineTestCase):
    """Full pipeline with the Codex CLI agent."""

    def test_step_01_start_env(self) -> None:
        self._step_start_env()

    def test_step_02_inject_failure(self) -> None:
        self._step_inject_failure()

    def test_step_03_run_cli_agent(self) -> None:
        self.assertIsNotNone(self.session_id)
        self._run_agent(agent_type="local_cli.codex_cli", model=CODEX_MODEL, max_steps=20)
        row = SessionStore().get_session(self.session_id)
        self.assertEqual(row.get("agent_type"), "local_cli.codex_cli")

    def test_step_04_check_workspace_and_messages(self) -> None:
        self.assertIsNotNone(self.session_dir)

        workspace = self.session_dir / "codex_workspace"
        self.assertTrue(workspace.is_dir())
        self.assertTrue((workspace / ".git").is_dir())
        self.assertTrue((workspace / ".codex_home").is_dir())

        config_text = (workspace / ".codex_home" / "config.toml").read_text(encoding="utf-8")
        self.assertIn("NIKA_SESSION_ID", config_text)
        self.assertIn(self.session_id, config_text)
        self.assertIn("[mcp_servers.", config_text)
        self.assertIn('default_tools_approval_mode = "approve"', config_text)

        diag_output = workspace / "diagnosis_output.txt"
        self.assertTrue(diag_output.exists())
        self.assertGreater(diag_output.stat().st_size, 0)

        messages = self._load_jsonl("messages.jsonl")
        agents = {e["agent"] for e in messages}
        self.assertIn(DIAGNOSIS, agents)
        self.assertIn(SUBMISSION, agents)

        mcp_events = [e for e in messages if e.get("event") == "mcp_config"]
        diag_mcp = next((e for e in mcp_events if e.get("agent") == DIAGNOSIS), None)
        self.assertIsNotNone(diag_mcp)
        servers = diag_mcp.get("servers", [])
        self.assertIn("kathara_base_mcp_server", servers)
        self.assertIn("kathara_frr_mcp_server", servers)
        self.assertNotIn("kathara_bmv2_mcp_server", servers)
        self.assertNotIn("kathara_telemetry_mcp_server", servers)

        sub_mcp = next((e for e in mcp_events if e.get("agent") == SUBMISSION), None)
        self.assertIsNotNone(sub_mcp)
        self.assertIn("task_mcp_server", sub_mcp.get("servers", []))

        start_events = [e for e in messages if e.get("event") == "subprocess_start"]
        self.assertGreaterEqual(len(start_events), 2)

        codex_events = [e for e in messages if "codex_event" in e]
        self.assertGreater(len(codex_events), 0)
        rendered_count = sum(1 for e in codex_events if format_codex_event(e["codex_event"]))
        self.assertGreater(rendered_count, 0)

    def test_step_05_check_submission(self) -> None:
        self.assertIsNotNone(self.session_dir)
        self.assertTrue((self.session_dir / "submission.json").exists())
        assert_submission_fields(self, self.session_dir)

    def test_step_06_session_close(self) -> None:
        self._step_close_and_verify("local_cli.codex_cli")

    def test_step_07_eval_metrics(self) -> None:
        self._step_eval_metrics()


if __name__ == "__main__":
    unittest.main()
