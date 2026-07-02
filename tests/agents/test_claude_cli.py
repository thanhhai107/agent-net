"""Claude Code CLI agent tests: unit checks + ``simple_bgp`` / ``link_down`` pipeline."""

from __future__ import annotations

import json
import os
import unittest
import unittest.mock

from agent.local_cli.claude_cli.claude_display import format_claude_event
from agent.local_cli.claude_cli.claude_worker import ClaudeWorker, _build_mcp_json
from agent.local_cli.claude_cli.config import (
    default_claude_model,
    has_env_claude_credentials,
    prepare_claude_subprocess_env,
    use_bare_claude_mode,
)
from agent.utils.phases import DIAGNOSIS, SUBMISSION
from nika.cli.main import app
from nika.utils.agent_config import resolve_agent_model
from nika.utils.session_store import SessionStore
from tests.agents._assertions import assert_submission_fields
from tests.integration_base import OrderedPipelineTestCase
from tests.integration_pipeline import CommonPipelineSteps, claude_cli_available, load_test_env

load_test_env()


# ---------------------------------------------------------------------------
# Unit tests (no Docker)
# ---------------------------------------------------------------------------


class ClaudeConfigTest(unittest.TestCase):
    """Claude env model and auth helpers."""

    def test_default_model_reads_anthropic_model(self) -> None:
        with unittest.mock.patch.dict(
            os.environ,
            {"ANTHROPIC_MODEL": "model-a", "CLAUDE_CODE_SUBAGENT_MODEL": "model-b"},
            clear=True,
        ):
            self.assertEqual(default_claude_model(), "model-a")

    def test_default_model_missing_raises(self) -> None:
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                default_claude_model()

    def test_prepare_env_maps_auth_token_to_api_key(self) -> None:
        env = prepare_claude_subprocess_env({"ANTHROPIC_AUTH_TOKEN": "tok"})
        self.assertEqual(env["ANTHROPIC_API_KEY"], "tok")

    def test_use_bare_mode_when_env_credentials_present(self) -> None:
        with unittest.mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key"}, clear=True):
            self.assertTrue(use_bare_claude_mode())
            self.assertTrue(has_env_claude_credentials())


class ClaudeAgentConfigTest(unittest.TestCase):
    """CLI env resolution for the Claude CLI agent."""

    def test_model_from_env(self) -> None:
        with unittest.mock.patch.dict(os.environ, {"ANTHROPIC_MODEL": "deepseek-v4-pro[1m]"}, clear=True):
            self.assertEqual(resolve_agent_model("local_cli.claude_cli", None), "deepseek-v4-pro[1m]")


class ClaudeMcpJsonTest(unittest.TestCase):
    """MCP config JSON generation for Claude Code CLI."""

    def test_produces_valid_json_with_mcp_servers_key(self) -> None:
        json_str = _build_mcp_json(
            {
                "kathara_base_mcp_server": {
                    "command": "python3",
                    "args": ["/path/base.py"],
                    "env": {"NIKA_SESSION_ID": "sess-abc"},
                }
            }
        )
        config = json.loads(json_str)
        self.assertIn("mcpServers", config)
        srv = config["mcpServers"]["kathara_base_mcp_server"]
        self.assertEqual(srv["type"], "stdio")
        self.assertEqual(srv["command"], "python3")
        self.assertEqual(srv["args"], ["/path/base.py"])
        self.assertEqual(srv["env"]["NIKA_SESSION_ID"], "sess-abc")

    def test_multiple_servers_all_present(self) -> None:
        json_str = _build_mcp_json(
            {
                "kathara_base_mcp_server": {"command": "python3", "args": ["/path/base.py"]},
                "task_mcp_server": {"command": "python3", "args": ["/path/task.py"]},
            }
        )
        config = json.loads(json_str)
        self.assertIn("kathara_base_mcp_server", config["mcpServers"])
        self.assertIn("task_mcp_server", config["mcpServers"])

    def test_server_without_env_omits_env_key(self) -> None:
        json_str = _build_mcp_json({"task_mcp_server": {"command": "python3", "args": ["/path/task.py"]}})
        config = json.loads(json_str)
        self.assertNotIn("env", config["mcpServers"]["task_mcp_server"])


class ClaudeWorkerConfigTest(unittest.TestCase):
    """ClaudeWorker constructor validation."""

    def test_rejects_invalid_phase(self) -> None:
        with self.assertRaises(ValueError):
            ClaudeWorker(
                session_id="sess-123",
                session_dir="/tmp/sess-123",
                phase="invalid_phase",
            )


class ClaudeDisplayTest(unittest.TestCase):
    """Claude Code stream-json event formatting."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sample_model = default_claude_model()

    def test_system_init_event_with_string_servers(self) -> None:
        event = {
            "type": "system",
            "subtype": "init",
            "model": self.sample_model,
            "mcp_servers": ["kathara_base_mcp_server"],
        }
        rendered = format_claude_event(event)
        self.assertIn(self.sample_model, rendered or "")
        self.assertIn("kathara_base_mcp_server", rendered or "")

    def test_system_init_event_with_dict_servers(self) -> None:
        event = {
            "type": "system",
            "subtype": "init",
            "model": self.sample_model,
            "mcp_servers": [{"name": "kathara_base_mcp_server", "status": "connected"}],
        }
        rendered = format_claude_event(event)
        self.assertIn("kathara_base_mcp_server", rendered or "")

    def test_system_init_without_servers(self) -> None:
        event = {
            "type": "system",
            "subtype": "init",
            "model": self.sample_model,
            "mcp_servers": [],
        }
        rendered = format_claude_event(event)
        self.assertIsNotNone(rendered)
        self.assertNotIn("mcp:", rendered or "")

    def test_thinking_tokens_skipped(self) -> None:
        event = {"type": "system", "subtype": "thinking_tokens", "estimated_tokens": 42}
        self.assertIsNone(format_claude_event(event))

    def test_assistant_text_message(self) -> None:
        event = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "BGP peer is unreachable."}]},
        }
        rendered = format_claude_event(event)
        self.assertIn("BGP peer is unreachable.", rendered or "")

    def test_assistant_tool_use_block(self) -> None:
        event = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "mcp__kathara_frr_mcp_server__frr_show_ip_route",
                        "input": {"host_name": "r1"},
                    }
                ]
            },
        }
        rendered = format_claude_event(event)
        self.assertIn("mcp__kathara_frr_mcp_server__frr_show_ip_route", rendered or "")

    def test_result_success(self) -> None:
        event = {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "Diagnosis complete.",
            "usage": {"input_tokens": 500, "output_tokens": 80},
        }
        rendered = format_claude_event(event)
        self.assertIn("in=500", rendered or "")
        self.assertIn("out=80", rendered or "")

    def test_result_error(self) -> None:
        event = {
            "type": "result",
            "is_error": True,
            "result": "Not logged in",
        }
        rendered = format_claude_event(event)
        self.assertIn("Not logged in", rendered or "")

    def test_unknown_type_returns_none(self) -> None:
        self.assertIsNone(format_claude_event({"type": "some_unknown"}))


# ---------------------------------------------------------------------------
# Integration pipeline (Docker + Claude CLI)
# ---------------------------------------------------------------------------


@unittest.skipUnless(claude_cli_available(), "Claude Code CLI and credentials required")
class ClaudeAgentPipelineTest(CommonPipelineSteps, OrderedPipelineTestCase):
    """Full pipeline with the Claude Code CLI agent."""

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.session_id and not cls.env_destroyed:
            try:
                cls.runner.invoke(app, ["session", "close", "--session_id", cls.session_id, "-y"])
            except Exception:
                pass

    def test_step_01_start_env(self) -> None:
        self._step_start_env()

    def test_step_02_inject_failure(self) -> None:
        self._step_inject_failure()

    def test_step_03_run_claude_agent(self) -> None:
        self.assertIsNotNone(self.session_id)
        result = self.runner.invoke(
            app,
            [
                "agent",
                "run",
                "--agent",
                "local_cli.claude_cli",
                "--max-steps",
                "20",
                "--session_id",
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
        self.assertEqual(row.get("agent_type"), "local_cli.claude_cli")

    def test_step_04_check_workspace_and_messages(self) -> None:
        self.assertIsNotNone(self.session_dir)

        workspace = self.session_dir / "claude_workspace"
        self.assertTrue(workspace.is_dir())

        diag_config = workspace / "diagnosis_mcp_config.json"
        sub_config = workspace / "submission_mcp_config.json"
        self.assertTrue(diag_config.exists())
        self.assertTrue(sub_config.exists())

        diag_cfg = json.loads(diag_config.read_text())
        self.assertIn("mcpServers", diag_cfg)
        self.assertIn("kathara_base_mcp_server", diag_cfg["mcpServers"])
        self.assertIn("kathara_frr_mcp_server", diag_cfg["mcpServers"])
        self.assertNotIn("kathara_bmv2_mcp_server", diag_cfg["mcpServers"])

        sub_cfg = json.loads(sub_config.read_text())
        self.assertIn("task_mcp_server", sub_cfg["mcpServers"])

        diag_cfg_str = diag_config.read_text()
        self.assertIn("NIKA_SESSION_ID", diag_cfg_str)
        self.assertIn(self.session_id, diag_cfg_str)

        messages = self._load_jsonl("messages.jsonl")
        agents = {e["agent"] for e in messages}
        self.assertIn(DIAGNOSIS, agents)
        self.assertIn(SUBMISSION, agents)

        mcp_events = [e for e in messages if e.get("event") == "mcp_config"]
        diag_mcp = next((e for e in mcp_events if e.get("agent") == DIAGNOSIS), None)
        self.assertIsNotNone(diag_mcp)
        self.assertIn("kathara_base_mcp_server", diag_mcp.get("servers", []))
        self.assertIn("kathara_frr_mcp_server", diag_mcp.get("servers", []))

        sub_mcp = next((e for e in mcp_events if e.get("agent") == SUBMISSION), None)
        self.assertIsNotNone(sub_mcp)
        self.assertIn("task_mcp_server", sub_mcp.get("servers", []))

        start_events = [e for e in messages if e.get("event") == "subprocess_start"]
        self.assertGreaterEqual(len(start_events), 2)

        claude_events = [e for e in messages if "claude_event" in e]
        self.assertGreater(len(claude_events), 0)
        rendered_count = sum(1 for e in claude_events if format_claude_event(e["claude_event"]))
        self.assertGreater(rendered_count, 0)

        result_events = [e for e in claude_events if e["claude_event"].get("type") == "result"]
        self.assertTrue(result_events)

    def test_step_05_check_submission(self) -> None:
        self.assertIsNotNone(self.session_dir)
        self.assertTrue((self.session_dir / "submission.json").exists())
        assert_submission_fields(self, self.session_dir)

    def test_step_06_session_close(self) -> None:
        self._step_close_and_verify("local_cli.claude_cli")

    def test_step_07_eval_metrics(self) -> None:
        self._step_eval_metrics()


if __name__ == "__main__":
    unittest.main()
