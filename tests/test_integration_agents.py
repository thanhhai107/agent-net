"""Integrated agent pipeline tests.

Covers all four agent types (mock, codex_cli, claude_cli, react) against a real Docker
scenario with a live Kathara lab.  The file is organized as follows:

Unit tests (no Docker/LLM required)
------------------------------------
* :class:`BuildMcpTomlTest`  — Codex CLI MCP TOML config generation
* :class:`BuildMcpJsonTest`  — Claude Code MCP JSON config generation
* :class:`AgentConfigTest` — agent CLI env resolution (``nika.utils.agent_config``)
* :class:`ClaudeConfigTest`  — Claude env model and auth helpers
* :class:`CodexDisplayTest`  — ``codex exec --json`` event formatter
* :class:`ClaudeDisplayTest` — ``claude --output-format stream-json`` event formatter

Pipeline integration tests (Docker required)
---------------------------------------------
Each pipeline test deploys a ``simple_bgp`` lab, injects a ``link_down``
fault, runs the agent, verifies all artefacts, closes the session, and
evaluates metrics.  Steps are ordered so that a failing step short-circuits
subsequent ones via ``assertEqual``/``assertIsNotNone`` guards.

* :class:`MockAgentPipelineTest`  — mock agent (no LLM; always runnable)
  Also validates MCP session context, diagnosis tools, and task MCP submission.
* :class:`CodexCliAgentPipelineTest`   — Codex CLI agent
  (requires ``codex`` in PATH and OpenAI credentials)
* :class:`ClaudeAgentPipelineTest` — Claude Code CLI agent
  (requires ``claude`` in PATH; ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN`` or ``claude auth login``)
* :class:`ReactAgentPipelineTest` — LangChain ReAct agent
  (requires DEEPSEEK_API_KEY; skipped when absent)

Prerequisites
-------------
- Docker must be running
- Kathara images must be available (``kathara/nika-frr``, ``kathara/nika-base``)
- For the CLI agent: ``codex login`` or ``OPENAI_API_KEY`` env var
- For the Claude agent: credentials via ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN``
  (optional ``ANTHROPIC_BASE_URL`` for Anthropic-compatible APIs) or ``claude auth login``;
  model from ``ANTHROPIC_MODEL`` when ``-m`` is omitted
- For the React agent: ``DEEPSEEK_API_KEY`` in ``.env``

Run
---
::

    # All tests (Docker + credentials required):
    uv run python -m unittest tests/test_integration_agents.py -v

    # Fast unit tests only:
    uv run python -m unittest tests.test_integration_agents.BuildMcpTomlTest -v
    uv run python -m unittest tests.test_integration_agents.AgentConfigTest -v
    uv run python -m unittest tests.test_integration_agents.ClaudeConfigTest -v
    uv run python -m unittest tests.test_integration_agents.BuildMcpJsonTest -v
    uv run python -m unittest tests.test_integration_agents.CodexDisplayTest -v
    uv run python -m unittest tests.test_integration_agents.ClaudeDisplayTest -v

    # Single agent pipeline:
    uv run python -m unittest tests.test_integration_agents.MockAgentPipelineTest -v
    uv run python -m unittest tests.test_integration_agents.CodexCliAgentPipelineTest -v
    uv run python -m unittest tests.test_integration_agents.ClaudeAgentPipelineTest -v
    uv run python -m unittest tests.test_integration_agents.ReactAgentPipelineTest -v
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import unittest
import unittest.mock
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

from agent.codex_cli.codex_display import format_codex_event
from agent.codex_cli.codex_worker import CodexWorker, _build_mcp_toml
from agent.claude_cli.claude_display import format_claude_event
from agent.claude_cli.config import (
    claude_credentials_available,
    default_claude_model,
    has_env_claude_credentials,
    prepare_claude_subprocess_env,
    resolve_claude_model,
    use_bare_claude_mode,
)
from agent.claude_cli.claude_worker import ClaudeWorker, _build_mcp_json
from agent.utils.mcp_servers import MCPServerConfig
from agent.utils.phases import DIAGNOSIS, SUBMISSION
from langchain_mcp_adapters.client import MultiServerMCPClient
from nika.codex_cli.main import app
from nika.utils.agent_config import (
    ENV_AGENT_TYPE,
    ENV_CODEX_MODEL,
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
    resolve_reasoning_effort,
)
from nika.utils.session_index import SessionIndex
from nika.utils.session_store import SessionStore
from tests.integration_base import OrderedPipelineTestCase

SCENARIO = "simple_bgp"
PROBLEM = "link_down"
CODEX_MODEL = "gpt-5.4-mini"
REACT_PROVIDER = "deepseek"
REACT_MODEL = "deepseek-chat"


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------

def _codex_cli_available() -> bool:
    if shutil.which("codex") is None:
        return False
    return bool(os.environ.get("OPENAI_API_KEY")) or (Path.home() / ".codex" / "auth.json").is_file()


def _claude_cli_available() -> bool:
    return claude_credentials_available()


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
    return [str(item["text"]) if isinstance(item, dict) and "text" in item else str(item) for item in result]


# ===========================================================================
# Unit tests — no Docker/LLM required
# ===========================================================================


class BuildMcpTomlTest(unittest.TestCase):
    """Unit tests for Codex CLI MCP config TOML generation."""

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
                ENV_AGENT_TYPE: "codex_cli",
                ENV_LLM_PROVIDER: "deepseek",
                ENV_MAX_STEPS: "30",
                ENV_CODEX_MODEL: "gpt-5.4-mini",
                "NIKA_CODEX_REASONING_EFFORT": "medium",
            },
            clear=False,
        ):
            self.assertEqual(resolve_agent_type(None), "codex_cli")
            self.assertIsNone(resolve_llm_provider(None, agent_type="codex_cli"))
            self.assertEqual(resolve_max_steps(None), 30)
            self.assertEqual(resolve_reasoning_effort(None), "medium")

    def test_missing_config_raises(self) -> None:
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                resolve_agent_type(None)
            with self.assertRaises(ValueError):
                resolve_llm_provider(None, agent_type="react")
            with self.assertRaises(ValueError):
                resolve_max_steps(None)

    def test_llm_provider_optional_for_non_react(self) -> None:
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(resolve_llm_provider(None, agent_type="mock"))
            self.assertIsNone(resolve_llm_provider(None, agent_type="codex_cli"))
            self.assertIsNone(resolve_llm_provider(None, agent_type="claude_cli"))

    def test_agent_model_per_type(self) -> None:
        with unittest.mock.patch.dict(
            os.environ,
            {
                ENV_REACT_MODEL: "deepseek-chat",
                ENV_CODEX_MODEL: "gpt-5.4-mini",
                ENV_MOCK_MODEL: "mock-v1",
                "ANTHROPIC_MODEL": "deepseek-v4-pro[1m]",
            },
            clear=False,
        ):
            self.assertEqual(resolve_agent_model("react", None), "deepseek-chat")
            self.assertEqual(resolve_agent_model("codex_cli", None), "gpt-5.4-mini")
            self.assertEqual(resolve_agent_model("mock", None), "mock-v1")
            self.assertEqual(resolve_agent_model("claude_cli", None), "deepseek-v4-pro[1m]")
            self.assertEqual(resolve_agent_model("react", "override"), "override")

    def test_judge_from_env(self) -> None:
        with unittest.mock.patch.dict(
            os.environ,
            {ENV_JUDGE_PROVIDER: "deepseek", ENV_JUDGE_MODEL: "deepseek-chat"},
            clear=False,
        ):
            self.assertEqual(resolve_judge_provider(None), "deepseek")
            self.assertEqual(resolve_judge_model(None), "deepseek-chat")

    def test_judge_missing_raises(self) -> None:
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                resolve_judge_provider(None)


class ClaudeConfigTest(unittest.TestCase):
    """Unit tests for Claude env model and auth helpers."""

    def test_default_model_reads_anthropic_model(self) -> None:
        with unittest.mock.patch.dict(
            os.environ,
            {"ANTHROPIC_MODEL": "model-a", "CLAUDE_CODE_SUBAGENT_MODEL": "model-b"},
            clear=False,
        ):
            self.assertEqual(default_claude_model(), "model-a")

    def test_default_model_missing_raises(self) -> None:
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                default_claude_model()

    def test_resolve_claude_model_prefers_explicit(self) -> None:
        self.assertEqual(resolve_claude_model("custom-model"), "custom-model")

    def test_prepare_env_maps_auth_token_to_api_key(self) -> None:
        env = prepare_claude_subprocess_env({"ANTHROPIC_AUTH_TOKEN": "tok"})
        self.assertEqual(env["ANTHROPIC_API_KEY"], "tok")

    def test_use_bare_mode_when_env_credentials_present(self) -> None:
        with unittest.mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key"}, clear=False):
            self.assertTrue(use_bare_claude_mode())
            self.assertTrue(has_env_claude_credentials())


class BuildMcpJsonTest(unittest.TestCase):
    """Unit tests for Claude Code MCP config JSON generation."""

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
        json_str = _build_mcp_json(
            {"task_mcp_server": {"command": "python3", "args": ["/path/task.py"]}}
        )
        config = json.loads(json_str)
        self.assertNotIn("env", config["mcpServers"]["task_mcp_server"])


class ClaudeWorkerConfigTest(unittest.TestCase):
    """Unit tests for ClaudeWorker constructor validation."""

    def test_rejects_invalid_phase(self) -> None:
        with self.assertRaises(ValueError):
            ClaudeWorker(
                session_id="sess-123",
                session_dir="/tmp/sess-123",
                phase="invalid_phase",
            )

    def test_accepts_valid_phases(self) -> None:
        for phase in (DIAGNOSIS, SUBMISSION):
            worker = ClaudeWorker(
                session_id="sess-123",
                session_dir="/tmp/sess-123",
                phase=phase,
            )
            self.assertEqual(worker.phase, phase)


class CodexWorkerConfigTest(unittest.TestCase):
    """Unit tests for CodexWorker constructor validation."""

    def test_rejects_invalid_reasoning_effort(self) -> None:
        with self.assertRaises(ValueError):
            CodexWorker(
                session_id="sess-123",
                session_dir="/tmp/sess-123",
                phase=DIAGNOSIS,
                reasoning_effort="turbo",
            )


class CodexDisplayTest(unittest.TestCase):
    """Unit tests for Codex JSONL terminal event formatting."""

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


class ClaudeDisplayTest(unittest.TestCase):
    """Unit tests for Claude Code stream-json event formatting."""

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
        """Claude Code returns mcp_servers as a list of dicts with a 'name' key."""
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
            "message": {
                "content": [{"type": "text", "text": "BGP peer is unreachable."}]
            },
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


# ===========================================================================
# Pipeline integration tests — Docker + live Kathara lab required
# ===========================================================================


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
                "failure", "inject", PROBLEM,
                "--session-id", self.session_id,
                "--set", "host_name=pc1",
                "--set", "intf_name=eth0",
            ]
        )
        row = SessionStore().get_session(self.session_id)
        self.assertIn(PROBLEM, row.get("problem_names", []))  # type: ignore[attr-defined]
        self.assertIn("task_description", row)  # type: ignore[attr-defined]
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
        self.assertGreaterEqual(metrics["detection_score"], 0.0)  # type: ignore[attr-defined]
        self.assertGreaterEqual(metrics["tool_calls"], min_tool_calls)  # type: ignore[attr-defined]

        run = self._load_json("run.json")  # type: ignore[attr-defined]
        self.assertIn("eval_metrics", run)  # type: ignore[attr-defined]

        index_row = SessionIndex().get_row(self.session_id)  # type: ignore[attr-defined]
        self.assertIsNotNone(index_row)  # type: ignore[attr-defined]
        self.assertIsNotNone(index_row.get("detection_score"))  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Mock agent pipeline
# ---------------------------------------------------------------------------


class MockAgentPipelineTest(_CommonPipelineSteps, OrderedPipelineTestCase):
    """Full pipeline with the mock agent — also validates MCP infrastructure."""

    def test_step_01_start_env(self) -> None:
        """Deploy simple_bgp and capture the session id."""
        self._step_start_env()

    def test_step_02_verify_session_and_cli(self) -> None:
        """Confirm the session is active and basic CLI commands work."""
        self.assertIsNotNone(self.session_id)

        row = SessionStore().get_session(self.session_id)
        self.assertEqual(row["status"], "running")
        self.assertEqual(row["scenario_name"], SCENARIO)
        self.assertIsNotNone(row.get("lab_name"))

        self._invoke_ok(["env", "ps"])
        self._invoke_ok(["session", "ps"])
        self._invoke_ok(["session", "inspect"])
        self._invoke_ok(["failure", "ps"])
        self._invoke_ok(["exec", "pc1", "hostname"])
        desc = self._invoke_ok(["failure", "describe", PROBLEM])
        self.assertIn(PROBLEM, desc)

    def test_step_03_inject_failure(self) -> None:
        """Inject a fault and record ground truth."""
        self._step_inject_failure()

    def test_step_04_mcp_session_context(self) -> None:
        """mcp_session_context resolves lab_name and session_dir from NIKA_SESSION_ID."""
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
        """Diagnosis MCP tools run correctly against the live lab."""
        self.assertIsNotNone(self.session_id)

        mcp_config = MCPServerConfig(session_id=self.session_id)
        self.assertEqual(mcp_config._server_env()["NIKA_SESSION_ID"], self.session_id)

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
            self.assertNotIn("NIKA_SESSION_ID is not set", result)

            self.assertIn("exec_shell", tools)
            exec_result = str(await tools["exec_shell"].ainvoke({"host_name": "pc1", "command": "hostname"}))
            self.assertGreater(len(exec_result), 0)

        asyncio.run(_run())

    def test_step_06_task_mcp_submit(self) -> None:
        """task_mcp_server.submit() writes submission.json with correct content."""
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

        submission = json.loads((self.session_dir / "submission.json").read_text())
        self.assertTrue(submission["is_anomaly"])
        self.assertIn("pc1", submission["faulty_devices"])
        self.assertIn(PROBLEM, submission["root_cause_name"])

    def test_step_07_run_mock_agent(self) -> None:
        """Mock agent completes the full diagnosis + submission pipeline."""
        self.assertIsNotNone(self.session_id)
        self._invoke_ok(
            ["agent", "run", "--agent", "mock", "--model", "mock-v1",
             "--session-id", self.session_id]
        )

        messages = self._load_jsonl("messages.jsonl")
        agents = {e["agent"] for e in messages}
        self.assertIn(DIAGNOSIS, agents)
        self.assertIn(SUBMISSION, agents)

        diag_events = [e["event"] for e in messages if e["agent"] == DIAGNOSIS]
        self.assertIn("tool_start", diag_events)
        self.assertIn("llm_end", diag_events)

        sub_tools = [
            e["tool"]["name"]
            for e in messages
            if e["agent"] == SUBMISSION and e["event"] == "tool_start"
        ]
        self.assertIn("list_avail_problems", sub_tools)
        self.assertIn("submit", sub_tools)

        submission = self._load_json("submission.json")
        for field in ("is_anomaly", "faulty_devices", "root_cause_name"):
            self.assertIn(field, submission)

        run = self._load_json("run.json")
        self.assertEqual(run["agent_type"], "mock")

    def test_step_08_session_close(self) -> None:
        """Close the session and verify it is marked finished."""
        self._step_close_and_verify("mock")

    def test_step_09_eval_metrics(self) -> None:
        """Compute rule-based metrics and verify required fields."""
        self._step_eval_metrics(min_tool_calls=1)
        metrics = self._load_json("eval_metrics.json")
        self.assertEqual(metrics["detection_score"], 1.0)
        self.assertEqual(metrics["rca_accuracy"], 1.0)


# ---------------------------------------------------------------------------
# Codex CLI agent pipeline
# ---------------------------------------------------------------------------


@unittest.skipUnless(_codex_cli_available(), "Codex CLI and OpenAI credentials required")
class CodexCliAgentPipelineTest(_CommonPipelineSteps, OrderedPipelineTestCase):
    """Full pipeline with the Codex CLI agent — validates workspace artefacts."""

    @classmethod
    def tearDownClass(cls) -> None:
        """Undeploy if still running; preserve artefacts for inspection."""
        if cls.session_id and not cls.env_destroyed:
            try:
                cls.runner.invoke(app, ["session", "close", cls.session_id, "-y"])
            except Exception:
                pass

    def test_step_01_start_env(self) -> None:
        self._step_start_env()

    def test_step_02_inject_failure(self) -> None:
        self._step_inject_failure()

    def test_step_03_run_cli_agent(self) -> None:
        """Run the Codex CLI agent through the full diagnosis → submission pipeline."""
        self.assertIsNotNone(self.session_id)
        result = self.runner.invoke(
            app,
            ["agent", "run", "--agent", "codex_cli",
             "--model", CODEX_MODEL, "--session-id", self.session_id],
        )
        self.assertEqual(
            result.exit_code, 0,
            f"agent run exited {result.exit_code}:\n{result.output}"
            + (f"\nException: {result.exception}" if result.exception else ""),
        )
        row = SessionStore().get_session(self.session_id)
        self.assertEqual(row.get("agent_type"), "codex_cli")

    def test_step_04_check_workspace_and_messages(self) -> None:
        """Verify codex_workspace artefacts and messages.jsonl events."""
        self.assertIsNotNone(self.session_dir)

        workspace = self.session_dir / "codex_workspace"
        self.assertTrue(workspace.is_dir(), "codex_workspace/ must exist after agent run")
        self.assertTrue((workspace / ".git").is_dir(), "workspace must be a git repo")
        self.assertTrue((workspace / ".codex_home").is_dir(), "isolated .codex_home must exist")

        config_text = (workspace / ".codex_home" / "config.toml").read_text(encoding="utf-8")
        self.assertIn("NIKA_SESSION_ID", config_text, "Session ID must appear in the MCP server env block")
        self.assertIn(self.session_id, config_text, "Session ID value must match the running session")
        self.assertIn("[mcp_servers.", config_text, "config.toml must contain at least one [mcp_servers.*] section")
        self.assertIn(
            'default_tools_approval_mode = "approve"',
            config_text,
            "Codex exec needs auto-approved MCP tools in non-interactive mode",
        )

        diag_output = workspace / "diagnosis_output.txt"
        self.assertTrue(diag_output.exists(), "diagnosis_output.txt must be written by --output-last-message")
        self.assertGreater(diag_output.stat().st_size, 0, "diagnosis output must be non-empty")

        messages = self._load_jsonl("messages.jsonl")
        agents = {e["agent"] for e in messages}
        self.assertIn(DIAGNOSIS, agents, f"diagnosis phase must log events under {DIAGNOSIS!r}")
        self.assertIn(SUBMISSION, agents, f"submission phase must log events under {SUBMISSION!r}")

        mcp_events = [e for e in messages if e.get("event") == "mcp_config"]
        self.assertTrue(len(mcp_events) >= 1, "At least one mcp_config event must be logged")
        diag_mcp = next((e for e in mcp_events if e.get("agent") == DIAGNOSIS), None)
        self.assertIsNotNone(diag_mcp, "diagnosis phase must log an mcp_config event")
        servers = diag_mcp.get("servers", [])
        self.assertIn("kathara_base_mcp_server", servers, "base server must always be selected for diagnosis")
        self.assertIn("kathara_frr_mcp_server", servers, "'simple_bgp' contains 'bgp' → frr server must be selected")
        self.assertNotIn("kathara_bmv2_mcp_server", servers, "bmv2 server must NOT be selected for a pure BGP scenario")
        self.assertNotIn(
            "kathara_telemetry_mcp_server", servers, "telemetry server must NOT be selected for a pure BGP scenario"
        )

        sub_mcp = next((e for e in mcp_events if e.get("agent") == SUBMISSION), None)
        self.assertIsNotNone(sub_mcp, "submission phase must log an mcp_config event")
        self.assertIn("task_mcp_server", sub_mcp.get("servers", []), "task MCP server must be selected for submission")

        start_events = [e for e in messages if e.get("event") == "subprocess_start"]
        self.assertGreaterEqual(len(start_events), 2, "subprocess_start must be logged for both phases")

        codex_events = [e for e in messages if "codex_event" in e]
        self.assertGreater(
            len(codex_events),
            0,
            "messages.jsonl must contain codex exec --json events under 'codex_event'",
        )
        rendered_count = sum(1 for e in codex_events if format_codex_event(e["codex_event"]))
        self.assertGreater(
            rendered_count,
            0,
            "at least one codex event from the pipeline must format for terminal display",
        )

        agent_messages = [
            e
            for e in codex_events
            if e["codex_event"].get("type") == "item.completed"
            and (e["codex_event"].get("item") or {}).get("type") == "agent_message"
        ]
        self.assertTrue(agent_messages, "pipeline must produce at least one agent_message codex event")
        rendered_agent = format_codex_event(agent_messages[0]["codex_event"])
        self.assertIsNotNone(rendered_agent)
        self.assertIn("Agent:", rendered_agent or "")

        mcp_calls = [
            e
            for e in codex_events
            if (e["codex_event"].get("item") or {}).get("type") == "mcp_tool_call"
        ]
        self.assertTrue(mcp_calls, "pipeline must produce at least one mcp_tool_call codex event")
        self.assertIn("MCP", format_codex_event(mcp_calls[0]["codex_event"]) or "")

        turn_completed = [e for e in codex_events if e["codex_event"].get("type") == "turn.completed"]
        self.assertTrue(turn_completed, "pipeline must produce at least one turn.completed codex event")
        self.assertIn("Turn completed", format_codex_event(turn_completed[0]["codex_event"]) or "")

    def test_step_05_check_submission(self) -> None:
        """submission.json must be written with required fields."""
        self.assertIsNotNone(self.session_dir)
        submission_path = self.session_dir / "submission.json"
        self.assertTrue(
            submission_path.exists(),
            "submission.json must be written by the task MCP server's submit() tool.\n"
            "If this fails, codex did not complete the submission phase — check "
            "messages.jsonl for subprocess_error events.",
        )
        submission = json.loads(submission_path.read_text(encoding="utf-8"))
        for field in ("is_anomaly", "faulty_devices", "root_cause_name"):
            self.assertIn(field, submission)

    def test_step_06_session_close(self) -> None:
        self._step_close_and_verify("codex_cli")

    def test_step_07_eval_metrics(self) -> None:
        self._step_eval_metrics()


# ---------------------------------------------------------------------------
# Claude Code CLI agent pipeline
# ---------------------------------------------------------------------------


@unittest.skipUnless(_claude_cli_available(), "Claude Code CLI and credentials required")
class ClaudeAgentPipelineTest(_CommonPipelineSteps, OrderedPipelineTestCase):
    """Full pipeline with the Claude Code CLI agent — validates workspace artefacts."""

    @classmethod
    def tearDownClass(cls) -> None:
        """Undeploy if still running; preserve artefacts for inspection."""
        if cls.session_id and not cls.env_destroyed:
            try:
                cls.runner.invoke(app, ["session", "close", cls.session_id, "-y"])
            except Exception:
                pass

    def test_step_01_start_env(self) -> None:
        self._step_start_env()

    def test_step_02_inject_failure(self) -> None:
        self._step_inject_failure()

    def test_step_03_run_claude_agent(self) -> None:
        """Run the Claude Code CLI agent through the full diagnosis → submission pipeline."""
        self.assertIsNotNone(self.session_id)
        result = self.runner.invoke(
            app,
            ["agent", "run", "--agent", "claude_cli",
             "--session-id", self.session_id],
        )
        self.assertEqual(
            result.exit_code, 0,
            f"agent run exited {result.exit_code}:\n{result.output}"
            + (f"\nException: {result.exception}" if result.exception else ""),
        )
        row = SessionStore().get_session(self.session_id)
        self.assertEqual(row.get("agent_type"), "claude_cli")

    def test_step_04_check_workspace_and_messages(self) -> None:
        """Verify claude_workspace artefacts and messages.jsonl events."""
        self.assertIsNotNone(self.session_dir)

        workspace = self.session_dir / "claude_workspace"
        self.assertTrue(workspace.is_dir(), "claude_workspace/ must exist after agent run")

        # MCP config JSON files for both phases.
        diag_config = workspace / "diagnosis_mcp_config.json"
        sub_config = workspace / "submission_mcp_config.json"
        self.assertTrue(diag_config.exists(), "diagnosis_mcp_config.json must be written")
        self.assertTrue(sub_config.exists(), "submission_mcp_config.json must be written")

        diag_cfg = json.loads(diag_config.read_text())
        self.assertIn("mcpServers", diag_cfg)
        self.assertIn("kathara_base_mcp_server", diag_cfg["mcpServers"])
        self.assertIn("kathara_frr_mcp_server", diag_cfg["mcpServers"])
        self.assertNotIn("kathara_bmv2_mcp_server", diag_cfg["mcpServers"])

        sub_cfg = json.loads(sub_config.read_text())
        self.assertIn("task_mcp_server", sub_cfg["mcpServers"])

        # NIKA_SESSION_ID must be present in the env blocks.
        diag_cfg_str = diag_config.read_text()
        self.assertIn("NIKA_SESSION_ID", diag_cfg_str)
        self.assertIn(self.session_id, diag_cfg_str)

        messages = self._load_jsonl("messages.jsonl")
        agents = {e["agent"] for e in messages}
        self.assertIn(DIAGNOSIS, agents)
        self.assertIn(SUBMISSION, agents)

        mcp_events = [e for e in messages if e.get("event") == "mcp_config"]
        diag_mcp = next((e for e in mcp_events if e.get("agent") == DIAGNOSIS), None)
        self.assertIsNotNone(diag_mcp, "diagnosis phase must log an mcp_config event")
        servers = diag_mcp.get("servers", [])
        self.assertIn("kathara_base_mcp_server", servers)
        self.assertIn("kathara_frr_mcp_server", servers)

        sub_mcp = next((e for e in mcp_events if e.get("agent") == SUBMISSION), None)
        self.assertIsNotNone(sub_mcp, "submission phase must log an mcp_config event")
        self.assertIn("task_mcp_server", sub_mcp.get("servers", []))

        # subprocess_start events confirm claude was invoked for both phases.
        start_events = [e for e in messages if e.get("event") == "subprocess_start"]
        self.assertGreaterEqual(len(start_events), 2)

        # Claude stream-json events must be logged and renderable.
        claude_events = [e for e in messages if "claude_event" in e]
        self.assertGreater(len(claude_events), 0, "messages.jsonl must contain claude stream-json events")
        rendered_count = sum(1 for e in claude_events if format_claude_event(e["claude_event"]))
        self.assertGreater(rendered_count, 0)

        # A result event must be present (final response).
        result_events = [
            e for e in claude_events if e["claude_event"].get("type") == "result"
        ]
        self.assertTrue(result_events, "pipeline must produce at least one result event")

    def test_step_05_check_submission(self) -> None:
        """submission.json must be written with required fields."""
        self.assertIsNotNone(self.session_dir)
        submission_path = self.session_dir / "submission.json"
        self.assertTrue(
            submission_path.exists(),
            "submission.json must be written by the task MCP server's submit() tool.",
        )
        submission = json.loads(submission_path.read_text())
        for field in ("is_anomaly", "faulty_devices", "root_cause_name"):
            self.assertIn(field, submission)

    def test_step_06_session_close(self) -> None:
        self._step_close_and_verify("claude_cli")

    def test_step_07_eval_metrics(self) -> None:
        self._step_eval_metrics()


# ---------------------------------------------------------------------------
# React (LangChain) agent pipeline
# ---------------------------------------------------------------------------


@unittest.skipUnless(_deepseek_api_key_available(), "DEEPSEEK_API_KEY required for react agent")
class ReactAgentPipelineTest(_CommonPipelineSteps, OrderedPipelineTestCase):
    """Full pipeline with the LangChain ReAct agent using the DeepSeek provider."""

    def test_step_01_start_env(self) -> None:
        self._step_start_env()

    def test_step_02_inject_failure(self) -> None:
        self._step_inject_failure()

    def test_step_03_run_react_agent(self) -> None:
        """Run the ReAct agent through the full diagnosis → submission pipeline."""
        self.assertIsNotNone(self.session_id)
        result = self.runner.invoke(
            app,
            [
                "agent", "run", "--agent", "react",
                "--provider", REACT_PROVIDER,
                "--model", REACT_MODEL,
                "--max-steps", "20",
                "--session-id", self.session_id,
            ],
        )
        self.assertEqual(
            result.exit_code, 0,
            f"agent run exited {result.exit_code}:\n{result.output}"
            + (f"\nException: {result.exception}" if result.exception else ""),
        )
        row = SessionStore().get_session(self.session_id)
        self.assertEqual(row.get("agent_type"), "react")

    def test_step_04_check_messages(self) -> None:
        """messages.jsonl must contain diagnosis and submission phase events."""
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

        sub_tools = [
            e["tool"]["name"]
            for e in messages
            if e["agent"] == SUBMISSION and e["event"] == "tool_start" and "tool" in e
        ]
        self.assertIn("list_avail_problems", sub_tools)
        self.assertIn("submit", sub_tools)

    def test_step_05_check_submission(self) -> None:
        """submission.json must be written with required fields."""
        self.assertIsNotNone(self.session_dir)
        submission = json.loads((self.session_dir / "submission.json").read_text())
        for field in ("is_anomaly", "faulty_devices", "root_cause_name"):
            self.assertIn(field, submission)

    def test_step_06_session_close(self) -> None:
        self._step_close_and_verify("react")

    def test_step_07_eval_metrics(self) -> None:
        self._step_eval_metrics()


if __name__ == "__main__":
    unittest.main()
