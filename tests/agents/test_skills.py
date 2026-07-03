"""Tests for the shared NIKA agent skill library."""

from __future__ import annotations

import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from agent.utils.skills import (
    ENV_ENABLE_SKILLS,
    ENV_SKILLS_DIR,
    claude_skills_package_dir,
    diagnosis_prompt_with_skills,
    prepare_claude_workspace,
    prepare_codex_workspace,
    resolve_skills_root,
    skills_enabled,
)
from agent.utils.template import OVERALL_DIAGNOSIS_PROMPT, SKILLS_PROMPT_SUFFIX
from nika.cli.main import app
from nika.utils.agent_config import ENV_CODEX_MODEL, ENV_CODEX_SDK_MODEL
from nika.utils.session_store import SessionStore
from tests.agents._assertions import (
    assert_skill_invoked,
    assert_submission_fields,
    marker_before_first_mcp_tool,
    skill_invoked,
)
from tests.integration_base import OrderedPipelineTestCase
from tests.integration_pipeline import (
    CommonPipelineSteps,
    claude_cli_available,
    claude_sdk_available,
    codex_cli_available,
    codex_sdk_available,
    load_test_env,
)

load_test_env()

CODEX_MODEL = (
    os.environ.get(ENV_CODEX_SDK_MODEL, "").strip()
    or os.environ.get(ENV_CODEX_MODEL, "").strip()
    or "gpt-5.4-mini"
)


class SkillsConfigTest(unittest.TestCase):
    def test_resolve_skills_root_default(self) -> None:
        root = resolve_skills_root()
        self.assertTrue((root / "skills" / "nika-test-skill" / "SKILL.md").is_file())

    def test_resolve_skills_root_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            override = Path(tmp)
            with unittest.mock.patch.dict(os.environ, {ENV_SKILLS_DIR: str(override)}, clear=False):
                self.assertEqual(resolve_skills_root(), override.resolve())

    def test_skills_enabled_default(self) -> None:
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(skills_enabled())

    def test_skills_enabled_false(self) -> None:
        with unittest.mock.patch.dict(os.environ, {ENV_ENABLE_SKILLS: "false"}, clear=True):
            self.assertFalse(skills_enabled())

    def test_claude_skills_package_dir_when_disabled(self) -> None:
        with unittest.mock.patch.dict(os.environ, {ENV_ENABLE_SKILLS: "false"}, clear=True):
            self.assertIsNone(claude_skills_package_dir())

    def test_claude_skills_package_dir_when_enabled(self) -> None:
        with unittest.mock.patch.dict(os.environ, {ENV_ENABLE_SKILLS: "true"}, clear=True):
            package = claude_skills_package_dir()
            self.assertIsNotNone(package)
            assert package is not None
            self.assertTrue((package / ".claude" / "CLAUDE.md").is_file())

    def test_diagnosis_prompt_with_skills(self) -> None:
        with unittest.mock.patch.dict(os.environ, {ENV_ENABLE_SKILLS: "true"}, clear=True):
            prompt = diagnosis_prompt_with_skills(OVERALL_DIAGNOSIS_PROMPT)
            self.assertIn(SKILLS_PROMPT_SUFFIX, prompt)
        with unittest.mock.patch.dict(os.environ, {ENV_ENABLE_SKILLS: "false"}, clear=True):
            self.assertEqual(
                diagnosis_prompt_with_skills(OVERALL_DIAGNOSIS_PROMPT),
                OVERALL_DIAGNOSIS_PROMPT,
            )


class SkillsWorkspaceTest(unittest.TestCase):
    def test_prepare_claude_workspace_links_dot_claude(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            with unittest.mock.patch.dict(os.environ, {ENV_ENABLE_SKILLS: "true"}, clear=True):
                prepare_claude_workspace(workspace)
            link = workspace / ".claude"
            self.assertTrue(link.exists())
            self.assertTrue((link / "skills" / "nika-test-skill" / "SKILL.md").exists())

    def test_prepare_codex_workspace_links_agents_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            with unittest.mock.patch.dict(os.environ, {ENV_ENABLE_SKILLS: "true"}, clear=True):
                prepare_codex_workspace(workspace)
            self.assertTrue((workspace / ".agents" / "skills" / "nika-test-skill" / "SKILL.md").exists())
            self.assertTrue((workspace / "AGENTS.md").is_file())


class SkillAssertionTest(unittest.TestCase):
    def test_skill_invoked_from_tool_start(self) -> None:
        messages = [
            {
                "event": "tool_start",
                "tool": {"name": "Skill"},
                "input": "{'skill': 'nika-test-skill'}",
            }
        ]
        self.assertTrue(skill_invoked(messages))

    def test_skill_invoked_from_claude_event(self) -> None:
        messages = [
            {
                "event": "assistant",
                "claude_event": {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Skill",
                                "input": {"skill": "nika-test-skill"},
                            }
                        ]
                    },
                },
            }
        ]
        self.assertTrue(skill_invoked(messages))

    def test_marker_before_first_mcp_tool(self) -> None:
        messages = [
            {
                "event": "llm_end",
                "text": "NIKA_TEST_SKILL_ACTIVE",
            },
            {
                "event": "tool_start",
                "tool": {"name": "get_reachability"},
                "input": "{}",
            },
        ]
        self.assertTrue(marker_before_first_mcp_tool(messages))

    def test_marker_must_precede_mcp_tools(self) -> None:
        messages = [
            {
                "event": "tool_start",
                "tool": {"name": "get_reachability"},
                "input": "{}",
            },
            {
                "event": "llm_end",
                "text": "NIKA_TEST_SKILL_ACTIVE",
            },
        ]
        self.assertFalse(marker_before_first_mcp_tool(messages))


def _skills_env_patch() -> dict[str, str]:
    return {ENV_ENABLE_SKILLS: "true"}


class _SkillPipelineMixin:
    agent_id: str
    agent_model: str | None = None
    max_steps: str = "20"

    def _agent_run_args(self) -> list[str]:
        args = [
            "agent",
            "run",
            "--agent",
            self.agent_id,
            "--max-steps",
            self.max_steps,
            "--session_id",
            self.session_id,
        ]
        if self.agent_model:
            args.extend(["--model", self.agent_model])
        return args

    def test_step_03_run_agent_with_skills(self) -> None:
        self.assertIsNotNone(self.session_id)
        with unittest.mock.patch.dict(os.environ, _skills_env_patch(), clear=False):
            result = self.runner.invoke(app, self._agent_run_args())
        self.assertEqual(
            result.exit_code,
            0,
            f"agent run exited {result.exit_code}:\n{result.output}"
            + (f"\nException: {result.exception}" if result.exception else ""),
        )
        row = SessionStore().get_session(self.session_id)
        self.assertEqual(row.get("agent_type"), self.agent_id)

    def test_step_04_check_skill_invocation(self) -> None:
        self.assertIsNotNone(self.session_dir)
        messages = self._load_jsonl("messages.jsonl")
        assert_skill_invoked(self, messages)

    def test_step_05_check_submission(self) -> None:
        self.assertIsNotNone(self.session_dir)
        self.assertTrue((self.session_dir / "submission.json").exists())
        assert_submission_fields(self, self.session_dir)


@unittest.skipUnless(claude_sdk_available(), "claude-agent-sdk + ANTHROPIC credentials required")
class ClaudeSdkSkillPipelineTest(_SkillPipelineMixin, CommonPipelineSteps, OrderedPipelineTestCase):
    agent_id = "sdk.claude_sdk"

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

    def test_step_06_session_close(self) -> None:
        self._step_close_and_verify(self.agent_id)


@unittest.skipUnless(claude_cli_available(), "Claude CLI + ANTHROPIC credentials required")
class ClaudeCliSkillPipelineTest(_SkillPipelineMixin, CommonPipelineSteps, OrderedPipelineTestCase):
    agent_id = "local_cli.claude_cli"

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

    def test_step_06_session_close(self) -> None:
        self._step_close_and_verify(self.agent_id)


@unittest.skipUnless(codex_cli_available(), "Codex CLI and OpenAI credentials required")
class CodexCliSkillPipelineTest(_SkillPipelineMixin, CommonPipelineSteps, OrderedPipelineTestCase):
    agent_id = "local_cli.codex_cli"
    agent_model = CODEX_MODEL

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

    def test_step_06_session_close(self) -> None:
        self._step_close_and_verify(self.agent_id)


@unittest.skipUnless(codex_sdk_available(), "openai-codex + ~/.codex/auth.json required")
class CodexSdkSkillPipelineTest(_SkillPipelineMixin, CommonPipelineSteps, OrderedPipelineTestCase):
    agent_id = "sdk.codex_sdk"
    agent_model = CODEX_MODEL

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

    def test_step_06_session_close(self) -> None:
        self._step_close_and_verify(self.agent_id)


if __name__ == "__main__":
    unittest.main()
