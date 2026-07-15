"""Centralized unit tests for shared agent CLI/env configuration."""

from __future__ import annotations

import os
import unittest
import unittest.mock

from nika.utils.agent_config import (
    ENV_AGENT_TYPE,
    ENV_JUDGE_MODEL,
    ENV_JUDGE_PROVIDER,
    ENV_LLM_PROVIDER,
    ENV_MAX_STEPS,
    SUPPORTED_AGENT_TYPES,
    resolve_agent_model,
    resolve_agent_type,
    resolve_judge_model,
    resolve_judge_provider,
    resolve_llm_provider,
    resolve_max_steps,
)
from tests.support.integration_pipeline import load_test_env

load_test_env()


class AgentConfigTest(unittest.TestCase):
    def test_only_three_public_agent_types_are_supported(self) -> None:
        self.assertEqual(
            SUPPORTED_AGENT_TYPES,
            ("react", "plan-execute", "reflexion"),
        )
        with self.assertRaisesRegex(ValueError, "Unsupported agent type"):
            resolve_agent_type("unsupported")

    def test_cli_values_override_env(self) -> None:
        with unittest.mock.patch.dict(
            os.environ,
            {
                ENV_AGENT_TYPE: "mock",
                ENV_LLM_PROVIDER: "deepseek",
                ENV_MAX_STEPS: "99",
            },
            clear=True,
        ):
            self.assertEqual(resolve_agent_type("byo.langgraph"), "react")
            self.assertEqual(
                resolve_llm_provider("openai", agent_type="react"), "openai"
            )
            self.assertEqual(resolve_max_steps(20), 20)
            self.assertEqual(resolve_agent_model("react", "override"), "override")

    def test_required_shared_values_raise_when_missing(self) -> None:
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_agent_type(None), "react")
            self.assertEqual(resolve_max_steps(None), 50)
            self.assertEqual(resolve_llm_provider(None, agent_type="react"), "custom")

    def test_internal_mock_does_not_require_llm_provider(self) -> None:
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(resolve_llm_provider(None, agent_type="mock"))

    def test_judge_values_from_baseline(self) -> None:
        with unittest.mock.patch.dict(
            os.environ,
            {ENV_JUDGE_PROVIDER: "deepseek", ENV_JUDGE_MODEL: "deepseek-chat"},
            clear=True,
        ):
            self.assertEqual(resolve_judge_provider(None), "custom")
            self.assertEqual(resolve_judge_model(None), "openai/gpt-oss-120b")

    def test_agent_models_use_shared_baseline(self) -> None:
        self.assertEqual(resolve_agent_model("react", None), "openai/gpt-oss-120b")


if __name__ == "__main__":
    unittest.main()
