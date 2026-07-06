"""Mock agent unit tests (CLI config resolution)."""

from __future__ import annotations

import os
import unittest
import unittest.mock

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
from tests.integration_pipeline import load_test_env

load_test_env()


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


if __name__ == "__main__":
    unittest.main()
