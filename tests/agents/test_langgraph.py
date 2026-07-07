"""LangGraph agent tests: unit checks + ``simple_bgp`` / ``link_down`` pipeline."""

from __future__ import annotations

import os
import unittest
import unittest.mock

from nika.utils.agent_config import (
    ENV_LANGGRAPH_MODEL,
    ENV_LLM_PROVIDER,
    resolve_agent_model,
    resolve_llm_provider,
)
from nika.utils.session_store import SessionStore
from tests.agents._assertions import assert_phase_messages, assert_submission_fields
from tests.integration_base import OrderedPipelineTestCase
from tests.integration_pipeline import (
    CommonPipelineSteps,
    deepseek_api_key_available,
    load_test_env,
)

load_test_env()

LANGGRAPH_PROVIDER = "deepseek"
LANGGRAPH_MODEL = "deepseek-chat"


# ---------------------------------------------------------------------------
# Unit tests (no Docker)
# ---------------------------------------------------------------------------


class LangGraphAgentConfigTest(unittest.TestCase):
    """CLI env resolution for the langgraph agent."""

    def test_cli_overrides_env(self) -> None:
        with unittest.mock.patch.dict(
            os.environ,
            {ENV_LLM_PROVIDER: "deepseek", ENV_LANGGRAPH_MODEL: "deepseek-chat"},
            clear=True,
        ):
            self.assertEqual(
                resolve_llm_provider("openai", agent_type="byo.langgraph"), "openai"
            )

    def test_llm_provider_required(self) -> None:
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                resolve_llm_provider(None, agent_type="byo.langgraph")

    def test_model_from_env(self) -> None:
        with unittest.mock.patch.dict(
            os.environ, {ENV_LANGGRAPH_MODEL: "deepseek-chat"}, clear=True
        ):
            self.assertEqual(
                resolve_agent_model("byo.langgraph", None), "deepseek-chat"
            )
            self.assertEqual(
                resolve_agent_model("byo.langgraph", "override"), "override"
            )


# ---------------------------------------------------------------------------
# Integration pipeline (Docker + DeepSeek)
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    deepseek_api_key_available(), "DEEPSEEK_API_KEY required for byo.langgraph agent"
)
class LangGraphAgentPipelineTest(CommonPipelineSteps, OrderedPipelineTestCase):
    """Full pipeline with the LangGraph agent."""

    def test_step_01_start_env(self) -> None:
        self._step_start_env()

    def test_step_02_inject_failure(self) -> None:
        self._step_inject_failure()

    def test_step_03_run_langgraph_agent(self) -> None:
        self.assertIsNotNone(self.session_id)
        self._run_agent(
            agent_type="byo.langgraph",
            llm_provider=LANGGRAPH_PROVIDER,
            model=LANGGRAPH_MODEL,
            max_steps=20,
        )
        row = SessionStore().get_session(self.session_id)
        self.assertEqual(row.get("agent_type"), "byo.langgraph")

    def test_step_04_check_messages(self) -> None:
        self.assertIsNotNone(self.session_dir)
        assert_phase_messages(self, self._load_jsonl("messages.jsonl"))

    def test_step_05_check_submission(self) -> None:
        self.assertIsNotNone(self.session_dir)
        assert_submission_fields(self, self.session_dir)

    def test_step_06_session_close(self) -> None:
        self._step_close_and_verify("byo.langgraph")

    def test_step_07_eval_metrics(self) -> None:
        self._step_eval_metrics()


if __name__ == "__main__":
    unittest.main()
