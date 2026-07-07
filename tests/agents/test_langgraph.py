"""LangGraph agent tests: unit checks + ``simple_bgp`` / ``link_down`` pipeline."""

from __future__ import annotations

import unittest

from nika.utils.session_store import SessionStore
from tests.agents._assertions import assert_phase_messages, assert_submission_fields
from tests.integration_base import OrderedPipelineTestCase
from tests.integration_pipeline import (
    ClabCommonPipelineSteps,
    CommonPipelineSteps,
    _min3clos_prerequisites,
    deepseek_api_key_available,
    load_test_env,
)

load_test_env()

LANGGRAPH_PROVIDER = "deepseek"
LANGGRAPH_MODEL = "deepseek-chat"


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


@unittest.skipUnless(
    _min3clos_prerequisites() and deepseek_api_key_available(),
    "containerlab/gnmic/Docker or DEEPSEEK_API_KEY not available",
)
class LangGraphClabPipelineTest(ClabCommonPipelineSteps, OrderedPipelineTestCase):
    """Full containerlab pipeline with the LangGraph agent."""

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
