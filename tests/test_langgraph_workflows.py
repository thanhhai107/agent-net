"""Unit tests for the advanced LangGraph troubleshooting workflows."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langgraph.errors import GraphRecursionError
from pydantic import ValidationError

from agent.langgraph.plan_execute_agent import PlanExecuteAgent
from agent.langgraph.reflection_agent import ReflectionAgent
from agent.langgraph.workflow_models import (
    DiagnosisCritique,
    InvestigationPlan,
    PlanStep,
    ReplanDecision,
    StepResult,
)
from agent.registry import create_agent
from agent.utils.loggers import AgentCallbackLogger
from nika.codex_cli.commands.agent import SUPPORTED_AGENT_TYPES


class WorkflowModelTest(unittest.TestCase):
    def test_plan_requires_at_least_one_valid_step(self) -> None:
        with self.assertRaises(ValidationError):
            InvestigationPlan(objective="Diagnose", steps=[])

        with self.assertRaises(ValidationError):
            PlanStep(step_id="", action="ping", expected_evidence="reachability")

    def test_replan_decision_enforces_finish_contract(self) -> None:
        with self.assertRaises(ValidationError):
            ReplanDecision(completed=True)

        with self.assertRaises(ValidationError):
            ReplanDecision(completed=False, remaining_steps=[])

    def test_critique_requires_revision_instructions(self) -> None:
        with self.assertRaises(ValidationError):
            DiagnosisCritique(
                evidence_sufficient=False,
                anomaly_assessment="Unclear",
                localization_assessment="Unsupported",
                root_cause_assessment="Unsupported",
                revision_instructions=[],
            )


class WorkflowRegistrationTest(unittest.TestCase):
    def test_cli_lists_all_agent_types(self) -> None:
        self.assertEqual(
            SUPPORTED_AGENT_TYPES,
            ("react", "plan-execute", "reflection", "mock", "cli"),
        )

    def test_registry_constructs_new_agent_types(self) -> None:
        with (
            patch("agent.registry.PlanExecuteAgent") as plan_agent,
            patch("agent.registry.ReflectionAgent") as reflection_agent,
        ):
            create_agent(
                "plan-execute",
                session_id="session",
                llm_backend="openai",
                model="model",
                max_steps=7,
            )
            create_agent(
                "reflection",
                session_id="session",
                llm_backend="deepseek",
                model="model",
                max_steps=9,
            )

        plan_agent.assert_called_once_with(
            session_id="session",
            llm_backend="openai",
            model="model",
            max_steps=7,
        )
        reflection_agent.assert_called_once_with(
            session_id="session",
            llm_backend="deepseek",
            model="model",
            max_steps=9,
        )


class WorkflowLoggingTest(unittest.TestCase):
    def test_phase_metadata_is_written_without_changing_agent_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = AgentCallbackLogger(
                agent="diagnosis_agent",
                session_dir=tmp,
                extra_fields={"phase": "critic"},
            )
            logger._log("test", {"value": 1})
            entry = json.loads(
                (Path(tmp) / "messages.jsonl").read_text(encoding="utf-8")
            )

        self.assertEqual(entry["agent"], "diagnosis_agent")
        self.assertEqual(entry["phase"], "critic")
        self.assertEqual(entry["event"], "test")


class PlanExecuteBehaviorTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.agent = PlanExecuteAgent.__new__(PlanExecuteAgent)
        self.agent.max_steps = 2
        self.agent.session_dir = self.tmp.name

    def tearDown(self) -> None:
        self.tmp.cleanup()

    async def test_executor_records_success_and_advances_one_item(self) -> None:
        first = PlanStep(
            step_id="reachability",
            action="Check reachability",
            expected_evidence="Failed pairs",
        )
        second = PlanStep(
            step_id="routes",
            action="Inspect routes",
            expected_evidence="Missing route",
        )
        self.agent.executor = AsyncMock()
        self.agent.executor.ainvoke.return_value = {
            "messages": [SimpleNamespace(content="pc1 cannot reach pc2")]
        }

        update = await self.agent._execute(
            {
                "task_description": "Diagnose",
                "objective": "Find the fault",
                "plan": [first, second],
                "completed_steps": [],
                "executed_steps": 0,
            }
        )

        self.assertEqual(update["plan"], [second])
        self.assertEqual(update["executed_steps"], 1)
        self.assertTrue(update["completed_steps"][0].succeeded)

    async def test_executor_failure_becomes_observation_for_replanner(self) -> None:
        step = PlanStep(
            step_id="routes",
            action="Inspect routes",
            expected_evidence="Route state",
        )
        self.agent.executor = AsyncMock()
        self.agent.executor.ainvoke.side_effect = RuntimeError("tool unavailable")

        update = await self.agent._execute(
            {
                "task_description": "Diagnose",
                "objective": "Find the fault",
                "plan": [step],
                "completed_steps": [],
                "executed_steps": 0,
            }
        )

        result = update["completed_steps"][0]
        self.assertFalse(result.succeeded)
        self.assertIn("tool unavailable", result.observation)

    async def test_replanner_can_finish_early(self) -> None:
        step = PlanStep(
            step_id="reachability",
            action="Check reachability",
            expected_evidence="Failed pairs",
        )
        self.agent.replanner = AsyncMock()
        self.agent.replanner.ainvoke.return_value = ReplanDecision(
            completed=True,
            diagnosis_report="Link failure on r1.",
        )

        update = await self.agent._replan(
            {
                "objective": "Find the fault",
                "task_description": "Diagnose",
                "plan": [],
                "completed_steps": [
                    StepResult(step=step, observation="Packet loss")
                ],
            }
        )

        self.assertEqual(update["diagnosis_report"], "Link failure on r1.")
        self.assertEqual(
            self.agent._route_after_replan(
                {**update, "executed_steps": 1}
            ),
            "submission",
        )

    async def test_planner_failure_stops_before_executor_and_submission(self) -> None:
        self.agent.planner = AsyncMock()
        self.agent.planner.ainvoke.side_effect = RuntimeError("invalid plan")

        update = await self.agent._plan({"task_description": "Diagnose"})

        self.assertTrue(update["planning_failed"])
        self.assertEqual(update["plan"], [])
        self.assertEqual(self.agent._route_after_plan(update), "end")

    async def test_replanner_can_replace_remaining_plan(self) -> None:
        old_step = PlanStep(
            step_id="old",
            action="Inspect routes",
            expected_evidence="Route state",
        )
        revised_step = PlanStep(
            step_id="new",
            action="Inspect interface counters",
            expected_evidence="Drops",
        )
        self.agent.replanner = AsyncMock()
        self.agent.replanner.ainvoke.return_value = {
            "completed": False,
            "remaining_steps": [revised_step.model_dump()],
        }

        update = await self.agent._replan(
            {
                "objective": "Find the fault",
                "task_description": "Diagnose",
                "plan": [old_step],
                "completed_steps": [],
            }
        )

        self.assertEqual(update["plan"], [revised_step])

    def test_plan_item_limit_routes_to_synthesis(self) -> None:
        self.assertEqual(
            self.agent._route_after_replan(
                {
                    "diagnosis_report": "",
                    "executed_steps": 2,
                    "plan": [
                        PlanStep(
                            step_id="extra",
                            action="More checks",
                            expected_evidence="More data",
                        )
                    ],
                }
            ),
            "synthesis",
        )

    async def test_empty_report_skips_submission(self) -> None:
        self.agent.submission_agent = AsyncMock()

        update = await self.agent._submit({"diagnosis_report": ""})

        self.assertEqual(update, {})
        self.agent.submission_agent.ainvoke.assert_not_awaited()


class ReflectionBehaviorTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.agent = ReflectionAgent.__new__(ReflectionAgent)
        self.agent.max_steps = 3
        self.agent.session_dir = self.tmp.name

    def tearDown(self) -> None:
        self.tmp.cleanup()

    async def test_critic_failure_preserves_initial_report(self) -> None:
        self.agent.critic = AsyncMock()
        self.agent.critic.ainvoke.side_effect = RuntimeError("invalid response")

        update = await self.agent._critique(
            {
                "task_description": "Diagnose",
                "initial_report": "Initial diagnosis",
            }
        )

        self.assertTrue(update["critic_failed"])
        self.assertEqual(update["diagnosis_report"], "Initial diagnosis")

    def test_critic_failure_routes_directly_to_submission(self) -> None:
        self.assertEqual(
            self.agent._route_after_critique({"critic_failed": True}),
            "submission",
        )

    async def test_reviser_receives_critique_and_returns_revised_report(self) -> None:
        self.agent.reviser = AsyncMock()
        self.agent.reviser.ainvoke.return_value = {
            "messages": [SimpleNamespace(content="Evidence-backed diagnosis")]
        }
        critique = DiagnosisCritique(
            evidence_sufficient=False,
            anomaly_assessment="Anomaly is plausible",
            localization_assessment="Needs one more check",
            root_cause_assessment="Weakly supported",
            missing_evidence=["routing table"],
            revision_instructions=["Inspect the routing table"],
        )

        update = await self.agent._revise(
            {
                "task_description": "Diagnose",
                "initial_report": "Initial diagnosis",
                "critique": critique,
            }
        )

        self.assertEqual(update["diagnosis_report"], "Evidence-backed diagnosis")
        call = self.agent.reviser.ainvoke.await_args
        self.assertIn("routing table", call.args[0]["messages"][0].content)

    async def test_reviser_failure_falls_back_to_initial_report(self) -> None:
        self.agent.reviser = AsyncMock()
        self.agent.reviser.ainvoke.side_effect = GraphRecursionError()
        critique = DiagnosisCritique(
            evidence_sufficient=False,
            anomaly_assessment="Unclear",
            localization_assessment="Unclear",
            root_cause_assessment="Unclear",
            revision_instructions=["Collect more evidence"],
        )

        update = await self.agent._revise(
            {
                "task_description": "Diagnose",
                "initial_report": "Initial diagnosis",
                "critique": critique,
            }
        )

        self.assertEqual(update["diagnosis_report"], "Initial diagnosis")

    async def test_empty_report_skips_submission(self) -> None:
        self.agent.submission_agent = AsyncMock()

        update = await self.agent._submit({"diagnosis_report": ""})

        self.assertEqual(update, {})
        self.agent.submission_agent.ainvoke.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
