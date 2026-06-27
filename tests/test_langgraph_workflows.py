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
from agent.langgraph.reflexion_agent import ReflexionAgent
from agent.langgraph.workflow_models import (
    InvestigationPlan,
    PlanStep,
    ReflexionEvaluation,
    ReflexionMemory,
    ReplanDecision,
    StepResult,
)
from agent.llm.model_factory import (
    DEFAULT_LLM_BACKEND,
    DEFAULT_MODEL,
    NETMIND_BASE_URL,
    NETMIND_MAX_RETRIES,
    NETMIND_SUPPORTED_MODELS,
    NETMIND_TIMEOUT_SECONDS,
    load_model,
)
from agent.registry import create_agent
from agent.utils.loggers import AgentCallbackLogger
from nika.codex_cli.commands.agent import (
    SUPPORTED_AGENT_TYPES,
    SUPPORTED_LLM_BACKENDS,
)


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

    def test_reflexion_success_requires_high_quality_score(self) -> None:
        with self.assertRaises(ValidationError):
            ReflexionEvaluation(
                success=True,
                quality_score=0.6,
                evidence_sufficient=True,
                anomaly_assessment="Anomaly confirmed",
                localization_assessment="Device identified",
                root_cause_assessment="Cause identified",
            )


class WorkflowRegistrationTest(unittest.TestCase):
    def test_cli_lists_all_agent_types(self) -> None:
        self.assertEqual(
            SUPPORTED_AGENT_TYPES,
            (
                "react",
                "plan-execute",
                "reflexion",
                "mock",
                "cli",
            ),
        )

    def test_registry_constructs_new_agent_types(self) -> None:
        with (
            patch("agent.registry.BasicReActAgent") as react_agent,
            patch("agent.registry.PlanExecuteAgent") as plan_agent,
            patch("agent.registry.ReflexionAgent") as reflexion_agent,
            patch("agent.registry.ProceduralMemoryModule") as memory_module,
            patch("agent.registry.MemoryAugmentedAgent") as memory_adapter,
        ):
            create_agent(
                "plan-execute",
                session_id="session",
                llm_backend="openai",
                model="model",
                max_steps=7,
            )
            create_agent(
                "reflexion",
                session_id="session",
                llm_backend="deepseek",
                model="model",
                max_steps=9,
                max_attempts=4,
            )
            create_agent(
                "react",
                session_id="session",
                llm_backend="openai",
                model="model",
                max_steps=11,
                tool_evolution_enabled=True,
                tool_library_id="experiment-a",
                tool_evolution_mode="mastery",
                memory_mode="read",
                memory_bank="experiment",
                memory_top_k=4,
                memory_token_budget=900,
            )

        plan_agent.assert_called_once_with(
            session_id="session",
            llm_backend="openai",
            model="model",
            max_steps=7,
            oracle_routing=False,
            tool_evolution_enabled=False,
            tool_library_id="default",
            tool_evolution_mode="dual",
        )
        reflexion_agent.assert_called_once_with(
            session_id="session",
            llm_backend="deepseek",
            model="model",
            max_steps=9,
            max_attempts=4,
            oracle_routing=False,
            tool_evolution_enabled=False,
            tool_library_id="default",
            tool_evolution_mode="dual",
        )
        react_agent.assert_called_once_with(
            session_id="session",
            llm_backend="openai",
            model="model",
            max_steps=11,
            oracle_routing=False,
            tool_evolution_enabled=True,
            tool_library_id="experiment-a",
            tool_evolution_mode="mastery",
            use_problem_tool_hints=False,
        )
        memory_module.assert_called_once_with(
            bank_id="experiment",
            llm_backend="openai",
            model="model",
        )
        memory_adapter.assert_called_once_with(
            react_agent.return_value,
            memory_module.return_value,
            memory_mode="read",
            memory_top_k=4,
            memory_token_budget=900,
        )

    def test_cli_lists_netmind_backend(self) -> None:
        self.assertIn("netmind", SUPPORTED_LLM_BACKENDS)
        self.assertEqual(
            NETMIND_SUPPORTED_MODELS,
            (
                "MiniMax/MiniMax-M2.7",
                "Qwen/Qwen3.5-122B-A10B-FP8",
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
                "zai-org/GLM-4.7",
            ),
        )

    def test_tool_evolution_rejects_non_langgraph_workflows(self) -> None:
        with self.assertRaisesRegex(ValueError, "supports react"):
            create_agent(
                "mock",
                session_id="session",
                llm_backend="openai",
                model="model",
                tool_evolution_enabled=True,
            )

    def test_tool_evolve_is_not_an_agent_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported agent type"):
            create_agent(
                "tool-evolve",
                session_id="session",
                llm_backend="openai",
                model="model",
            )

    def test_memory_composes_with_each_supported_workflow(self) -> None:
        cases = (
            ("react", "agent.registry.BasicReActAgent", {}),
            ("plan-execute", "agent.registry.PlanExecuteAgent", {}),
            (
                "reflexion",
                "agent.registry.ReflexionAgent",
                {"max_attempts": 4},
            ),
        )
        for agent_type, target, extra in cases:
            with self.subTest(agent_type=agent_type):
                with (
                    patch(target) as workflow,
                    patch("agent.registry.ProceduralMemoryModule") as memory_module,
                    patch("agent.registry.MemoryAugmentedAgent") as adapter,
                ):
                    result = create_agent(
                        agent_type,
                        session_id="session",
                        llm_backend="openai",
                        model="model",
                        max_steps=7,
                        memory_mode="evolve",
                        memory_bank="experiment",
                        **extra,
                    )

                self.assertIs(result, adapter.return_value)
                self.assertFalse(workflow.call_args.kwargs["use_problem_tool_hints"])
                adapter.assert_called_once_with(
                    workflow.return_value,
                    memory_module.return_value,
                    memory_mode="evolve",
                    memory_top_k=5,
                    memory_token_budget=1500,
                )

    def test_memory_rejects_unsupported_workflow(self) -> None:
        with self.assertRaisesRegex(ValueError, "supported only"):
            create_agent(
                "mock",
                session_id="session",
                llm_backend="mock",
                model="model",
                memory_mode="read",
            )


class ModelFactoryTest(unittest.TestCase):
    def test_default_model_is_netmind_gpt_oss_120b(self) -> None:
        self.assertEqual(DEFAULT_LLM_BACKEND, "netmind")
        self.assertEqual(DEFAULT_MODEL, "openai/gpt-oss-120b")

    @patch.dict("os.environ", {}, clear=True)
    def test_netmind_requires_api_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "NETMIND_API_KEY"):
            load_model("netmind", "NetMind/NetMind-X1")

    @patch.dict(
        "os.environ",
        {"NETMIND_API_KEY": "test-key"},
        clear=True,
    )
    @patch("agent.llm.model_factory.ChatOpenAI")
    def test_load_model_defaults_to_netmind_gpt_oss_120b(self, chat_openai) -> None:
        load_model()

        self.assertEqual(chat_openai.call_args.kwargs["model"], DEFAULT_MODEL)
        self.assertEqual(chat_openai.call_args.kwargs["api_key"], "test-key")
        self.assertEqual(chat_openai.call_args.kwargs["base_url"], NETMIND_BASE_URL)

    @patch.dict(
        "os.environ",
        {
            "NETMIND_API_KEY": "test-key",
            "NETMIND_BASE_URL": "https://netmind.example/v1",
        },
        clear=True,
    )
    @patch("agent.llm.model_factory.ChatOpenAI")
    def test_netmind_uses_openai_compatible_endpoint(self, chat_openai) -> None:
        load_model("netmind", "openai/gpt-oss-20b")

        chat_openai.assert_called_once_with(
            model="openai/gpt-oss-20b",
            api_key="test-key",
            base_url="https://netmind.example/v1",
            temperature=0,
            timeout=NETMIND_TIMEOUT_SECONDS,
            max_retries=NETMIND_MAX_RETRIES,
        )

    @patch.dict(
        "os.environ",
        {"NETMIND_API_KEY": "test-key"},
        clear=True,
    )
    @patch("agent.llm.model_factory.ChatOpenAI")
    def test_netmind_uses_default_base_url(self, chat_openai) -> None:
        load_model("netmind", "Qwen/Qwen3.5-122B-A10B-FP8")

        self.assertEqual(chat_openai.call_args.kwargs["base_url"], NETMIND_BASE_URL)
        self.assertEqual(
            chat_openai.call_args.kwargs["timeout"],
            NETMIND_TIMEOUT_SECONDS,
        )
        self.assertEqual(
            chat_openai.call_args.kwargs["max_retries"],
            NETMIND_MAX_RETRIES,
        )

    @patch.dict(
        "os.environ",
        {
            "NETMIND_API_KEY": "test-key",
            "NETMIND_TIMEOUT_SECONDS": "12.5",
            "NETMIND_MAX_RETRIES": "2",
        },
        clear=True,
    )
    @patch("agent.llm.model_factory.ChatOpenAI")
    def test_netmind_allows_timeout_and_retry_overrides(self, chat_openai) -> None:
        load_model("netmind", "MiniMax/MiniMax-M2.7")

        self.assertEqual(chat_openai.call_args.kwargs["timeout"], 12.5)
        self.assertEqual(chat_openai.call_args.kwargs["max_retries"], 2)

    @patch.dict(
        "os.environ",
        {
            "NETMIND_API_KEY": "test-key",
            "NETMIND_TIMEOUT_SECONDS": "never",
        },
        clear=True,
    )
    def test_netmind_rejects_invalid_timeout(self) -> None:
        with self.assertRaisesRegex(ValueError, "NETMIND_TIMEOUT_SECONDS"):
            load_model("netmind", "openai/gpt-oss-20b")

    @patch.dict(
        "os.environ",
        {"NETMIND_API_KEY": "test-key"},
        clear=True,
    )
    def test_netmind_rejects_model_outside_whitelist(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported NetMind model"):
            load_model("netmind", "Qwen/Qwen3-4B-Instruct-2507")


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
                "completed_steps": [StepResult(step=step, observation="Packet loss")],
            }
        )

        self.assertEqual(update["diagnosis_report"], "Link failure on r1.")
        self.assertEqual(
            self.agent._route_after_replan({**update, "executed_steps": 1}),
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


class ReflexionBehaviorTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.agent = ReflexionAgent.__new__(ReflexionAgent)
        self.agent.max_steps = 3
        self.agent.max_attempts = 3
        self.agent.session_dir = self.tmp.name

    def tearDown(self) -> None:
        self.tmp.cleanup()

    async def test_retry_attempt_receives_episodic_memory(self) -> None:
        self.agent.actor = AsyncMock()
        self.agent.actor.ainvoke.return_value = {
            "messages": [SimpleNamespace(content="Evidence-backed diagnosis")]
        }
        memory = ReflexionMemory(
            summary="The first attempt over-focused on routing.",
            lessons=["A paused host is direct evidence of host failure."],
            next_strategy=["Inspect the affected host before the control plane."],
            evidence_to_collect=["Host runtime state"],
            avoid_repeating=["Do not infer host health from BGP health."],
        )

        update = await self.agent._attempt(
            {
                "task_description": "Diagnose",
                "attempt_count": 1,
                "diagnosis_report": "Old report",
                "memories": [memory],
            }
        )

        self.assertEqual(update["attempt_count"], 2)
        self.assertEqual(update["attempt_report"], "Evidence-backed diagnosis")
        call = self.agent.actor.ainvoke.await_args
        prompt = call.args[0]["messages"][0].content
        self.assertIn("paused host", prompt)
        self.assertIn('"attempt": 2', prompt)

    async def test_attempt_failure_is_preserved_for_evaluation(self) -> None:
        self.agent.actor = AsyncMock()
        self.agent.actor.ainvoke.side_effect = GraphRecursionError()

        update = await self.agent._attempt(
            {
                "task_description": "Diagnose",
                "attempt_count": 0,
                "diagnosis_report": "",
                "memories": [],
            }
        )

        self.assertEqual(update["attempt_count"], 1)
        self.assertEqual(update["attempt_report"], "")
        self.assertTrue(update["attempt_error"])

    async def test_evaluator_returns_structured_success(self) -> None:
        self.agent.evaluator = AsyncMock()
        self.agent.evaluator.ainvoke.return_value = ReflexionEvaluation(
            success=True,
            quality_score=0.95,
            evidence_sufficient=True,
            anomaly_assessment="Anomaly confirmed",
            localization_assessment="pc_0_0 confirmed",
            root_cause_assessment="Host is paused",
        )

        update = await self.agent._evaluate(
            {
                "task_description": "Diagnose",
                "attempt_count": 1,
                "attempt_report": "pc_0_0 is paused",
                "attempt_error": "",
                "best_score": -1.0,
            }
        )

        self.assertTrue(update["evaluation"].success)
        self.assertFalse(update["evaluation_failed"])
        self.assertEqual(update["diagnosis_report"], "pc_0_0 is paused")

    async def test_lower_scoring_attempt_does_not_replace_best_report(self) -> None:
        self.agent.evaluator = AsyncMock()
        self.agent.evaluator.ainvoke.return_value = ReflexionEvaluation(
            success=False,
            quality_score=0.4,
            evidence_sufficient=False,
            anomaly_assessment="Anomaly plausible",
            localization_assessment="Localization uncertain",
            root_cause_assessment="Cause speculative",
            failure_reasons=["Insufficient direct evidence"],
        )

        update = await self.agent._evaluate(
            {
                "task_description": "Diagnose",
                "attempt_count": 2,
                "attempt_report": "A weaker second report",
                "attempt_error": "",
                "diagnosis_report": "The stronger first report",
                "best_score": 0.7,
            }
        )

        self.assertNotIn("diagnosis_report", update)
        self.assertNotIn("best_score", update)

    def test_successful_evaluation_routes_to_submission(self) -> None:
        evaluation = ReflexionEvaluation(
            success=True,
            quality_score=0.9,
            evidence_sufficient=True,
            anomaly_assessment="Anomaly confirmed",
            localization_assessment="pc_0_0 confirmed",
            root_cause_assessment="Host crash confirmed",
        )
        self.assertEqual(
            self.agent._route_after_evaluation(
                {
                    "evaluation": evaluation,
                    "attempt_count": 1,
                    "diagnosis_report": "Complete report",
                }
            ),
            "submission",
        )

    def test_failed_evaluation_routes_to_reflect(self) -> None:
        evaluation = ReflexionEvaluation(
            success=False,
            quality_score=0.4,
            evidence_sufficient=False,
            anomaly_assessment="Anomaly is plausible",
            localization_assessment="Needs one more check",
            root_cause_assessment="Weakly supported",
            missing_evidence=["routing table"],
            failure_reasons=["Root cause is speculative"],
        )

        self.assertEqual(
            self.agent._route_after_evaluation(
                {
                    "evaluation": evaluation,
                    "attempt_count": 1,
                    "diagnosis_report": "Incomplete report",
                }
            ),
            "reflect",
        )

    def test_last_failed_attempt_submits_best_available_report(self) -> None:
        evaluation = ReflexionEvaluation(
            success=False,
            quality_score=0.5,
            evidence_sufficient=False,
            anomaly_assessment="Unclear",
            localization_assessment="Unclear",
            root_cause_assessment="Unclear",
            failure_reasons=["Evidence remains incomplete"],
        )

        self.assertEqual(
            self.agent._route_after_evaluation(
                {
                    "evaluation": evaluation,
                    "attempt_count": 3,
                    "diagnosis_report": "Best available report",
                }
            ),
            "submission",
        )

    async def test_reflexion_memory_is_appended(self) -> None:
        self.agent.reflector = AsyncMock()
        memory = ReflexionMemory(
            summary="The attempt stopped at a healthy control-plane check.",
            lessons=["Healthy BGP does not prove host health."],
            next_strategy=["Inspect host runtime state first."],
        )
        self.agent.reflector.ainvoke.return_value = memory
        evaluation = ReflexionEvaluation(
            success=False,
            quality_score=0.3,
            evidence_sufficient=False,
            anomaly_assessment="Anomaly not resolved",
            localization_assessment="No exact device",
            root_cause_assessment="Unsupported",
            missing_evidence=["Host runtime state"],
        )

        update = await self.agent._reflect(
            {
                "task_description": "Diagnose",
                "attempt_count": 1,
                "attempt_report": "BGP is healthy",
                "attempt_error": "",
                "evaluation": evaluation,
                "memories": [],
            }
        )

        self.assertEqual(update["memories"], [memory])
        call = self.agent.reflector.ainvoke.await_args
        self.assertIn("Host runtime state", call.args[0][1].content)

    async def test_empty_report_skips_submission(self) -> None:
        self.agent.submission_agent = AsyncMock()

        update = await self.agent._submit({"diagnosis_report": ""})

        self.assertEqual(update, {})
        self.agent.submission_agent.ainvoke.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
