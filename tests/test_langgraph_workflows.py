"""Unit tests for the advanced LangGraph troubleshooting workflows."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.errors import GraphRecursionError
from pydantic import ValidationError

from agent.composition import (
    AgentRunConfig,
    MemoryConfig,
    ToolEvolutionConfig,
    workflow_agent_kwargs,
)
from agent.langgraph.plan_execute_agent import PlanExecuteAgent
from agent.langgraph.reflexion_agent import ReflexionAgent
from agent.langgraph.langfuse_tracing import (
    callback_config,
    create_langfuse_callbacks,
)
from agent.langgraph.workflow_models import (
    InvestigationPlan,
    PlanStep,
    ReflexionEvaluation,
    ReflexionMemory,
    ReplanDecision,
    StepResult,
)
from agent.llm.model_factory import (
    CUSTOM_DEFAULT_API_BASE,
    CUSTOM_MAX_RETRIES,
    CUSTOM_RECOMMENDED_MODELS,
    CUSTOM_TIMEOUT_SECONDS,
    DEFAULT_LLM_BACKEND,
    DEFAULT_MODEL,
    GLM47ChatOpenAI,
    _extract_glm_tool_calls,
    _normalize_glm_tool_calls,
    load_model,
)
from agent.registry import create_agent
from agent.utils.loggers import AgentCallbackLogger
from nika.codex_cli.commands.agent import (
    SUPPORTED_AGENT_TYPES,
    SUPPORTED_LLM_BACKENDS,
)


def _agent_run_config(
    agent_type: str,
    *,
    session_id: str = "session",
    llm_backend: str = "openai",
    model: str = "model",
    max_steps: int = 7,
    max_attempts: int = 3,
    tool_evolution: ToolEvolutionConfig | None = None,
    memory: MemoryConfig | None = None,
) -> AgentRunConfig:
    return AgentRunConfig(
        agent_type=agent_type,
        session_id=session_id,
        llm_backend=llm_backend,
        model=model,
        max_steps=max_steps,
        max_attempts=max_attempts,
        tool_evolution=tool_evolution or ToolEvolutionConfig(),
        memory=memory or MemoryConfig(),
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
    def test_langfuse_auth_errors_disable_tracing_without_failing(self) -> None:
        with (
            patch("agent.langgraph.langfuse_tracing.CallbackHandler") as handler,
            patch("agent.langgraph.langfuse_tracing.get_client") as get_client,
        ):
            get_client.return_value.auth_check.side_effect = TimeoutError("slow")

            self.assertEqual(create_langfuse_callbacks(), [])
            handler.assert_called_once_with()

    def test_callback_config_omits_empty_callbacks(self) -> None:
        self.assertEqual(callback_config([]), {})
        callback = object()
        self.assertEqual(callback_config([callback]), {"callbacks": [callback]})

    def test_cli_lists_all_agent_types(self) -> None:
        self.assertEqual(
            SUPPORTED_AGENT_TYPES,
            (
                "react",
                "plan-execute",
                "reflexion",
                "mock",
                "cli",
                "codex_cli",
                "claude_cli",
            ),
        )

    def test_registry_constructs_new_agent_types(self) -> None:
        with (
            patch("agent.registry.BasicReActAgent") as react_agent,
            patch("agent.registry.PlanExecuteAgent") as plan_agent,
            patch("agent.registry.ReflexionAgent") as reflexion_agent,
            patch("agent.composition.ProceduralMemoryModule") as memory_module,
            patch("agent.composition.MemoryAugmentedAgent") as memory_adapter,
        ):
            create_agent(
                _agent_run_config("plan-execute")
            )
            create_agent(
                _agent_run_config(
                    "reflexion",
                    llm_backend="deepseek",
                    max_steps=9,
                    max_attempts=4,
                )
            )
            create_agent(
                _agent_run_config(
                    "react",
                    max_steps=11,
                    tool_evolution=ToolEvolutionConfig(
                        enabled=True,
                        library_id="experiment-a",
                    ),
                    memory=MemoryConfig(
                        mode="read",
                        bank="experiment",
                        top_k=4,
                        token_budget=900,
                    ),
                )
            )

        plan_agent.assert_called_once_with(
            session_id="session",
            llm_backend="openai",
            model="model",
            max_steps=7,
            tool_evolution_enabled=False,
            tool_library_id="default",
        )
        reflexion_agent.assert_called_once_with(
            session_id="session",
            llm_backend="deepseek",
            model="model",
            max_steps=9,
            max_attempts=4,
            tool_evolution_enabled=False,
            tool_library_id="default",
        )
        react_agent.assert_called_once_with(
            session_id="session",
            llm_backend="openai",
            model="model",
            max_steps=11,
            tool_evolution_enabled=True,
            tool_library_id="experiment-a",
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

    def test_cli_lists_custom_backend(self) -> None:
        self.assertIn("custom", SUPPORTED_LLM_BACKENDS)
        self.assertNotIn("netmind", SUPPORTED_LLM_BACKENDS)
        self.assertEqual(
            CUSTOM_RECOMMENDED_MODELS,
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
                _agent_run_config(
                    "mock",
                    tool_evolution=ToolEvolutionConfig(enabled=True),
                )
            )

    def test_tool_evolve_is_not_an_agent_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported agent type"):
            create_agent(
                _agent_run_config("tool-evolve")
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
                    patch("agent.composition.ProceduralMemoryModule") as memory_module,
                    patch("agent.composition.MemoryAugmentedAgent") as adapter,
                ):
                    result = create_agent(
                        _agent_run_config(
                            agent_type,
                            memory=MemoryConfig(mode="evolve", bank="experiment"),
                            **extra,
                        )
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
                _agent_run_config(
                    "mock",
                    llm_backend="mock",
                    memory=MemoryConfig(mode="read"),
                )
            )

    def test_composition_config_builds_extension_kwargs(self) -> None:
        config = AgentRunConfig(
            agent_type="react",
            session_id="session",
            llm_backend="openai",
            model="model",
            max_steps=9,
            tool_evolution=ToolEvolutionConfig(
                enabled=True,
                library_id="tools-a",
            ),
            memory=MemoryConfig(mode="read", bank="memory-a"),
        )

        kwargs = workflow_agent_kwargs(config)

        self.assertEqual(kwargs["tool_library_id"], "tools-a")
        self.assertNotIn("policy_overlay_path", kwargs)
        self.assertFalse(kwargs["use_problem_tool_hints"])


class ModelFactoryTest(unittest.TestCase):
    def test_default_model_is_custom_gpt_oss_120b(self) -> None:
        self.assertEqual(DEFAULT_LLM_BACKEND, "custom")
        self.assertEqual(DEFAULT_MODEL, "openai/gpt-oss-120b")

    @patch.dict("os.environ", {}, clear=True)
    @patch("agent.llm.model_factory.ChatOpenAI")
    def test_load_model_defaults_to_custom_gpt_oss_120b(self, chat_openai) -> None:
        load_model()

        self.assertEqual(chat_openai.call_args.kwargs["model"], DEFAULT_MODEL)
        self.assertEqual(chat_openai.call_args.kwargs["api_key"], "dummy")
        self.assertEqual(chat_openai.call_args.kwargs["base_url"], CUSTOM_DEFAULT_API_BASE)

    @patch.dict("os.environ", {"CUSTOM_API_KEY": "test-key"}, clear=True)
    @patch("agent.llm.model_factory.ChatOpenAI")
    def test_custom_uses_default_base_url(self, chat_openai) -> None:
        load_model()

        self.assertEqual(chat_openai.call_args.kwargs["api_key"], "test-key")
        self.assertEqual(chat_openai.call_args.kwargs["base_url"], CUSTOM_DEFAULT_API_BASE)
        self.assertEqual(
            chat_openai.call_args.kwargs["timeout"],
            CUSTOM_TIMEOUT_SECONDS,
        )
        self.assertEqual(
            chat_openai.call_args.kwargs["max_retries"],
            CUSTOM_MAX_RETRIES,
        )

    @patch.dict(
        "os.environ",
        {
            "CUSTOM_API_KEY": "test-key",
            "CUSTOM_API_BASE": "https://custom.example/v1",
        },
        clear=True,
    )
    @patch("agent.llm.model_factory.ChatOpenAI")
    def test_custom_uses_openai_compatible_endpoint(self, chat_openai) -> None:
        load_model("custom", "openai/gpt-oss-20b")

        chat_openai.assert_called_once_with(
            model="openai/gpt-oss-20b",
            api_key="test-key",
            base_url="https://custom.example/v1",
            temperature=0,
            timeout=CUSTOM_TIMEOUT_SECONDS,
            max_retries=CUSTOM_MAX_RETRIES,
        )

    @patch.dict(
        "os.environ",
        {"CUSTOM_API_KEY": "test-key"},
        clear=True,
    )
    def test_custom_glm_47_uses_tool_call_adapter(self) -> None:
        model = load_model("custom", "zai-org/GLM-4.7")

        self.assertIsInstance(model, GLM47ChatOpenAI)

    def test_netmind_backend_is_no_longer_supported(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported llm backend: netmind"):
            load_model("netmind", "openai/gpt-oss-20b")

    def test_glm_tool_call_xml_is_normalized(self) -> None:
        extracted = _extract_glm_tool_calls(
            (
                '<tool_call>{"name":"ping_pair","arguments":'
                '{"host_a":"pc1","host_b":"pc2"}}</tool_call>'
            )
        )

        self.assertIsNotNone(extracted)
        calls, cleaned = extracted
        self.assertEqual(cleaned, "")
        self.assertEqual(calls[0]["name"], "ping_pair")
        self.assertEqual(calls[0]["args"], {"host_a": "pc1", "host_b": "pc2"})
        self.assertTrue(calls[0]["id"].startswith("call_glm_"))

    def test_glm_tool_call_xml_name_without_json_is_normalized(self) -> None:
        extracted = _extract_glm_tool_calls(
            "Check reachability.<tool_call>get_reachability</tool_call>"
        )

        self.assertIsNotNone(extracted)
        calls, cleaned = extracted
        self.assertEqual(cleaned, "Check reachability.")
        self.assertEqual(calls[0]["name"], "get_reachability")
        self.assertEqual(calls[0]["args"], {})

    def test_glm_tool_call_xml_arg_key_value_is_normalized(self) -> None:
        extracted = _extract_glm_tool_calls(
            (
                "<tool_call>frr_show_bgp_summary"
                "<arg_key>router_name</arg_key>"
                "<arg_value>router_core_1</arg_value>"
                "</tool_call>"
            )
        )

        self.assertIsNotNone(extracted)
        calls, cleaned = extracted
        self.assertEqual(cleaned, "")
        self.assertEqual(calls[0]["name"], "frr_show_bgp_summary")
        self.assertEqual(calls[0]["args"], {"router_name": "router_core_1"})

    def test_glm_tool_call_accepts_openai_function_payload(self) -> None:
        extracted = _extract_glm_tool_calls(
            (
                '<tool_call>{"id":"call_1","function":'
                '{"name":"get_host_net_config","arguments":'
                '"{\\"host_name\\":\\"pc1\\"}"}}</tool_call>'
            )
        )

        self.assertIsNotNone(extracted)
        calls, _ = extracted
        self.assertEqual(
            calls[0],
            {
                "name": "get_host_net_config",
                "args": {"host_name": "pc1"},
                "id": "call_1",
                "type": "tool_call",
            },
        )

    def test_glm_chat_result_normalization_sets_langchain_tool_calls(self) -> None:
        result = ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=(
                            "Need evidence.\n"
                            '<tool_call>{"name":"get_reachability",'
                            '"arguments":{}}</tool_call>'
                        )
                    )
                )
            ]
        )

        normalized = _normalize_glm_tool_calls(result)

        message = normalized.generations[0].message
        self.assertEqual(message.content, "Need evidence.")
        self.assertEqual(message.tool_calls[0]["name"], "get_reachability")
        self.assertEqual(message.tool_calls[0]["args"], {})

    @patch.dict(
        "os.environ",
        {
            "CUSTOM_API_KEY": "test-key",
            "CUSTOM_TIMEOUT_SECONDS": "12.5",
            "CUSTOM_MAX_RETRIES": "2",
        },
        clear=True,
    )
    @patch("agent.llm.model_factory.ChatOpenAI")
    def test_custom_allows_timeout_and_retry_overrides(self, chat_openai) -> None:
        load_model("custom", "MiniMax/MiniMax-M2.7")

        self.assertEqual(chat_openai.call_args.kwargs["timeout"], 12.5)
        self.assertEqual(chat_openai.call_args.kwargs["max_retries"], 2)

    @patch.dict(
        "os.environ",
        {
            "CUSTOM_API_KEY": "test-key",
            "CUSTOM_TIMEOUT_SECONDS": "never",
        },
        clear=True,
    )
    def test_custom_rejects_invalid_timeout(self) -> None:
        with self.assertRaisesRegex(ValueError, "CUSTOM_TIMEOUT_SECONDS"):
            load_model("custom", "openai/gpt-oss-20b")

    @patch.dict(
        "os.environ",
        {"CUSTOM_API_KEY": "test-key"},
        clear=True,
    )
    @patch("agent.llm.model_factory.ChatOpenAI")
    def test_custom_allows_arbitrary_model_ids(self, chat_openai) -> None:
        load_model("custom", "Qwen/Qwen3-4B-Instruct-2507")

        self.assertEqual(
            chat_openai.call_args.kwargs["model"],
            "Qwen/Qwen3-4B-Instruct-2507",
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
