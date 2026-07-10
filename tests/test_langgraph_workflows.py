"""Unit tests for the advanced LangGraph troubleshooting workflows."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.errors import GraphRecursionError
from pydantic import ValidationError

from agent.composition import (
    AgentRunConfig,
    MemoryConfig,
    ToolEvolutionConfig,
    workflow_agent_kwargs,
)
from agent.langgraph.plan_execute_agent import (
    EXECUTOR_PROMPT,
    PLANNER_PROMPT,
    REPLANNER_PROMPT,
    SYNTHESIS_PROMPT,
    PlanExecuteAgent,
)
from agent.langgraph.evidence_gate import (
    ToolObservation,
    evidence_gate_enabled,
    evaluate_fault_family_evidence,
)
from agent.langgraph.phases.diagnosis import DiagnosisPhase
from agent.langgraph.react_agent import BasicReActAgent
from agent.langgraph.reflexion_agent import (
    ACTOR_PROMPT,
    EVALUATOR_PROMPT,
    REFLEXION_PROMPT,
    ReflexionAgent,
)
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
from agent.memory.runtime import strip_integrated_learning_guidance
from agent.llm.model_factory import (
    DEFAULT_LLM_BACKEND,
    DEFAULT_MODEL,
    GLM47ChatOpenAI,
    NETMIND_API_URL,
    _extract_glm_tool_calls,
    _normalize_glm_tool_calls,
    load_model,
)
from agent.registry import create_agent
from agent.utils.loggers import AgentCallbackLogger
from agent.utils.template import (
    DISCRIMINATING_EVIDENCE_PROMPT,
    EVIDENCE_CONTRACT_PROMPT,
    OVERALL_DIAGNOSIS_PROMPT,
)
from agent.utils.tracing import session_problem_label
from nika.cli.commands.agent import (
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
    def test_session_problem_label_uses_problem_name_when_present(self) -> None:
        session = SimpleNamespace(
            problem_names=["link_down"],
            root_cause_name="no_fault",
        )

        self.assertEqual(session_problem_label(session), "link_down")

    def test_session_problem_label_handles_no_fault_controls(self) -> None:
        session = SimpleNamespace(problem_names=[], root_cause_name="no_fault")

        self.assertEqual(session_problem_label(session), "no_fault")

    def test_evidence_contract_is_shared_by_diagnosis_prompts(self) -> None:
        contract_anchor = "guidance only; they are not evidence"
        prompts = [
            OVERALL_DIAGNOSIS_PROMPT,
            PLANNER_PROMPT,
            EXECUTOR_PROMPT,
            REPLANNER_PROMPT,
            SYNTHESIS_PROMPT,
            ACTOR_PROMPT,
            EVALUATOR_PROMPT,
            REFLEXION_PROMPT,
        ]

        self.assertIn(contract_anchor, EVIDENCE_CONTRACT_PROMPT)
        for prompt in prompts:
            with self.subTest(prompt=prompt[:40]):
                self.assertIn(contract_anchor, prompt)

    def test_strip_integrated_learning_guidance_keeps_only_observation(self) -> None:
        text = (
            "eth0 is down\n\n"
            "[Integrated learning guidance - not evidence]\n"
            "Active Skill-MDP option: seed_react_decision"
        )

        self.assertEqual(strip_integrated_learning_guidance(text), "eth0 is down")

    def test_diagnosis_prompt_has_evidence_based_stop_condition(self) -> None:
        self.assertIn("Stop calling tools", OVERALL_DIAGNOSIS_PROMPT)
        self.assertIn("Final report format", OVERALL_DIAGNOSIS_PROMPT)
        self.assertIn(
            "Discriminating evidence policy",
            DISCRIMINATING_EVIDENCE_PROMPT,
        )
        self.assertIn("Discriminating evidence policy", EVIDENCE_CONTRACT_PROMPT)
        self.assertNotIn("DNS/resolver", OVERALL_DIAGNOSIS_PROMPT)
        self.assertNotIn("BGP", OVERALL_DIAGNOSIS_PROMPT)

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


class EvidenceGateTest(unittest.TestCase):
    def test_dns_gate_blocks_reachability_only_report(self) -> None:
        result = evaluate_fault_family_evidence(
            task_description="Users cannot resolve local DNS names.",
            diagnosis_report=(
                "Anomaly exists. The root cause is DNS service failure on dns_server."
            ),
            observations=[
                ToolObservation(
                    tool="ping_pair",
                    summary="pc_1_1 can ping dns_server successfully.",
                )
            ],
            available_tools=[
                "ping_pair",
                "curl_web_test",
                "cat_file",
                "systemctl_ops",
                "netstat",
            ],
        )

        self.assertFalse(result.sufficient)
        self.assertIn("DNS/resolver", result.families)
        self.assertTrue(
            any("DNS/resolver" in item for item in result.missing_evidence)
        )
        self.assertIn("Evidence gate blocked finalization", result.prompt)

    def test_dns_gate_accepts_resolution_or_resolver_evidence(self) -> None:
        result = evaluate_fault_family_evidence(
            task_description="Users cannot resolve local DNS names.",
            diagnosis_report=(
                "Anomaly exists. DNS lookup for web0.local fails from pc_1_1."
            ),
            observations=[
                ToolObservation(
                    tool="exec_shell",
                    summary="nslookup web0.local returns SERVFAIL from dns_server.",
                )
            ],
            available_tools=["exec_shell", "curl_web_test", "cat_file"],
        )

        self.assertTrue(result.sufficient)
        self.assertEqual(result.missing_evidence, ())

    def test_bgp_gate_blocks_ping_only_route_report(self) -> None:
        result = evaluate_fault_family_evidence(
            task_description="BGP missing route advertisement between leaves.",
            diagnosis_report=(
                "Anomaly exists. Root cause is missing BGP advertisement on leaf_router_0_0."
            ),
            observations=[
                ToolObservation(
                    tool="ping_pair",
                    summary="Traffic fails between hosts in different racks.",
                )
            ],
            available_tools=[
                "ping_pair",
                "frr_show_bgp_summary",
                "frr_get_bgp_conf",
                "frr_show_ip_route",
            ],
        )

        self.assertFalse(result.sufficient)
        self.assertIn("BGP control plane", result.families)
        self.assertTrue(
            any("frr_show_bgp_summary" in item for item in result.suggested_steps)
        )

    def test_gate_ignores_topology_catalog_when_report_claims_one_family(self) -> None:
        result = evaluate_fault_family_evidence(
            task_description=(
                "Network Description: enterprise topology with DNS, DHCP, OSPF, "
                "HTTP services, ACLs, and BGP edge connectivity."
            ),
            diagnosis_report=(
                "Anomaly exists. The root cause is DNS resolver service failure."
            ),
            observations=[
                ToolObservation(
                    tool="ping_pair",
                    summary="Affected host can ping the resolver address.",
                )
            ],
            available_tools=[
                "curl_web_test",
                "cat_file",
                "systemctl_ops",
                "netstat",
                "frr_get_ospf_conf",
                "frr_show_bgp_summary",
                "get_host_net_config",
            ],
        )

        self.assertFalse(result.sufficient)
        self.assertEqual(result.families, ("DNS/resolver",))
        self.assertTrue(
            all("BGP" not in item and "OSPF" not in item for item in result.suggested_steps)
        )

    def test_gate_does_not_turn_topology_only_task_into_protocol_sprawl(self) -> None:
        result = evaluate_fault_family_evidence(
            task_description=(
                "Network Description: enterprise topology with DNS, DHCP, OSPF, "
                "HTTP services, ACLs, and BGP edge connectivity."
            ),
            diagnosis_report="No anomaly is confirmed from the current evidence.",
            observations=[
                ToolObservation(
                    tool="get_reachability",
                    summary="All tested host pairs are reachable.",
                )
            ],
            available_tools=[
                "curl_web_test",
                "get_host_net_config",
                "frr_get_ospf_conf",
                "frr_show_bgp_summary",
            ],
        )

        self.assertTrue(result.sufficient)
        self.assertEqual(result.families, ())
        self.assertEqual(result.suggested_steps, ())


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

    def test_memory_runtime_config_is_forwarded_by_agent_facades(self) -> None:
        cases = (
            (BasicReActAgent, "_refresh_diagnosis_agent"),
            (PlanExecuteAgent, "_refresh_executor"),
            (ReflexionAgent, "_refresh_actor"),
        )
        for agent_cls, refresh_name in cases:
            with self.subTest(agent_cls=agent_cls.__name__):
                agent = agent_cls.__new__(agent_cls)
                agent.session_dir = "/tmp/session"
                agent._diagnosis_phase = Mock()
                setattr(agent, refresh_name, Mock())

                agent.install_memory_runtime(
                    memory=object(),
                    memory_mode="evolve",
                    task_description="task",
                    top_k=7,
                    token_budget=2100,
                    skill_selector_mode="llm_topk_lcb",
                    meta_controller_mode="llm",
                    max_skill_age=6,
                    selector_min_lcb=-0.02,
                    selector_nominee_k=4,
                )

                kwargs = agent._diagnosis_phase.install_memory_runtime.call_args.kwargs
                self.assertEqual(kwargs["session_dir"], "/tmp/session")
                self.assertEqual(kwargs["max_skill_age"], 6)
                self.assertEqual(kwargs["selector_min_lcb"], -0.02)
                self.assertEqual(kwargs["selector_nominee_k"], 4)
                getattr(agent, refresh_name).assert_called_once_with()

    def test_cli_lists_all_agent_types(self) -> None:
        self.assertEqual(
            SUPPORTED_AGENT_TYPES,
            (
                "react",
                "plan-execute",
                "reflexion",
                "mock",
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
                        skill_selector_mode="llm_topk_lcb",
                        meta_controller_mode="llm",
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
            tool_doc_chars=500,
            tool_prompt_doc_limit=6,
            tool_scoped_prompt_doc_limit=4,
            tool_planned_checks=4,
            tool_next_checks=2,
        )
        reflexion_agent.assert_called_once_with(
            session_id="session",
            llm_backend="deepseek",
            model="model",
            max_steps=9,
            max_attempts=4,
            tool_evolution_enabled=False,
            tool_library_id="default",
            tool_doc_chars=500,
            tool_prompt_doc_limit=6,
            tool_scoped_prompt_doc_limit=4,
            tool_planned_checks=4,
            tool_next_checks=2,
        )
        react_agent.assert_called_once_with(
            session_id="session",
            llm_backend="openai",
            model="model",
            max_steps=11,
            tool_evolution_enabled=True,
            tool_library_id="experiment-a",
            tool_doc_chars=500,
            tool_prompt_doc_limit=6,
            tool_scoped_prompt_doc_limit=4,
            tool_planned_checks=4,
            tool_next_checks=2,
        )
        memory_module.assert_called_once_with(
            bank_id="experiment",
            llm_backend="openai",
            model="model",
            pool_size=32,
            evolution_threshold=3,
            best_of_n=3,
            ppo_epsilon=0.2,
            include_expert_seeds=False,
        )
        memory_adapter.assert_called_once_with(
            react_agent.return_value,
            memory_module.return_value,
            memory_mode="read",
            memory_top_k=4,
            memory_token_budget=900,
            memory_skill_selector_mode="llm_topk_lcb",
            memory_meta_controller_mode="llm",
            memory_max_skill_age=4,
            memory_selector_min_lcb=-0.05,
            memory_selector_nominee_k=3,
        )

    def test_cli_lists_custom_backend(self) -> None:
        self.assertIn("custom", SUPPORTED_LLM_BACKENDS)
        self.assertNotIn("netmind", SUPPORTED_LLM_BACKENDS)

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
                self.assertNotIn("use_problem_tool_hints", workflow.call_args.kwargs)
                adapter.assert_called_once_with(
                    workflow.return_value,
                    memory_module.return_value,
                    memory_mode="evolve",
                    memory_top_k=5,
                    memory_token_budget=1500,
                    memory_skill_selector_mode="lcb",
                    memory_meta_controller_mode="heuristic",
                    memory_max_skill_age=4,
                    memory_selector_min_lcb=-0.05,
                    memory_selector_nominee_k=3,
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
                tool_doc_chars=640,
                prompt_doc_limit=5,
                scoped_prompt_doc_limit=3,
                planned_checks=2,
                next_checks=1,
                convergence_threshold=0.8,
            ),
            memory=MemoryConfig(
                mode="read",
                bank="memory-a",
                top_k=7,
                token_budget=2100,
                max_skill_age=6,
                selector_min_lcb=-0.02,
                selector_nominee_k=4,
                pool_size=24,
                evolution_threshold=2,
                best_of_n=5,
                ppo_epsilon=0.15,
            ),
        )

        kwargs = workflow_agent_kwargs(config)

        self.assertEqual(kwargs["tool_library_id"], "tools-a")
        self.assertEqual(kwargs["tool_doc_chars"], 640)
        self.assertEqual(kwargs["tool_prompt_doc_limit"], 5)
        self.assertEqual(kwargs["tool_scoped_prompt_doc_limit"], 3)
        self.assertEqual(kwargs["tool_planned_checks"], 2)
        self.assertEqual(kwargs["tool_next_checks"], 1)
        self.assertNotIn("use_problem_tool_hints", kwargs)


class ModelFactoryTest(unittest.TestCase):
    def test_default_model_is_custom_gpt_oss_20b(self) -> None:
        self.assertEqual(DEFAULT_LLM_BACKEND, "custom")
        self.assertEqual(DEFAULT_MODEL, "openai/gpt-oss-20b")

    @patch.dict("os.environ", {"CUSTOM_API_KEY": "password"}, clear=True)
    @patch("agent.llm.model_factory.NetmindChatOpenAI")
    def test_load_model_defaults_to_custom_gpt_oss_20b(self, chat_openai) -> None:
        load_model()

        self.assertEqual(chat_openai.call_args.kwargs["model"], DEFAULT_MODEL)
        self.assertEqual(chat_openai.call_args.kwargs["api_key"], "password")
        self.assertEqual(chat_openai.call_args.kwargs["base_url"], NETMIND_API_URL)

    @patch.dict("os.environ", {"CUSTOM_API_KEY": "password"}, clear=True)
    @patch("agent.llm.model_factory.NetmindChatOpenAI")
    def test_custom_netmind_default_uses_custom_api_key(self, chat_openai) -> None:
        load_model()

        self.assertEqual(chat_openai.call_args.kwargs["api_key"], "password")
        self.assertEqual(chat_openai.call_args.kwargs["base_url"], NETMIND_API_URL)
        self.assertEqual(chat_openai.call_args.kwargs["timeout"], 90.0)
        self.assertEqual(chat_openai.call_args.kwargs["max_retries"], 0)

    @patch.dict("os.environ", {}, clear=True)
    def test_netmind_requires_custom_api_key_password(self) -> None:
        with self.assertRaisesRegex(ValueError, "CUSTOM_API_KEY is required"):
            load_model("custom", "openai/gpt-oss-20b")

    @patch.dict(
        "os.environ",
        {
            "CUSTOM_API_URL": "https://stream-netmind.viettel.vn/gateway/v1",
            "CUSTOM_API_KEY": "password",
        },
        clear=True,
    )
    @patch("agent.llm.model_factory.NetmindChatOpenAI")
    def test_netmind_url_uses_netmind_adapter_without_model_filter(
        self,
        chat_openai,
    ) -> None:
        load_model("custom", "any/provider-model")

        self.assertEqual(chat_openai.call_args.kwargs["model"], "any/provider-model")
        self.assertEqual(chat_openai.call_args.kwargs["api_key"], "password")
        self.assertEqual(chat_openai.call_args.kwargs["base_url"], NETMIND_API_URL)

    @patch.dict(
        "os.environ",
        {
            "CUSTOM_API_KEY": "test-key",
            "CUSTOM_API_URL": "https://custom.example/v1",
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
            timeout=90.0,
            max_retries=0,
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
            "CUSTOM_API_URL": "https://custom.example/v1",
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
        {
            "CUSTOM_API_KEY": "test-key",
            "CUSTOM_API_URL": "https://custom.example/v1",
        },
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


class DiagnosisPhasePromptTest(unittest.TestCase):
    def test_get_agent_keeps_learning_context_out_of_static_system_prompt(self) -> None:
        phase = DiagnosisPhase.__new__(DiagnosisPhase)
        phase.llm = object()
        phase.tools = []
        phase.prompt_suffix = lambda **_: "\nSkill-Pro stale option"

        with patch("agent.langgraph.phases.diagnosis.create_agent") as create:
            phase.get_agent()
            static_prompt = create.call_args.kwargs["system_prompt"]
            create.reset_mock()

            phase.get_agent(include_learning_context=True)
            explicit_prompt = create.call_args.kwargs["system_prompt"]

        self.assertNotIn("Skill-Pro stale option", static_prompt)
        self.assertIn("Skill-Pro stale option", explicit_prompt)

    def test_prompt_suffix_lets_skill_runtime_own_draft_context(self) -> None:
        class FakeDraftRuntime:
            def __init__(self) -> None:
                self.calls = 0

            def prompt_suffix(self) -> str:
                self.calls += 1
                return "\n\nDRAFT tool documentation memory:\nstandalone"

        class FakeSkillRuntime:
            def __init__(self) -> None:
                self.calls: list[bool] = []

            def prompt_suffix(self, *, activate_skill: bool = True) -> str:
                self.calls.append(activate_skill)
                return "\n\nIntegrated Skill-Pro + DRAFT diagnosis loop:\nlinked"

        phase = DiagnosisPhase.__new__(DiagnosisPhase)
        phase.tool_evolution_runtime = FakeDraftRuntime()
        phase.skill_tool_runtime = FakeSkillRuntime()

        suffix = phase.prompt_suffix(activate_skill=False)

        self.assertIn("Integrated Skill-Pro + DRAFT diagnosis loop", suffix)
        self.assertIn("linked", suffix)
        self.assertNotIn("standalone", suffix)
        self.assertEqual(phase.tool_evolution_runtime.calls, 0)
        self.assertEqual(phase.skill_tool_runtime.calls, [False])


class ReactBehaviorTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.agent = BasicReActAgent.__new__(BasicReActAgent)
        self.agent.max_steps = 3
        self.agent.session_dir = self.tmp.name
        self.agent._diagnosis_phase = SimpleNamespace(
            prompt_suffix=lambda **_: ""
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    async def test_diagnosis_call_injects_action_time_learning_context(self) -> None:
        calls: list[bool] = []

        def prompt_suffix(*, activate_skill=True) -> str:
            calls.append(activate_skill)
            return "\nSkill-Pro active option: react_route"

        self.agent._diagnosis_phase = SimpleNamespace(prompt_suffix=prompt_suffix)
        self.agent.diagnosis_agent = AsyncMock()
        self.agent.diagnosis_agent.ainvoke.return_value = {
            "messages": [SimpleNamespace(content="Evidence-backed report")]
        }

        update = await self.agent.diagnosis_agent_builder(
            {"messages": [HumanMessage(content="Diagnose")]}
        )
        payload = self.agent.diagnosis_agent.ainvoke.call_args.args[0]
        messages = payload["messages"]

        self.assertEqual(calls, [True])
        self.assertEqual(messages[0].content, "Diagnose")
        self.assertIn("Current integrated learning context", messages[1].content)
        self.assertIn("react_route", messages[1].content)
        self.assertEqual(update["diagnosis_report"], "Evidence-backed report")

    async def test_diagnosis_retries_when_evidence_gate_blocks_report(self) -> None:
        self.agent.evidence_gate_retries = 1
        self.agent.diagnosis_tool_names = ["ping_pair", "exec_shell"]
        self.agent.skill_tool_runtime = None
        self.agent.diagnosis_agent = AsyncMock()
        self.agent.diagnosis_agent.ainvoke.side_effect = [
            {
                "messages": [
                    AIMessage(
                        content=(
                            "Anomaly exists. Root cause is DNS service failure on dns_server."
                        )
                    )
                ]
            },
            {
                "messages": [
                    ToolMessage(
                        content="nslookup web0.local returns SERVFAIL from dns_server.",
                        tool_call_id="call_1",
                        name="exec_shell",
                    ),
                    AIMessage(
                        content=(
                            "Anomaly exists. DNS resolution fails based on nslookup "
                            "SERVFAIL from dns_server."
                        )
                    ),
                ]
            },
        ]

        update = await self.agent.diagnosis_agent_builder(
            {
                "task_description": "Users cannot resolve local DNS names.",
                "messages": [
                    HumanMessage(content="Users cannot resolve local DNS names.")
                ],
            }
        )

        self.assertEqual(self.agent.diagnosis_agent.ainvoke.await_count, 2)
        retry_payload = self.agent.diagnosis_agent.ainvoke.call_args_list[1].args[0]
        self.assertIn(
            "Evidence gate blocked finalization",
            retry_payload["messages"][-1].content,
        )
        self.assertIn("DNS resolution fails", update["diagnosis_report"])

    async def test_diagnosis_does_not_retry_when_evidence_gate_disabled(self) -> None:
        self.agent.evidence_gate_enabled = False
        self.agent.evidence_gate_retries = 1
        self.agent.diagnosis_tool_names = ["ping_pair", "exec_shell"]
        self.agent.skill_tool_runtime = None
        self.agent.diagnosis_agent = AsyncMock()
        self.agent.diagnosis_agent.ainvoke.return_value = {
            "messages": [
                AIMessage(
                    content="Anomaly exists. Root cause is DNS service failure on dns_server."
                )
            ]
        }

        update = await self.agent.diagnosis_agent_builder(
            {
                "task_description": "Users cannot resolve local DNS names.",
                "messages": [
                    HumanMessage(content="Users cannot resolve local DNS names.")
                ],
            }
        )

        self.assertEqual(self.agent.diagnosis_agent.ainvoke.await_count, 1)
        self.assertIn("DNS service failure", update["diagnosis_report"])

    def test_evidence_gate_enabled_env_override(self) -> None:
        with patch.dict("os.environ", {"NIKA_EVIDENCE_GATE_ENABLED": "false"}):
            self.assertFalse(evidence_gate_enabled(default=True))
        with patch.dict("os.environ", {"NIKA_EVIDENCE_GATE_ENABLED": "true"}):
            self.assertTrue(evidence_gate_enabled(default=False))

    async def test_diagnosis_max_steps_matches_upstream_error_contract(
        self,
    ) -> None:
        self.agent.diagnosis_agent = AsyncMock()
        self.agent.diagnosis_agent.ainvoke.side_effect = GraphRecursionError()
        self.agent.skill_tool_runtime = SimpleNamespace(
            snapshot=lambda: {
                "recent_transitions": [
                    {
                        "tool": "get_host_net_config",
                        "tool_input": {"host_name": "pc_0_0"},
                        "observation_summary": "pc_0_0 eth0 state DOWN; ip_route is empty",
                    }
                ]
            }
        )

        update = await self.agent.diagnosis_agent_builder(
            {"messages": [HumanMessage(content="Diagnose")]}
        )

        self.assertTrue(update["is_max_steps_reached"])
        self.assertEqual(update["diagnosis_report"], "ERROR_MAX_STEPS_REACHED")
        self.assertEqual(
            update["messages"][-1].content,
            "Error: diagnosis did not finish within max steps.",
        )

class ReactGraphRoutingTest(unittest.TestCase):
    def test_graph_stops_before_submission_after_diagnosis_max_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            diagnosis_agent = SimpleNamespace(ainvoke=AsyncMock())
            diagnosis_agent.ainvoke.side_effect = GraphRecursionError()
            submission_agent = SimpleNamespace(ainvoke=AsyncMock())
            submission_agent.ainvoke.return_value = {
                "messages": [AIMessage(content="submitted")]
            }
            skill_runtime = SimpleNamespace(
                snapshot=lambda: {
                    "recent_transitions": [
                        {
                            "tool": "get_host_net_config",
                            "tool_input": {"host_name": "pc_0_0"},
                            "observation_summary": "pc_0_0 eth0 state DOWN",
                        }
                    ]
                }
            )
            diagnosis_phase = SimpleNamespace(
                llm=object(),
                tool_evolution_runtime=None,
                skill_tool_runtime=skill_runtime,
                tools=[],
                load_tools=AsyncMock(),
                get_agent=lambda: diagnosis_agent,
                prompt_suffix=lambda **_: "",
            )
            submission_phase = SimpleNamespace(
                load_tools=AsyncMock(),
                get_agent=lambda: submission_agent,
            )

            with (
                patch("agent.langgraph.react_agent.DiagnosisPhase") as diag_cls,
                patch("agent.langgraph.react_agent.SubmissionPhase") as sub_cls,
                patch("agent.langgraph.react_agent.Session") as session_cls,
                patch(
                    "agent.langgraph.react_agent.create_langfuse_callbacks",
                    return_value=[],
                ),
            ):
                session = session_cls.return_value
                session.load_running_session.return_value = None
                session.session_dir = tmp
                session.scenario_name = "dc_clos_bgp"
                session.problem_names = ["link_down"]
                session.scenario_topo_size = "s"
                diag_cls.return_value = diagnosis_phase
                sub_cls.return_value = submission_phase

                agent = BasicReActAgent(session_id="s1", max_steps=3)
                result = asyncio.run(
                    agent.graph.ainvoke(
                        {"messages": [HumanMessage(content="Diagnose")]}
                    )
                )

            submission_agent.ainvoke.assert_not_awaited()
            self.assertEqual(result["diagnosis_report"], "ERROR_MAX_STEPS_REACHED")
            self.assertTrue(result["is_max_steps_reached"])
            self.assertEqual(
                result["messages"][-1].content,
                "Error: diagnosis did not finish within max steps.",
            )


class PlanExecuteBehaviorTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.agent = PlanExecuteAgent.__new__(PlanExecuteAgent)
        self.agent.max_steps = 2
        self.agent.session_dir = self.tmp.name
        self.agent._diagnosis_phase = SimpleNamespace(
            prompt_suffix=lambda **_: ""
        )

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

    async def test_executor_strips_learning_guidance_from_completed_evidence(self) -> None:
        step = PlanStep(
            step_id="reachability",
            action="Check reachability",
            expected_evidence="Failed pairs",
        )
        self.agent.executor = AsyncMock()
        self.agent.executor.ainvoke.return_value = {
            "messages": [
                SimpleNamespace(
                    content=(
                        "pc1 cannot reach pc2\n\n"
                        "[Integrated learning guidance - not evidence]\n"
                        "Active Skill-MDP option: seed_react_decision"
                    )
                )
            ]
        }

        update = await self.agent._execute(
            {
                "task_description": "Diagnose",
                "objective": "Find the fault",
                "plan": [step],
                "completed_steps": [],
                "executed_steps": 0,
            }
        )

        observation = update["completed_steps"][0].observation
        self.assertEqual(observation, "pc1 cannot reach pc2")
        self.assertNotIn("Integrated learning guidance", observation)

    async def test_executor_prompt_refreshes_learning_context_per_step(self) -> None:
        step = PlanStep(
            step_id="routes",
            action="Inspect routes",
            expected_evidence="Route state",
        )
        calls: list[str] = []

        def prompt_suffix(*, activate_skill=True) -> str:
            calls.append(str(activate_skill))
            return "\nSkill-Pro active option: followup_route"

        self.agent._diagnosis_phase = SimpleNamespace(prompt_suffix=prompt_suffix)
        self.agent.executor = AsyncMock()
        self.agent.executor.ainvoke.return_value = {
            "messages": [SimpleNamespace(content="route missing")]
        }

        await self.agent._execute(
            {
                "task_description": "Diagnose",
                "objective": "Find the fault",
                "plan": [step],
                "completed_steps": [],
                "executed_steps": 0,
            }
        )
        payload = self.agent.executor.ainvoke.call_args.args[0]
        prompt = payload["messages"][0].content

        self.assertEqual(calls, ["True"])
        self.assertIn("Current integrated learning context", prompt)
        self.assertIn("followup_route", prompt)

    def test_refresh_executor_keeps_static_prompt_and_defers_learning_context(self) -> None:
        self.agent.llm = object()
        self.agent._diagnosis_phase = SimpleNamespace(
            tool_evolution_runtime=None,
            skill_tool_runtime=None,
            tools=[],
            prompt_suffix=lambda **_: "\nSkill-Pro stale option",
        )
        with patch("agent.langgraph.plan_execute_agent.create_agent") as create:
            self.agent._refresh_executor()

        kwargs = create.call_args.kwargs
        self.assertIn("network troubleshooting executor", kwargs["system_prompt"])
        self.assertNotIn("Skill-Pro stale option", kwargs["system_prompt"])

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

    async def test_planner_prompt_includes_integrated_learning_context(self) -> None:
        self.agent._diagnosis_phase = SimpleNamespace(
            prompt_suffix=lambda **_: "\nSkill-Pro + DRAFT suffix"
        )
        self.agent.planner = AsyncMock()
        self.agent.planner.ainvoke.return_value = InvestigationPlan(
            objective="Find the fault",
            steps=[
                PlanStep(
                    step_id="reachability",
                    action="Check reachability",
                    expected_evidence="Packet loss",
                )
            ],
        )

        await self.agent._plan({"task_description": "Diagnose"})
        messages = self.agent.planner.ainvoke.call_args.args[0]

        self.assertIn(
            "Integrated learning context",
            messages[0].content,
        )
        self.assertIn("Skill-Pro + DRAFT suffix", messages[0].content)

    async def test_planner_reads_learning_context_without_activating_skill(self) -> None:
        calls: list[bool] = []

        def prompt_suffix(*, activate_skill=True) -> str:
            calls.append(activate_skill)
            return "\nSkill-Pro candidate context"

        self.agent._diagnosis_phase = SimpleNamespace(prompt_suffix=prompt_suffix)
        self.agent.planner = AsyncMock()
        self.agent.planner.ainvoke.return_value = InvestigationPlan(
            objective="Find the fault",
            steps=[
                PlanStep(
                    step_id="reachability",
                    action="Check reachability",
                    expected_evidence="Packet loss",
                )
            ],
        )

        await self.agent._plan({"task_description": "Diagnose"})

        self.assertEqual(calls, [False])

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

    def test_plan_item_limit_stops_before_synthesis_and_submission(self) -> None:
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
            "end",
        )

    async def test_executor_max_steps_stops_plan_execute_before_submission(self) -> None:
        step = PlanStep(
            step_id="routes",
            action="Inspect routes",
            expected_evidence="Route state",
        )
        self.agent.executor = AsyncMock()
        self.agent.executor.ainvoke.side_effect = GraphRecursionError()

        update = await self.agent._execute(
            {
                "task_description": "Diagnose",
                "objective": "Find the fault",
                "plan": [step],
                "completed_steps": [],
                "executed_steps": 0,
            }
        )

        self.assertTrue(update["is_max_steps_reached"])
        self.assertEqual(update["diagnosis_report"], "ERROR_MAX_STEPS_REACHED")
        self.assertEqual(
            self.agent._route_after_replan(update),
            "end",
        )

    async def test_workflow_recursion_returns_missing_submission_state(self) -> None:
        self.agent.session = SimpleNamespace(
            scenario_name="simple_bgp",
            problem_names=["link_down"],
            root_cause_name="link_down",
            scenario_topo_size="",
            model="model",
        )
        self.agent.langfuse_callbacks = []
        self.agent.tool_evolution_runtime = None
        self.agent.graph = AsyncMock()
        self.agent.graph.ainvoke.side_effect = GraphRecursionError()

        with patch(
            "agent.langgraph.plan_execute_agent.write_tool_evolution_session"
        ) as write_session:
            result = await self.agent.run("Diagnose")

        self.assertTrue(result["is_max_steps_reached"])
        self.assertEqual(result["diagnosis_report"], "ERROR_MAX_STEPS_REACHED")
        self.assertIn("did not finish within max steps", result["messages"][0].content)
        write_session.assert_called_once_with(None, self.tmp.name)

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
        self.agent._diagnosis_phase = SimpleNamespace(
            prompt_suffix=lambda **_: ""
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_compact_trace_strips_integrated_learning_guidance(self) -> None:
        trace = ReflexionAgent._compact_trace(
            [
                SimpleNamespace(
                    content=(
                        "route missing\n\n"
                        "[Integrated learning guidance - not evidence]\n"
                        "DRAFT next checks: inspect interface"
                    )
                )
            ]
        )

        self.assertEqual(trace[0]["content"], "route missing")
        self.assertNotIn("DRAFT next checks", trace[0]["content"])

    def test_refresh_actor_keeps_static_prompt_and_defers_learning_context(self) -> None:
        self.agent.llm = object()
        self.agent._diagnosis_phase = SimpleNamespace(
            tool_evolution_runtime=None,
            skill_tool_runtime=None,
            tools=[],
            prompt_suffix=lambda **_: "\nSkill-Pro stale option",
        )
        with patch("agent.langgraph.reflexion_agent.create_agent") as create:
            self.agent._refresh_actor()

        kwargs = create.call_args.kwargs
        self.assertIn("iterative Reflexion", kwargs["system_prompt"])
        self.assertNotIn("Skill-Pro stale option", kwargs["system_prompt"])

    async def test_attempt_prompt_refreshes_learning_context_per_attempt(self) -> None:
        calls: list[str] = []

        def prompt_suffix(*, activate_skill=True) -> str:
            calls.append(str(activate_skill))
            return "\nSkill-Pro active option: retry_bgp"

        self.agent._diagnosis_phase = SimpleNamespace(prompt_suffix=prompt_suffix)
        self.agent.actor = AsyncMock()
        self.agent.actor.ainvoke.return_value = {
            "messages": [SimpleNamespace(content="Evidence-backed diagnosis")]
        }

        await self.agent._attempt(
            {
                "task_description": "Diagnose",
                "attempt_count": 0,
                "diagnosis_report": "",
                "memories": [],
            }
        )
        payload = self.agent.actor.ainvoke.call_args.args[0]
        prompt = payload["messages"][0].content

        self.assertEqual(calls, ["True"])
        self.assertIn("Integrated learning context for this Reflexion attempt", prompt)
        self.assertIn("retry_bgp", prompt)

    async def test_retry_attempt_receives_episodic_memory(self) -> None:
        self.agent._diagnosis_phase = SimpleNamespace(
            prompt_suffix=lambda **_: ""
        )
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

    async def test_attempt_max_steps_stops_reflexion_before_submission(self) -> None:
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
        self.assertEqual(update["attempt_error"], "ERROR_MAX_STEPS_REACHED")
        self.assertTrue(update["is_max_steps_reached"])
        self.assertEqual(update["diagnosis_report"], "ERROR_MAX_STEPS_REACHED")
        self.assertEqual(
            self.agent._route_after_evaluation(update),
            "end",
        )

    async def test_workflow_recursion_returns_missing_submission_state(self) -> None:
        self.agent.session = SimpleNamespace(
            scenario_name="simple_bgp",
            problem_names=["link_down"],
            root_cause_name="link_down",
            scenario_topo_size="",
            model="model",
        )
        self.agent.langfuse_callbacks = []
        self.agent.tool_evolution_runtime = None
        self.agent.graph = AsyncMock()
        self.agent.graph.ainvoke.side_effect = GraphRecursionError()

        with patch(
            "agent.langgraph.reflexion_agent.write_tool_evolution_session"
        ) as write_session:
            result = await self.agent.run("Diagnose")

        self.assertTrue(result["is_max_steps_reached"])
        self.assertEqual(result["diagnosis_report"], "ERROR_MAX_STEPS_REACHED")
        self.assertEqual(result["attempt_error"], "ERROR_MAX_STEPS_REACHED")
        self.assertIn("did not finish within max steps", result["messages"][0].content)
        write_session.assert_called_once_with(None, self.tmp.name)

    async def test_attempt_non_recursion_failure_is_preserved_for_evaluation(self) -> None:
        self.agent.actor = AsyncMock()
        self.agent.actor.ainvoke.side_effect = RuntimeError("temporary failure")

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
        self.assertIn("temporary failure", update["attempt_error"])
        self.assertNotIn("is_max_steps_reached", update)

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
