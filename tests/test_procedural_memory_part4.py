"""Tests for Skill-Pro Procedural Memory."""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.tools import StructuredTool

from agent.procedural_memory.models import (
    EvaluationEvidence,
    ProceduralSkill,
    SemanticGradientDraft,
    SkillCandidateDraft,
    SkillStep,
)
from agent.procedural_memory.runtime import SkillToolRuntime
from agent.procedural_memory.policy_scorer import (
    BehavioralReplayPolicyScorer,
    PolicyReplayDraft,
    PolicyReplayItem,
    StructuredReplayPolicyScorer,
)
from agent.procedural_memory.service import (
    ProceduralMemoryModule,
)
from agent.procedural_memory.workflow import (
    update_procedural_memory_from_session,
)


class _FakeSkillController:
    def __init__(
        self,
        skill_ids: list[str],
        *,
        termination_status: str = "CONTINUE",
    ) -> None:
        self.skill_ids = list(skill_ids)
        self.termination_status = termination_status
        self.prompts: list[str] = []
        self.selection_prompts: list[str] = []

    def with_structured_output(self, _schema):
        owner = self

        class StructuredSelector:
            def invoke(self, prompt: str):
                owner.selection_prompts.append(prompt)
                skill_id = owner.skill_ids.pop(0) if owner.skill_ids else ""
                return {"skill_id": skill_id, "reason": "test selection"}

        return StructuredSelector()

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return SimpleNamespace(content=f"<status>{self.termination_status}</status>")


class _FakeEvolutionModel:
    def __init__(self) -> None:
        self.schema: type | None = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, prompt: str):
        if self.schema is SkillCandidateDraft:
            return SkillCandidateDraft(
                title="Evidence-guided diagnosis",
                initiation="When current diagnostic evidence is incomplete.",
                policy=[
                    "Inspect the current diagnostic evidence.",
                    "Cross-check the leading diagnosis with independent evidence.",
                ],
                termination="Stop after the current diagnosis is supported.",
            )
        if self.schema is PolicyReplayDraft:
            return PolicyReplayDraft(
                scores=[
                    PolicyReplayItem(
                        experience_id=experience_id,
                        candidate_alignment=0.8,
                        baseline_alignment=0.2,
                    )
                    for experience_id in re.findall(
                        r'"experience_id": "([^"]+)"', prompt
                    )
                ]
            )
        return SemanticGradientDraft(
            critique="The trajectory supports a reusable evidence check.",
            proposed_update="Cross-check the diagnosis before termination.",
            initiation="When current diagnostic evidence is incomplete.",
            policy=[
                "Inspect the current diagnostic evidence.",
                "Cross-check the leading diagnosis.",
            ],
            termination="Stop after the current diagnosis is supported.",
            is_related=True,
        )




class SkillProProceduralMemoryTestPart4(unittest.TestCase):
    @staticmethod
    def _enable_fake_evolution(module: ProceduralMemoryModule) -> None:
        model = _FakeEvolutionModel()
        module.llm_backend = "custom"
        module.model = "test-model"
        module._training_llm_instance = model
        module.policy_scorer = BehavioralReplayPolicyScorer(lambda: model)


    @staticmethod
    def _seed_running_baseline(
        module: ProceduralMemoryModule,
        scenario: str,
        value: float = 0.0,
    ) -> None:
        state = module.store.load()
        state.baselines[scenario] = value
        module.store.save(state)


    def test_skill_runtime_avoids_immediate_reselect_of_completed_option(self) -> None:
        def ping_host(host: str) -> str:
            return f"{host} reachable"

        def show_route(router: str) -> str:
            return f"{router} route table"

        with tempfile.TemporaryDirectory() as tmp:
            ping_tool = StructuredTool.from_function(
                ping_host,
                name="ping_host",
                description="Ping one host.",
            )
            route_tool = StructuredTool.from_function(
                show_route,
                name="show_route",
                description="Show route table.",
            )
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            state = module.store.load()
            state.skills["one_step_ping"] = ProceduralSkill(
                skill_id="one_step_ping",
                title="One-step reachability skill",
                activation_condition="Use for host reachability failure with ping_host.",
                execution_steps=[
                    SkillStep(
                        order=1,
                        action="Call ping_host once and interpret reachability.",
                        tool_name="ping_host",
                    )
                ],
                termination_condition="Stop after one concrete diagnostic action is selected.",
                tools=["ping_host"],
                status="validated",
                score=3.0,
            )
            state.skills["followup_route"] = ProceduralSkill(
                skill_id="followup_route",
                title="Follow-up route evidence skill",
                activation_condition=(
                    "Use after reachability evidence to inspect routing with show_route."
                ),
                execution_steps=[
                    SkillStep(
                        order=1,
                        action="Call show_route to explain the reachability result.",
                        tool_name="show_route",
                    )
                ],
                termination_condition="Stop after route evidence is interpreted.",
                tools=["show_route"],
                status="validated",
                score=2.5,
            )
            module.store.save(state)
            runtime = SkillToolRuntime(
                procedural_memory=module,
                allow_training_updates=False,
                session=SimpleNamespace(
                    session_id="s2",
                    scenario_name="simple_bgp",
                    scenario_topo_size="small",
                ),
                task_description="Host reachability failure",
                tools=[ping_tool, route_tool],
                session_dir=tmp,
                meta_controller_llm=_FakeSkillController(
                    ["one_step_ping", "followup_route"],
                    termination_status="DONE",
                ),
            )
            prompt = runtime.prompt_suffix()
            output = runtime.wrap_tools([ping_tool])[0].invoke({"host": "pc1"})
            snapshot = runtime.snapshot()
            rows = [
                json.loads(line)
                for line in (Path(tmp) / "messages.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]

        post_tool_activations = [
            row
            for row in rows
            if row.get("event") == "skill_activation"
            and row.get("source") == "post_tool"
        ]
        self.assertIn("one_step_ping", prompt)
        self.assertEqual(snapshot["active_skill_id"], "followup_route")
        self.assertEqual(snapshot["skill_cooldowns"], {})
        self.assertTrue(post_tool_activations)
        self.assertIn("one_step_ping", post_tool_activations[-1]["cooldown_exclusions"])
        self.assertEqual(output, "pc1 reachable")
        self.assertNotIn("Active Skill-MDP option:", output)


    def test_skill_runtime_logs_tool_deviation_without_posthoc_reselection(
        self,
    ) -> None:
        def ping_host(host: str) -> str:
            return f"{host} reachable"

        def show_route(router: str) -> str:
            return f"{router} route table"

        with tempfile.TemporaryDirectory() as tmp:
            ping_tool = StructuredTool.from_function(
                ping_host,
                name="ping_host",
                description="Ping one host.",
            )
            route_tool = StructuredTool.from_function(
                show_route,
                name="show_route",
                description="Show route table.",
            )
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            state = module.store.load()
            state.skills["prompt_ping"] = ProceduralSkill(
                skill_id="prompt_ping",
                title="Prompt-time reachability skill",
                activation_condition="Use for host reachability failure with ping_host.",
                execution_steps=[
                    SkillStep(
                        order=1,
                        action="Call ping_host before any route-table check.",
                        tool_name="ping_host",
                    )
                ],
                termination_condition="Stop after ping evidence is interpreted.",
                tools=["ping_host"],
                status="validated",
                score=2.0,
            )
            state.skills["route_skill"] = ProceduralSkill(
                skill_id="route_skill",
                title="Route-table skill",
                activation_condition="Use only after show_route is selected.",
                execution_steps=[
                    SkillStep(
                        order=1,
                        action="Call show_route.",
                        tool_name="show_route",
                    )
                ],
                termination_condition="Stop after route evidence is interpreted.",
                tools=["show_route"],
                status="validated",
                score=0.2,
            )
            module.store.save(state)
            runtime = SkillToolRuntime(
                procedural_memory=module,
                allow_training_updates=False,
                session=SimpleNamespace(
                    session_id="s2",
                    scenario_name="simple_bgp",
                    scenario_topo_size="small",
                ),
                task_description="Host reachability failure",
                tools=[ping_tool, route_tool],
                session_dir=tmp,
                meta_controller_llm=_FakeSkillController(
                    ["prompt_ping"],
                    termination_status="CONTINUE",
                ),
            )
            runtime.prompt_suffix()
            wrapped_route = runtime.wrap_tools([route_tool])[0]
            wrapped_route.invoke({"router": "r1"})
            snapshot = runtime.snapshot()
            rows = [
                json.loads(line)
                for line in (Path(tmp) / "messages.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]

        deviations = [
            row for row in rows if row.get("event") == "skill_policy_deviation"
        ]
        transitions = [row for row in rows if row.get("event") == "skill_transition"]
        self.assertEqual(snapshot["active_skill_id"], "prompt_ping")
        self.assertTrue(deviations)
        self.assertEqual(deviations[-1]["active_skill_id"], "prompt_ping")
        self.assertEqual(deviations[-1]["tool"], "show_route")
        self.assertEqual(transitions[-1]["active_skill_id"], "prompt_ping")
        self.assertEqual(transitions[-1]["activation_id"], "s2:1")


    def test_skill_tool_wrapper_preserves_content_and_artifact_tools(self) -> None:
        def reachability(host: str) -> tuple[str, dict[str, str]]:
            return f"{host} unreachable", {"raw": f"{host} unreachable"}

        with tempfile.TemporaryDirectory() as tmp:
            tool = StructuredTool.from_function(
                reachability,
                name="get_reachability",
                description="Get host reachability.",
                response_format="content_and_artifact",
            )
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            module.learn_from_episode(
                evidence=EvaluationEvidence(
                    session_id="s1",
                    task_description="Host reachability failure",
                    scenario="simple_bgp",
                    metrics={
                        "detection_score": 1.0,
                        "localization_accuracy": 1.0,
                        "rca_accuracy": 1.0,
                    },
                    success=True,
                ),
                tool_steps=[
                    SkillStep(
                        order=1,
                        action="Check host reachability before diagnosis.",
                        tool_name="get_reachability",
                    )
                ],
            )
            runtime = SkillToolRuntime(
                procedural_memory=module,
                allow_training_updates=False,
                session=SimpleNamespace(
                    session_id="s2",
                    scenario_name="simple_bgp",
                    scenario_topo_size="small",
                ),
                task_description="Client cannot reach server",
                tools=[tool],
                session_dir=tmp,
            )
            output = runtime.wrap_tools([tool])[0].run(
                {"host": "pc1"},
                tool_call_id="call-1",
            )

        self.assertEqual(output.artifact, {"raw": "pc1 unreachable"})
        self.assertIn("pc1 unreachable", str(output.content))
        self.assertNotIn(
            "Integrated training guidance - not evidence",
            str(output.content),
        )


    def test_trajectory_buffer_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            module.learn_from_episode(
                evidence=EvaluationEvidence(
                    session_id="s1",
                    task_description="HTTP ACL appears to block traffic",
                    scenario="ospf_enterprise_dhcp",
                    metrics={
                        "detection_score": 1.0,
                        "localization_accuracy": 1.0,
                        "rca_accuracy": 1.0,
                    },
                    steps=30,
                    tool_calls=20,
                    success=True,
                ),
                tool_steps=[
                    SkillStep(
                        order=1,
                        action="Test HTTP reachability.",
                        skill_id="seed_react_decision",
                        activation_id="s1:1",
                        tool_name="curl",
                    )
                ],
            )
            state = module.store.load()
            stats = module.store.bank_stats()

        self.assertEqual(len(state.experiences), 1)
        self.assertEqual(stats["experiences"], 1)


    def test_seed_parent_refines_to_versioned_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=1,
            )
            self._enable_fake_evolution(module)
            self._seed_running_baseline(module, "dc_clos_bgp")
            steps = [
                SkillStep(
                    order=1,
                    action="Check BGP summary.",
                    skill_id="seed_react_decision",
                    activation_id="s1:1",
                    tool_name="frr_show_bgp_summary",
                )
            ]
            bootstrap = module.learn_from_episode(
                evidence=EvaluationEvidence(
                    session_id="s1",
                    task_description="BGP route is missing",
                    scenario="dc_clos_bgp",
                    metrics={
                        "detection_score": 1.0,
                        "localization_f1": 0.5,
                        "rca_f1": 0.5,
                    },
                    steps=30,
                    tool_calls=20,
                    success=True,
                ),
                tool_steps=steps,
            )
            refined_report = module.learn_from_episode(
                evidence=EvaluationEvidence(
                    session_id="s2",
                    task_description="BGP route is missing again",
                    scenario="dc_clos_bgp",
                    metrics={
                        "detection_score": 1.0,
                        "localization_accuracy": 1.0,
                        "rca_accuracy": 1.0,
                    },
                    steps=3,
                    tool_calls=2,
                    success=True,
                ),
                tool_steps=[
                    steps[0].model_copy(
                        update={
                            "skill_id": bootstrap["skill_id"],
                            "activation_id": "s2:1",
                        }
                    )
                ],
            )
            state = module.store.load()
            refined = state.skills[refined_report["skill_id"]]

        self.assertEqual(bootstrap["status"], "deferred")
        self.assertEqual(refined_report["status"], "accepted")
        self.assertEqual(refined.parent_id, "seed_react_decision")
        self.assertGreaterEqual(refined.version, 1)
        self.assertEqual(state.skills["seed_react_decision"].status, "validated")
        self.assertIsNone(refined_report["decision"]["replaced_skill_id"])


    def test_llm_semantic_gradient_updates_candidate_skill(self) -> None:
        prompts: list[str] = []

        class FakeModel:
            schema: type | None = None

            def with_structured_output(self, schema):
                self.schema = schema
                return self

            def invoke(self, prompt):
                prompts.append(prompt)
                if self.schema is SkillCandidateDraft:
                    return SkillCandidateDraft(
                        title="BGP route verification",
                        initiation="When current BGP route evidence is incomplete.",
                        policy=[
                            "Inspect current BGP route evidence.",
                            "Cross-check current neighbor evidence.",
                        ],
                        termination=(
                            "Terminate only after route and neighbor evidence agree."
                        ),
                    )
                if self.schema is PolicyReplayDraft:
                    return PolicyReplayDraft(
                        scores=[
                            PolicyReplayItem(
                                experience_id=experience_id,
                                candidate_alignment=0.8,
                                baseline_alignment=0.2,
                            )
                            for experience_id in re.findall(
                                r'"experience_id": "([^"]+)"', prompt
                            )
                        ]
                    )
                return SemanticGradientDraft(
                    source_session_id="s1",
                    critique="Preserve BGP route checks but do not rely on leaf_router_0_1 alone.",
                    proposed_update=(
                        "For bgp_missing_route_advertisement, terminate only after "
                        "route and neighbor evidence agree."
                    ),
                    termination=(
                        "Terminate only after route and neighbor evidence agree "
                        "without naming leaf_router_0_1."
                    ),
                )

        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                llm_backend="openai",
                model="test-model",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=2,
            )
            self._seed_running_baseline(module, "dc_clos_bgp")
            with patch(
                "agent.procedural_memory.service.load_model", return_value=FakeModel()
            ):
                first = module.learn_from_episode(
                    evidence=EvaluationEvidence(
                        session_id="s0",
                        task_description="BGP missing route advertisement",
                        scenario="dc_clos_bgp",
                        root_cause=["bgp_missing_route_advertisement"],
                        faulty_devices=["leaf_router_0_1"],
                        metrics={
                            "detection_score": 1.0,
                            "localization_accuracy": 1.0,
                            "rca_accuracy": 1.0,
                        },
                        steps=5,
                        tool_calls=3,
                        success=True,
                    ),
                    tool_steps=[
                        SkillStep(
                            order=1,
                            action="Check BGP neighbors.",
                            skill_id="seed_react_decision",
                            activation_id="s0:1",
                            tool_name="frr_show_bgp_summary",
                        )
                    ],
                )
                report = module.learn_from_episode(
                    evidence=EvaluationEvidence(
                        session_id="s1",
                        task_description="BGP missing route advertisement",
                        scenario="dc_clos_bgp",
                        root_cause=["bgp_missing_route_advertisement"],
                        faulty_devices=["leaf_router_0_1"],
                        metrics={
                            "detection_score": 1.0,
                            "localization_accuracy": 1.0,
                            "rca_accuracy": 1.0,
                        },
                        steps=5,
                        tool_calls=3,
                        success=True,
                    ),
                    tool_steps=[
                        SkillStep(
                            order=1,
                            action="Check BGP neighbors.",
                            skill_id="seed_react_decision",
                            activation_id="s1:1",
                            tool_name="frr_show_bgp_summary",
                        )
                    ],
                )
            state = module.store.load()
            skill = state.skills[report["skill_id"]]
            stats = module.store.bank_stats()

        self.assertEqual(first["status"], "deferred")
        self.assertEqual(report["status"], "accepted")
        self.assertEqual(report["semantic_gradient_source"], "llm")
        self.assertEqual(skill.semantic_gradients[0].gradient_source, "llm")
        self.assertIn("route and neighbor evidence", skill.termination_condition)
        self.assertNotIn("bgp_missing_route_advertisement", skill.termination_condition)
        self.assertNotIn("leaf_router_0_1", skill.termination_condition)
        self.assertEqual(stats["llm_semantic_gradients"], 1)
        self.assertEqual(len(prompts), 6)
        self.assertTrue(any("batch semantic-gradient aggregator" in p for p in prompts))
        self.assertEqual(
            sum("Skill Evolver" in prompt for prompt in prompts),
            2,
        )
        self.assertNotIn("bgp_missing_route_advertisement", "\n".join(prompts))
        self.assertNotIn("leaf_router_0_1", "\n".join(prompts))


    def test_llm_semantic_gradient_failure_is_reported(self) -> None:
        class FailingModel:
            def with_structured_output(self, _schema):
                return self

            def invoke(self, _prompt):
                raise TimeoutError("skill timeout")

        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                llm_backend="custom",
                model="test-model",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=2,
                policy_scorer=StructuredReplayPolicyScorer(),
            )
            with (
                patch.dict(
                    os.environ,
                    {
                        "NIKA_TRAINING_LLM_BACKEND": "custom",
                        "NIKA_TRAINING_LLM_MODEL": "training-model",
                    },
                ),
                patch(
                    "agent.procedural_memory.service.load_model",
                    return_value=FailingModel(),
                ) as load_model,
            ):
                first = module.learn_from_episode(
                    evidence=EvaluationEvidence(
                        session_id="s0",
                        task_description="BGP missing route advertisement",
                        scenario="dc_clos_bgp",
                        metrics={
                            "detection_score": 1.0,
                            "localization_accuracy": 1.0,
                            "rca_accuracy": 1.0,
                        },
                        steps=5,
                        tool_calls=3,
                        success=True,
                    ),
                    tool_steps=[
                        SkillStep(
                            order=1,
                            action="Check BGP neighbors.",
                            skill_id="seed_react_decision",
                            activation_id="s0:1",
                            tool_name="frr_show_bgp_summary",
                        )
                    ],
                )
                report = module.learn_from_episode(
                    evidence=EvaluationEvidence(
                        session_id="s1",
                        task_description="BGP missing route advertisement",
                        scenario="dc_clos_bgp",
                        metrics={
                            "detection_score": 1.0,
                            "localization_accuracy": 1.0,
                            "rca_accuracy": 1.0,
                        },
                        steps=5,
                        tool_calls=3,
                        success=True,
                    ),
                    tool_steps=[
                        SkillStep(
                            order=1,
                            action="Check BGP neighbors.",
                            skill_id="seed_react_decision",
                            activation_id="s1:1",
                            tool_name="frr_show_bgp_summary",
                        )
                    ],
                )

        self.assertEqual(first["status"], "deferred")
        load_model.assert_called_once()
        args, kwargs = load_model.call_args
        self.assertEqual(args[:2], ("custom", "training-model"))
        self.assertEqual(kwargs["max_retries"], 0)
        self.assertEqual(report["semantic_gradient_source"], "llm")
        self.assertTrue(report["semantic_gradient_llm_attempted"])
        self.assertTrue(report["semantic_gradient_llm_failed"])
        self.assertIn("TimeoutError", report["semantic_gradient_llm_error"])


    def test_offline_workflow_writes_skill_update_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp) / "session"
            session_dir.mkdir()
            (session_dir / "run.json").write_text(
                json.dumps(
                    {
                        "session_id": "s1",
                        "status": "finished",
                        "agent_type": "react",
                        "model": "test-model",
                        "scenario_name": "enterprise",
                        "scenario_topo_size": "small",
                        "problem_names": ["dns_record_error"],
                        "root_cause_name": "dns_record_error",
                        "procedural_memory_enabled": True,
                        "allow_training_updates": True,
                        "procedural_memory_bank": "skill",
                    }
                ),
                encoding="utf-8",
            )
            (session_dir / "ground_truth.json").write_text(
                json.dumps(
                    {
                        "root_cause_name": ["dns_record_error"],
                        "faulty_devices": ["dns_server"],
                    }
                ),
                encoding="utf-8",
            )
            (session_dir / "messages.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "agent": "diagnosis_agent",
                                "event": "tool_start",
                                "tool": {"name": "dig"},
                                "input": "{'host': 'client'}",
                            }
                        )
                    ]
                ),
                encoding="utf-8",
            )
            (session_dir / "procedural_memory_runtime_session.json").write_text(
                json.dumps(
                    {
                        "prompt_added_tokens": 80,
                        "tool_description_added_tokens": 20,
                        "total_added_tokens": 100,
                        "prompt_injection_count": 2,
                        "tool_description_injection_count": 4,
                    }
                ),
                encoding="utf-8",
            )
            with patch(
                "agent.procedural_memory.store.PROCEDURAL_MEMORY_DIR",
                Path(tmp) / "procedural_memory",
            ):
                report = asyncio.run(
                    update_procedural_memory_from_session(
                        run_meta={
                            "procedural_memory_enabled": True,
                            "allow_training_updates": True,
                            "procedural_memory_bank": "skill",
                            "procedural_memory_pool_size": 24,
                            "procedural_memory_update_threshold": 2,
                            "procedural_memory_best_of_n": 5,
                            "procedural_memory_ppo_epsilon": 0.15,
                            "procedural_memory_max_skill_age": 6,
                            "session_id": "s1",
                            "task_description": "DNS record resolves to wrong host",
                            "scenario_name": "enterprise",
                        },
                        metrics={
                            "detection_score": 1.0,
                            "localization_accuracy": 1.0,
                            "rca_accuracy": 1.0,
                            "steps": 4,
                            "tool_calls": 2,
                        },
                        session_dir=session_dir,
                    )
                )
            self.assertTrue((session_dir / "procedural_memory_update.json").exists())

        self.assertEqual(report["method"], "Skill-Pro")
        self.assertEqual(report["procedural_memory_config"]["pool_size"], 24)
        self.assertEqual(report["procedural_memory_config"]["evolution_threshold"], 2)
        self.assertEqual(report["procedural_memory_config"]["best_of_n"], 5)
        self.assertEqual(report["procedural_memory_config"]["ppo_epsilon"], 0.15)
        self.assertEqual(report["procedural_memory_config"]["max_skill_age"], 6)
        self.assertEqual(
            report["procedural_memory_config"]["selection_policy"],
            "llm_direct_epsilon_greedy",
        )
        self.assertEqual(report["procedural_memory_config"]["selection_epsilon"], 0.25)
        self.assertEqual(report["total_added_tokens"], 100)
        self.assertEqual(report["delta_prompt_tokens_per_step"], 25.0)
        self.assertEqual(report["prompt_added_tokens"], 80)
        self.assertEqual(report["tool_description_added_tokens"], 20)


    def test_evaluation_workflow_skips_memory_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = asyncio.run(
                update_procedural_memory_from_session(
                    run_meta={
                        "procedural_memory_enabled": True,
                        "allow_training_updates": False,
                    },
                    metrics={},
                    session_dir=tmp,
                )
            )

        self.assertEqual(report["status"], "skipped")
        self.assertIn("disabled", report["reason"])

