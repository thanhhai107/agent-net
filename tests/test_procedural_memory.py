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
from unittest.mock import AsyncMock, patch

from langchain_core.tools import StructuredTool

from agent.procedural_memory.attributes import infer_procedural_memory_attributes
from agent.procedural_memory.models import (
    EvaluationEvidence,
    ProceduralMemoryQuery,
    ProceduralSkill,
    SemanticGradient,
    SemanticGradientDraft,
    SkillCandidateDraft,
    SkillComponentGradient,
    SkillExperience,
    SkillStep,
    SkillTransition,
)
from agent.procedural_memory.runtime import SkillToolRuntime
from agent.procedural_memory.policy_scorer import (
    BehavioralReplayPolicyScorer,
    PolicyReplayDraft,
    PolicyReplayItem,
    PolicyReplayResult,
    PolicyLogprobScorer,
    PolicyStepLogprob,
    StructuredReplayPolicyScorer,
)
from agent.procedural_memory.service import (
    ProceduralMemoryModule,
    _evidence_score,
    _metric_success,
)
from agent.procedural_memory.workflow import (
    update_procedural_memory_from_session,
    extract_skill_steps,
)
from agent.tool_refinement.runtime import ToolRefinementRuntime
from agent.tool_refinement.store import ToolRefinementStore
from nika.evaluator.result_log import build_eval_result_from_session_dir
from nika.workflows.eval.session import run_eval_metrics


class SkillProProceduralMemoryTest(unittest.TestCase):
    @staticmethod
    def _seed_running_baseline(
        module: ProceduralMemoryModule,
        scenario: str,
        value: float = 0.0,
    ) -> None:
        state = module.store.load()
        state.baselines[scenario] = value
        module.store.save(state)

    def test_ready_secondary_skill_is_scheduled_before_most_used_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="scheduler",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=2,
            )
            state = module.store.load()
            state.experiences.append(
                SkillExperience(
                    experience_id="secondary-history",
                    session_id="old",
                    reward=0.5,
                    skill_ids=["seed_self_consistency_check"],
                    transitions=[SkillTransition(action="inspect")],
                )
            )

            selected = module._runtime_parent_from_steps(
                state,
                [
                    SkillStep(order=1, skill_id="seed_react_decision", action="a"),
                    SkillStep(order=2, skill_id="seed_react_decision", action="b"),
                    SkillStep(
                        order=3,
                        skill_id="seed_self_consistency_check",
                        action="c",
                    ),
                ],
            )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.skill_id, "seed_self_consistency_check")

    def test_epsilon_exploration_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="exploration",
                store_path=Path(tmp) / "skills.json",
            )
            query = ProceduralMemoryQuery(
                text="Investigate endpoint reachability",
                top_k=1,
            )
            first = module.select_skill(
                query=query,
                record_reuse=False,
                exploration_epsilon=1.0,
                exploration_key="session-1:0",
            )
            second = module.select_skill(
                query=query,
                record_reuse=False,
                exploration_epsilon=1.0,
                exploration_key="session-1:0",
            )

        self.assertEqual(
            first.skill.skill_id if first else None,
            second.skill.skill_id if second else None,
        )

    def test_selection_epsilon_decay_uses_source_episode_scale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="epsilon-decay",
                store_path=Path(tmp) / "skills.json",
            )
            state = module.store.load()
            state.iteration = 100
            module.store.save(state)

            epsilon = module.decayed_selection_epsilon(0.3)

        self.assertAlmostEqual(epsilon, 0.25)

    def test_epsilon_exploration_runs_when_retrieval_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="empty-retrieval-exploration",
                store_path=Path(tmp) / "skills.json",
            )
            query = ProceduralMemoryQuery(text="unseen state", top_k=1)
            selected = None
            with patch.object(module, "retrieve", return_value=[]):
                for index in range(20):
                    selected = module.select_skill(
                        query=query,
                        record_reuse=False,
                        exploration_epsilon=1.0,
                        exploration_key=f"empty:{index}",
                    )
                    if selected is not None:
                        break

        self.assertIsNotNone(selected)
        self.assertIn("epsilon_exploration", selected.reasons)

    def test_epsilon_exploration_respects_cooldown_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="exploration-cooldown",
                store_path=Path(tmp) / "skills.json",
            )
            query = ProceduralMemoryQuery(text="diagnostic decision", top_k=2)
            excluded = "seed_react_decision"
            selected_ids = {
                selected.skill.skill_id
                for index in range(30)
                if (
                    selected := module.select_skill(
                        query=query,
                        record_reuse=False,
                        exclude_skill_ids={excluded},
                        allow_excluded_fallback=False,
                        exploration_epsilon=1.0,
                        exploration_key=f"cooldown:{index}",
                    )
                )
                is not None
            }

        self.assertNotIn(excluded, selected_ids)

    def test_maintenance_retires_semantic_duplicate_after_warmup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="semantic-dedup",
                store_path=Path(tmp) / "skills.json",
            )
            state = module.store.load()
            common = {
                "title": "Reachability evidence",
                "activation_condition": "When endpoint reachability evidence is incomplete.",
                "termination_condition": "Stop after reachability evidence is confirmed.",
                "status": "validated",
                "maturity": 3,
            }
            state.skills["learned-a"] = ProceduralSkill(
                skill_id="learned-a",
                execution_steps=[
                    SkillStep(order=1, action="Inspect endpoint reachability evidence.")
                ],
                avg_gain=0.2,
                frequency=5,
                **common,
            )
            state.skills["learned-b"] = ProceduralSkill(
                skill_id="learned-b",
                execution_steps=[
                    SkillStep(
                        order=1,
                        action="Inspect endpoint reachability evidence carefully.",
                    )
                ],
                avg_gain=0.1,
                frequency=5,
                **common,
            )

            module._maintain(state)

        self.assertEqual(state.skills["learned-a"].status, "validated")
        self.assertEqual(state.skills["learned-b"].status, "retired")
        self.assertTrue(
            any(item["stage"] == "semantic duplicate" for item in state.maintenance_log)
        )

    def test_custom_api_uses_teacher_forced_logprob_scorer(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(
                os.environ,
                {
                    "CUSTOM_API_URL": "https://example.test/v1",
                    "CUSTOM_API_KEY": "test-password",
                },
            ),
        ):
            module = ProceduralMemoryModule(
                bank_id="api-logprob",
                llm_backend="custom",
                model="provider/model",
                store_path=Path(tmp) / "skills.json",
            )

        self.assertIsInstance(module.policy_scorer, PolicyLogprobScorer)

    def test_ppo_gate_uses_target_action_logprob_ratio(self) -> None:
        class LogprobScorer:
            def score_batch(self, *, candidate, baseline, experiences):
                del candidate, baseline
                return PolicyReplayResult(
                    scores=[],
                    method="policy_logprob",
                    step_logprobs=[
                        PolicyStepLogprob(
                            experience_id=experience.experience_id,
                            transition_index=index,
                            candidate_logprob=-1.0,
                            baseline_logprob=-2.0,
                        )
                        for experience in experiences
                        for index, _ in enumerate(experience.transitions)
                    ],
                )

        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="logprob-gate",
                store_path=Path(tmp) / "skills.json",
                policy_scorer=LogprobScorer(),
                ppo_epsilon=0.2,
            )
            candidate = ProceduralSkill(
                skill_id="candidate",
                title="Candidate",
                activation_condition="When routing evidence is incomplete.",
                execution_steps=[SkillStep(order=1, action="Inspect routes.")],
                termination_condition="Stop after route evidence.",
            )
            decision = module.ppo_gate(
                candidate=candidate,
                evidence=EvaluationEvidence(session_id="logprob-gate-1"),
                samples=[
                    SkillExperience(
                        experience_id="logprob-exp",
                        session_id="logprob-gate-1",
                        reward=1.0,
                        baseline=0.0,
                        advantage=1.0,
                        transitions=[
                            SkillTransition(
                                state="Routing evidence is incomplete.",
                                action="show_routes({})",
                                done=True,
                            )
                        ],
                    )
                ],
            )

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.verification_method, "policy_logprob")
        self.assertAlmostEqual(decision.j_score, 1.2)
        self.assertEqual(decision.candidate_alignment, -1.0)
        self.assertEqual(decision.baseline_alignment, -2.0)

    def test_verification_batch_rebases_advantage_to_current_running_baseline(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="rebase",
                store_path=Path(tmp) / "skills.json",
            )
            state = module.store.load()
            state.baselines["routing"] = 0.7
            original = SkillExperience(
                experience_id="old",
                session_id="old",
                scenario="routing",
                reward=0.9,
                baseline=0.2,
                advantage=0.7,
            )

            batch = module._verification_batch(
                state,
                None,
                generation_samples=[original],
            )

        self.assertAlmostEqual(batch[0].baseline, 0.7)
        self.assertAlmostEqual(batch[0].advantage, 0.2)
        self.assertEqual(original.baseline, 0.2)

    def test_ppo_gate_rejects_when_behavioral_verification_failed(self) -> None:
        class FailedReplayScorer:
            def score_batch(self, *, candidate, baseline, experiences):
                del candidate, baseline
                return PolicyReplayResult(
                    scores=[
                        PolicyReplayItem(
                            experience_id=experience.experience_id,
                            candidate_alignment=0.9,
                            baseline_alignment=0.1,
                        )
                        for experience in experiences
                    ],
                    method="structured_replay",
                    error="TimeoutError: behavioral replay timed out",
                )

        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="failed-gate",
                store_path=Path(tmp) / "skills.json",
                policy_scorer=FailedReplayScorer(),
            )
            candidate = ProceduralSkill(
                skill_id="candidate",
                title="Candidate",
                activation_condition="When routing evidence is incomplete.",
                execution_steps=[SkillStep(order=1, action="Inspect routes.")],
                termination_condition="Stop after route evidence.",
            )
            decision = module.ppo_gate(
                candidate=candidate,
                evidence=EvaluationEvidence(
                    session_id="failed-gate-1",
                    metrics={
                        "detection_score": 1.0,
                        "localization_f1": 1.0,
                        "rca_f1": 1.0,
                    },
                ),
                samples=[
                    SkillExperience(
                        experience_id="failed-gate-exp",
                        session_id="failed-gate-1",
                        reward=1.0,
                        baseline=0.0,
                        advantage=1.0,
                        transitions=[
                            SkillTransition(
                                state="Routing evidence is incomplete.",
                                action="Inspect routes.",
                                done=True,
                            )
                        ],
                    )
                ],
            )

        self.assertFalse(decision.accepted)
        self.assertIn("verification unavailable", decision.reason)
        self.assertIn("TimeoutError", decision.verification_error)

    def test_no_fault_success_requires_detection_only(self) -> None:
        metrics = {
            "detection_score": 1.0,
            "localization_f1": 0.0,
            "rca_f1": 0.0,
        }

        self.assertTrue(_metric_success(metrics, False))
        self.assertFalse(_metric_success(metrics, True))

    def test_no_fault_semantic_gradient_does_not_demand_rca(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="no-fault",
                store_path=Path(tmp) / "skills.json",
            )
            gradient = module.semantic_gradient(
                evidence=EvaluationEvidence(
                    session_id="no-fault-1",
                    task_description="Determine whether the network is healthy.",
                    ground_truth_is_anomaly=False,
                    metrics={"detection_score": 1.0},
                    success=True,
                ),
                tool_steps=[
                    SkillStep(
                        order=1,
                        action="Inspect current reachability.",
                        tool_name="get_reachability",
                        observation_summary="All expected paths are reachable.",
                    )
                ],
            )

        termination = gradient.component_update.termination
        self.assertIn("no-anomaly", termination)
        self.assertIn("leave localization and root cause empty", termination)

    def test_online_gain_uses_uniform_advantage_credit(self) -> None:
        skill = ProceduralSkill(
            skill_id="credit",
            title="Credit test",
            activation_condition="When evidence is available.",
            execution_steps=[SkillStep(order=1, action="Inspect evidence.")],
            termination_condition="Stop after inspection.",
        )

        skill.update_stats(
            reward=0.7,
            baseline=0.5,
            total_skill_calls=4,
            skill_call_count=2,
        )

        self.assertAlmostEqual(skill.total_gain, 0.1)
        self.assertEqual(skill.frequency, 2)
        self.assertAlmostEqual(skill.avg_gain, 0.05)
        self.assertEqual(skill.success_count, 1)
        self.assertEqual(skill.failure_count, 0)
        self.assertEqual(skill.maturity, 0)

    def test_each_learning_iteration_ages_the_whole_active_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="maturity",
                store_path=Path(tmp) / "skills.json",
            )
            module.learn_from_episode(
                evidence=EvaluationEvidence(
                    session_id="maturity-1",
                    task_description="Inspect current routing evidence.",
                    scenario="routing",
                    metrics={"detection_score": 1.0},
                ),
                tool_steps=[
                    SkillStep(
                        order=1,
                        action="Inspect routes.",
                        tool_name="show_routes",
                    )
                ],
            )
            state = module.store.load()

        active = [skill for skill in state.skills.values() if skill.status != "retired"]
        self.assertTrue(active)
        self.assertTrue(all(skill.maturity == 1 for skill in active))

    def test_credit_is_normalized_over_attributed_skill_steps_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="attributed-credit",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=10,
            )
            report = module.learn_from_episode(
                evidence=EvaluationEvidence(
                    session_id="attributed-credit-1",
                    task_description="Inspect current routing evidence.",
                    scenario="routing",
                    metrics={
                        "detection_score": 1.0,
                        "localization_f1": 1.0,
                        "rca_f1": 1.0,
                    },
                ),
                tool_steps=[
                    SkillStep(
                        order=1,
                        skill_id="seed_react_decision",
                        action="Inspect routes.",
                        tool_name="show_routes",
                    ),
                    SkillStep(
                        order=2,
                        action="Unattributed fallback action.",
                        tool_name="fallback_probe",
                    ),
                ],
            )
            skill = module.store.load().skills["seed_react_decision"]

        self.assertAlmostEqual(
            skill.total_gain,
            report["episode_reward"] - report["episode_baseline"],
        )
        self.assertEqual(skill.frequency, 1)

    def test_maintenance_uses_empirical_gain_instead_of_positive_prior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="maintenance",
                store_path=Path(tmp) / "skills.json",
            )
            state = module.store.load()
            state.skills["harmful"] = ProceduralSkill(
                skill_id="harmful",
                title="Harmful policy",
                activation_condition="When current evidence matches.",
                execution_steps=[SkillStep(order=1, action="Inspect evidence.")],
                termination_condition="Stop after inspection.",
                status="validated",
                prior_score=10.0,
                score=10.0,
                frequency=10,
                total_gain=-1.0,
                avg_gain=-0.1,
                maturity=10,
            )

            module._maintain(state)

        self.assertEqual(state.skills["harmful"].status, "retired")
        self.assertTrue(
            any(
                item.get("stage") == "non-positive online score"
                and item.get("skill_id") == "harmful"
                for item in state.maintenance_log
            )
        )

    def test_duplicate_maintenance_keeps_higher_gain_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="dedup-value",
                store_path=Path(tmp) / "skills.json",
            )
            state = module.store.load()

            def duplicate(skill_id: str, avg_gain: float) -> ProceduralSkill:
                return ProceduralSkill(
                    skill_id=skill_id,
                    title="Equivalent procedure",
                    activation_condition="When routing evidence is incomplete.",
                    execution_steps=[
                        SkillStep(
                            order=1,
                            action="Inspect current routing evidence.",
                            tool_name="show_routes",
                        )
                    ],
                    termination_condition="Stop after route evidence is observed.",
                    status="validated",
                    frequency=10,
                    total_gain=10 * avg_gain,
                    avg_gain=avg_gain,
                    maturity=3,
                )

            state.skills["lower"] = duplicate("lower", 0.1)
            state.skills["higher"] = duplicate("higher", 0.4)
            module._maintain(state)

        self.assertEqual(state.skills["lower"].status, "retired")
        self.assertEqual(state.skills["higher"].status, "validated")

    def test_experience_records_timestep_state_and_primitive_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="trajectory",
                store_path=Path(tmp) / "skills.json",
            )
            experience = module._experience_from_episode(
                evidence=EvaluationEvidence(
                    session_id="trajectory-1",
                    task_description="Investigate current reachability.",
                ),
                tool_steps=[
                    SkillStep(
                        order=1,
                        action="Ping the endpoint.",
                        tool_name="ping_host",
                        arguments_hint={"host": "client"},
                        observation_summary="No response.",
                    ),
                    SkillStep(
                        order=2,
                        action="Inspect the route.",
                        tool_name="show_route",
                        arguments_hint={"device": "router"},
                        observation_summary="Route is absent.",
                    ),
                ],
                reward=0.5,
                baseline=0.2,
                skill_ids=[],
                success=False,
            )

        self.assertEqual(
            experience.transitions[0].action, 'ping_host({"host":"client"})'
        )
        self.assertNotIn("No response", experience.transitions[0].state)
        self.assertIn("No response", experience.transitions[1].state)
        self.assertEqual(
            experience.transitions[1].action,
            'show_route({"device":"router"})',
        )

    def test_segment_replay_preserves_observations_from_other_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="segment-state",
                store_path=Path(tmp) / "skills.json",
            )
            evidence = EvaluationEvidence(
                session_id="segment-1",
                task_description="Investigate current routing state.",
            )
            segments = module._segment_experiences(
                evidence=evidence,
                tool_steps=[
                    SkillStep(
                        order=1,
                        skill_id="seed_react_decision",
                        action="Inspect routes.",
                        tool_name="show_routes",
                        observation_summary="Route is absent.",
                    ),
                    SkillStep(
                        order=2,
                        skill_id="seed_hypothesis_elimination",
                        action="Inspect neighbors.",
                        tool_name="show_neighbors",
                        observation_summary="Neighbor is inactive.",
                    ),
                    SkillStep(
                        order=3,
                        skill_id="seed_react_decision",
                        action="Inspect policy.",
                        tool_name="show_policy",
                        observation_summary="Policy rejects the route.",
                    ),
                ],
                reward=0.8,
                baseline=0.4,
                success=True,
                valid_skill_ids={
                    "seed_react_decision",
                    "seed_hypothesis_elimination",
                },
            )

        react = segments["seed_react_decision"]
        elimination = segments["seed_hypothesis_elimination"]
        self.assertEqual(react.step_count, 2)
        self.assertEqual(elimination.step_count, 1)
        self.assertIn("Route is absent", react.transitions[1].state)
        self.assertIn("Neighbor is inactive", react.transitions[1].state)
        self.assertTrue(react.transitions[-1].done)
        self.assertTrue(elimination.transitions[-1].done)

    def test_skill_representation_excludes_scenario_and_episode_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="procedural-only",
                store_path=Path(tmp) / "skills.json",
            )
            skill = module.propose_skill(
                evidence=EvaluationEvidence(
                    session_id="procedural-only-1",
                    task_description="Inspect routing evidence.",
                    scenario="training_scenario_only",
                ),
                tool_steps=[
                    SkillStep(
                        order=1,
                        action="Inspect the current route table.",
                        tool_name="show_routes",
                        arguments_hint={"device": "case_specific_router"},
                        observation_summary="case_specific_route_value",
                        status="success",
                    )
                ],
            )

        formatted = skill.format_for_llm()
        self.assertNotIn("training_scenario_only", formatted)
        self.assertNotIn("case_specific_router", formatted)
        self.assertNotIn("case_specific_route_value", formatted)
        self.assertEqual(skill.execution_steps[0].arguments_hint, {})
        self.assertEqual(skill.execution_steps[0].observation_summary, "")

    def test_retrieval_score_is_invariant_to_scenario_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="cross-task",
                store_path=Path(tmp) / "skills.json",
            )
            state = module.store.load()
            state.skills["dns_cross_task"] = ProceduralSkill(
                skill_id="dns_cross_task",
                title="DNS evidence procedure",
                activation_condition="Use when current observations show DNS lookup failure.",
                execution_steps=[
                    SkillStep(order=1, action="Inspect the current DNS response.")
                ],
                termination_condition="Stop after resolver evidence is observed.",
                scenarios=["training_scenario"],
                protocols=["dns"],
                services=["name_resolution"],
                symptoms=["dns_failure"],
                status="validated",
                score=0.8,
                prior_score=0.8,
            )
            module.store.save(state)

            def score_for(scenario: str) -> float:
                results = module.retrieve(
                    query=ProceduralMemoryQuery(
                        text="Current DNS lookup returns SERVFAIL.",
                        scenario=scenario,
                        protocols=["dns"],
                        services=["name_resolution"],
                        symptoms=["dns_failure"],
                        top_k=20,
                    )
                )
                return next(
                    item.score
                    for item in results
                    if item.skill.skill_id == "dns_cross_task"
                )

            train_score = score_for("training_scenario")
            transfer_score = score_for("unseen_scenario")

        self.assertEqual(train_score, transfer_score)

    def test_irrelevant_parent_gradients_create_a_new_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="new-skill",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=1,
            )
            self._seed_running_baseline(module, "generic")
            unrelated = SemanticGradient(
                source_session_id="new-1",
                critique="The active option was unrelated to the observed state.",
                proposed_update="Create a procedure for the observed action pattern.",
                component_update=SkillComponentGradient(is_related=False),
            )
            with (
                patch.object(module, "semantic_gradient", return_value=unrelated),
                patch.object(
                    module,
                    "aggregate_semantic_gradients",
                    wraps=module.aggregate_semantic_gradients,
                ) as aggregate,
            ):
                report = module.learn_from_episode(
                    evidence=EvaluationEvidence(
                        session_id="new-1",
                        task_description="Inspect an unknown service failure.",
                        scenario="generic",
                        metrics={
                            "detection_score": 1.0,
                            "localization_accuracy": 1.0,
                            "rca_accuracy": 1.0,
                        },
                    ),
                    tool_steps=[
                        SkillStep(
                            order=1,
                            action="Inspect service state.",
                            skill_id="seed_react_decision",
                            tool_name="inspect_service",
                            observation_summary="Service is unavailable.",
                        )
                    ],
                )
            skill = module.store.load().skills[report["skill_id"]]

        self.assertEqual(report["status"], "accepted")
        self.assertEqual(report["candidate_type"], "NEW")
        self.assertEqual(report["relevance_ratio"], 0.0)
        self.assertEqual(skill.parent_id, "")
        self.assertEqual(aggregate.call_args.kwargs["gradients"], [])

    def test_combined_runtime_injects_tool_refinement_guidance_once(self) -> None:
        def inspect(host: str) -> str:
            return host

        with tempfile.TemporaryDirectory() as tmp:
            tool = StructuredTool.from_function(
                inspect,
                name="inspect_host",
                description="Inspect one host.",
            )
            refinement_store = ToolRefinementStore(
                "combined",
                root=Path(tmp) / "tool_refinement",
            )
            refinement = ToolRefinementRuntime(
                session=SimpleNamespace(),
                primitive_tools=[tool],
                library_id="combined",
                store=refinement_store,
            )
            state = refinement_store.load()
            state.documents["inspect_host"].usage_notes = [
                "Use observed identifiers only."
            ]
            refinement_store.save(state)
            refinement = ToolRefinementRuntime(
                session=SimpleNamespace(),
                primitive_tools=[tool],
                library_id="combined",
                store=refinement_store,
            )
            primitive = refinement.build_tools(append_docs=False)[0]
            module = ProceduralMemoryModule(
                bank_id="combined",
                store_path=Path(tmp) / "skills.json",
            )
            runtime = SkillToolRuntime(
                procedural_memory=module,
                procedural_memory_mode="read",
                session=SimpleNamespace(),
                task_description="Inspect current connectivity.",
                tools=[primitive],
                tool_refinement_runtime=refinement,
            )

            description = runtime.wrap_tools([primitive])[0].description
            prompt = runtime.prompt_suffix()

        self.assertEqual(description.count("Use observed identifiers only."), 1)
        self.assertNotIn("DRAFT refined guidance", description)
        self.assertIn("DRAFT contract notes", description)
        self.assertIn("Tool Refinement contract deltas", prompt)
        self.assertIn("Use observed identifiers only.", prompt)

    def test_atomic_store_failure_preserves_previous_skill_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="atomic",
                store_path=Path(tmp) / "skills.json",
            )
            state = module.store.load()
            state.iteration = 1
            module.store.save(state)
            previous = module.store.state_path.read_text(encoding="utf-8")
            state.iteration = 2

            with patch(
                "agent.utils.atomic.os.replace",
                side_effect=OSError("simulated interruption"),
            ):
                with self.assertRaisesRegex(OSError, "simulated interruption"):
                    module.store.save(state)

            self.assertEqual(
                module.store.state_path.read_text(encoding="utf-8"),
                previous,
            )
            self.assertEqual(list(Path(tmp).glob(".*.tmp")), [])

    def test_batch_gradient_and_best_of_n_use_independent_llm_calls(self) -> None:
        schemas: list[type] = []
        prompts: list[str] = []
        candidate_index = 0

        class FakeModel:
            schema: type | None = None

            def with_structured_output(self, schema):
                self.schema = schema
                schemas.append(schema)
                return self

            def invoke(self, prompt):
                nonlocal candidate_index
                prompts.append(prompt)
                if self.schema is SkillCandidateDraft:
                    index = candidate_index
                    candidate_index += 1
                    return SkillCandidateDraft(
                        title=f"Candidate {index}",
                        initiation="When current route evidence is incomplete.",
                        policy=[
                            f"Inspect route evidence with independent strategy {index}.",
                            "Cross-check the observation before termination.",
                        ],
                        termination="Stop after route evidence is independently confirmed.",
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
                    critique="Route evidence needs a consistent cross-check.",
                    proposed_update="Require an independent route observation.",
                    policy=[
                        "Inspect route evidence.",
                        "Cross-check the route observation.",
                    ],
                    termination="Stop after the route observation is confirmed.",
                )

        def evidence(session_id: str) -> EvaluationEvidence:
            return EvaluationEvidence(
                session_id=session_id,
                task_description="Investigate a missing route using current evidence.",
                scenario="routing_family",
                metrics={
                    "detection_score": 1.0,
                    "localization_accuracy": 1.0,
                    "rca_accuracy": 1.0,
                },
                steps=4,
                tool_calls=2,
                success=True,
            )

        steps = [
            SkillStep(
                order=1,
                action="Inspect route evidence.",
                tool_name="show_route",
                observation_summary="The expected route is absent.",
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            fake_model = FakeModel()
            module = ProceduralMemoryModule(
                bank_id="batch",
                llm_backend="custom",
                model="test-model",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=2,
                best_of_n=3,
                policy_scorer=BehavioralReplayPolicyScorer(lambda: fake_model),
            )
            self._seed_running_baseline(module, "routing_family")
            with patch(
                "agent.procedural_memory.service.load_model", return_value=fake_model
            ) as load_model:
                first = module.learn_from_episode(
                    evidence=evidence("batch-1"), tool_steps=steps
                )
                second = module.learn_from_episode(
                    evidence=evidence("batch-2"), tool_steps=steps
                )
                third = module.learn_from_episode(
                    evidence=evidence("batch-3"), tool_steps=steps
                )

        self.assertEqual(first["status"], "deferred")
        self.assertEqual(second["status"], "accepted")
        self.assertEqual(third["status"], "deferred")
        self.assertEqual(second["semantic_gradient_count"], 2)
        self.assertEqual(second["decision"]["best_of_n"], 3)
        self.assertEqual(second["verification_method"], "behavioral_replay")
        self.assertEqual(schemas.count(SkillCandidateDraft), 3)
        self.assertEqual(sum("Skill Evolver" in prompt for prompt in prompts), 3)
        self.assertTrue(
            all(
                "Tool Refinement owns parameter schemas" in prompt
                for prompt in prompts
                if "Skill Evolver" in prompt
            )
        )
        self.assertTrue(
            any(
                "Keep tool parameter schemas" in prompt
                for prompt in prompts
                if "semantic-gradient critic" in prompt
            )
        )
        self.assertTrue(any("batch semantic-gradient aggregator" in p for p in prompts))
        load_model.assert_called_once()

    def test_attribute_mining_ignores_scenario_design_noise(self) -> None:
        attrs = infer_procedural_memory_attributes(
            (
                "Network Description: OSPF enterprise network with DHCP, DNS, "
                "HTTP web services and load balancer.\n\n"
                "Your goal is to analyze the network condition."
            ),
            scenario="ospf_enterprise_dhcp",
            topology_class="s",
            tools=[],
        )

        self.assertNotIn("ospf", attrs.protocols)
        self.assertNotIn("dhcp", attrs.protocols)
        self.assertNotIn("addressing", attrs.services)

    def test_learning_reward_is_macro_average_of_published_quality_metrics(
        self,
    ) -> None:
        detection_only = EvaluationEvidence(
            session_id="s-detect-only",
            metrics={
                "detection_score": 1.0,
                "localization_accuracy": 0.0,
                "localization_f1": 0.0,
                "rca_accuracy": 0.0,
                "rca_f1": 0.0,
            },
            steps=2,
            tool_calls=1,
            success=False,
        )
        partial = detection_only.model_copy(
            update={
                "metrics": {
                    "detection_score": 1.0,
                    "localization_accuracy": 1.0,
                    "localization_f1": 1.0,
                    "rca_accuracy": 0.0,
                    "rca_f1": 0.0,
                },
            }
        )
        complete = detection_only.model_copy(
            update={
                "metrics": {
                    "detection_score": 1.0,
                    "localization_accuracy": 1.0,
                    "localization_f1": 1.0,
                    "rca_accuracy": 1.0,
                    "rca_f1": 1.0,
                },
                "success": True,
            }
        )

        self.assertAlmostEqual(_evidence_score(detection_only), 1 / 3)
        self.assertAlmostEqual(_evidence_score(partial), 2 / 3)
        self.assertAlmostEqual(_evidence_score(complete), 1.0)
        expensive = complete.model_copy(update={"steps": 100, "tool_calls": 50})
        self.assertEqual(_evidence_score(expensive), _evidence_score(complete))

    def test_persisted_episode_redacts_hidden_answer_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            module.learn_from_episode(
                evidence=EvaluationEvidence(
                    session_id="s1",
                    task_description="BGP route is missing between leaves.",
                    scenario="dc_clos_bgp",
                    root_cause=["bgp_missing_route_advertisement"],
                    faulty_devices=["leaf_router_0_1"],
                    metrics={
                        "detection_score": 1.0,
                        "localization_accuracy": 1.0,
                        "rca_accuracy": 1.0,
                    },
                    steps=5,
                    tool_calls=2,
                    success=True,
                ),
                tool_steps=[
                    SkillStep(
                        order=1,
                        action="Check BGP neighbors.",
                        tool_name="frr_show_bgp_summary",
                    )
                ],
            )
            state = module.store.load()
            snapshot = "\n".join(module.store.snapshot_jsonl())

        self.assertEqual(state.episodes[0].root_cause, [])
        self.assertEqual(state.episodes[0].faulty_devices, [])
        self.assertNotIn("bgp_missing_route_advertisement", snapshot)
        self.assertNotIn("leaf_router_0_1", snapshot)

    def test_extracts_react_diagnosis_phase_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "messages.jsonl"
            rows = [
                {
                    "agent": "diagnosis",
                    "event": "tool_start",
                    "run_id": "1",
                    "tool": {"name": "ping_pair"},
                    "input": "{'host_a': 'pc1', 'host_b': 'dns'}",
                },
                {
                    "agent": "diagnosis",
                    "event": "tool_end",
                    "run_id": "1",
                    "output": (
                        "2 packets received\n\n"
                        "[Integrated learning guidance - not evidence]\n"
                        "Active Skill-MDP option: seed."
                    ),
                },
                {
                    "agent": "submission",
                    "event": "tool_start",
                    "tool": {"name": "submit"},
                    "input": "{}",
                },
            ]
            trace.write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )

            steps = extract_skill_steps(trace)

        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].tool_name, "ping_pair")
        self.assertEqual(steps[0].status, "success")
        self.assertIn("2 packets received", steps[0].observation_summary)
        self.assertNotIn("Integrated learning guidance", steps[0].observation_summary)

    def test_extracts_runtime_skill_transitions_before_plain_tool_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "messages.jsonl"
            policy_state = "state:" + ("s" * 4500)
            policy_context = "context:" + ("c" * 8500)
            rows = [
                {
                    "agent": "diagnosis",
                    "event": "tool_start",
                    "run_id": "1",
                    "tool": {"name": "ping_pair"},
                    "input": "{'host_a': 'pc1', 'host_b': 'pc2'}",
                },
                {
                    "agent": "diagnosis",
                    "event": "tool_end",
                    "run_id": "1",
                    "output": "plain callback output",
                },
                {
                    "agent": "procedural_memory_agent",
                    "phase": "skill_mdp_runtime",
                    "event": "skill_transition",
                    "active_skill_id": "seed_react_decision",
                    "activation_id": "session:1",
                    "tool": "ping_pair",
                    "tool_input": {"host_a": "pc1", "host_b": "pc2"},
                    "status": "success",
                    "observation_summary": "runtime interpreted output",
                    "policy_state": policy_state,
                    "policy_context": policy_context,
                },
            ]
            trace.write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )

            steps = extract_skill_steps(trace)

        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].skill_id, "seed_react_decision")
        self.assertEqual(steps[0].activation_id, "session:1")
        self.assertEqual(steps[0].tool_name, "ping_pair")
        self.assertEqual(steps[0].arguments_hint["host_a"], "pc1")
        self.assertIn("runtime interpreted output", steps[0].observation_summary)
        self.assertEqual(steps[0].policy_state, policy_state)
        self.assertEqual(steps[0].policy_context, policy_context)

    def test_ppo_gate_accepts_successful_skill_and_retrieves_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=1,
            )
            self._seed_running_baseline(module, "dc_clos_bgp")
            first = module.learn_from_episode(
                evidence=EvaluationEvidence(
                    session_id="s1",
                    task_description="BGP missing route advertisement between routers",
                    scenario="dc_clos_bgp",
                    root_cause=["bgp_missing_route_advertisement"],
                    metrics={
                        "detection_score": 1.0,
                        "localization_accuracy": 0.5,
                        "rca_accuracy": 0.5,
                    },
                    steps=8,
                    tool_calls=5,
                    success=True,
                ),
                tool_steps=[
                    SkillStep(
                        order=1,
                        action="Inspect BGP routes.",
                        tool_name="frr_show_bgp_summary",
                    )
                ],
            )
            report = module.learn_from_episode(
                evidence=EvaluationEvidence(
                    session_id="s2",
                    task_description="BGP missing route advertisement between routers",
                    scenario="dc_clos_bgp",
                    ground_truth_is_anomaly=True,
                    metrics={
                        "detection_score": 1.0,
                        "localization_accuracy": 1.0,
                        "rca_accuracy": 1.0,
                    },
                    steps=6,
                    tool_calls=4,
                    success=True,
                ),
                tool_steps=[
                    SkillStep(
                        order=1,
                        action="Inspect BGP routes.",
                        tool_name="frr_show_bgp_summary",
                    )
                ],
            )
            retrieved = module.retrieve(
                query=ProceduralMemoryQuery(
                    text="BGP route is not advertised",
                    scenario="dc_clos_bgp",
                    protocols=["bgp"],
                    symptoms=["missing_route"],
                    tools=["frr_show_bgp_summary"],
                    top_k=3,
                )
            )
            context = module.format_context(retrieved)
            last_evolution = module.store.load().evolution_log[-1]

        self.assertEqual(first["status"], "accepted")
        self.assertEqual(report["status"], "accepted")
        self.assertEqual(
            last_evolution["sample_experience_ids"],
            last_evolution["verification_experience_ids"],
        )
        self.assertGreater(report["episode_reward"], 0.0)
        self.assertGreater(report["episode_baseline"], 0.0)
        self.assertEqual(
            report["episode_advantage"],
            report["episode_reward"] - report["episode_baseline"],
        )
        self.assertTrue(report["episode_success"])
        self.assertIn(report["skill_id"], [item.skill.skill_id for item in retrieved])
        self.assertIn("Activation", context)
        self.assertNotIn("bgp_missing_route_advertisement", context)

    def test_retrieval_blocks_learned_skill_without_current_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            state = module.store.load()
            state.skills["missing_ip_skill"] = ProceduralSkill(
                skill_id="missing_ip_skill",
                title="Missing IP procedure",
                activation_condition="Use when current evidence shows a host has no IPv4 address.",
                execution_steps=[
                    SkillStep(
                        order=1,
                        action="Check the host interface address.",
                        tool_name="get_host_net_config",
                    )
                ],
                termination_condition="Stop after IP assignment evidence is collected.",
                scenarios=["ospf_enterprise_dhcp"],
                services=["addressing"],
                symptoms=["missing_ip"],
                tools=["get_host_net_config"],
                status="validated",
                score=1.0,
            )
            module.store.save(state)

            retrieved = module.retrieve(
                query=ProceduralMemoryQuery(
                    text="Generic enterprise network diagnosis with no current observations.",
                    scenario="ospf_enterprise_dhcp",
                    top_k=5,
                )
            )

        self.assertNotIn(
            "missing_ip_skill", [item.skill.skill_id for item in retrieved]
        )

    def test_retrieval_blocks_symptom_mismatched_learned_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            state = module.store.load()
            state.skills["missing_ip_skill"] = ProceduralSkill(
                skill_id="missing_ip_skill",
                title="Missing IP procedure",
                activation_condition="Use when current evidence shows a host has no IPv4 address.",
                execution_steps=[
                    SkillStep(
                        order=1,
                        action="Check the host interface address.",
                        tool_name="get_host_net_config",
                    )
                ],
                termination_condition="Stop after IP assignment evidence is collected.",
                scenarios=["ospf_enterprise_dhcp"],
                services=["addressing"],
                symptoms=["missing_ip"],
                tools=["get_host_net_config"],
                status="validated",
                score=1.0,
            )
            module.store.save(state)

            retrieved = module.retrieve(
                query=ProceduralMemoryQuery(
                    text="Current evidence shows the host uses an incorrect default gateway.",
                    scenario="ospf_enterprise_dhcp",
                    services=["addressing"],
                    symptoms=["bad_gateway"],
                    tools=["get_host_net_config"],
                    top_k=5,
                )
            )

        self.assertNotIn(
            "missing_ip_skill", [item.skill.skill_id for item in retrieved]
        )

    def test_retrieval_prefers_activation_fit_over_tool_catalog_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            state = module.store.load()
            state.skills["dns_policy"] = ProceduralSkill(
                skill_id="dns_policy",
                title="Name lookup policy",
                activation_condition=(
                    "Use when current observations show DNS resolver, name lookup, "
                    "nameserver, SERVFAIL, or port 53 symptoms."
                ),
                execution_steps=[
                    SkillStep(
                        order=1,
                        action="Run direct lookup timing.",
                        tool_name="curl_web_test",
                    )
                ],
                termination_condition="Stop after lookup/config/service evidence.",
                protocols=["dns"],
                services=["name_resolution"],
                symptoms=["lookup_failure"],
                tools=["curl_web_test", "systemctl_ops"],
                status="validated",
                score=0.4,
            )
            state.skills["bgp_tool_overlap"] = ProceduralSkill(
                skill_id="bgp_tool_overlap",
                title="BGP tool overlap",
                activation_condition=(
                    "Use when current observations show BGP neighbor or route symptoms."
                ),
                execution_steps=[
                    SkillStep(
                        order=1,
                        action="Check BGP neighbors.",
                        tool_name="frr_show_bgp_summary",
                    )
                ],
                termination_condition="Stop after BGP neighbor and route evidence.",
                protocols=["bgp"],
                services=["routing"],
                symptoms=["missing_route"],
                tools=["curl_web_test", "systemctl_ops"],
                status="validated",
                score=0.9,
            )
            module.store.save(state)

            retrieved = module.retrieve(
                query=ProceduralMemoryQuery(
                    text=(
                        "curl_web_test shows DNS name lookup SERVFAIL from "
                        "the configured resolver."
                    ),
                    protocols=["dns"],
                    services=["name_resolution"],
                    symptoms=["lookup_failure"],
                    tools=["curl_web_test", "systemctl_ops"],
                    top_k=2,
                )
            )

        self.assertTrue(retrieved)
        self.assertEqual(retrieved[0].skill.skill_id, "dns_policy")
        self.assertNotIn(
            "bgp_tool_overlap", [item.skill.skill_id for item in retrieved]
        )

    def test_partial_rca_episode_uses_continuous_advantage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=1,
            )
            self._seed_running_baseline(module, "ospf_enterprise_dhcp")
            before = set(module.store.load().skills)
            report = module.learn_from_episode(
                evidence=EvaluationEvidence(
                    session_id="partial",
                    task_description="Host reachability failure.",
                    scenario="ospf_enterprise_dhcp",
                    metrics={
                        "detection_score": 1.0,
                        "localization_accuracy": 1.0,
                        "localization_f1": 1.0,
                        "rca_accuracy": 0.0,
                        "rca_f1": 0.0,
                    },
                    steps=6,
                    tool_calls=4,
                    success=False,
                ),
                tool_steps=[
                    SkillStep(
                        order=1,
                        action="Check host network configuration.",
                        tool_name="get_host_net_config",
                        observation_summary="Host has no IPv4 address.",
                    )
                ],
            )
            after = set(module.store.load().skills)

        self.assertEqual(report["status"], "accepted")
        self.assertEqual(report["semantic_gradient_count"], 1)
        self.assertGreater(len(after), len(before))
        self.assertIsNotNone(report["decision"])
        self.assertGreater(report["episode_advantage"], 0.0)

    def test_partial_localization_episode_remains_a_ranked_experience(
        self,
    ) -> None:
        metrics = {
            "detection_score": 1.0,
            "localization_accuracy": 0.0,
            "localization_precision": 1.0,
            "localization_recall": 0.5,
            "localization_f1": 0.6667,
            "rca_accuracy": 1.0,
            "rca_f1": 1.0,
        }
        self.assertFalse(_metric_success(metrics))
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=1,
            )
            self._seed_running_baseline(module, "ospf_enterprise_dhcp")
            before = set(module.store.load().skills)
            report = module.learn_from_episode(
                evidence=EvaluationEvidence(
                    session_id="partial-localization",
                    task_description="Enterprise reachability failure.",
                    scenario="ospf_enterprise_dhcp",
                    metrics=metrics,
                    steps=7,
                    tool_calls=5,
                    success=True,
                ),
                tool_steps=[
                    SkillStep(
                        order=1,
                        action="Check host and route evidence.",
                        tool_name="get_host_net_config",
                        observation_summary="Only one affected component was localized.",
                    )
                ],
            )
            state = module.store.load()
            after = set(state.skills)
            experience = state.experiences[-1]

        self.assertEqual(report["status"], "accepted")
        self.assertEqual(report["semantic_gradient_count"], 1)
        self.assertFalse(report["episode_success"])
        self.assertFalse(experience.success)
        self.assertGreater(experience.reward, 0.0)
        self.assertEqual(len(state.golden_experiences), 1)
        self.assertAlmostEqual(
            state.baselines["ospf_enterprise_dhcp"],
            0.1 * experience.reward,
        )
        self.assertGreater(len(after), len(before))
        self.assertIsNotNone(report["decision"])

    def test_legacy_partial_experience_is_repaired_before_new_learning(self) -> None:
        partial_metrics = {
            "detection_score": 1.0,
            "localization_accuracy": 0.0,
            "localization_precision": 1.0,
            "localization_recall": 0.5,
            "localization_f1": 0.6667,
            "rca_accuracy": 1.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            state = module.store.load()
            state.episodes.append(
                EvaluationEvidence(
                    session_id="legacy-partial",
                    task_description="Legacy partial run.",
                    scenario="enterprise",
                    metrics=partial_metrics,
                    steps=7,
                    tool_calls=5,
                    success=True,
                )
            )
            legacy = SkillExperience(
                experience_id="exp-legacy-partial",
                session_id="legacy-partial",
                reward=0.75,
                baseline=0.0,
                advantage=0.75,
                transitions=[
                    SkillTransition(
                        state="Legacy partial run.",
                        action="Check broad evidence.",
                        tool_name="get_host_net_config",
                        observation_summary="Only partial localization was available.",
                        status="success",
                        done=True,
                    )
                ],
                success=True,
            )
            state.experiences.append(legacy)
            state.golden_experiences.append(legacy.model_copy(deep=True))
            module.store.save(state)

            module.learn_from_episode(
                evidence=EvaluationEvidence(
                    session_id="new-safe",
                    task_description="BGP route is missing.",
                    scenario="enterprise",
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
                        action="Inspect current BGP evidence.",
                        tool_name="frr_show_bgp_summary",
                    )
                ],
            )
            state = module.store.load()
            repaired = next(
                item
                for item in state.experiences
                if item.experience_id == "exp-legacy-partial"
            )

        self.assertFalse(repaired.success)
        self.assertGreater(repaired.reward, 0.0)
        self.assertIn(
            "exp-legacy-partial",
            {item.experience_id for item in state.golden_experiences},
        )
        self.assertTrue(
            any(
                entry.get("stage") == "normalize unsafe experience"
                for entry in state.maintenance_log
            )
        )

    def test_partial_episode_penalizes_reused_skill_online_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=1,
            )
            state = module.store.load()
            state.skills["active_bad_skill"] = ProceduralSkill(
                skill_id="active_bad_skill",
                title="Over-broad reachability policy",
                activation_condition="Use when current evidence shows host reachability failure.",
                execution_steps=[
                    SkillStep(
                        order=1,
                        action="Check broad reachability.",
                        tool_name="get_reachability",
                    )
                ],
                termination_condition="Stop after detecting an anomaly.",
                services=["routing"],
                symptoms=["reachability_loss"],
                tools=["get_reachability"],
                status="validated",
                score=0.6,
            )
            state.baselines["enterprise"] = 0.9
            module.store.save(state)

            report = module.learn_from_episode(
                evidence=EvaluationEvidence(
                    session_id="partial-reuse",
                    task_description="Host reachability failure.",
                    scenario="enterprise",
                    metrics={
                        "detection_score": 1.0,
                        "localization_accuracy": 1.0,
                        "rca_accuracy": 0.0,
                    },
                    steps=8,
                    tool_calls=5,
                    success=False,
                ),
                tool_steps=[
                    SkillStep(
                        order=1,
                        action="Check broad reachability.",
                        skill_id="active_bad_skill",
                        tool_name="get_reachability",
                        observation_summary="Reachability is unknown.",
                    )
                ],
            )
            skill = module.store.load().skills["active_bad_skill"]

        self.assertEqual(report["status"], "rejected")
        self.assertLess(skill.avg_gain, 0.0)
        self.assertEqual(skill.success_count, 0)
        self.assertEqual(skill.failure_count, 1)

    def test_deterministic_semantic_gradient_updates_components_for_partial_outcome(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            gradient = module.semantic_gradient(
                evidence=EvaluationEvidence(
                    session_id="partial-gradient",
                    task_description="Host reachability failure.",
                    metrics={
                        "detection_score": 1.0,
                        "localization_accuracy": 1.0,
                        "rca_accuracy": 0.0,
                    },
                    success=False,
                    steps=6,
                    tool_calls=4,
                ),
                tool_steps=[
                    SkillStep(
                        order=1,
                        action="Check reachability.",
                        tool_name="get_reachability",
                    )
                ],
            )

        self.assertIn("localization/RCA", gradient.critique)
        self.assertIn(
            "current observation history", gradient.component_update.initiation
        )
        self.assertIn("detection-only", gradient.component_update.termination)

    def test_proposed_skill_activation_uses_evidence_signature_not_scenario(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            skill = module.propose_skill(
                evidence=EvaluationEvidence(
                    session_id="general-skill",
                    task_description="Diagnose missing route.",
                    scenario="benchmark_specific_scenario",
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
                        action="Check route table.",
                        tool_name="frr_show_ip_route",
                        observation_summary="Route is missing.",
                    )
                ],
            )

        self.assertIn("evidence signature", skill.activation_condition)
        self.assertIn("observed tools", skill.activation_condition)
        self.assertNotIn("benchmark_specific_scenario", skill.activation_condition)
        self.assertNotIn("benchmark_specific_scenario", skill.skill_id)

    def test_refining_generic_seed_uses_evidence_signature_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            parent = module.store.load().skills["seed_explore_exploit"]

            skill = module.propose_skill(
                evidence=EvaluationEvidence(
                    session_id="refine-generic",
                    task_description="BGP route is missing.",
                    scenario="dc_clos_bgp",
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
                        action="Inspect BGP route state.",
                        tool_name="frr_show_bgp_summary",
                        observation_summary="BGP route is missing.",
                    )
                ],
                parent=parent,
            )

        self.assertIn("evidence signature", skill.activation_condition)
        self.assertIn("bgp", skill.activation_condition.lower())
        self.assertNotIn(
            "deciding between broad exploration", skill.activation_condition
        )

    def test_runtime_context_does_not_label_candidate_as_active_without_active_skill(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            state = module.store.load()
            state.skills["candidate_ping"] = ProceduralSkill(
                skill_id="candidate_ping",
                title="Candidate ping",
                activation_condition="Use for host reachability checks.",
                execution_steps=[
                    SkillStep(order=1, action="Call ping_pair.", tool_name="ping_pair")
                ],
                termination_condition="Stop after reachability evidence.",
                tools=["ping_pair"],
                status="validated",
                score=2.0,
            )
            module.store.save(state)
            retrieved = module.retrieve(
                query=ProceduralMemoryQuery(
                    text="Host reachability failure",
                    scenario="simple_bgp",
                    tools=["ping_pair"],
                    top_k=3,
                )
            )

            context = module.format_context(retrieved, active_skill_id="")

        self.assertIn("CANDIDATE Skill candidate_ping", context)
        self.assertNotIn("ACTIVE Skill candidate_ping", context)

    def test_unvalidated_candidate_is_not_retrievable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            state = module.store.load()
            state.skills["unverified"] = ProceduralSkill(
                skill_id="unverified",
                title="Unverified procedure",
                activation_condition="Use for current reachability evidence.",
                execution_steps=[
                    SkillStep(order=1, action="Inspect current reachability.")
                ],
                termination_condition="Stop after one observation.",
                status="candidate",
                score=100.0,
            )
            module.store.save(state)

            retrieved = module.retrieve(
                query=ProceduralMemoryQuery(
                    text="Current reachability evidence", top_k=20
                )
            )

        self.assertNotIn("unverified", {item.skill.skill_id for item in retrieved})

    def test_loading_bank_restores_seed_without_metric_specific_quarantine(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "skills.json"
            module = ProceduralMemoryModule(bank_id="skill", store_path=path)
            state = module.store.load()
            state.skills["seed_react_decision"].status = "retired"
            state.episodes.append(
                EvaluationEvidence(
                    session_id="failed-source",
                    metrics={
                        "detection_score": 1.0,
                        "localization_accuracy": 1.0,
                        "rca_accuracy": 0.0,
                    },
                )
            )
            state.skills["unsafe-learned"] = ProceduralSkill(
                skill_id="unsafe-learned",
                title="Unsafe learned procedure",
                activation_condition="Use after one broad observation.",
                execution_steps=[SkillStep(order=1, action="Stop early.")],
                termination_condition="Stop immediately.",
                source_sessions=["failed-source"],
                status="validated",
                origin="learned",
            )
            module.store.save(state)

            reloaded = ProceduralMemoryModule(bank_id="skill", store_path=path)
            repaired = reloaded.store.load()

        self.assertEqual(repaired.skills["seed_react_decision"].status, "validated")
        self.assertEqual(repaired.skills["unsafe-learned"].status, "validated")

    def test_runtime_context_redacts_known_root_cause_ids_from_dirty_skills(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            state = module.store.load()
            state.skills["dirty_dns"] = ProceduralSkill(
                skill_id="dirty_dns",
                title="Dirty DNS skill",
                activation_condition="Use when dns_record_error is suspected.",
                execution_steps=[
                    SkillStep(
                        order=1,
                        action="Confirm dns_record_error with DNS checks.",
                        tool_name="dig_query",
                    )
                ],
                termination_condition="Stop after dns_record_error is proven.",
                tools=["dig_query"],
                status="validated",
                score=2.0,
            )
            module.store.save(state)
            retrieved = module.retrieve(
                query=ProceduralMemoryQuery(
                    text="DNS lookup failure",
                    services=["dns"],
                    tools=["dig_query"],
                    top_k=1,
                )
            )

            context = module.format_context(retrieved)

        self.assertIn("[redacted]", context)
        self.assertNotIn("dns_record_error", context)

    def test_default_learning_defers_evolution_until_batch_is_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            report = module.learn_from_episode(
                evidence=EvaluationEvidence(
                    session_id="s1",
                    task_description="BGP missing route advertisement between routers",
                    scenario="dc_clos_bgp",
                    metrics={
                        "detection_score": 1.0,
                        "localization_accuracy": 1.0,
                        "rca_accuracy": 1.0,
                    },
                    steps=8,
                    tool_calls=5,
                    success=True,
                ),
                tool_steps=[
                    SkillStep(
                        order=1,
                        action="Inspect BGP routes.",
                        tool_name="frr_show_bgp_summary",
                    )
                ],
            )
            state = module.store.load()
            stats = module.store.bank_stats()

        self.assertEqual(report["status"], "deferred")
        self.assertEqual(report["sample_count"], 1)
        self.assertEqual(report["required_sample_count"], 6)
        self.assertEqual(report["decision"], None)
        self.assertEqual(len(state.experiences), 1)
        self.assertEqual(stats["ppo_decisions"], 0)
        self.assertEqual(state.evolution_log[-1]["action"], "deferred")

    def test_evolution_batch_does_not_reuse_consumed_experiences(self) -> None:
        def evidence(session_id: str) -> EvaluationEvidence:
            return EvaluationEvidence(
                session_id=session_id,
                task_description="BGP missing route advertisement between routers",
                scenario="dc_clos_bgp",
                metrics={
                    "detection_score": 1.0,
                    "localization_accuracy": 1.0,
                    "rca_accuracy": 1.0,
                },
                steps=8,
                tool_calls=5,
                success=True,
            )

        steps = [
            SkillStep(
                order=1,
                action="Inspect BGP routes.",
                tool_name="frr_show_bgp_summary",
            )
        ]

        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=2,
                policy_scorer=StructuredReplayPolicyScorer(),
            )
            self._seed_running_baseline(module, "dc_clos_bgp")
            first = module.learn_from_episode(
                evidence=evidence("s1"),
                tool_steps=steps,
            )
            second = module.learn_from_episode(
                evidence=evidence("s2"),
                tool_steps=[
                    steps[0].model_copy(update={"skill_id": first["skill_id"]})
                ],
            )
            third = module.learn_from_episode(
                evidence=evidence("s3"),
                tool_steps=steps,
            )
            state = module.store.load()
            used = [exp for exp in state.experiences if exp.used_for_evolution]
            unused = [exp for exp in state.experiences if not exp.used_for_evolution]
            gate_events = [
                item for item in state.evolution_log if item["action"] != "deferred"
            ]

        self.assertEqual(first["status"], "deferred")
        self.assertEqual(second["status"], "accepted")
        self.assertEqual(third["status"], "deferred")
        self.assertEqual(len(used), 2)
        self.assertEqual(len(unused), 1)
        self.assertEqual(len(gate_events), 1)
        self.assertEqual(
            set(gate_events[0]["sample_experience_ids"]),
            {exp.experience_id for exp in used},
        )

    def test_evolution_batch_returns_full_odd_threshold_when_pool_is_large(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=3,
            )
            state = module.store.load()
            parent = state.skills["seed_explore_exploit"]
            for index, reward in enumerate([0.1, 0.3, 0.5, 0.7, 0.9], start=1):
                state.experiences.append(
                    SkillExperience(
                        experience_id=f"exp-{index}",
                        session_id=f"s{index}",
                        reward=reward,
                        baseline=0.0,
                        advantage=reward,
                        skill_ids=[parent.skill_id],
                        transitions=[
                            SkillTransition(
                                state="BGP route missing",
                                action="Inspect BGP routes.",
                                tool_name="frr_show_bgp_summary",
                                observation_summary="Route evidence collected.",
                                status="success",
                                done=True,
                            )
                        ],
                        success=True,
                    )
                )

            batch = module._evolution_batch(state, parent)

        self.assertEqual(len(batch), 3)
        self.assertEqual(
            {experience.experience_id for experience in batch},
            {"exp-1", "exp-4", "exp-5"},
        )

    def test_generic_seed_evolution_batch_uses_parent_buffer_across_domains(
        self,
    ) -> None:
        def experience(
            exp_id: str,
            reward: float,
            *,
            action: str,
            tool_name: str,
            observation: str,
        ) -> SkillExperience:
            return SkillExperience(
                experience_id=exp_id,
                session_id=exp_id,
                reward=reward,
                baseline=0.0,
                advantage=reward,
                skill_ids=["seed_explore_exploit"],
                trajectory=action,
                transitions=[
                    SkillTransition(
                        state=action,
                        action=action,
                        tool_name=tool_name,
                        observation_summary=observation,
                        status="success",
                        done=True,
                    )
                ],
                success=True,
            )

        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=3,
            )
            state = module.store.load()
            parent = state.skills["seed_explore_exploit"]
            current = experience(
                "bgp-current",
                0.9,
                action="Inspect BGP route advertisement.",
                tool_name="frr_show_bgp_summary",
                observation="BGP route is missing.",
            )
            state.experiences.extend(
                [
                    experience(
                        "bgp-low",
                        0.2,
                        action="Check BGP neighbor state.",
                        tool_name="frr_show_bgp_summary",
                        observation="BGP neighbor and route evidence collected.",
                    ),
                    experience(
                        "ospf-high",
                        0.8,
                        action="Check OSPF neighbor state.",
                        tool_name="frr_get_ospf_conf",
                        observation="OSPF neighbor missing.",
                    ),
                    experience(
                        "p4-high",
                        0.7,
                        action="Inspect P4 table.",
                        tool_name="bmv2_table_dump",
                        observation="P4 table entry missing.",
                    ),
                    current,
                    experience(
                        "bgp-mid",
                        0.6,
                        action="Inspect BGP route table.",
                        tool_name="frr_show_ip_route",
                        observation="BGP missing route evidence.",
                    ),
                ]
            )

            batch = module._evolution_batch(state, parent, current=current)

        self.assertEqual(len(batch), 3)
        self.assertEqual(
            {item.experience_id for item in batch},
            {"bgp-low", "bgp-current", "ospf-high"},
        )

    def test_parent_buffer_is_not_partitioned_by_evidence_family(self) -> None:
        def experience(exp_id: str, text: str) -> SkillExperience:
            return SkillExperience(
                experience_id=exp_id,
                session_id=exp_id,
                reward=1.0,
                advantage=1.0,
                skill_ids=["seed_react_decision"],
                trajectory=text,
                transitions=[
                    SkillTransition(
                        action="Inspect current reachability.",
                        tool_name="get_reachability",
                        observation_summary=text,
                        status="success",
                        done=True,
                    )
                ],
                success=True,
            )

        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=3,
            )
            state = module.store.load()
            parent = state.skills["seed_react_decision"]
            current = experience(
                "service-current",
                "DNS resolver and DHCP server are reachable over ICMP.",
            )
            state.experiences.extend(
                [
                    current,
                    experience("service-2", "DNS lookup and DHCP service evidence."),
                    experience(
                        "service-3", "DNS server response and DHCP lease evidence."
                    ),
                    experience(
                        "p4-no-fault", "P4 hosts have successful ICMP reachability."
                    ),
                ]
            )

            batch = module._evolution_batch(state, parent, current=current)

        self.assertEqual(len(batch), 3)
        self.assertIn("p4-no-fault", {item.experience_id for item in batch})

    def test_refining_generic_seed_preserves_seed_and_resets_candidate_stats(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=1,
            )
            self._seed_running_baseline(module, "routing")
            first = module.learn_from_episode(
                evidence=EvaluationEvidence(
                    session_id="successful-refinement-holdout",
                    task_description="A route is missing after a neighbor failure.",
                    scenario="routing",
                    metrics={
                        "detection_score": 1.0,
                        "localization_accuracy": 0.5,
                        "rca_accuracy": 0.5,
                    },
                    steps=3,
                    tool_calls=2,
                    success=True,
                ),
                tool_steps=[
                    SkillStep(
                        order=1,
                        skill_id="seed_react_decision",
                        action="Inspect route state.",
                        tool_name="show_route",
                        observation_summary="The expected route is missing.",
                    )
                ],
            )
            report = module.learn_from_episode(
                evidence=EvaluationEvidence(
                    session_id="successful-refinement",
                    task_description="A route is missing after a neighbor failure.",
                    scenario="routing",
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
                    SkillStep(
                        order=1,
                        skill_id="seed_react_decision",
                        action="Inspect route state.",
                        tool_name="show_route",
                        observation_summary="The expected route is missing.",
                    )
                ],
            )
            state = module.store.load()
            candidate = state.skills[report["skill_id"]]

        self.assertEqual(first["status"], "accepted")
        self.assertEqual(report["status"], "accepted")
        self.assertEqual(state.skills["seed_react_decision"].status, "validated")
        self.assertIsNone(report["decision"]["replaced_skill_id"])
        self.assertEqual(candidate.success_count, 0)
        self.assertEqual(candidate.failure_count, 0)
        self.assertEqual(candidate.frequency, 0)
        self.assertEqual(candidate.maturity, 0)
        self.assertEqual(
            report["decision"]["candidate_score"],
            report["decision"]["baseline_score"],
        )

    def test_seed_skill_pool_is_available_in_fresh_bank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=1,
            )
            retrieved = module.retrieve(
                query=ProceduralMemoryQuery(
                    text="Plan a diagnosis with little evidence",
                    scenario="simple_bgp",
                    top_k=3,
                )
            )
            state = module.store.load()
            context = module.format_context(retrieved)

        self.assertEqual(
            len(
                [skill_id for skill_id in state.skills if skill_id.startswith("seed_")]
            ),
            6,
        )
        self.assertNotIn("seed_bgp_config_disambiguation", state.skills)
        self.assertNotIn("seed_name_resolution_ladder", state.skills)
        self.assertNotIn("seed_host_addressing_ladder", state.skills)
        self.assertNotIn("seed_routing_adjacency_ladder", state.skills)
        self.assertTrue(
            any(item.skill.skill_id.startswith("seed_") for item in retrieved)
        )
        self.assertIn("Skill-MDP", context)

    def test_existing_expert_seed_entries_are_removed_from_a_bank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "skills.json"
            module = ProceduralMemoryModule(bank_id="skill", store_path=path)
            state = module.store.load()
            state.skills["legacy_expert_seed"] = ProceduralSkill(
                skill_id="legacy_expert_seed",
                title="Legacy expert seed",
                activation_condition="Legacy condition.",
                execution_steps=[SkillStep(order=1, action="Legacy action.")],
                termination_condition="Legacy termination.",
                origin="expert_seed",
                status="validated",
            )
            module.store.save(state)

            cleaned = ProceduralMemoryModule(bank_id="skill", store_path=path)

        self.assertNotIn("legacy_expert_seed", cleaned.store.load().skills)

    def test_failed_domain_specific_skill_does_not_dominate_other_domains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=1,
            )
            state = module.store.load()
            state.skills["overfit_ospf"] = ProceduralSkill(
                skill_id="overfit_ospf",
                title="Overfit OSPF",
                activation_condition="Use for OSPF DHCP DNS enterprise incidents.",
                execution_steps=[
                    SkillStep(order=1, action="Check OSPF, DHCP, and DNS first.")
                ],
                termination_condition="Stop after OSPF/DHCP/DNS checks.",
                scenarios=["ospf_enterprise_dhcp"],
                protocols=["ospf", "dhcp", "dns"],
                services=["routing", "name_resolution", "addressing"],
                tools=["frr_show_ip_route"],
                status="validated",
                score=0.95,
                frequency=10,
                success_count=1,
                failure_count=9,
                avg_gain=-0.05,
                total_gain=-0.5,
                maturity=10,
            )
            module.store.save(state)

            retrieved = module.retrieve(
                query=ProceduralMemoryQuery(
                    text="BGP route is not advertised",
                    scenario="dc_clos_bgp",
                    protocols=["bgp"],
                    services=["routing"],
                    tools=["frr_show_bgp_summary"],
                    top_k=3,
                )
            )

        self.assertTrue(retrieved)
        self.assertNotEqual(retrieved[0].skill.skill_id, "overfit_ospf")

    def test_unstable_seed_child_is_filtered_and_not_used_as_parent(self) -> None:
        bad_skill_id = "seed_react_decision_v1_badbad"
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=1,
            )
            state = module.store.load()
            state.skills[bad_skill_id] = ProceduralSkill(
                skill_id=bad_skill_id,
                title="Contaminated reachability child",
                activation_condition=(
                    "Observed ping from host to server returned Network is unreachable."
                ),
                execution_steps=[
                    SkillStep(
                        order=1,
                        action="Check DHCP, DNS, and OSPF before localizing.",
                        tool_name="ping_pair",
                    )
                ],
                termination_condition="Stop after broad enterprise checks.",
                parent_id="seed_react_decision",
                version=1,
                protocols=["bgp", "ospf", "dhcp", "dns", "http", "icmp"],
                services=["routing", "name_resolution", "addressing"],
                symptoms=["unreachable"],
                tools=["ping_pair", "get_host_net_config"],
                status="validated",
                score=0.95,
                frequency=179,
                success_count=20,
                failure_count=57,
                total_gain=-3.58,
                avg_gain=-0.02,
                maturity=10,
            )
            module.store.save(state)

            retrieved = module.retrieve(
                query=ProceduralMemoryQuery(
                    text=(
                        "Observed ping from host to server returned Network is "
                        "unreachable while checking DHCP."
                    ),
                    protocols=["dhcp"],
                    services=["addressing"],
                    symptoms=["unreachable"],
                    tools=["ping_pair", "get_host_net_config"],
                    top_k=10,
                )
            )
            parent = module._runtime_parent_from_steps(
                module.store.load(),
                [
                    SkillStep(
                        order=1,
                        action="Follow contaminated policy.",
                        skill_id=bad_skill_id,
                        tool_name="ping_pair",
                        observation_summary="Network is unreachable.",
                    )
                ],
            )

        self.assertNotIn(bad_skill_id, [item.skill.skill_id for item in retrieved])
        self.assertIsNone(parent)

    def test_clean_episode_with_unstable_runtime_skill_learns_new_skill(self) -> None:
        bad_skill_id = "seed_react_decision_v1_badbad"
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=1,
            )
            state = module.store.load()
            state.skills[bad_skill_id] = ProceduralSkill(
                skill_id=bad_skill_id,
                title="Contaminated reachability child",
                activation_condition="Observed ping returned Network is unreachable.",
                execution_steps=[
                    SkillStep(order=1, action="Check too many unrelated systems.")
                ],
                termination_condition="Stop after broad checks.",
                parent_id="seed_react_decision",
                version=1,
                protocols=["ospf", "dhcp", "dns"],
                services=["routing", "name_resolution", "addressing"],
                symptoms=["unreachable"],
                tools=["ping_pair"],
                status="validated",
                score=0.95,
                frequency=179,
                success_count=20,
                failure_count=57,
                total_gain=-3.58,
                avg_gain=-0.02,
                maturity=10,
            )
            module.store.save(state)
            self._seed_running_baseline(module, "dc_clos_bgp")

            first = module.learn_from_episode(
                evidence=EvaluationEvidence(
                    session_id="clean-after-bad-parent",
                    task_description="BGP route is missing after endpoint checks.",
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
                        action="Inspect BGP route state.",
                        skill_id=bad_skill_id,
                        tool_name="frr_show_bgp_summary",
                        observation_summary="BGP route is missing.",
                    )
                ],
            )
            state = module.store.load()
            learned = state.skills[first["skill_id"]]

        self.assertEqual(first["status"], "accepted")
        self.assertNotEqual(learned.parent_id, bad_skill_id)
        self.assertEqual(state.skills[bad_skill_id].status, "retired")

    def test_verification_prefers_held_out_after_in_batch_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=1,
            )
            self._seed_running_baseline(module, "ospf")
            good = EvaluationEvidence(
                session_id="s1",
                task_description="OSPF neighbor missing",
                scenario="ospf",
                root_cause=["ospf_neighbor_missing"],
                metrics={
                    "detection_score": 1.0,
                    "localization_accuracy": 1.0,
                    "rca_accuracy": 1.0,
                },
                steps=5,
                tool_calls=3,
                success=True,
            )
            bad = good.model_copy(
                update={
                    "session_id": "s2",
                    "metrics": {
                        "detection_score": 1.0,
                        "localization_accuracy": 0.0,
                        "rca_accuracy": 0.0,
                    },
                    "steps": 40,
                    "tool_calls": 20,
                    "success": False,
                }
            )
            steps = [
                SkillStep(
                    order=1,
                    action="Check OSPF neighbors.",
                    tool_name="frr_show_ip_ospf_neighbor",
                )
            ]
            first = module.learn_from_episode(evidence=good, tool_steps=steps)
            second = module.learn_from_episode(evidence=bad, tool_steps=steps)
            event = module.store.load().evolution_log[-1]

        self.assertEqual(first["status"], "accepted")
        self.assertIn(second["status"], {"accepted", "rejected"})
        self.assertEqual(
            event["sample_experience_ids"], event["verification_experience_ids"]
        )

    def test_ppo_gate_uses_replayed_transition_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=1,
            )
            baseline = ProceduralSkill(
                skill_id="baseline_unrelated",
                title="Unrelated DHCP check",
                activation_condition="Use for address assignment issues.",
                execution_steps=[
                    SkillStep(order=1, action="Inspect DHCP lease state.")
                ],
                termination_condition="Stop after DHCP lease evidence.",
                tools=["dhcp_lease_dump"],
                status="validated",
                score=0.1,
            )
            candidate = ProceduralSkill(
                skill_id="candidate_bgp_replay",
                title="BGP neighbor replay check",
                activation_condition="Use for BGP route loss or missing advertisements.",
                execution_steps=[
                    SkillStep(
                        order=1,
                        action="Run frr_show_bgp_summary and inspect neighbor state.",
                    )
                ],
                termination_condition="Stop after BGP neighbor and route evidence agree.",
                tools=["frr_show_bgp_summary"],
                status="validated",
                score=0.2,
            )
            experience = SkillExperience(
                experience_id="exp-bgp",
                session_id="s1",
                reward=0.8,
                baseline=0.2,
                advantage=0.6,
                skill_ids=["baseline_unrelated"],
                transitions=[
                    SkillTransition(
                        state="BGP route missing",
                        action="Check BGP neighbors.",
                        tool_name="frr_show_bgp_summary",
                        observation_summary="Neighbor idle; no advertised prefixes.",
                        status="success",
                        done=True,
                    )
                ],
                success=True,
            )
            decision = module.ppo_gate(
                candidate=candidate,
                baseline=baseline,
                evidence=EvaluationEvidence(
                    session_id="s1",
                    task_description="BGP route missing",
                    scenario="dc_clos_bgp",
                    metrics={
                        "detection_score": 1.0,
                        "localization_accuracy": 1.0,
                        "rca_accuracy": 1.0,
                    },
                    success=True,
                ),
                samples=[experience],
                candidate_type="REFINE",
            )

        self.assertTrue(decision.accepted)
        self.assertGreater(decision.candidate_alignment, decision.baseline_alignment)
        self.assertGreater(decision.j_score, 0.0)

    def test_skill_mdp_selects_active_skill_and_records_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=1,
            )
            module.learn_from_episode(
                evidence=EvaluationEvidence(
                    session_id="s1",
                    task_description="DNS record resolves to the wrong backend",
                    scenario="ospf_enterprise_dhcp",
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
                        order=1, action="Query DNS from the client.", tool_name="dig"
                    )
                ],
            )
            before_selection = module.store.load()
            self.assertEqual(
                sum(skill.reuse_count for skill in before_selection.skills.values()),
                0,
            )

            active = module.select_skill(
                query=ProceduralMemoryQuery(
                    text="DNS record gives the wrong address",
                    scenario="ospf_enterprise_dhcp",
                    protocols=["dns"],
                    services=["name_resolution"],
                    symptoms=[],
                    tools=["dig"],
                )
            )
            state = module.store.load()
            active_id = active.skill.skill_id if active is not None else ""
            skill = state.skills[active_id]

        self.assertIsNotNone(active)
        self.assertEqual(active.skill.skill_id, skill.skill_id)
        self.assertEqual(skill.reuse_count, 1)

    def test_lcb_gate_allows_untried_seed_skill_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            state = module.store.load()
            state.skills["seed_hypothesis"] = ProceduralSkill(
                skill_id="seed_hypothesis",
                title="Hypothesis elimination",
                activation_condition="Use when several root-cause hypotheses remain plausible.",
                execution_steps=[
                    SkillStep(order=1, action="Collect discriminating evidence.")
                ],
                termination_condition="Stop when one supported hypothesis remains.",
                status="validated",
                score=0.25,
                frequency=0,
                avg_gain=0.0,
                maturity=8,
            )
            module.store.save(state)

            active = module.select_skill(
                query=ProceduralMemoryQuery(
                    text="Several root-cause hypotheses remain plausible.",
                    scenario="simple_bgp",
                    top_k=3,
                )
            )

        self.assertIsNotNone(active)
        self.assertIn(
            active.skill.skill_id,
            {"seed_hypothesis", "seed_hypothesis_elimination"},
        )
        self.assertEqual(active.skill.frequency, 0)

    def test_select_skill_skips_lcb_rejected_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            state = module.store.load()
            state.skills["risky_high_score"] = ProceduralSkill(
                skill_id="risky_high_score",
                title="Risky high score",
                activation_condition="Use for host reachability failure.",
                execution_steps=[
                    SkillStep(
                        order=1, action="Check reachability.", tool_name="ping_host"
                    )
                ],
                termination_condition="Stop after reachability evidence.",
                tools=["ping_host"],
                status="validated",
                score=1.0,
                frequency=1,
                total_gain=-1.0,
                avg_gain=-1.0,
                maturity=5,
            )
            state.skills["stable_lower_score"] = ProceduralSkill(
                skill_id="stable_lower_score",
                title="Stable lower score",
                activation_condition="Use for host reachability failure.",
                execution_steps=[
                    SkillStep(
                        order=1, action="Check reachability.", tool_name="ping_host"
                    )
                ],
                termination_condition="Stop after reachability evidence.",
                tools=["ping_host"],
                status="validated",
                score=0.5,
                frequency=20,
                total_gain=6.0,
                avg_gain=0.3,
                maturity=5,
            )
            module.store.save(state)

            active = module.select_skill(
                query=ProceduralMemoryQuery(
                    text="Host reachability failure",
                    scenario="simple_bgp",
                    tools=["ping_host"],
                    top_k=5,
                )
            )
            state = module.store.load()

        self.assertIsNotNone(active)
        self.assertEqual(active.skill.skill_id, "stable_lower_score")
        self.assertEqual(state.skills["risky_high_score"].reuse_count, 0)
        self.assertEqual(state.skills["stable_lower_score"].reuse_count, 1)

    def test_runtime_prompt_query_does_not_use_full_tool_catalog_by_default(
        self,
    ) -> None:
        def ping_host(host: str) -> str:
            return f"{host} reachable"

        def show_route(router: str) -> str:
            return f"{router} routes"

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
            runtime = SkillToolRuntime(
                procedural_memory=module,
                procedural_memory_mode="read",
                session=SimpleNamespace(
                    session_id="s2",
                    scenario_name="simple_bgp",
                    scenario_topo_size="small",
                ),
                task_description="Host reachability failure",
                tools=[ping_tool, route_tool],
                session_dir=tmp,
            )

            initial = runtime._query(extra_text="decision prompt before next action")
            runtime.recent_transitions.append(
                {
                    "tool": "ping_host",
                    "status": "success",
                    "observation_summary": "pc1 reachable",
                }
            )
            observed = runtime._query(extra_text="decision prompt before next action")

        self.assertEqual(initial.tools, [])
        self.assertEqual(observed.tools, ["ping_host"])
        self.assertNotIn("show_route", observed.tools)

    def test_runtime_snapshot_tracks_learning_prompt_overhead(self) -> None:
        def ping_host(host: str) -> str:
            return f"{host} reachable"

        with tempfile.TemporaryDirectory() as tmp:
            tool = StructuredTool.from_function(
                ping_host,
                name="ping_host",
                description="Ping one host.",
            )
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            runtime = SkillToolRuntime(
                procedural_memory=module,
                procedural_memory_mode="read",
                session=SimpleNamespace(
                    session_id="s2",
                    scenario_name="simple_bgp",
                    scenario_topo_size="small",
                ),
                task_description="Host reachability failure",
                tools=[tool],
                session_dir=tmp,
            )

            wrapped = runtime.wrap_tools([tool])[0]
            runtime.prompt_suffix()
            wrapped.invoke({"host": "pc1"})
            snapshot = runtime.snapshot()

        self.assertGreater(snapshot["prompt_added_tokens"], 0)
        self.assertEqual(snapshot["config"]["max_skill_age"], 8)
        self.assertEqual(
            snapshot["selection_policy"],
            "epsilon_then_similarity_top_k_online_value",
        )
        self.assertEqual(snapshot["tool_description_added_tokens"], 0)
        self.assertGreater(snapshot["followup_added_tokens"], 0)
        self.assertEqual(snapshot["prompt_injection_count"], 1)
        self.assertEqual(snapshot["tool_description_injection_count"], 0)
        self.assertEqual(snapshot["followup_guidance_count"], 1)
        self.assertEqual(
            snapshot["total_added_tokens"],
            snapshot["prompt_added_tokens"]
            + snapshot["tool_description_added_tokens"]
            + snapshot["followup_added_tokens"],
        )

    def test_episode_learning_credits_runtime_active_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=1,
            )
            self._seed_running_baseline(module, "simple_bgp")
            report = module.learn_from_episode(
                evidence=EvaluationEvidence(
                    session_id="s-runtime",
                    task_description="Tool feedback should drive the next diagnostic action",
                    scenario="simple_bgp",
                    metrics={
                        "detection_score": 1.0,
                        "localization_accuracy": 1.0,
                        "rca_accuracy": 1.0,
                    },
                    steps=4,
                    tool_calls=2,
                    success=True,
                ),
                tool_steps=[
                    SkillStep(
                        order=1,
                        action="Use active ReAct skill with ping evidence.",
                        skill_id="seed_react_decision",
                        activation_id="s-runtime:1",
                        tool_name="ping_pair",
                        observation_summary="packet loss observed",
                        status="success",
                    ),
                    SkillStep(
                        order=2,
                        action="Use active ReAct skill with route evidence.",
                        skill_id="seed_react_decision",
                        activation_id="s-runtime:1",
                        tool_name="frr_show_ip_route",
                        observation_summary="route missing",
                        status="success",
                    ),
                ],
            )
            state = module.store.load()
            credited = state.skills["seed_react_decision"]
            experience = state.experiences[-1]
            stats = module.store.bank_stats()

        self.assertIn("seed_react_decision", report["runtime_skill_ids"])
        self.assertEqual(credited.frequency, 1)
        self.assertEqual(experience.transitions[0].skill_id, "seed_react_decision")
        self.assertEqual(experience.transitions[1].skill_id, "seed_react_decision")
        self.assertEqual(experience.transitions[0].activation_id, "s-runtime:1")
        self.assertEqual(experience.transitions[1].activation_id, "s-runtime:1")
        self.assertEqual(report["status"], "accepted")
        self.assertEqual(stats["ppo_decisions"], 1)
        self.assertIsNotNone(stats["last_candidate_alignment"])

    def test_runtime_skill_frequency_counts_distinct_activations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            state = module.store.load()
            steps = [
                SkillStep(
                    order=1,
                    action="Inspect reachability.",
                    skill_id="seed_react_decision",
                    activation_id="session:1",
                ),
                SkillStep(
                    order=2,
                    action="Inspect routes in the same activation.",
                    skill_id="seed_react_decision",
                    activation_id="session:1",
                ),
                SkillStep(
                    order=3,
                    action="Re-activate after new evidence.",
                    skill_id="seed_react_decision",
                    activation_id="session:2",
                ),
            ]

            counts = module._runtime_skill_counts(state, steps)

        self.assertEqual(counts["seed_react_decision"], 2)

    def test_skill_runtime_logs_termination_for_completed_one_step_skill(self) -> None:
        def ping_host(host: str) -> str:
            return f"{host} reachable"

        with tempfile.TemporaryDirectory() as tmp:
            tool = StructuredTool.from_function(
                ping_host,
                name="ping_host",
                description="Ping one host.",
            )
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            state = module.store.load()
            state.skills["one_step_ping"] = ProceduralSkill(
                skill_id="one_step_ping",
                title="One step reachability check",
                activation_condition="Use for host reachability failure with ping_host.",
                execution_steps=[
                    SkillStep(
                        order=1,
                        action="Call ping_host once and interpret reachability.",
                    )
                ],
                termination_condition=(
                    "Stop after one concrete diagnostic action is selected."
                ),
                tools=["ping_host"],
                status="validated",
                score=1.5,
            )
            module.store.save(state)
            runtime = SkillToolRuntime(
                procedural_memory=module,
                procedural_memory_mode="read",
                session=SimpleNamespace(
                    session_id="s2",
                    scenario_name="simple_bgp",
                    scenario_topo_size="small",
                ),
                task_description="Host reachability failure",
                tools=[tool],
                session_dir=tmp,
            )

            runtime.prompt_suffix()
            decision_snapshot = runtime.before_tool(
                tool_name="ping_host", tool_input={"host": "pc1"}
            )
            runtime.after_tool(
                tool_name="ping_host",
                tool_input={"host": "pc1"},
                result="pc1 reachable",
                decision_snapshot=decision_snapshot,
            )
            runtime.before_tool(tool_name="ping_host", tool_input={"host": "pc2"})
            rows = [
                json.loads(line)
                for line in (Path(tmp) / "messages.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]

        terminations = [row for row in rows if row.get("event") == "skill_termination"]
        self.assertTrue(terminations)
        self.assertEqual(
            terminations[-1]["reason"],
            "termination_condition_satisfied",
        )

    def test_skill_runtime_llm_meta_controller_can_terminate_skill(self) -> None:
        def ping_host(host: str) -> str:
            return f"{host} reachable"

        class FakeMetaController:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            def invoke(self, prompt: str):
                self.prompts.append(prompt)
                return SimpleNamespace(content="<status>DONE</status>")

        with tempfile.TemporaryDirectory() as tmp:
            tool = StructuredTool.from_function(
                ping_host,
                name="ping_host",
                description="Ping one host.",
            )
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            state = module.store.load()
            state.skills["meta_ping"] = ProceduralSkill(
                skill_id="meta_ping",
                title="Meta-controller reachability skill",
                activation_condition="Use for host reachability failure with ping_host.",
                execution_steps=[
                    SkillStep(
                        order=1,
                        action="Call ping_host and interpret the reachability result.",
                        tool_name="ping_host",
                    )
                ],
                termination_condition=(
                    "Stop when current observations show endpoint reachability "
                    "has been interpreted."
                ),
                tools=["ping_host"],
                status="validated",
                score=3.0,
            )
            module.store.save(state)
            meta = FakeMetaController()
            runtime = SkillToolRuntime(
                procedural_memory=module,
                procedural_memory_mode="read",
                session=SimpleNamespace(
                    session_id="s2",
                    scenario_name="simple_bgp",
                    scenario_topo_size="small",
                ),
                task_description="Host reachability failure",
                tools=[tool],
                session_dir=tmp,
                meta_controller_llm=meta,
            )

            runtime.prompt_suffix()
            decision_snapshot = runtime.before_tool(
                tool_name="ping_host", tool_input={"host": "pc1"}
            )
            runtime.after_tool(
                tool_name="ping_host",
                tool_input={"host": "pc1"},
                result="pc1 reachable",
                decision_snapshot=decision_snapshot,
            )
            snapshot = runtime.snapshot()
            rows = [
                json.loads(line)
                for line in (Path(tmp) / "messages.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]

        meta_events = [
            row for row in rows if row.get("event") == "skill_meta_controller"
        ]
        terminations = [row for row in rows if row.get("event") == "skill_termination"]
        self.assertTrue(snapshot["meta_controller_available"])
        self.assertTrue(meta.prompts)
        self.assertIn("[ACTIVE OPTION]", meta.prompts[-1])
        self.assertEqual(meta_events[-1]["status"], "DONE")
        self.assertEqual(terminations[-1]["reason"], "meta_controller_done")
        self.assertEqual(terminations[-1]["source"], "post_tool")

    def test_skill_runtime_caches_llm_meta_controller_for_same_state(self) -> None:
        def ping_host(host: str) -> str:
            return f"{host} reachable"

        class FakeMetaController:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            def invoke(self, prompt: str):
                self.prompts.append(prompt)
                return SimpleNamespace(content="<status>CONTINUE</status>")

        with tempfile.TemporaryDirectory() as tmp:
            tool = StructuredTool.from_function(
                ping_host,
                name="ping_host",
                description="Ping one host.",
            )
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            state = module.store.load()
            state.skills["meta_ping"] = ProceduralSkill(
                skill_id="meta_ping",
                title="Meta-controller reachability skill",
                activation_condition="Use for host reachability failure with ping_host.",
                execution_steps=[
                    SkillStep(
                        order=1,
                        action="Call ping_host and interpret the reachability result.",
                        tool_name="ping_host",
                    )
                ],
                termination_condition=(
                    "Stop when current observations show endpoint reachability "
                    "has been interpreted."
                ),
                tools=["ping_host"],
                status="validated",
                score=3.0,
            )
            module.store.save(state)
            meta = FakeMetaController()
            runtime = SkillToolRuntime(
                procedural_memory=module,
                procedural_memory_mode="read",
                session=SimpleNamespace(
                    session_id="s2",
                    scenario_name="simple_bgp",
                    scenario_topo_size="small",
                ),
                task_description="Host reachability failure",
                tools=[tool],
                session_dir=tmp,
                meta_controller_llm=meta,
            )

            runtime.prompt_suffix()
            decision_snapshot = runtime.before_tool(
                tool_name="ping_host", tool_input={"host": "pc1"}
            )
            runtime.after_tool(
                tool_name="ping_host",
                tool_input={"host": "pc1"},
                result="pc1 reachable",
                decision_snapshot=decision_snapshot,
            )
            runtime.prompt_suffix()
            snapshot = runtime.snapshot()
            rows = [
                json.loads(line)
                for line in (Path(tmp) / "messages.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]

        meta_events = [
            row for row in rows if row.get("event") == "skill_meta_controller"
        ]
        self.assertEqual(len(meta.prompts), 1)
        self.assertEqual(snapshot["meta_controller_cache_hits"], 1)
        self.assertEqual(meta_events[-1]["status"], "cached")

    def test_skill_runtime_selects_active_skill_in_prompt_before_tool_choice(
        self,
    ) -> None:
        def ping_host(host: str) -> str:
            return f"{host} reachable"

        with tempfile.TemporaryDirectory() as tmp:
            tool = StructuredTool.from_function(
                ping_host,
                name="ping_host",
                description="Ping one host.",
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
                        action="Call ping_host before narrowing the fault.",
                        tool_name="ping_host",
                    )
                ],
                termination_condition="Stop after ping evidence is interpreted.",
                tools=["ping_host"],
                status="validated",
                score=2.0,
            )
            module.store.save(state)
            runtime = SkillToolRuntime(
                procedural_memory=module,
                procedural_memory_mode="read",
                session=SimpleNamespace(
                    session_id="s2",
                    scenario_name="simple_bgp",
                    scenario_topo_size="small",
                ),
                task_description="Host reachability failure",
                tools=[tool],
                session_dir=tmp,
            )

            decision_context = "Assigned step: verify client-to-server reachability."
            prompt = runtime.prompt_suffix(decision_context=decision_context)
            runtime.prompt_suffix(decision_context=decision_context)
            policy_state = runtime._decision_policy_state
            snapshot = runtime.snapshot()
            state = module.store.load()
            rows = [
                json.loads(line)
                for line in (Path(tmp) / "messages.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]

        prompt_activations = [
            row
            for row in rows
            if row.get("event") == "skill_activation" and row.get("source") == "prompt"
        ]
        self.assertIn("[ACTIVE SKILL-MDP OPTION]", prompt)
        self.assertIn("procedural guidance, not as evidence", prompt)
        self.assertIn("prompt_ping", prompt)
        self.assertIn(decision_context, prompt)
        self.assertIn(decision_context, policy_state)
        self.assertEqual(snapshot["active_skill_id"], "prompt_ping")
        self.assertEqual(snapshot["prompt_selection_count"], 1)
        self.assertEqual(state.skills["prompt_ping"].reuse_count, 0)
        self.assertEqual(len(prompt_activations), 1)

    def test_skill_runtime_does_not_attribute_skill_after_action_selection(
        self,
    ) -> None:
        def ping_host(host: str) -> str:
            return f"{host} reachable"

        with tempfile.TemporaryDirectory() as tmp:
            tool = StructuredTool.from_function(
                ping_host,
                name="ping_host",
                description="Ping one host.",
            )
            module = ProceduralMemoryModule(
                bank_id="pre-action-only",
                store_path=Path(tmp) / "skills.json",
            )
            runtime = SkillToolRuntime(
                procedural_memory=module,
                procedural_memory_mode="read",
                session=SimpleNamespace(
                    session_id="s2",
                    scenario_name="simple_bgp",
                    scenario_topo_size="small",
                ),
                task_description="Host reachability failure",
                tools=[tool],
                session_dir=tmp,
            )

            snapshot = runtime.before_tool(
                tool_name="ping_host", tool_input={"host": "pc1"}
            )
            runtime.after_tool(
                tool_name="ping_host",
                tool_input={"host": "pc1"},
                result="pc1 reachable",
                decision_snapshot=snapshot,
            )
            rows = [
                json.loads(line)
                for line in (Path(tmp) / "messages.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]

        transition = next(row for row in rows if row.get("event") == "skill_transition")
        self.assertEqual(snapshot["active_skill_id"], "")
        self.assertEqual(transition["active_skill_id"], "")
        self.assertFalse(
            any(
                row.get("event") == "skill_activation"
                and row.get("source") == "tool_fallback"
                for row in rows
            )
        )

    def test_skill_runtime_read_only_prompt_does_not_activate_or_record_reuse(
        self,
    ) -> None:
        def ping_host(host: str) -> str:
            return f"{host} reachable"

        with tempfile.TemporaryDirectory() as tmp:
            tool = StructuredTool.from_function(
                ping_host,
                name="ping_host",
                description="Ping one host.",
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
                        action="Call ping_host before narrowing the fault.",
                        tool_name="ping_host",
                    )
                ],
                termination_condition="Stop after ping evidence is interpreted.",
                tools=["ping_host"],
                status="validated",
                score=2.0,
            )
            module.store.save(state)
            runtime = SkillToolRuntime(
                procedural_memory=module,
                procedural_memory_mode="read",
                session=SimpleNamespace(
                    session_id="s2",
                    scenario_name="simple_bgp",
                    scenario_topo_size="small",
                ),
                task_description="Host reachability failure",
                tools=[tool],
                session_dir=tmp,
            )

            prompt = runtime.prompt_suffix(activate_skill=False)
            snapshot = runtime.snapshot()
            state = module.store.load()
            log_path = Path(tmp) / "messages.jsonl"

        self.assertEqual(prompt, "")
        self.assertEqual(snapshot["active_skill_id"], "")
        self.assertEqual(snapshot["prompt_selection_count"], 0)
        self.assertEqual(state.skills["prompt_ping"].reuse_count, 0)
        self.assertFalse(log_path.exists())

    def test_skill_runtime_does_not_reselect_cooldown_option_after_termination(
        self,
    ) -> None:
        def ping_host(host: str) -> str:
            return f"{host} reachable"

        with tempfile.TemporaryDirectory() as tmp:
            tool = StructuredTool.from_function(
                ping_host,
                name="ping_host",
                description="Ping one host.",
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
                score=2.0,
            )
            module.store.save(state)
            runtime = SkillToolRuntime(
                procedural_memory=module,
                procedural_memory_mode="read",
                session=SimpleNamespace(
                    session_id="s2",
                    scenario_name="simple_bgp",
                    scenario_topo_size="small",
                ),
                task_description="Host reachability failure",
                tools=[tool],
                session_dir=tmp,
            )
            runtime.prompt_suffix()
            output = runtime.wrap_tools([tool])[0].invoke({"host": "pc1"})
            snapshot = runtime.snapshot()
            rows = [
                json.loads(line)
                for line in (Path(tmp) / "messages.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]

        post_tool_terms = [
            row
            for row in rows
            if row.get("event") == "skill_termination"
            and row.get("source") == "post_tool"
        ]
        post_tool_activations = [
            row
            for row in rows
            if row.get("event") == "skill_activation"
            and row.get("source") == "post_tool"
        ]
        self.assertTrue(post_tool_terms)
        self.assertEqual(
            post_tool_terms[-1]["reason"],
            "termination_condition_satisfied",
        )
        self.assertTrue(post_tool_activations)
        self.assertNotEqual(
            post_tool_activations[-1]["active_skill_id"], "one_step_ping"
        )
        self.assertNotEqual(snapshot["active_skill_id"], "one_step_ping")
        self.assertEqual(snapshot["skill_age"], 0)
        self.assertNotIn("one_step_ping", output)

    def test_skill_runtime_refreshes_once_after_parallel_tool_batch(self) -> None:
        def ping_host(host: str) -> str:
            return f"{host} reachable"

        with tempfile.TemporaryDirectory() as tmp:
            tool = StructuredTool.from_function(
                ping_host,
                name="ping_host",
                description="Ping one host.",
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
                score=2.0,
            )
            module.store.save(state)
            runtime = SkillToolRuntime(
                procedural_memory=module,
                procedural_memory_mode="read",
                session=SimpleNamespace(
                    session_id="s2",
                    scenario_name="simple_bgp",
                    scenario_topo_size="small",
                ),
                task_description="Host reachability failure",
                tools=[tool],
                session_dir=tmp,
            )
            runtime.prompt_suffix()
            runtime.before_tool(tool_name="ping_host", tool_input={"host": "pc1"})
            runtime.before_tool(tool_name="ping_host", tool_input={"host": "pc2"})
            runtime.after_tool(
                tool_name="ping_host",
                tool_input={"host": "pc1"},
                result="pc1 reachable",
            )
            mid_snapshot = runtime.snapshot()
            runtime.after_tool(
                tool_name="ping_host",
                tool_input={"host": "pc2"},
                result="pc2 reachable",
            )
            snapshot = runtime.snapshot()
            rows = [
                json.loads(line)
                for line in (Path(tmp) / "messages.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]

        terminations = [row for row in rows if row.get("event") == "skill_termination"]
        post_tool_terms = [
            row for row in terminations if row.get("source") == "post_tool"
        ]
        batch_reselects = [
            row
            for row in rows
            if row.get("event") == "skill_activation"
            and row.get("source") == "tool_after_termination"
        ]
        self.assertEqual(mid_snapshot["inflight_tool_calls"], 1)
        self.assertEqual(mid_snapshot["post_tool_selection_count"], 0)
        self.assertEqual(mid_snapshot["followup_guidance_count"], 0)
        self.assertEqual(snapshot["inflight_tool_calls"], 0)
        self.assertEqual(snapshot["followup_guidance_count"], 1)
        self.assertEqual(len(post_tool_terms), 1)
        self.assertEqual(terminations, post_tool_terms)
        self.assertFalse(batch_reselects)

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
                procedural_memory_mode="read",
                session=SimpleNamespace(
                    session_id="s2",
                    scenario_name="simple_bgp",
                    scenario_topo_size="small",
                ),
                task_description="Host reachability failure",
                tools=[ping_tool, route_tool],
                session_dir=tmp,
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
        self.assertIn("followup_route", output)
        self.assertNotIn("Active Skill-MDP option: one_step_ping", output)

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
                procedural_memory_mode="read",
                session=SimpleNamespace(
                    session_id="s2",
                    scenario_name="simple_bgp",
                    scenario_topo_size="small",
                ),
                task_description="Host reachability failure",
                tools=[ping_tool, route_tool],
                session_dir=tmp,
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
                procedural_memory_mode="read",
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
        self.assertIn(
            "Integrated learning guidance - not evidence",
            str(output.content),
        )

    def test_experience_and_golden_pools_are_persisted(self) -> None:
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
                        order=1, action="Test HTTP reachability.", tool_name="curl"
                    )
                ],
            )
            state = module.store.load()
            stats = module.store.bank_stats()

        self.assertEqual(len(state.experiences), 1)
        self.assertEqual(len(state.golden_experiences), 1)
        self.assertEqual(stats["experiences"], 1)
        self.assertEqual(stats["golden_experiences"], 1)

    def test_golden_pool_ranks_trajectory_experience_by_reward(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            module.store.record_experience(
                SkillExperience(
                    experience_id="exp-failed",
                    session_id="failed",
                    reward=-0.1,
                    baseline=0.0,
                    advantage=-0.1,
                    transitions=[
                        SkillTransition(
                            state="Failed diagnosis.",
                            action="Check broad evidence.",
                            tool_name="get_reachability",
                            status="success",
                            done=True,
                        )
                    ],
                    success=False,
                )
            )
            state = module.store.load()

        self.assertEqual(len(state.experiences), 1)
        self.assertEqual(len(state.golden_experiences), 1)

    def test_parent_skill_refines_to_versioned_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=1,
            )
            self._seed_running_baseline(module, "dc_clos_bgp")
            steps = [
                SkillStep(
                    order=1,
                    action="Check BGP summary.",
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
                    steps[0].model_copy(update={"skill_id": bootstrap["skill_id"]})
                ],
            )
            state = module.store.load()
            refined = state.skills[refined_report["skill_id"]]

        self.assertEqual(bootstrap["status"], "accepted")
        self.assertEqual(refined_report["status"], "accepted")
        self.assertEqual(refined.parent_id, bootstrap["skill_id"])
        self.assertGreaterEqual(refined.version, 1)
        self.assertEqual(state.skills[bootstrap["skill_id"]].status, "validated")
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
        self.assertGreaterEqual(len(prompts), 2 + module.best_of_n)
        self.assertTrue(any("batch semantic-gradient aggregator" in p for p in prompts))
        self.assertEqual(
            sum("Skill Evolver" in prompt for prompt in prompts),
            module.best_of_n,
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
                        "NIKA_LEARNING_LLM_BACKEND": "custom",
                        "NIKA_LEARNING_LLM_MODEL": "learning-model",
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
                            tool_name="frr_show_bgp_summary",
                        )
                    ],
                )

        self.assertEqual(first["status"], "deferred")
        load_model.assert_called_once()
        args, kwargs = load_model.call_args
        self.assertEqual(args[:2], ("custom", "learning-model"))
        self.assertEqual(kwargs["max_retries"], 0)
        self.assertEqual(report["semantic_gradient_source"], "deterministic")
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
                        "procedural_memory_mode": "evolve",
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
                        "followup_added_tokens": 20,
                        "total_added_tokens": 120,
                        "prompt_injection_count": 2,
                        "tool_description_injection_count": 4,
                        "followup_guidance_count": 1,
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
                            "procedural_memory_mode": "evolve",
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
            "epsilon_then_similarity_top_k_online_value",
        )
        self.assertEqual(report["procedural_memory_config"]["selection_epsilon"], 0.3)
        self.assertEqual(report["total_added_tokens"], 120)
        self.assertEqual(report["delta_prompt_tokens_per_step"], 30.0)
        self.assertEqual(report["prompt_added_tokens"], 80)
        self.assertEqual(report["tool_description_added_tokens"], 20)
        self.assertEqual(report["followup_added_tokens"], 20)

    def _legacy_eval_metrics_embeds_memory_update_report(self) -> None:
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
                        "procedural_memory_mode": "evolve",
                        "procedural_memory_bank": "skill",
                    }
                ),
                encoding="utf-8",
            )
            (session_dir / "ground_truth.json").write_text(
                json.dumps(
                    {
                        "is_anomaly": True,
                        "root_cause_name": ["dns_record_error"],
                        "faulty_devices": ["dns_server"],
                    }
                ),
                encoding="utf-8",
            )
            (session_dir / "submission.json").write_text(
                json.dumps(
                    {
                        "is_anomaly": True,
                        "root_cause_name": ["dns_record_error"],
                        "faulty_devices": ["dns_server"],
                    }
                ),
                encoding="utf-8",
            )
            (session_dir / "messages.jsonl").write_text("", encoding="utf-8")
            updates: list[tuple[str, object]] = []

            class FakeSession:
                def __init__(self) -> None:
                    self.session_dir = str(session_dir)
                    self.session_id = "s1"
                    self.procedural_memory_mode = "evolve"
                    self.procedural_memory_bank = "skill"
                    self.llm_backend = "custom"
                    self.model = "test-model"
                    self.tool_refinement_enabled = False
                    self.store = None

                def load_closed_session(self, *, session_id=None) -> None:
                    self.session_id = session_id or self.session_id

                def update_run_meta(self, key: str, value: object) -> None:
                    updates.append((key, value))
                    setattr(self, key, value)

            procedural_memory_report = {
                "status": "accepted",
                "skill_id": "skill_dns",
                "runtime_skill_ids": ["seed_react_decision"],
                "episode_reward": 0.81,
                "episode_baseline": 0.34,
                "episode_advantage": 0.47,
                "episode_success": True,
                "total_added_tokens": 120,
                "delta_prompt_tokens_per_step": 30.0,
                "prompt_added_tokens": 80,
                "tool_description_added_tokens": 20,
                "followup_added_tokens": 20,
                "skills": 7,
                "decision": {
                    "j_score": 0.42,
                    "candidate_alignment": 0.73,
                    "baseline_alignment": 0.21,
                    "verification_method": "alignment_surrogate",
                    "verified_success_count": 2,
                },
                "semantic_gradient_source": "llm",
                "semantic_gradient_llm_failed": False,
                "semantic_gradient_count": 3,
                "verification_method": "alignment_surrogate",
            }
            with (
                patch("nika.workflows.eval.session.Session", FakeSession),
                patch(
                    "agent.procedural_memory.workflow.update_procedural_memory_from_session",
                    new=AsyncMock(return_value=procedural_memory_report),
                ),
            ):
                run_eval_metrics(session_id="s1")

            metrics = json.loads((session_dir / "eval_metrics.json").read_text())
            result = build_eval_result_from_session_dir(session_dir)

        self.assertEqual(metrics["procedural_memory"], procedural_memory_report)
        self.assertEqual(result.procedural_memory_update_status, "accepted")
        self.assertEqual(result.procedural_memory_skill_id, "skill_dns")
        self.assertEqual(
            result.procedural_memory_runtime_skill_ids, ["seed_react_decision"]
        )
        self.assertEqual(result.procedural_memory_episode_reward, 0.81)
        self.assertEqual(result.procedural_memory_episode_baseline, 0.34)
        self.assertEqual(result.procedural_memory_episode_advantage, 0.47)
        self.assertTrue(result.procedural_memory_episode_success)
        self.assertEqual(result.procedural_memory_total_added_tokens, 120)
        self.assertEqual(result.procedural_memory_delta_prompt_tokens_per_step, 30.0)
        self.assertEqual(result.procedural_memory_prompt_added_tokens, 80)
        self.assertEqual(result.procedural_memory_tool_description_added_tokens, 20)
        self.assertEqual(result.procedural_memory_followup_added_tokens, 20)
        self.assertEqual(result.procedural_memory_ppo_j_score, 0.42)
        self.assertEqual(result.procedural_memory_candidate_alignment, 0.73)
        self.assertEqual(result.procedural_memory_baseline_alignment, 0.21)
        self.assertEqual(result.procedural_memory_semantic_gradient_count, 3)
        self.assertEqual(
            result.procedural_memory_verification_method, "alignment_surrogate"
        )
        self.assertEqual(result.procedural_memory_verified_success_count, 2)
        self.assertEqual(result.procedural_memory_skills, 7)
        self.assertIn(("procedural_memory", procedural_memory_report), updates)
