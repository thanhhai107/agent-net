"""Tests for Skill-Pro Procedural Memory."""

from __future__ import annotations

import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


from agent.procedural_memory.models import (
    EvaluationEvidence,
    PPOGateDecision,
    ProceduralSkill,
    SemanticGradientDraft,
    SkillCandidateDraft,
    SkillExperience,
    SkillStep,
    SkillTransition,
)
from agent.procedural_memory.runtime import SkillToolRuntime
from agent.procedural_memory.policy_context import (
    build_runtime_skill_policy_prefix,
    build_skill_policy_suffix,
)
from agent.procedural_memory.policy_scorer import (
    BehavioralReplayPolicyScorer,
    PolicyReplayDraft,
    PolicyReplayItem,
    PolicyReplayResult,
    PolicyLogprobScorer,
    PolicyStepLogprob,
)
from agent.procedural_memory.service import (
    ProceduralMemoryModule,
    _metric_success,
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




class SkillProProceduralMemoryTestPart1(unittest.TestCase):
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


    def test_evaluation_runtime_does_not_mutate_frozen_bank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "skills.json"
            training = ProceduralMemoryModule(
                bank_id="frozen",
                store_path=state_path,
            )
            before = training.store.state_hash()
            evaluation_module = ProceduralMemoryModule(
                bank_id="frozen",
                store_path=state_path,
                read_only=True,
            )
            controller = _FakeSkillController(["seed_react_decision"])
            runtime = SkillToolRuntime(
                procedural_memory=evaluation_module,
                allow_training_updates=False,
                session=SimpleNamespace(session_id="evaluation"),
                task_description="Inspect current reachability.",
                tools=[],
                meta_controller_llm=controller,
            )
            runtime.prompt_suffix()
            after = training.store.state_hash()
            snapshot = runtime.snapshot()

            with self.assertRaises(PermissionError):
                evaluation_module.store.save(evaluation_module.store.load())

        self.assertEqual(before, after)
        self.assertTrue(snapshot["state_unchanged"])
        self.assertFalse(snapshot["allow_training_updates"])
        self.assertEqual(snapshot["selection_policy"], "llm_direct")


    def test_ready_skill_is_scheduled_from_global_trajectory_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="scheduler",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=2,
            )
            state = module.store.load()
            state.experiences.extend(
                SkillExperience(
                    experience_id=f"secondary-history-{index}",
                    session_id=f"old-{index}",
                    reward=0.5,
                    skill_ids=["seed_self_consistency_check"],
                    transitions=[SkillTransition(action="inspect")],
                )
                for index in range(2)
            )

            selected = module._next_evolution_parent(state)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.skill_id, "seed_self_consistency_check")


    def test_offline_accepted_probationary_skill_stays_probationary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "skills.json"
            module = ProceduralMemoryModule(
                bank_id="legacy-accepted",
                store_path=store_path,
            )
            state = module.store.load()
            state.skills["accepted_legacy_skill"] = ProceduralSkill(
                skill_id="accepted_legacy_skill",
                title="Accepted legacy skill",
                activation_condition="Use when independent evidence is required.",
                execution_steps=[
                    SkillStep(
                        order=1,
                        action="Collect independent evidence.",
                    )
                ],
                termination_condition="Stop after collecting independent evidence.",
                status="probationary",
            )
            state.ppo_decisions.append(
                PPOGateDecision(
                    accepted=True,
                    reason="accepted by the persisted gate",
                    candidate_score=0.7,
                    baseline_score=0.5,
                    candidate_skill_id="accepted_legacy_skill",
                )
            )
            module.store.save(state)

            reloaded = ProceduralMemoryModule(
                bank_id="legacy-accepted",
                store_path=store_path,
            )
            self.assertEqual(
                reloaded.store.load().skills["accepted_legacy_skill"].status,
                "probationary",
            )


    def test_policy_suffix_respects_hard_context_budget(self) -> None:
        suffix = build_skill_policy_suffix(
            "state " * 1000,
            None,
            max_tokens=120,
        )

        self.assertLessEqual(len(suffix), 120 * 4)


    def test_policy_suffix_allows_diagnosis_report_when_evidence_is_complete(
        self,
    ) -> None:
        suffix = build_skill_policy_suffix(
            "The collected observations support a complete diagnosis.",
            None,
        )

        self.assertIn("If evidence is incomplete", suffix)
        self.assertIn("return a concise diagnosis report", suffix)
        self.assertIn("Do not submit from the diagnosis phase", suffix)
        self.assertNotIn("tool call only", suffix)


    def test_runtime_records_the_exact_budgeted_skill_system_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="runtime-policy-context",
                store_path=Path(tmp) / "skills.json",
            )
            controller = _FakeSkillController(["seed_react_decision"])
            runtime = SkillToolRuntime(
                procedural_memory=module,
                allow_training_updates=False,
                session=SimpleNamespace(session_id="runtime-context-1"),
                task_description="Inspect current reachability.",
                tools=[],
                token_budget=400,
                meta_controller_llm=controller,
            )

            runtime.prompt_suffix(decision_context="Latest user-visible state")
            snapshot = runtime.before_tool(tool_name="inspect", tool_input={})

        self.assertEqual(snapshot["policy_token_budget"], "200")
        self.assertEqual(
            snapshot["policy_context"],
            build_runtime_skill_policy_prefix(
                runtime.active_skill.skill,
                max_tokens=200,
            ),
        )


    def test_explicit_store_path_is_used_as_the_state_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom-bank.json"
            module = ProceduralMemoryModule(
                bank_id="explicit-path",
                store_path=path,
            )

            self.assertEqual(module.store.state_path, path)
            self.assertTrue(path.exists())


    def test_explicit_store_path_migrates_the_legacy_nested_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom-bank.json"
            legacy = Path(tmp) / "legacy-bank" / "skills.json"
            legacy_store = ProceduralMemoryModule(
                bank_id="legacy-bank",
                store_path=legacy,
            )
            state = legacy_store.store.load()
            state.iteration = 7
            legacy_store.store.save(state)

            migrated = ProceduralMemoryModule(
                bank_id="legacy-bank",
                store_path=path,
            )

            self.assertTrue(path.exists())
            self.assertEqual(migrated.store.load().iteration, 7)


    def test_persisted_history_and_retired_skills_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="bounded-history",
                store_path=Path(tmp) / "skills.json",
                pool_size=1,
                experience_pool_size=2,
            )
            state = module.store.load()
            state.episodes = [
                EvaluationEvidence(session_id=f"episode-{index}")
                for index in range(105)
            ]
            for index in range(18):
                skill_id = f"retired-{index:02d}"
                state.skills[skill_id] = ProceduralSkill(
                    skill_id=skill_id,
                    title=skill_id,
                    activation_condition="When old evidence applies.",
                    execution_steps=[SkillStep(order=1, action="Inspect evidence.")],
                    termination_condition="Stop after inspection.",
                    status="retired",
                )

            module._save_state(state)
            saved = module.store.load()

        self.assertEqual(len(saved.episodes), 100)
        self.assertEqual(
            sum(
                skill.status == "retired" and skill.origin == "learned"
                for skill in saved.skills.values()
            ),
            16,
        )


    def test_epsilon_exploration_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="exploration",
                store_path=Path(tmp) / "skills.json",
            )
            first_triggered, first = module.exploration_selection(
                epsilon=1.0,
                key="session-1:0",
                record_reuse=False,
            )
            second_triggered, second = module.exploration_selection(
                epsilon=1.0,
                key="session-1:0",
                record_reuse=False,
            )

        self.assertTrue(first_triggered)
        self.assertTrue(second_triggered)
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

        self.assertAlmostEqual(epsilon, 0.05)


    def test_epsilon_exploration_falls_back_when_retrieval_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="empty-retrieval-exploration",
                store_path=Path(tmp) / "skills.json",
            )
            with patch.object(module, "selection_candidates", return_value=[]):
                triggered, selected = module.exploration_selection(
                    epsilon=1.0,
                    key="empty",
                    record_reuse=False,
                )

        self.assertTrue(triggered)
        self.assertIsNone(selected)


    def test_epsilon_exploration_respects_cooldown_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="exploration-cooldown",
                store_path=Path(tmp) / "skills.json",
            )
            excluded = "seed_react_decision"
            selected_ids = {
                selected.skill.skill_id
                for index in range(30)
                if (
                    selected := module.exploration_selection(
                        epsilon=1.0,
                        key=f"cooldown:{index}",
                        record_reuse=False,
                        exclude_skill_ids={excluded},
                    )[1]
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


    def test_custom_api_uses_behavioral_replay_without_logprob_endpoint(self) -> None:
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

        self.assertIsInstance(module.policy_scorer, BehavioralReplayPolicyScorer)


    def test_behavioral_replay_does_not_silently_fallback_without_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="behavioral-no-backend",
                store_path=Path(tmp) / "skills.json",
            )

        self.assertIsInstance(module.policy_scorer, BehavioralReplayPolicyScorer)


    def test_explicit_logprob_endpoint_enables_teacher_forced_scorer(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(
                os.environ,
                {
                    "NIKA_SKILL_LOGPROB_URL": "https://logprob.example.test/v1",
                    "NIKA_SKILL_LOGPROB_API_KEY": "test-password",
                },
            ),
        ):
            module = ProceduralMemoryModule(
                bank_id="api-logprob",
                llm_backend="custom",
                model="provider/model",
                verifier="policy_logprob",
                store_path=Path(tmp) / "skills.json",
            )

        self.assertIsInstance(module.policy_scorer, PolicyLogprobScorer)


    def test_policy_logprob_rejects_missing_endpoint(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(
                os.environ,
                {
                    "NIKA_SKILL_LOGPROB_URL": "",
                    "NIKA_SKILL_LOGPROB_API_KEY": "",
                    "CUSTOM_API_KEY": "",
                },
            ),
            self.assertRaisesRegex(ValueError, "NIKA_SKILL_LOGPROB_URL"),
        ):
            ProceduralMemoryModule(
                bank_id="api-logprob-missing-endpoint",
                llm_backend="custom",
                model="provider/model",
                verifier="policy_logprob",
                store_path=Path(tmp) / "skills.json",
            )


    def test_unknown_verifier_is_rejected(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(ValueError, "Unsupported verifier"),
        ):
            ProceduralMemoryModule(
                bank_id="unknown-verifier",
                verifier="implicit_backup",
                store_path=Path(tmp) / "skills.json",
            )


    def test_logprob_replay_uses_recorded_behavior_policy_context(self) -> None:
        scorer = PolicyLogprobScorer(
            base_url="https://example.test/v1",
            api_key="test-password",
            model="provider/model",
        )
        scorer._score_targets = MagicMock(side_effect=[[-1.0, -1.5], [-2.0, -2.5]])
        candidate = ProceduralSkill(
            skill_id="candidate",
            title="Candidate",
            activation_condition="When evidence is incomplete.",
            execution_steps=[SkillStep(order=1, action="Inspect evidence.")],
            termination_condition="Stop after sufficient evidence.",
        )
        experiences = [
            SkillExperience(
                experience_id="mixed-behavior",
                session_id="s1",
                reward=1.0,
                transitions=[
                    SkillTransition(
                        state="state one",
                        action="first_action",
                        policy_context="recorded policy one",
                    ),
                    SkillTransition(
                        state="state two",
                        action="second_action",
                        policy_context="recorded policy two",
                    ),
                ],
            )
        ]

        result = scorer._score_batch(
            candidate=candidate,
            experiences=experiences,
        )

        self.assertEqual(result.method, "policy_logprob")
        baseline_rows = scorer._score_targets.call_args_list[1].args[0]
        self.assertEqual(
            baseline_rows,
            [
                ("recorded policy one", "first_action"),
                ("recorded policy two", "second_action"),
            ],
        )


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
        self.assertAlmostEqual(decision.parent_j_score, 1.0)
        self.assertAlmostEqual(decision.j_score, 0.2)
        self.assertAlmostEqual(decision.delta_j_score, 0.2)
        self.assertEqual(decision.candidate_alignment, -1.0)
        self.assertEqual(decision.baseline_alignment, -2.0)


    def test_verification_batch_preserves_behavior_time_baseline(
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
                generation_samples=[
                    original,
                    original.model_copy(
                        update={"experience_id": "new", "session_id": "new"}
                    ),
                ],
            )

        self.assertAlmostEqual(batch[0].baseline, 0.2)
        self.assertAlmostEqual(batch[0].advantage, 0.7)
        self.assertEqual(original.baseline, 0.2)


    def test_ppo_gate_rejects_policy_identical_to_parent(self) -> None:
        class IdenticalLogprobScorer:
            def score_batch(self, *, candidate, baseline, experiences):
                del candidate, baseline
                return PolicyReplayResult(
                    scores=[],
                    method="policy_logprob",
                    step_logprobs=[
                        PolicyStepLogprob(
                            experience_id=experience.experience_id,
                            transition_index=index,
                            candidate_logprob=-2.0,
                            baseline_logprob=-2.0,
                        )
                        for experience in experiences
                        for index, _ in enumerate(experience.transitions)
                    ],
                )

        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="identical-policy-gate",
                store_path=Path(tmp) / "skills.json",
                policy_scorer=IdenticalLogprobScorer(),
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
                evidence=EvaluationEvidence(session_id="same-policy"),
                samples=[
                    SkillExperience(
                        experience_id="same-policy-exp",
                        session_id="same-policy",
                        reward=1.0,
                        advantage=1.0,
                        transitions=[SkillTransition(action="show_routes({})")],
                    )
                ],
            )

        self.assertFalse(decision.accepted)
        self.assertAlmostEqual(decision.delta_j_score, 0.0)


    def test_legacy_activation_records_cannot_cross_holdout_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="trajectory-holdout",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=2,
            )
            state = module.store.load()
            state.experiences.extend(
                [
                    SkillExperience(
                        experience_id="same-activation-1",
                        session_id="same-session",
                        reward=0.4,
                        skill_ids=["seed_react_decision"],
                        transitions=[SkillTransition(action="first")],
                    ),
                    SkillExperience(
                        experience_id="same-activation-2",
                        session_id="same-session",
                        reward=0.4,
                        skill_ids=["seed_react_decision"],
                        transitions=[SkillTransition(action="second")],
                    ),
                    SkillExperience(
                        experience_id="other-trajectory",
                        session_id="other-session",
                        reward=0.8,
                        skill_ids=["seed_react_decision"],
                        transitions=[SkillTransition(action="other")],
                    ),
                ]
            )
            parent = state.skills["seed_react_decision"]

            evolution_batch = module._evolution_batch(state, parent)
            holdout = module._verification_batch(
                state,
                generation_samples=evolution_batch,
            )

        self.assertEqual(len(evolution_batch), 2)
        self.assertEqual(
            {item.session_id for item in evolution_batch},
            {"same-session", "other-session"},
        )
        self.assertEqual(len(holdout), 1)


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
        self.assertIn("verification deferred", decision.reason)
        self.assertIn("TimeoutError", decision.verification_error)


    def test_no_fault_success_requires_detection_only(self) -> None:
        metrics = {
            "detection_score": 1.0,
            "localization_f1": 0.0,
            "rca_f1": 0.0,
        }

        self.assertTrue(_metric_success(metrics, False))
        self.assertFalse(_metric_success(metrics, True))


    def test_semantic_gradient_requires_training_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="no-fault",
                store_path=Path(tmp) / "skills.json",
            )
            with self.assertRaisesRegex(RuntimeError, "not configured"):
                module.semantic_gradient(
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
                        )
                    ],
                )


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
        self.assertEqual(skill.frequency, 1)
        self.assertEqual(skill.episode_exposures, 1)
        self.assertEqual(skill.activation_count, 2)
        self.assertAlmostEqual(skill.avg_gain, 0.1)
        self.assertEqual(skill.success_count, 1)
        self.assertEqual(skill.failure_count, 0)
        self.assertEqual(skill.maturity, 0)


    def test_each_training_iteration_ages_the_whole_active_pool(self) -> None:
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
                        skill_id="seed_react_decision",
                        activation_id="maturity-1:1",
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
                        activation_id="attributed-credit-1:1",
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

