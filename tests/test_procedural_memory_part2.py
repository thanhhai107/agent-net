"""Tests for Skill-Pro Procedural Memory."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.tools import StructuredTool

from agent.procedural_memory.attributes import infer_procedural_memory_attributes
from agent.procedural_memory.models import (
    EvaluationEvidence,
    PPOGateDecision,
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
)
from agent.procedural_memory.service import (
    ProceduralMemoryModule,
    _evidence_score,
    _metric_success,
)
from agent.procedural_memory.workflow import (
    extract_skill_steps,
)
from agent.tool_refinement.runtime import ToolRefinementRuntime
from agent.tool_refinement.store import ToolRefinementStore


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




class SkillProProceduralMemoryTestPart2(unittest.TestCase):
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
                item.get("stage") == "validated online rollback"
                and item.get("skill_id") == "harmful"
                for item in state.maintenance_log
            )
        )


    def test_probation_promotes_positive_gain_and_rolls_back_negative_gain(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="probation-lifecycle",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=3,
                acceptance_margin=0.01,
            )
            state = module.store.load()

            def probationary(skill_id: str, gain: float) -> ProceduralSkill:
                return ProceduralSkill(
                    skill_id=skill_id,
                    title=skill_id,
                    activation_condition=f"Use for {skill_id} evidence.",
                    execution_steps=[
                        SkillStep(order=1, action=f"Inspect {skill_id} evidence.")
                    ],
                    termination_condition="Return control after inspection.",
                    status="probationary",
                    frequency=3,
                    total_gain=3 * gain,
                    avg_gain=gain,
                    success_count=1 if gain > 0 else 0,
                )

            state.skills["helpful"] = probationary("helpful", 0.20)
            state.skills["harmful"] = probationary("harmful", -0.05)
            module._maintain(state)

        self.assertEqual(state.skills["helpful"].status, "validated")
        self.assertEqual(state.skills["harmful"].status, "retired")
        self.assertTrue(
            any(
                item.get("stage") == "promote probationary skill"
                and item.get("skill_id") == "helpful"
                for item in state.maintenance_log
            )
        )
        self.assertTrue(
            any(
                item.get("stage") == "probationary online rollback"
                and item.get("skill_id") == "harmful"
                for item in state.maintenance_log
            )
        )


    def test_freeze_retires_unresolved_probation_and_persists_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="freeze-bank",
                store_path=Path(tmp) / "skills.json",
            )
            state = module.store.load()
            state.skills["pending"] = ProceduralSkill(
                skill_id="pending",
                title="Pending",
                activation_condition="When evidence is incomplete.",
                execution_steps=[SkillStep(order=1, action="Inspect evidence.")],
                termination_condition="Stop after inspection.",
                status="probationary",
                frequency=1,
            )
            module.store.save(state)

            manifest = module.freeze_for_evaluation(
                output_path=Path(tmp) / "frozen.jsonl"
            )
            frozen = module.store.load()
            current_hash = module.bank_state_hash()

        self.assertEqual(frozen.skills["pending"].status, "retired")
        self.assertEqual(manifest["state_hash"], current_hash)
        self.assertEqual(manifest["retired_probationary_skill_ids"], ["pending"])


    def test_new_skill_id_includes_normalized_candidate_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="content-id",
                store_path=Path(tmp) / "skills.json",
            )
            evidence = EvaluationEvidence(
                session_id="content-id-session",
                task_description="Inspect current BGP evidence.",
                scenario="dc_clos_bgp",
            )
            steps = [
                SkillStep(
                    order=1,
                    action="Inspect route evidence.",
                    tool_name="show_route",
                )
            ]
            gradient = SemanticGradient(
                source_session_id=evidence.session_id,
                critique="Improve the procedure.",
                proposed_update="Use supported evidence.",
            )
            first = module.propose_skill(
                evidence=evidence,
                tool_steps=steps,
                critique=gradient,
                sampled_candidate=SkillCandidateDraft(
                    title="Route procedure",
                    initiation="When a route is missing.",
                    policy=["Inspect the route table."],
                    termination="Stop after route evidence.",
                ),
            )
            second = module.propose_skill(
                evidence=evidence,
                tool_steps=steps,
                critique=gradient,
                sampled_candidate=SkillCandidateDraft(
                    title="Neighbor procedure",
                    initiation="When a route is missing.",
                    policy=["Inspect neighbor state."],
                    termination="Stop after neighbor evidence.",
                ),
            )

        self.assertNotEqual(first.skill_id, second.skill_id)


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


    def test_replay_groups_activations_by_skill_and_episode(self) -> None:
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
                        activation_id="segment-1:1",
                        action="Inspect routes.",
                        tool_name="show_routes",
                        observation_summary="Route is absent.",
                    ),
                    SkillStep(
                        order=2,
                        skill_id="seed_hypothesis_elimination",
                        activation_id="segment-1:2",
                        action="Inspect neighbors.",
                        tool_name="show_neighbors",
                        observation_summary="Neighbor is inactive.",
                    ),
                    SkillStep(
                        order=3,
                        skill_id="seed_react_decision",
                        activation_id="segment-1:3",
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

        self.assertEqual(
            list(segments),
            ["seed_react_decision", "seed_hypothesis_elimination"],
        )
        react_episode = segments["seed_react_decision"]
        hypothesis_episode = segments["seed_hypothesis_elimination"]
        self.assertEqual(react_episode.step_count, 2)
        self.assertEqual(react_episode.skill_ids, ["seed_react_decision"])
        self.assertEqual(hypothesis_episode.step_count, 1)
        self.assertEqual(
            hypothesis_episode.skill_ids,
            ["seed_hypothesis_elimination"],
        )
        self.assertEqual(
            [item.activation_id for item in react_episode.transitions],
            ["segment-1:1", "segment-1:3"],
        )
        self.assertFalse(react_episode.transitions[0].done)
        self.assertTrue(react_episode.transitions[-1].done)
        self.assertTrue(hypothesis_episode.transitions[-1].done)


    def test_replay_rejects_skill_id_without_activation_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="missing-activation",
                store_path=Path(tmp) / "skills.json",
            )
            segments = module._segment_experiences(
                evidence=EvaluationEvidence(session_id="missing-activation-1"),
                tool_steps=[
                    SkillStep(
                        order=1,
                        skill_id="seed_react_decision",
                        action="Inspect current evidence.",
                    )
                ],
                reward=0.5,
                baseline=0.0,
                success=True,
                valid_skill_ids={"seed_react_decision"},
            )

        self.assertEqual(segments, {})


    def test_incomplete_llm_candidate_is_rejected_without_mutating_draft(self) -> None:
        draft = SkillCandidateDraft(
            title="Candidate",
            initiation="When evidence is incomplete.",
            policy=["Inspect current evidence."],
            termination="",
        )
        redacted = ProceduralMemoryModule._redact_skill_candidate(
            draft,
            EvaluationEvidence(session_id="incomplete-candidate"),
        )

        self.assertIsNone(redacted)
        self.assertEqual(draft.title, "Candidate")


    def test_evolution_batch_is_keyed_by_activated_parent(self) -> None:
        def experience(exp_id: str, skill_id: str) -> SkillExperience:
            return SkillExperience(
                experience_id=exp_id,
                session_id=exp_id,
                reward=0.1,
                skill_ids=[skill_id],
                transitions=[
                    SkillTransition(
                        skill_id=skill_id,
                        action="inspect",
                        tool_name="get_reachability",
                        done=True,
                    )
                ],
            )

        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="context-family",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=2,
            )
            state = module.store.load()
            parent = state.skills["seed_react_decision"]
            state.experiences.extend(
                [
                    experience("react-1", parent.skill_id),
                    experience("react-2", parent.skill_id),
                    experience("other", "seed_hypothesis_elimination"),
                ]
            )
            batch = module._evolution_batch(state, parent)

        self.assertEqual(
            {item.experience_id for item in batch},
            {"react-1", "react-2"},
        )


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
                critique=SemanticGradient(
                    source_session_id="procedural-only-1",
                    critique="Preserve the observed evidence order.",
                    proposed_update="Generalize the observed route procedure.",
                ),
                sampled_candidate=SkillCandidateDraft(
                    title="Route evidence procedure",
                    initiation="When current route evidence is incomplete.",
                    policy=["Inspect the current route table."],
                    termination="Stop after current route evidence is supported.",
                ),
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
                verifier="structured_replay",
            )
            self._seed_running_baseline(module, "generic")
            unrelated = SemanticGradient(
                source_session_id="new-1",
                critique="The active option was unrelated to the observed state.",
                proposed_update="Create a procedure for the observed action pattern.",
                component_update=SkillComponentGradient(is_related=False),
            )
            with (
                patch.object(
                    module,
                    "semantic_gradient_from_experience",
                    return_value=unrelated,
                ),
                patch.object(
                    module,
                    "aggregate_semantic_gradients",
                    return_value=unrelated,
                ) as aggregate,
                patch.object(
                    module,
                    "_llm_skill_candidate",
                    return_value=SkillCandidateDraft(
                        title="Service evidence procedure",
                        initiation="When current service evidence is incomplete.",
                        policy=["Inspect current service state."],
                        termination="Stop after current service state is supported.",
                    ),
                ),
                patch.object(
                    module,
                    "ppo_gate",
                    return_value=PPOGateDecision(
                        accepted=True,
                        reason="candidate passed offline prescreen",
                        candidate_score=0.2,
                        baseline_score=0.1,
                        j_score=0.1,
                        delta_j_score=0.1,
                        candidate_type="NEW",
                    ),
                ),
            ):
                first = module.learn_from_episode(
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
                            activation_id="new-1:1",
                            tool_name="inspect_service",
                            observation_summary="Service is unavailable.",
                        )
                    ],
                )
                report = module.learn_from_episode(
                    evidence=EvaluationEvidence(
                        session_id="new-2",
                        task_description="Inspect another unknown service failure.",
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
                            activation_id="new-2:1",
                            tool_name="inspect_service",
                            observation_summary="Service is unavailable.",
                        )
                    ],
                )
            skill = module.store.load().skills[report["skill_id"]]

        self.assertEqual(first["status"], "deferred")
        self.assertEqual(report["status"], "accepted")
        self.assertEqual(report["candidate_type"], "NEW")
        self.assertEqual(report["relevance_ratio"], 0.0)
        self.assertEqual(skill.parent_id, "seed_react_decision")
        self.assertEqual(aggregate.call_args.kwargs["gradients"], [unrelated])


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
            state.documents["inspect_host"].published = True
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
                allow_training_updates=False,
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
        self.assertNotIn("Tool Refinement contract deltas", prompt)
        self.assertNotIn("Use observed identifiers only.", prompt)


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


    def test_best_of_n_reuses_one_batch_semantic_gradient(self) -> None:
        schemas: list[type] = []
        prompts: list[str] = []

        class FakeModel:
            schema: type | None = None

            def with_structured_output(self, schema):
                self.schema = schema
                schemas.append(schema)
                return self

            def invoke(self, prompt):
                prompts.append(prompt)
                if self.schema is SkillCandidateDraft:
                    return SkillCandidateDraft(
                        title="Independent route candidate",
                        initiation="When current route evidence is incomplete.",
                        policy=[
                            "Inspect route evidence.",
                            "Cross-check the observation before termination.",
                        ],
                        termination=(
                            "Stop after route evidence is independently confirmed."
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
                skill_id="seed_react_decision",
                activation_id="batch:1",
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
        self.assertEqual(second["semantic_gradient_count"], 1)
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
        self.assertEqual(
            sum("batch semantic-gradient aggregator" in prompt for prompt in prompts),
            1,
        )
        load_model.assert_called_once()


    def test_attribute_mining_ignores_scenario_design_noise(self) -> None:
        attrs = infer_procedural_memory_attributes(
            (
                "Network Description: OSPF enterprise network with DHCP, DNS, "
                "HTTP web services and load balancer.\n\n"
                "Your goal is to analyze the network condition."
            ),
            tools=[],
        )

        self.assertNotIn("ospf", attrs.protocols)
        self.assertNotIn("dhcp", attrs.protocols)
        self.assertNotIn("addressing", attrs.services)


    def test_training_reward_prioritizes_localization_and_rca(
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

        self.assertAlmostEqual(_evidence_score(detection_only), 0.10)
        self.assertAlmostEqual(_evidence_score(partial), 0.45)
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
                    ground_truth_is_anomaly=True,
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
                        skill_id="seed_react_decision",
                        activation_id="s1:1",
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
                {
                    "agent": "procedural_memory_agent",
                    "phase": "skill_mdp_runtime",
                    "event": "skill_terminal_transition",
                    "active_skill_id": "seed_react_decision",
                    "activation_id": "session:1",
                    "action": "No anomaly is supported by the collected evidence.",
                    "status": "success",
                    "policy_state": policy_state,
                    "policy_context": policy_context,
                },
            ]
            trace.write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )

            steps = extract_skill_steps(trace)

        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0].skill_id, "seed_react_decision")
        self.assertEqual(steps[0].activation_id, "session:1")
        self.assertEqual(steps[0].tool_name, "ping_pair")
        self.assertEqual(steps[0].arguments_hint["host_a"], "pc1")
        self.assertIn("runtime interpreted output", steps[0].observation_summary)
        self.assertEqual(steps[0].policy_state, policy_state)
        self.assertEqual(steps[0].policy_context, policy_context)
        self.assertEqual(steps[1].tool_name, "")
        self.assertIn("No anomaly", steps[1].action)
        self.assertEqual(steps[1].activation_id, "session:1")


    def test_ppo_gate_accepts_successful_skill_for_read_time_use(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=1,
            )
            self._enable_fake_evolution(module)
            self._seed_running_baseline(module, "dc_clos_bgp")
            first = module.learn_from_episode(
                evidence=EvaluationEvidence(
                    session_id="s1",
                    task_description="BGP missing route advertisement between routers",
                    scenario="dc_clos_bgp",
                    root_cause=["bgp_missing_route_advertisement"],
                    ground_truth_is_anomaly=True,
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
                        skill_id="seed_react_decision",
                        activation_id="s1:1",
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
                        skill_id="seed_react_decision",
                        activation_id="s2:1",
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
            probationary = module.retrieve(
                query=ProceduralMemoryQuery(
                    text="BGP route is not advertised",
                    scenario="dc_clos_bgp",
                    protocols=["bgp"],
                    symptoms=["missing_route"],
                    tools=["frr_show_bgp_summary"],
                    top_k=10,
                ),
                include_probationary=True,
            )
            last_evolution = module.store.load().evolution_log[-1]

        self.assertEqual(first["status"], "deferred")
        self.assertEqual(report["status"], "accepted")
        self.assertTrue(
            set(last_evolution["generation_experience_ids"]).isdisjoint(
                last_evolution["verification_experience_ids"]
            )
        )
        self.assertGreater(report["episode_reward"], 0.0)
        self.assertGreater(report["episode_baseline"], 0.0)
        self.assertEqual(
            report["episode_advantage"],
            report["episode_reward"] - report["episode_baseline"],
        )
        self.assertTrue(report["episode_success"])
        self.assertNotIn(
            report["skill_id"], [item.skill.skill_id for item in retrieved]
        )
        self.assertIn(
            report["skill_id"], [item.skill.skill_id for item in probationary]
        )
        self.assertEqual(report["skill_status"], "probationary")


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
                        skill_id="seed_react_decision",
                        activation_id="partial:1",
                        tool_name="get_host_net_config",
                        observation_summary="Host has no IPv4 address.",
                    )
                ],
            )
            after = set(module.store.load().skills)

        self.assertEqual(report["status"], "deferred")
        self.assertEqual(report["semantic_gradient_count"], 0)
        self.assertEqual(after, before)
        self.assertIsNone(report["decision"])
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
                        skill_id="seed_react_decision",
                        activation_id="partial-localization:1",
                        tool_name="get_host_net_config",
                        observation_summary="Only one affected component was localized.",
                    )
                ],
            )
            state = module.store.load()
            after = set(state.skills)
            experience = state.experiences[-1]

        self.assertEqual(report["status"], "deferred")
        self.assertEqual(report["semantic_gradient_count"], 0)
        self.assertFalse(report["episode_success"])
        self.assertFalse(experience.success)
        self.assertGreater(experience.reward, 0.0)
        self.assertAlmostEqual(
            state.baselines["ospf_enterprise_dhcp::any-topology::unknown"],
            0.1 * experience.reward,
        )
        self.assertEqual(after, before)
        self.assertIsNone(report["decision"])


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
                        activation_id="partial-reuse:1",
                        tool_name="get_reachability",
                        observation_summary="Reachability is unknown.",
                    )
                ],
            )
            skill = module.store.load().skills["active_bad_skill"]

        self.assertEqual(report["status"], "deferred")
        self.assertLess(skill.avg_gain, 0.0)
        self.assertEqual(skill.success_count, 0)
        self.assertEqual(skill.failure_count, 1)


    def test_semantic_gradient_uses_llm_without_rule_fallback(self) -> None:
        class FakeModel:
            def with_structured_output(self, _schema):
                return self

            def invoke(self, _prompt):
                return SemanticGradientDraft(
                    source_session_id="partial-gradient",
                    critique="The current policy ended before RCA was supported.",
                    proposed_update="Add one independent RCA check.",
                    policy=["Verify RCA with independent current evidence."],
                    termination="Stop after RCA is supported.",
                )

        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                llm_backend="custom",
                model="test-model",
                store_path=Path(tmp) / "skills.json",
            )
            with patch(
                "agent.procedural_memory.service.load_model",
                return_value=FakeModel(),
            ):
                gradient = module.semantic_gradient(
                    evidence=EvaluationEvidence(
                        session_id="partial-gradient",
                        task_description="Host reachability failure.",
                    ),
                    tool_steps=[
                        SkillStep(
                            order=1,
                            action="Check reachability.",
                            tool_name="get_reachability",
                        )
                    ],
                )

        self.assertEqual(gradient.gradient_source, "llm")
        self.assertIn("independent RCA", gradient.proposed_update)


    def test_sampled_skill_activation_excludes_scenario_metadata(
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
                critique=SemanticGradient(
                    source_session_id="general-skill",
                    critique="Preserve the route evidence check.",
                    proposed_update="Generalize the route check.",
                ),
                sampled_candidate=SkillCandidateDraft(
                    title="Route evidence procedure",
                    initiation=(
                        "Use when current observations match an evidence signature "
                        "from observed tools."
                    ),
                    policy=["Check current route evidence."],
                    termination="Stop after current route evidence is supported.",
                ),
            )

        self.assertIn("evidence signature", skill.activation_condition)
        self.assertIn("observed tools", skill.activation_condition)
        self.assertNotIn("benchmark_specific_scenario", skill.activation_condition)
        self.assertNotIn("benchmark_specific_scenario", skill.skill_id)


    def test_refining_generic_seed_uses_sampled_evidence_activation(self) -> None:
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
                critique=SemanticGradient(
                    source_session_id="refine-generic",
                    critique="Specialize initiation from current evidence.",
                    proposed_update="Use the observed BGP route evidence.",
                ),
                sampled_candidate=SkillCandidateDraft(
                    title="BGP route evidence procedure",
                    initiation=(
                        "Use when the current BGP evidence signature shows missing "
                        "route information."
                    ),
                    policy=["Inspect current BGP route state."],
                    termination="Stop after current BGP route evidence is supported.",
                ),
            )

        self.assertIn("evidence signature", skill.activation_condition)
        self.assertIn("bgp", skill.activation_condition.lower())
        self.assertNotIn(
            "deciding between broad exploration", skill.activation_condition
        )

