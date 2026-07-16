"""Tests for Skill-Pro Procedural Memory."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from langchain_core.tools import StructuredTool

from agent.procedural_memory.models import (
    EvaluationEvidence,
    ProceduralMemoryQuery,
    ProceduralSkill,
    SemanticGradientDraft,
    SkillCandidateDraft,
    SkillExperience,
    SkillStep,
    SkillTransition,
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




class SkillProProceduralMemoryTestPart3(unittest.TestCase):
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


    def test_loading_bank_preserves_online_seed_quarantine(
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

        self.assertEqual(repaired.skills["seed_react_decision"].status, "retired")
        self.assertEqual(repaired.skills["unsafe-learned"].status, "validated")


    def test_episode_without_attributed_skill_updates_baseline_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="all-episodes-baseline",
                store_path=Path(tmp) / "skills.json",
            )
            evidence = EvaluationEvidence(
                session_id="clean-1",
                scenario="routing",
                ground_truth_is_anomaly=False,
                metrics={"detection_score": 0.8},
            )

            first = module.learn_from_episode(evidence=evidence, tool_steps=[])
            second = module.learn_from_episode(evidence=evidence, tool_steps=[])
            state = module.store.load()

        self.assertEqual(first["status"], "deferred")
        self.assertEqual(second["status"], "deferred")
        self.assertEqual(state.iteration, 1)
        self.assertEqual(len(state.episodes), 1)
        self.assertAlmostEqual(
            state.baselines["routing::any-topology::clean"],
            0.08,
        )
        self.assertEqual(state.experiences, [])


    def test_default_training_defers_evolution_until_batch_is_ready(self) -> None:
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
                        skill_id="seed_react_decision",
                        activation_id="s1:1",
                        tool_name="frr_show_bgp_summary",
                    )
                ],
            )
            state = module.store.load()
            stats = module.store.bank_stats()

        self.assertEqual(report["status"], "deferred")
        self.assertEqual(report["sample_count"], 1)
        self.assertEqual(report["required_sample_count"], 3)
        self.assertEqual(report["decision"], None)
        self.assertEqual(len(state.experiences), 1)
        self.assertEqual(report["semantic_gradient_source"], "pending")
        self.assertEqual(stats["ppo_decisions"], 0)
        self.assertEqual(state.evolution_log[-1]["action"], "deferred")


    def test_evolution_failure_preserves_experiences_for_retry(self) -> None:
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
                skill_id="seed_react_decision",
                activation_id="routing:1",
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
        self.assertEqual(second["status"], "deferred")
        self.assertEqual(third["status"], "deferred")
        self.assertEqual(len(used), 0)
        self.assertEqual(len(unused), 3)
        self.assertEqual(len(gate_events), 0)


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
            {"exp-1", "exp-3", "exp-5"},
        )


    def test_generic_seed_evolution_batch_uses_parent_buffer(
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

            parent = state.skills["seed_explore_exploit"]
            batch = module._evolution_batch(state, parent)

        self.assertEqual(len(batch), 3)
        self.assertEqual(
            {item.experience_id for item in batch},
            {"bgp-low", "p4-high", "bgp-current"},
        )


    def test_generic_seed_batch_excludes_other_parent_only(self) -> None:
        def experience(
            exp_id: str,
            text: str,
            skill_id: str = "seed_react_decision",
        ) -> SkillExperience:
            return SkillExperience(
                experience_id=exp_id,
                session_id=exp_id,
                reward=1.0,
                advantage=1.0,
                skill_ids=[skill_id],
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
                        "other-parent",
                        "P4 hosts have successful ICMP reachability.",
                        "seed_hypothesis_elimination",
                    ),
                ]
            )

            parent = state.skills["seed_react_decision"]
            batch = module._evolution_batch(state, parent)

        self.assertEqual(len(batch), 3)
        self.assertNotIn("other-parent", {item.experience_id for item in batch})


    def test_refining_generic_seed_preserves_seed_and_resets_candidate_stats(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=1,
            )
            self._enable_fake_evolution(module)
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
                        activation_id="successful-refinement-holdout:1",
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
                        activation_id="successful-refinement:1",
                        action="Inspect route state.",
                        tool_name="show_route",
                        observation_summary="The expected route is missing.",
                    )
                ],
            )
            state = module.store.load()
            candidate = state.skills[report["skill_id"]]

        self.assertEqual(first["status"], "deferred")
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
            parent = module._next_evolution_parent(module.store.load())

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
                        activation_id="clean-after-bad-parent:1",
                        tool_name="frr_show_bgp_summary",
                        observation_summary="BGP route is missing.",
                    )
                ],
            )
            state = module.store.load()
            experience = state.experiences[-1]

        self.assertEqual(first["status"], "deferred")
        self.assertEqual(experience.skill_ids, [bad_skill_id])
        self.assertEqual(state.skills[bad_skill_id].status, "retired")


    def test_verification_prefers_held_out_after_in_batch_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=1,
            )
            self._enable_fake_evolution(module)
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
                    skill_id="seed_react_decision",
                    activation_id="ospf:1",
                    tool_name="frr_show_ip_ospf_neighbor",
                )
            ]
            first = module.learn_from_episode(evidence=good, tool_steps=steps)
            second = module.learn_from_episode(evidence=bad, tool_steps=steps)
            event = module.store.load().evolution_log[-1]

        self.assertEqual(first["status"], "deferred")
        self.assertIn(second["status"], {"accepted", "rejected"})
        self.assertTrue(
            set(event["generation_experience_ids"]).isdisjoint(
                event["verification_experience_ids"]
            )
        )


    def test_ppo_gate_uses_replayed_transition_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=1,
                verifier="structured_replay",
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
            before_selection = module.store.load()
            self.assertEqual(
                sum(skill.reuse_count for skill in before_selection.skills.values()),
                0,
            )

            active = module.activate_skill("seed_react_decision")
            state = module.store.load()
            active_id = active.skill.skill_id if active is not None else ""
            skill = state.skills[active_id]

        self.assertIsNotNone(active)
        self.assertEqual(active.skill.skill_id, skill.skill_id)
        self.assertEqual(skill.reuse_count, 1)


    def test_selection_pool_includes_untried_seed_skill(self) -> None:
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

            candidate_ids = {skill.skill_id for skill in module.selection_candidates()}

        self.assertIn("seed_hypothesis", candidate_ids)


    def test_selection_pool_filters_unstable_candidate(self) -> None:
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
                frequency=10,
                total_gain=-10.0,
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

            candidate_ids = {skill.skill_id for skill in module.selection_candidates()}
            active = module.activate_skill("stable_lower_score")
            state = module.store.load()

        self.assertIsNotNone(active)
        self.assertNotIn("risky_high_score", candidate_ids)
        self.assertIn("stable_lower_score", candidate_ids)
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
                allow_training_updates=False,
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


    def test_runtime_snapshot_tracks_training_prompt_overhead(self) -> None:
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
                allow_training_updates=False,
                session=SimpleNamespace(
                    session_id="s2",
                    scenario_name="simple_bgp",
                    scenario_topo_size="small",
                ),
                task_description="Host reachability failure",
                tools=[tool],
                session_dir=tmp,
                meta_controller_llm=_FakeSkillController(["seed_react_decision"]),
            )

            wrapped = runtime.wrap_tools([tool])[0]
            runtime.prompt_suffix()
            wrapped.invoke({"host": "pc1"})
            snapshot = runtime.snapshot()

        self.assertGreater(snapshot["prompt_added_tokens"], 0)
        self.assertEqual(snapshot["config"]["max_skill_age"], 8)
        self.assertEqual(
            snapshot["selection_policy"],
            "llm_direct",
        )
        self.assertEqual(snapshot["tool_description_added_tokens"], 0)
        self.assertEqual(snapshot["prompt_injection_count"], 1)
        self.assertEqual(snapshot["tool_description_injection_count"], 0)
        self.assertEqual(
            snapshot["total_added_tokens"],
            snapshot["prompt_added_tokens"] + snapshot["tool_description_added_tokens"],
        )


    def test_episode_training_credits_runtime_active_skill(self) -> None:
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
        self.assertEqual(report["status"], "deferred")
        self.assertEqual(stats["ppo_decisions"], 0)
        self.assertIsNone(stats["last_candidate_alignment"])


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
                allow_training_updates=False,
                session=SimpleNamespace(
                    session_id="s2",
                    scenario_name="simple_bgp",
                    scenario_topo_size="small",
                ),
                task_description="Host reachability failure",
                tools=[tool],
                session_dir=tmp,
                meta_controller_llm=_FakeSkillController(
                    ["one_step_ping", ""],
                    termination_status="DONE",
                ),
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
            "meta_controller_done",
        )


    def test_skill_runtime_llm_meta_controller_can_terminate_skill(self) -> None:
        def ping_host(host: str) -> str:
            return f"{host} reachable"

        class FakeMetaController(_FakeSkillController):
            def __init__(self) -> None:
                super().__init__(["meta_ping", ""], termination_status="DONE")

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
                allow_training_updates=False,
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

        class FakeMetaController(_FakeSkillController):
            def __init__(self) -> None:
                super().__init__(["meta_ping"], termination_status="CONTINUE")

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
                allow_training_updates=False,
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
                allow_training_updates=False,
                session=SimpleNamespace(
                    session_id="s2",
                    scenario_name="simple_bgp",
                    scenario_topo_size="small",
                ),
                task_description="Host reachability failure",
                tools=[tool],
                session_dir=tmp,
                meta_controller_llm=_FakeSkillController(["prompt_ping"]),
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
        self.assertNotIn(decision_context, prompt)
        self.assertIn(decision_context, policy_state)
        self.assertIn("neither proves the diagnosis", prompt)
        self.assertIn("return a concise diagnosis report", prompt)
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
                allow_training_updates=False,
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
                allow_training_updates=False,
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
                allow_training_updates=False,
                session=SimpleNamespace(
                    session_id="s2",
                    scenario_name="simple_bgp",
                    scenario_topo_size="small",
                ),
                task_description="Host reachability failure",
                tools=[tool],
                session_dir=tmp,
                meta_controller_llm=_FakeSkillController(
                    ["one_step_ping", ""],
                    termination_status="DONE",
                ),
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
            "meta_controller_done",
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
                allow_training_updates=False,
                session=SimpleNamespace(
                    session_id="s2",
                    scenario_name="simple_bgp",
                    scenario_topo_size="small",
                ),
                task_description="Host reachability failure",
                tools=[tool],
                session_dir=tmp,
                meta_controller_llm=_FakeSkillController(
                    ["one_step_ping", ""],
                    termination_status="DONE",
                ),
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
        self.assertEqual(snapshot["inflight_tool_calls"], 0)
        self.assertEqual(len(post_tool_terms), 1)
        self.assertEqual(terminations, post_tool_terms)
        self.assertFalse(batch_reselects)
