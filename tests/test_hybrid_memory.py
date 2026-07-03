"""Tests for Skill-Pro procedural memory."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.tools import StructuredTool

from agent.memory.models import (
    EvaluationEvidence,
    MemoryQuery,
    ProceduralSkill,
    SemanticGradient,
    SemanticGradientDraft,
    SkillComponentGradient,
    SkillExperience,
    SkillStep,
    SkillTransition,
)
from agent.memory.adapter import MemoryAugmentedAgent
from agent.memory.runtime import SkillToolRuntime
from agent.memory.service import ProceduralMemoryModule, _evidence_score
from agent.memory.workflow import evolve_session_memory, extract_skill_steps
from agent.tool_evolution.curator import rewrite_documentation
from agent.tool_evolution.models import DraftExploration, ToolDocumentation
from agent.tool_evolution.runtime import ToolEvolutionRuntime
from agent.tool_evolution.store import ToolEvolutionStore
from nika.evaluator.result_log import build_eval_result_from_session_dir
from nika.workflows.eval.session import run_eval_metrics


class SkillProMemoryTest(unittest.TestCase):
    def test_detection_only_episode_gets_no_positive_learning_reward(self) -> None:
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

        self.assertEqual(_evidence_score(detection_only), 0.0)
        self.assertGreater(_evidence_score(complete), _evidence_score(partial))
        self.assertGreater(_evidence_score(partial), _evidence_score(detection_only))

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
            trace.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

            steps = extract_skill_steps(trace)

        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].tool_name, "ping_pair")
        self.assertEqual(steps[0].status, "success")
        self.assertIn("2 packets received", steps[0].observation_summary)
        self.assertNotIn("Integrated learning guidance", steps[0].observation_summary)

    def test_extracts_runtime_skill_transitions_before_plain_tool_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "messages.jsonl"
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
                    "agent": "memory_agent",
                    "phase": "skill_mdp_runtime",
                    "event": "skill_transition",
                    "active_skill_id": "seed_react_decision",
                    "tool": "ping_pair",
                    "tool_input": {"host_a": "pc1", "host_b": "pc2"},
                    "status": "success",
                    "observation_summary": "runtime interpreted output",
                },
            ]
            trace.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

            steps = extract_skill_steps(trace)

        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].skill_id, "seed_react_decision")
        self.assertEqual(steps[0].tool_name, "ping_pair")
        self.assertEqual(steps[0].arguments_hint["host_a"], "pc1")
        self.assertIn("runtime interpreted output", steps[0].observation_summary)

    def test_ppo_gate_accepts_successful_skill_and_retrieves_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=1,
            )
            report = module.learn_from_episode(
                evidence=EvaluationEvidence(
                    session_id="s1",
                    task_description="BGP missing route advertisement between routers",
                    scenario="dc_clos_bgp",
                    root_cause=["bgp_missing_route_advertisement"],
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
            retrieved = module.retrieve(
                query=MemoryQuery(
                    text="BGP route is not advertised",
                    scenario="dc_clos_bgp",
                    protocols=["bgp"],
                    symptoms=["missing_route"],
                    tools=["frr_show_bgp_summary"],
                    top_k=3,
                )
            )
            context = module.format_context(retrieved)

        self.assertEqual(report["status"], "accepted")
        self.assertGreater(report["episode_reward"], 0.0)
        self.assertEqual(report["episode_baseline"], 0.0)
        self.assertEqual(report["episode_advantage"], report["episode_reward"])
        self.assertTrue(report["episode_success"])
        self.assertIn(report["skill_id"], [item.skill.skill_id for item in retrieved])
        self.assertIn("Activation", context)
        self.assertNotIn("bgp_missing_route_advertisement", context)

    def test_runtime_context_does_not_label_candidate_as_active_without_active_skill(self) -> None:
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
                query=MemoryQuery(
                    text="Host reachability failure",
                    scenario="simple_bgp",
                    tools=["ping_pair"],
                    top_k=3,
                )
            )

            context = module.format_context(retrieved, active_skill_id="")

        self.assertIn("CANDIDATE Skill candidate_ping", context)
        self.assertNotIn("ACTIVE Skill candidate_ping", context)

    def test_runtime_context_redacts_known_root_cause_ids_from_dirty_skills(self) -> None:
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
                query=MemoryQuery(
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
        self.assertEqual(report["required_sample_count"], 3)
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
            )
            first = module.learn_from_episode(
                evidence=evidence("s1"),
                tool_steps=steps,
            )
            second = module.learn_from_episode(
                evidence=evidence("s2"),
                tool_steps=steps,
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
        self.assertIsNotNone(second["decision"])
        self.assertEqual(third["status"], "deferred")
        self.assertEqual(third["sample_count"], 1)
        self.assertEqual(len(used), 2)
        self.assertEqual(len(unused), 1)
        self.assertEqual(len(gate_events), 1)
        self.assertEqual(
            set(gate_events[0]["sample_experience_ids"]),
            {exp.experience_id for exp in used},
        )

    def test_seed_skill_pool_is_available_in_fresh_bank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=1,
            )
            retrieved = module.retrieve(
                query=MemoryQuery(
                    text="Plan a diagnosis with little evidence",
                    scenario="simple_bgp",
                    top_k=3,
                )
            )
            state = module.store.load()
            context = module.format_context(retrieved)

        self.assertGreaterEqual(
            len([skill_id for skill_id in state.skills if skill_id.startswith("seed_")]),
            6,
        )
        self.assertTrue(any(item.skill.skill_id.startswith("seed_") for item in retrieved))
        self.assertIn("Skill-MDP", context)

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
                query=MemoryQuery(
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

    def test_ppo_gate_rejects_weaker_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=1,
            )
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
            steps = [SkillStep(order=1, action="Check OSPF neighbors.", tool_name="frr_show_ip_ospf_neighbor")]
            first = module.learn_from_episode(evidence=good, tool_steps=steps)
            second = module.learn_from_episode(evidence=bad, tool_steps=steps)

        self.assertEqual(first["status"], "accepted")
        self.assertEqual(second["status"], "rejected")

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
                    SkillStep(order=1, action="Query DNS from the client.", tool_name="dig")
                ],
            )
            before_selection = module.store.load()
            self.assertEqual(
                sum(skill.reuse_count for skill in before_selection.skills.values()),
                0,
            )

            active = module.select_skill(
                query=MemoryQuery(
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

    def test_llm_topk_lcb_selector_uses_llm_nominees_then_lcb(self) -> None:
        class FakeSelector:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            def invoke(self, prompt: str):
                self.prompts.append(prompt)
                return SimpleNamespace(
                    content=(
                        "<choice>weak_high_score</choice>\n"
                        "<choice>strong_lower_score</choice>"
                    )
                )

        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            state = module.store.load()
            state.skills["weak_high_score"] = ProceduralSkill(
                skill_id="weak_high_score",
                title="Weak high score",
                activation_condition="Use for host reachability failure with ping_host.",
                execution_steps=[
                    SkillStep(order=1, action="Call ping_host.", tool_name="ping_host")
                ],
                termination_condition="Stop after ping evidence.",
                tools=["ping_host"],
                status="validated",
                score=5.0,
                frequency=10,
                avg_gain=-0.5,
                maturity=5,
            )
            state.skills["strong_lower_score"] = ProceduralSkill(
                skill_id="strong_lower_score",
                title="Strong lower score",
                activation_condition="Use for host reachability failure with ping_host.",
                execution_steps=[
                    SkillStep(
                        order=1,
                        action="Call ping_host then inspect route.",
                        tool_name="ping_host",
                    )
                ],
                termination_condition="Stop after ping and route evidence.",
                tools=["ping_host"],
                status="validated",
                score=1.0,
                frequency=10,
                avg_gain=0.7,
                maturity=5,
            )
            module.store.save(state)
            selector = FakeSelector()

            active = module.select_skill_llm_topk_lcb(
                query=MemoryQuery(
                    text="Host reachability failure",
                    scenario="simple_bgp",
                    tools=["ping_host"],
                    top_k=5,
                ),
                llm_agent=selector,
            )
            state = module.store.load()

        self.assertIsNotNone(active)
        self.assertIn("[AVAILABLE SKILL-MDP OPTIONS]", selector.prompts[-1])
        self.assertEqual(active.skill.skill_id, "strong_lower_score")
        self.assertEqual(state.skills["strong_lower_score"].reuse_count, 1)
        self.assertEqual(state.skills["weak_high_score"].reuse_count, 0)

    def test_llm_topk_lcb_selector_falls_back_when_llm_returns_none(self) -> None:
        class FakeSelector:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            def invoke(self, prompt: str):
                self.prompts.append(prompt)
                return SimpleNamespace(content="<choice>NONE</choice>")

        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            state = module.store.load()
            state.skills["generic_reachability"] = ProceduralSkill(
                skill_id="generic_reachability",
                title="Generic reachability",
                activation_condition="Use for host reachability failure.",
                execution_steps=[
                    SkillStep(order=1, action="Check reachability.", tool_name="ping_host")
                ],
                termination_condition="Stop after reachability evidence.",
                tools=["ping_host"],
                status="validated",
                score=1.0,
                frequency=3,
                avg_gain=0.3,
            )
            module.store.save(state)
            selector = FakeSelector()

            active = module.select_skill_llm_topk_lcb(
                query=MemoryQuery(
                    text="Host reachability failure",
                    scenario="simple_bgp",
                    tools=["ping_host"],
                    top_k=5,
                ),
                llm_agent=selector,
            )
            state = module.store.load()

        self.assertIsNotNone(active)
        self.assertEqual(active.skill.skill_id, "generic_reachability")
        self.assertEqual(state.skills["generic_reachability"].reuse_count, 1)
        self.assertIn("<choice>NONE</choice>", selector.prompts[-1])

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
                query=MemoryQuery(
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
                    SkillStep(order=1, action="Check reachability.", tool_name="ping_host")
                ],
                termination_condition="Stop after reachability evidence.",
                tools=["ping_host"],
                status="validated",
                score=1.0,
                frequency=1,
                avg_gain=-1.0,
                maturity=5,
            )
            state.skills["stable_lower_score"] = ProceduralSkill(
                skill_id="stable_lower_score",
                title="Stable lower score",
                activation_condition="Use for host reachability failure.",
                execution_steps=[
                    SkillStep(order=1, action="Check reachability.", tool_name="ping_host")
                ],
                termination_condition="Stop after reachability evidence.",
                tools=["ping_host"],
                status="validated",
                score=0.5,
                frequency=20,
                avg_gain=0.3,
                maturity=5,
            )
            module.store.save(state)

            active = module.select_skill(
                query=MemoryQuery(
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

    def test_llm_topk_lcb_selector_skips_lcb_rejected_nominee(self) -> None:
        class FakeSelector:
            def invoke(self, _prompt: str):
                return SimpleNamespace(
                    content=(
                        "<choice>risky_high_score</choice>\n"
                        "<choice>stable_lower_score</choice>"
                    )
                )

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
                    SkillStep(order=1, action="Check reachability.", tool_name="ping_host")
                ],
                termination_condition="Stop after reachability evidence.",
                tools=["ping_host"],
                status="validated",
                score=1.0,
                frequency=1,
                avg_gain=-1.0,
                maturity=5,
            )
            state.skills["stable_lower_score"] = ProceduralSkill(
                skill_id="stable_lower_score",
                title="Stable lower score",
                activation_condition="Use for host reachability failure.",
                execution_steps=[
                    SkillStep(order=1, action="Check reachability.", tool_name="ping_host")
                ],
                termination_condition="Stop after reachability evidence.",
                tools=["ping_host"],
                status="validated",
                score=0.5,
                frequency=20,
                avg_gain=0.3,
                maturity=5,
            )
            module.store.save(state)

            active = module.select_skill_llm_topk_lcb(
                query=MemoryQuery(
                    text="Host reachability failure",
                    scenario="simple_bgp",
                    tools=["ping_host"],
                    top_k=5,
                ),
                llm_agent=FakeSelector(),
            )
            state = module.store.load()

        self.assertIsNotNone(active)
        self.assertEqual(active.skill.skill_id, "stable_lower_score")
        self.assertEqual(state.skills["risky_high_score"].reuse_count, 0)
        self.assertEqual(state.skills["stable_lower_score"].reuse_count, 1)

    def test_runtime_prompt_selection_can_use_llm_topk_lcb_selector(self) -> None:
        def ping_host(host: str) -> str:
            return f"{host} reachable"

        class FakeSelector:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            def invoke(self, prompt: str):
                self.prompts.append(prompt)
                return SimpleNamespace(content="<choice>strong_ping</choice>")

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
            state.skills["strong_ping"] = ProceduralSkill(
                skill_id="strong_ping",
                title="Strong ping selector skill",
                activation_condition="Use for host reachability failure with ping_host.",
                execution_steps=[
                    SkillStep(order=1, action="Call ping_host.", tool_name="ping_host")
                ],
                termination_condition="Stop after ping evidence.",
                tools=["ping_host"],
                status="validated",
                score=1.0,
                frequency=3,
                avg_gain=0.4,
            )
            module.store.save(state)
            selector = FakeSelector()
            runtime = SkillToolRuntime(
                memory=module,
                memory_mode="read",
                session=SimpleNamespace(
                    session_id="s2",
                    scenario_name="simple_bgp",
                    scenario_topo_size="small",
                ),
                task_description="Host reachability failure",
                tools=[tool],
                session_dir=tmp,
                meta_controller_llm=selector,
                skill_selector_mode="llm_topk_lcb",
            )

            prompt = runtime.prompt_suffix()
            snapshot = runtime.snapshot()
            state = module.store.load()

        self.assertIn("strong_ping", prompt)
        self.assertEqual(snapshot["skill_selector_mode"], "llm_topk_lcb")
        self.assertEqual(snapshot["active_skill_id"], "strong_ping")
        self.assertEqual(snapshot["prompt_selection_count"], 1)
        self.assertEqual(state.skills["strong_ping"].reuse_count, 1)
        self.assertTrue(selector.prompts)

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
                memory=module,
                memory_mode="read",
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
        self.assertGreater(snapshot["tool_description_added_tokens"], 0)
        self.assertGreater(snapshot["followup_added_tokens"], 0)
        self.assertEqual(snapshot["prompt_injection_count"], 1)
        self.assertEqual(snapshot["tool_description_injection_count"], 1)
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
                        tool_name="ping_pair",
                        observation_summary="packet loss observed",
                        status="success",
                    ),
                    SkillStep(
                        order=2,
                        action="Use active ReAct skill with route evidence.",
                        skill_id="seed_react_decision",
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
        self.assertEqual(credited.frequency, 2)
        self.assertEqual(experience.transitions[0].skill_id, "seed_react_decision")
        self.assertEqual(experience.transitions[1].skill_id, "seed_react_decision")
        self.assertGreaterEqual(stats["ppo_decisions"], 1)
        self.assertIn("last_candidate_alignment", stats)

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
                memory=module,
                memory_mode="read",
                session=SimpleNamespace(
                    session_id="s2",
                    scenario_name="simple_bgp",
                    scenario_topo_size="small",
                ),
                task_description="Host reachability failure",
                tools=[tool],
                session_dir=tmp,
            )

            runtime.before_tool(tool_name="ping_host", tool_input={"host": "pc1"})
            runtime.after_tool(
                tool_name="ping_host",
                tool_input={"host": "pc1"},
                result="pc1 reachable",
            )
            runtime.before_tool(tool_name="ping_host", tool_input={"host": "pc2"})
            rows = [
                json.loads(line)
                for line in (Path(tmp) / "messages.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
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
                memory=module,
                memory_mode="read",
                session=SimpleNamespace(
                    session_id="s2",
                    scenario_name="simple_bgp",
                    scenario_topo_size="small",
                ),
                task_description="Host reachability failure",
                tools=[tool],
                session_dir=tmp,
                meta_controller_llm=meta,
                meta_controller_mode="llm",
            )

            runtime.before_tool(tool_name="ping_host", tool_input={"host": "pc1"})
            runtime.after_tool(
                tool_name="ping_host",
                tool_input={"host": "pc1"},
                result="pc1 reachable",
            )
            snapshot = runtime.snapshot()
            rows = [
                json.loads(line)
                for line in (Path(tmp) / "messages.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

        meta_events = [
            row for row in rows if row.get("event") == "skill_meta_controller"
        ]
        terminations = [
            row for row in rows if row.get("event") == "skill_termination"
        ]
        self.assertEqual(snapshot["meta_controller_mode"], "llm")
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
                memory=module,
                memory_mode="read",
                session=SimpleNamespace(
                    session_id="s2",
                    scenario_name="simple_bgp",
                    scenario_topo_size="small",
                ),
                task_description="Host reachability failure",
                tools=[tool],
                session_dir=tmp,
                meta_controller_llm=meta,
                meta_controller_mode="llm",
            )

            runtime.before_tool(tool_name="ping_host", tool_input={"host": "pc1"})
            runtime.after_tool(
                tool_name="ping_host",
                tool_input={"host": "pc1"},
                result="pc1 reachable",
            )
            runtime.prompt_suffix()
            snapshot = runtime.snapshot()
            rows = [
                json.loads(line)
                for line in (Path(tmp) / "messages.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

        meta_events = [
            row for row in rows if row.get("event") == "skill_meta_controller"
        ]
        self.assertEqual(len(meta.prompts), 1)
        self.assertEqual(snapshot["meta_controller_cache_hits"], 1)
        self.assertEqual(meta_events[-1]["status"], "cached")

    def test_skill_runtime_selects_active_skill_in_prompt_before_tool_choice(self) -> None:
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
                memory=module,
                memory_mode="read",
                session=SimpleNamespace(
                    session_id="s2",
                    scenario_name="simple_bgp",
                    scenario_topo_size="small",
                ),
                task_description="Host reachability failure",
                tools=[tool],
                session_dir=tmp,
            )

            prompt = runtime.prompt_suffix()
            runtime.prompt_suffix()
            snapshot = runtime.snapshot()
            state = module.store.load()
            rows = [
                json.loads(line)
                for line in (Path(tmp) / "messages.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

        prompt_activations = [
            row
            for row in rows
            if row.get("event") == "skill_activation"
            and row.get("source") == "prompt"
        ]
        self.assertIn("Advisory Skill-MDP option selected before next LLM action", prompt)
        self.assertIn("not a final diagnosis stop condition", prompt)
        self.assertIn("prompt_ping", prompt)
        self.assertEqual(snapshot["active_skill_id"], "prompt_ping")
        self.assertEqual(snapshot["prompt_selection_count"], 1)
        self.assertEqual(state.skills["prompt_ping"].reuse_count, 1)
        self.assertEqual(len(prompt_activations), 1)

    def test_skill_runtime_read_only_prompt_does_not_activate_or_record_reuse(self) -> None:
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
                memory=module,
                memory_mode="read",
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

        self.assertIn("prompt_ping", prompt)
        self.assertIn("read-only planning context", prompt)
        self.assertNotIn("selected before next LLM action", prompt)
        self.assertEqual(snapshot["active_skill_id"], "")
        self.assertEqual(snapshot["prompt_selection_count"], 0)
        self.assertEqual(state.skills["prompt_ping"].reuse_count, 0)
        self.assertFalse(log_path.exists())

    def test_skill_runtime_refreshes_option_in_tool_output_after_termination(self) -> None:
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
                memory=module,
                memory_mode="read",
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
                for line in (Path(tmp) / "messages.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
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
        self.assertEqual(snapshot["post_tool_selection_count"], 1)
        self.assertEqual(snapshot["skill_age"], 0)
        self.assertIn("Active Skill-MDP option", output)
        self.assertIn("Use current tool output as evidence", output)

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
                memory=module,
                memory_mode="read",
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
                for line in (Path(tmp) / "messages.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
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
                memory=module,
                memory_mode="read",
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
                for line in (Path(tmp) / "messages.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
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

    def test_skill_runtime_logs_tool_deviation_without_posthoc_reselection(self) -> None:
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
                memory=module,
                memory_mode="read",
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
                for line in (Path(tmp) / "messages.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
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

    def test_skill_prompt_links_active_skill_to_draft_tool_checks(self) -> None:
        def ping_host(host: str) -> str:
            return f"{host} reachable"

        with tempfile.TemporaryDirectory() as tmp:
            tool = StructuredTool.from_function(
                ping_host,
                name="ping_host",
                description="Ping one host.",
            )
            session = SimpleNamespace(
                session_id="s2",
                scenario_name="simple_bgp",
                scenario_topo_size="small",
                task_description="Host reachability failure",
            )
            draft_store = ToolEvolutionStore("draft", root=tmp)
            rewrite_documentation(
                draft_store,
                trials=[],
                tool_descriptions={"ping_host": "Ping one host."},
                metrics={},
                session_id="s1",
                task_description="Host reachability failure",
            )
            doc = draft_store.get_document("ping_host")
            assert doc is not None
            doc.usage_notes.append("Use exact host names from the active topology.")
            doc.exploration_suggestions.append(
                "Ping pc1 to verify endpoint reachability."
            )
            draft_store.upsert_document(doc)
            draft_state = draft_store.load()
            draft_state.explorations.append(
                DraftExploration(
                    exploration_id="explore_ping_pc1",
                    session_id="s1",
                    tool_name="ping_host",
                    intent="diagnosis_check",
                    user_query="Ping pc1.",
                    parameters={"host": "pc1"},
                    status="planned",
                    next_exploration="Ping pc1 to verify endpoint reachability.",
                )
            )
            draft_store.save(draft_state)
            draft_runtime = ToolEvolutionRuntime(
                session=session,
                primitive_tools=[tool],
                library_id="draft",
                store=draft_store,
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
                        action="Call ping_host and interpret reachability.",
                        tool_name="ping_host",
                    )
                ],
                termination_condition="Stop after ping evidence is interpreted.",
                tools=["ping_host"],
                status="validated",
                score=2.0,
            )
            module.store.save(state)
            selector = SimpleNamespace(
                prompts=[],
                invoke=lambda prompt: (
                    selector.prompts.append(prompt)
                    or SimpleNamespace(content="<choice>prompt_ping</choice>")
                ),
            )
            runtime = SkillToolRuntime(
                memory=module,
                memory_mode="read",
                session=session,
                task_description="Host reachability failure",
                tools=[tool],
                session_dir=tmp,
                tool_evolution_runtime=draft_runtime,
                meta_controller_llm=selector,
                skill_selector_mode="llm_topk_lcb",
            )

            prompt = runtime.prompt_suffix()

        self.assertIn("Active skill-tool links", prompt)
        self.assertIn("ping_host", prompt)
        self.assertIn("DRAFT tool documentation memory", prompt)
        self.assertIn("DRAFT checks", prompt)
        self.assertIn("DRAFT active exploration queue", selector.prompts[-1])
        self.assertIn("ping_host", selector.prompts[-1])

    def test_skill_runtime_logs_planned_draft_exploration_with_tool_call(self) -> None:
        def ping_host(host: str) -> str:
            return f"{host} reachable"

        with tempfile.TemporaryDirectory() as tmp:
            tool = StructuredTool.from_function(
                ping_host,
                name="ping_host",
                description="Ping one host.",
            )
            session = SimpleNamespace(
                session_id="s2",
                scenario_name="simple_bgp",
                scenario_topo_size="small",
                task_description="Host reachability failure",
            )
            draft_store = ToolEvolutionStore("draft", root=tmp)
            draft_state = draft_store.load()
            draft_state.documents["ping_host"] = ToolDocumentation(
                name="ping_host",
                description="Ping one host.",
            )
            draft_state.explorations.append(
                DraftExploration(
                    exploration_id="explore_ping_pc1",
                    session_id="s1",
                    tool_name="ping_host",
                    intent="diagnosis_check",
                    user_query="Check pc1 reachability.",
                    parameters={"host": "pc1"},
                    status="planned",
                    next_exploration="Ping pc1 to verify endpoint reachability.",
                )
            )
            draft_store.save(draft_state)
            draft_runtime = ToolEvolutionRuntime(
                session=session,
                primitive_tools=[tool],
                library_id="draft",
                store=draft_store,
            )
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            runtime = SkillToolRuntime(
                memory=module,
                memory_mode="read",
                session=session,
                task_description="Host reachability failure",
                tools=[tool],
                session_dir=tmp,
                tool_evolution_runtime=draft_runtime,
            )

            output = runtime.wrap_tools([tool])[0].invoke({"host": "pc1"})
            rows = [
                json.loads(line)
                for line in (Path(tmp) / "messages.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

        transitions = [
            row for row in rows if row.get("event") == "skill_transition"
        ]
        self.assertIn("DRAFT planned exploration advanced", output)
        self.assertEqual(transitions[-1]["draft_exploration_id"], "explore_ping_pc1")
        self.assertIn("Ping pc1", transitions[-1]["draft_next_exploration"])

    def test_skill_tool_wrapper_does_not_duplicate_existing_draft_guidance(self) -> None:
        def ping_host(host: str) -> str:
            return f"{host} reachable"

        with tempfile.TemporaryDirectory() as tmp:
            tool = StructuredTool.from_function(
                ping_host,
                name="ping_host",
                description="Ping one host.",
            )
            session = SimpleNamespace(
                session_id="s2",
                scenario_name="simple_bgp",
                scenario_topo_size="small",
                task_description="Host reachability failure",
            )
            draft_store = ToolEvolutionStore("draft", root=tmp)
            rewrite_documentation(
                draft_store,
                trials=[],
                tool_descriptions={"ping_host": "Ping one host."},
                metrics={},
                session_id="s1",
                task_description="Host reachability failure",
            )
            doc = draft_store.get_document("ping_host")
            assert doc is not None
            doc.usage_notes.append("Use exact host names from the active topology.")
            doc.exploration_suggestions.append(
                "Ping pc1 to verify endpoint reachability."
            )
            draft_store.upsert_document(doc)
            draft_runtime = ToolEvolutionRuntime(
                session=session,
                primitive_tools=[tool],
                library_id="draft",
                store=draft_store,
            )
            draft_tools = draft_runtime.build_tools()
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            runtime = SkillToolRuntime(
                memory=module,
                memory_mode="read",
                session=session,
                task_description="Host reachability failure",
                tools=draft_tools,
                session_dir=tmp,
                tool_evolution_runtime=draft_runtime,
            )

            wrapped = runtime.wrap_tools(draft_tools)[0]

        self.assertEqual(wrapped.description.count("DRAFT refined guidance:"), 1)
        self.assertEqual(wrapped.description.count("DRAFT planned active checks"), 1)
        self.assertNotIn("DRAFT tool guidance:", wrapped.description)
        self.assertNotIn("DRAFT active checks already reflected above", wrapped.description)

    def test_skill_tool_wrapper_scopes_guidance_to_linked_tools(self) -> None:
        def ping_host(host: str) -> str:
            return f"{host} reachable"

        def make_echo_tool(name: str) -> StructuredTool:
            def echo(value: str = "") -> str:
                return value

            echo.__name__ = name
            return StructuredTool.from_function(
                echo,
                name=name,
                description=f"{name} base.",
            )

        with tempfile.TemporaryDirectory() as tmp:
            tools = [
                StructuredTool.from_function(
                    ping_host,
                    name="ping_host",
                    description="Ping one host.",
                ),
                make_echo_tool("cat_file"),
                make_echo_tool("show_logs"),
                make_echo_tool("list_files"),
                make_echo_tool("read_config"),
                make_echo_tool("write_note"),
                make_echo_tool("noop_tool"),
            ]
            session = SimpleNamespace(
                session_id="s2",
                scenario_name="simple_bgp",
                scenario_topo_size="small",
                task_description="Host reachability failure",
            )
            draft_store = ToolEvolutionStore("draft", root=tmp)
            draft_state = draft_store.load()
            draft_state.explorations.append(
                DraftExploration(
                    exploration_id="explore_ping_host",
                    session_id="s1",
                    tool_name="ping_host",
                    intent="diagnosis_check",
                    user_query="Check pc1 reachability.",
                    parameters={"host": "pc1"},
                    status="planned",
                    next_exploration="Ping pc1 to verify endpoint reachability.",
                )
            )
            draft_store.save(draft_state)
            draft_runtime = ToolEvolutionRuntime(
                session=session,
                primitive_tools=tools,
                library_id="draft",
                store=draft_store,
            )
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            state = module.store.load()
            state.skills["specific_ping"] = ProceduralSkill(
                skill_id="specific_ping",
                title="Specific ping skill",
                activation_condition="Use ping_host for host reachability failure.",
                execution_steps=[
                    SkillStep(
                        order=1,
                        action="Call ping_host and interpret reachability.",
                        tool_name="ping_host",
                    )
                ],
                termination_condition="Stop after ping evidence is interpreted.",
                tools=["ping_host"],
                status="validated",
                score=3.0,
            )
            module.store.save(state)
            runtime = SkillToolRuntime(
                memory=module,
                memory_mode="read",
                session=session,
                task_description="Host reachability failure",
                tools=tools,
                session_dir=tmp,
                tool_evolution_runtime=draft_runtime,
            )

            wrapped = {tool.name: tool for tool in runtime.wrap_tools(tools)}

        self.assertIn("Integrated learning guidance", wrapped["ping_host"].description)
        self.assertIn("specific_ping", wrapped["ping_host"].description)
        self.assertIn("DRAFT tool guidance", wrapped["ping_host"].description)
        self.assertEqual(wrapped["cat_file"].description, "cat_file base.")
        self.assertNotIn("Integrated learning guidance", wrapped["cat_file"].description)
        self.assertNotIn("DRAFT tool guidance", wrapped["cat_file"].description)

    def test_skill_runtime_prioritizes_host_link_candidates_before_bgp_deep_dive(self) -> None:
        def make_echo_tool(name: str) -> StructuredTool:
            def echo(value: str = "") -> str:
                return value

            echo.__name__ = name
            return StructuredTool.from_function(
                echo,
                name=name,
                description=f"{name} base.",
            )

        with tempfile.TemporaryDirectory() as tmp:
            tools = [
                make_echo_tool("get_reachability"),
                make_echo_tool("ping_pair"),
                make_echo_tool("get_host_net_config"),
                make_echo_tool("ethtool"),
                make_echo_tool("ip_addr_statistics"),
                make_echo_tool("frr_show_bgp_summary"),
                make_echo_tool("frr_show_ip_route"),
                make_echo_tool("frr_get_bgp_conf"),
            ]
            session = SimpleNamespace(
                session_id="s2",
                scenario_name="dc_clos_bgp",
                scenario_topo_size="s",
                task_description=(
                    "Network Description: EBGP Clos. PCs: pc_0_0, pc_0_1. "
                    "Topology includes pc_0_0:eth0 and pc_0_1:eth0."
                ),
            )
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            runtime = SkillToolRuntime(
                memory=module,
                memory_mode="read",
                session=session,
                task_description=session.task_description,
                tools=tools,
                session_dir=tmp,
            )

            initial_candidates = runtime._fallback_tool_candidates()
            runtime.recent_observations.append(
                'get_reachability({}) -> {"results":[{"src":"pc_0_0",'
                '"dst":"pc_0_1","status":"unknown"}]}'
            )
            after_reachability_candidates = runtime._fallback_tool_candidates()

        self.assertIn("get_host_net_config", initial_candidates)
        self.assertIn("ethtool", initial_candidates)
        self.assertNotIn("frr_get_bgp_conf", initial_candidates)
        self.assertIn("get_host_net_config", after_reachability_candidates)
        self.assertIn("ethtool", after_reachability_candidates)
        self.assertNotIn("frr_show_bgp_summary", after_reachability_candidates)
        self.assertNotIn("frr_get_bgp_conf", after_reachability_candidates)

    def test_skill_runtime_followup_checks_endpoint_link_after_unknown_reachability(self) -> None:
        def reachability() -> str:
            return "unknown"

        with tempfile.TemporaryDirectory() as tmp:
            tool = StructuredTool.from_function(
                reachability,
                name="get_reachability",
                description="Check host reachability.",
            )
            session = SimpleNamespace(
                session_id="s2",
                scenario_name="dc_clos_bgp",
                scenario_topo_size="s",
                task_description="Check Clos host reachability.",
            )
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            runtime = SkillToolRuntime(
                memory=module,
                memory_mode="read",
                session=session,
                task_description=session.task_description,
                tools=[tool],
                session_dir=tmp,
            )

            output = runtime.after_tool(
                tool_name="get_reachability",
                tool_input={},
                result=(
                    '{"hosts":{"pc_0_0":"10.0.0.2","pc_0_1":"10.0.1.2"},'
                    '"results":[{"src":"pc_0_0","dst":"pc_0_1",'
                    '"status":"unknown"}]}'
                ),
            )

        self.assertIn("get_host_net_config", output)
        self.assertIn("ethtool", output)
        self.assertIn("Before deeper BGP", output)

    def test_skill_runtime_followup_stops_after_host_link_down_evidence(self) -> None:
        def host_config(host_name: str) -> str:
            return host_name

        with tempfile.TemporaryDirectory() as tmp:
            tool = StructuredTool.from_function(
                host_config,
                name="get_host_net_config",
                description="Inspect host network config.",
            )
            session = SimpleNamespace(
                session_id="s2",
                scenario_name="dc_clos_bgp",
                scenario_topo_size="s",
                task_description="Check Clos host reachability.",
            )
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            runtime = SkillToolRuntime(
                memory=module,
                memory_mode="read",
                session=session,
                task_description=session.task_description,
                tools=[tool],
                session_dir=tmp,
            )

            output = runtime.after_tool(
                tool_name="get_host_net_config",
                tool_input={"host_name": "pc_0_0"},
                result="pc_0_0 eth0 state DOWN; ip_route is empty",
            )

        self.assertIn("interface/link-down fault", output)
        self.assertNotIn("`link_down`", output)
        self.assertIn("pc_0_0", output)
        self.assertIn("stop calling diagnostic tools", output)

    def test_skill_runtime_link_down_evidence_uses_generic_device_names(self) -> None:
        devices = SkillToolRuntime._host_link_down_devices(
            "server-1 eth0 state DOWN; ip_route is empty"
        )

        self.assertEqual(devices, ["server-1"])

    def test_skill_runtime_does_not_treat_empty_route_as_link_down(self) -> None:
        devices = SkillToolRuntime._host_link_down_devices(
            'get_host_net_config({"host_name":"server-1"}) -> '
            '{"host_name":"server-1","ip_addr":"eth0 state UP",'
            '"ip_route":""}'
        )

        self.assertEqual(devices, [])

    def test_skill_runtime_link_down_extractor_ignores_tool_output_noise(self) -> None:
        text = "\n".join(
            [
                (
                    'frr_show_ip_route({"router_name":"leaf_router_0_1"}) -> '
                    "B>* 10.0.0.0/24 via 172.16.0.6 dev eth0; "
                    "table-direct vnc-direct"
                ),
                (
                    'get_host_net_config({"host_name":"pc_0_0"}) -> '
                    '{"host_name":"pc_0_0","ip_addr":"105: eth0: '
                    '<BROADCAST,MULTICAST> qdisc fq_codel state DOWN",'
                    '"id":"lc_4a6455c9-1725-4d39-883c-c7c7e9297940",'
                    '"ip_route":""}'
                ),
                (
                    'get_host_net_config({"host_name":"pc_0_1"}) -> '
                    '{"host_name":"pc_0_1","ip_addr":"eth0 state UP",'
                    '"ip_route":"default via 10.0.1.1 dev eth0"}'
                ),
            ]
        )

        devices = SkillToolRuntime._host_link_down_devices(text)

        self.assertEqual(devices, ["pc_0_0"])

    def test_skill_runtime_wraps_tools_and_feeds_online_guidance(self) -> None:
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
                        action="Ping endpoints to isolate the failed segment.",
                        tool_name="ping_host",
                    )
                ],
            )
            runtime = SkillToolRuntime(
                memory=module,
                memory_mode="read",
                session=SimpleNamespace(
                    session_id="s2",
                    scenario_name="simple_bgp",
                    scenario_topo_size="small",
                ),
                task_description="Client cannot reach server",
                tools=[tool],
                session_dir=tmp,
            )
            wrapped = runtime.wrap_tools([tool])[0]
            output = wrapped.invoke({"host": "pc1"})
            log_exists = (Path(tmp) / "messages.jsonl").exists()

        self.assertIn("Integrated learning guidance", wrapped.description)
        self.assertIn("Integrated learning guidance - not evidence", output)
        self.assertTrue(runtime.snapshot()["active_skill_id"])
        self.assertTrue(log_exists)

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
                memory=module,
                memory_mode="read",
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

    def test_memory_adapter_installs_integrated_runtime_when_supported(self) -> None:
        class RuntimeCapableAgent:
            def __init__(self) -> None:
                self.installed: dict[str, object] = {}
                self.seen_task = ""

            def install_memory_runtime(self, **kwargs) -> None:
                self.installed = kwargs

            async def run(self, task_description: str):
                self.seen_task = task_description
                return {"ok": True}

        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            agent = RuntimeCapableAgent()
            result = asyncio.run(
                MemoryAugmentedAgent(
                    agent,
                    module,
                    memory_mode="read",
                    memory_top_k=2,
                    memory_token_budget=300,
                    memory_skill_selector_mode="llm_topk_lcb",
                    memory_meta_controller_mode="llm",
                ).run("Diagnose BGP reachability")
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(agent.seen_task, "Diagnose BGP reachability")
        self.assertIs(agent.installed["memory"], module)
        self.assertEqual(agent.installed["top_k"], 2)
        self.assertEqual(agent.installed["skill_selector_mode"], "llm_topk_lcb")
        self.assertEqual(agent.installed["meta_controller_mode"], "llm")

    def test_memory_adapter_rejects_prompt_only_fallback(self) -> None:
        class PromptOnlyAgent:
            async def run(self, _task_description: str):
                return {"ok": True}

        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
            )
            agent = MemoryAugmentedAgent(
                PromptOnlyAgent(),
                module,
                memory_mode="read",
            )

            with self.assertRaisesRegex(RuntimeError, "integrated"):
                asyncio.run(agent.run("Diagnose BGP reachability"))

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
                    SkillStep(order=1, action="Test HTTP reachability.", tool_name="curl")
                ],
            )
            state = module.store.load()
            stats = module.store.bank_stats()

        self.assertEqual(len(state.experiences), 1)
        self.assertEqual(len(state.golden_experiences), 1)
        self.assertEqual(stats["experiences"], 1)
        self.assertEqual(stats["golden_experiences"], 1)

    def test_parent_skill_refines_to_versioned_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
                evolution_threshold=1,
            )
            steps = [
                SkillStep(order=1, action="Check BGP summary.", tool_name="frr_show_bgp_summary")
            ]
            first = module.learn_from_episode(
                evidence=EvaluationEvidence(
                    session_id="s1",
                    task_description="BGP route is missing",
                    scenario="dc_clos_bgp",
                    metrics={
                        "detection_score": 1.0,
                        "localization_accuracy": 1.0,
                        "rca_accuracy": 1.0,
                    },
                    steps=30,
                    tool_calls=20,
                    success=True,
                ),
                tool_steps=steps,
            )
            second = module.learn_from_episode(
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
                tool_steps=steps,
            )
            state = module.store.load()
            refined = state.skills[second["skill_id"]]

        self.assertEqual(first["status"], "accepted")
        self.assertEqual(second["status"], "accepted")
        self.assertTrue(refined.parent_id)
        self.assertGreaterEqual(refined.version, 1)

    def test_llm_semantic_gradient_updates_candidate_skill(self) -> None:
        prompts: list[str] = []

        class FakeModel:
            def with_structured_output(self, _schema):
                return self

            def invoke(self, prompt):
                prompts.append(prompt)
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
                evolution_threshold=1,
            )
            with patch("agent.memory.service.load_model", return_value=FakeModel()):
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

        self.assertEqual(report["status"], "accepted")
        self.assertEqual(report["semantic_gradient_source"], "llm")
        self.assertEqual(skill.semantic_gradients[0].gradient_source, "llm")
        self.assertIn("route and neighbor evidence", skill.termination_condition)
        self.assertNotIn("bgp_missing_route_advertisement", skill.termination_condition)
        self.assertNotIn("leaf_router_0_1", skill.termination_condition)
        self.assertEqual(stats["llm_semantic_gradients"], 1)
        self.assertEqual(len(prompts), 1)
        self.assertNotIn("bgp_missing_route_advertisement", prompts[0])
        self.assertNotIn("leaf_router_0_1", prompts[0])

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
                evolution_threshold=1,
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
                    "agent.memory.service.load_model",
                    return_value=FailingModel(),
                ) as load_model,
            ):
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
                        "memory_mode": "evolve",
                        "memory_bank": "skill",
                        "memory_skill_selector_mode": "llm_topk_lcb",
                        "memory_meta_controller_mode": "llm",
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
            (session_dir / "memory_runtime_session.json").write_text(
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
            report = asyncio.run(
                evolve_session_memory(
                    run_meta={
                        "memory_mode": "evolve",
                        "memory_bank": "skill",
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
            self.assertTrue((session_dir / "memory_update.json").exists())

        self.assertEqual(report["method"], "Skill-Pro")
        self.assertEqual(report["total_added_tokens"], 120)
        self.assertEqual(report["delta_prompt_tokens_per_step"], 30.0)
        self.assertEqual(report["prompt_added_tokens"], 80)
        self.assertEqual(report["tool_description_added_tokens"], 20)
        self.assertEqual(report["followup_added_tokens"], 20)

    def test_eval_metrics_embeds_memory_update_report(self) -> None:
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
                        "memory_mode": "evolve",
                        "memory_bank": "skill",
                        "memory_skill_selector_mode": "llm_topk_lcb",
                        "memory_meta_controller_mode": "llm",
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
                    self.memory_mode = "evolve"
                    self.memory_bank = "skill"
                    self.memory_skill_selector_mode = "llm_topk_lcb"
                    self.memory_meta_controller_mode = "llm"
                    self.llm_backend = "custom"
                    self.model = "test-model"
                    self.tool_evolution_enabled = False
                    self.store = None

                def load_closed_session(self, *, session_id=None) -> None:
                    self.session_id = session_id or self.session_id

                def update_run_meta(self, key: str, value: object) -> None:
                    updates.append((key, value))
                    setattr(self, key, value)

            memory_report = {
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
                },
                "semantic_gradient_source": "llm",
                "semantic_gradient_llm_failed": False,
            }
            with (
                patch("nika.workflows.eval.session.Session", FakeSession),
                patch(
                    "agent.memory.workflow.evolve_session_memory",
                    new=AsyncMock(return_value=memory_report),
                ),
            ):
                run_eval_metrics(session_id="s1")

            metrics = json.loads((session_dir / "eval_metrics.json").read_text())
            result = build_eval_result_from_session_dir(session_dir)

        self.assertEqual(metrics["memory_update"], memory_report)
        self.assertEqual(result.memory_update_status, "accepted")
        self.assertEqual(result.memory_skill_id, "skill_dns")
        self.assertEqual(result.memory_runtime_skill_ids, ["seed_react_decision"])
        self.assertEqual(result.memory_episode_reward, 0.81)
        self.assertEqual(result.memory_episode_baseline, 0.34)
        self.assertEqual(result.memory_episode_advantage, 0.47)
        self.assertTrue(result.memory_episode_success)
        self.assertEqual(result.memory_total_added_tokens, 120)
        self.assertEqual(result.memory_delta_prompt_tokens_per_step, 30.0)
        self.assertEqual(result.memory_prompt_added_tokens, 80)
        self.assertEqual(result.memory_tool_description_added_tokens, 20)
        self.assertEqual(result.memory_followup_added_tokens, 20)
        self.assertEqual(result.memory_ppo_j_score, 0.42)
        self.assertEqual(result.memory_candidate_alignment, 0.73)
        self.assertEqual(result.memory_baseline_alignment, 0.21)
        self.assertEqual(result.memory_skill_selector_mode, "llm_topk_lcb")
        self.assertEqual(result.memory_meta_controller_mode, "llm")
        self.assertEqual(result.memory_skills, 7)
        self.assertIn(("memory_update", memory_report), updates)
