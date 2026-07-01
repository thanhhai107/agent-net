"""Tests for Skill-Pro procedural memory."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.memory.models import EvaluationEvidence, MemoryQuery, SemanticGradient, SkillStep
from agent.memory.service import ProceduralMemoryModule
from agent.memory.workflow import evolve_session_memory


class SkillProMemoryTest(unittest.TestCase):
    def test_ppo_gate_accepts_successful_skill_and_retrieves_it(self) -> None:
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
        self.assertEqual(len(retrieved), 1)
        self.assertIn("Activation", context)
        self.assertNotIn("bgp_missing_route_advertisement", context)

    def test_ppo_gate_rejects_weaker_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                store_path=Path(tmp) / "skills.json",
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

    def test_llm_semantic_gradient_updates_candidate_skill(self) -> None:
        prompts: list[str] = []

        class FakeModel:
            def with_structured_output(self, _schema):
                return self

            def invoke(self, prompt):
                prompts.append(prompt)
                return SemanticGradient(
                    source_session_id="s1",
                    critique="Preserve BGP route checks but require independent evidence.",
                    proposed_update="Terminate only after route and neighbor evidence agree.",
                )

        with tempfile.TemporaryDirectory() as tmp:
            module = ProceduralMemoryModule(
                bank_id="skill",
                llm_backend="openai",
                model="test-model",
                store_path=Path(tmp) / "skills.json",
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
            skill = next(iter(state.skills.values()))
            stats = module.store.bank_stats()

        self.assertEqual(report["status"], "accepted")
        self.assertEqual(report["semantic_gradient_source"], "llm")
        self.assertEqual(skill.semantic_gradients[0].gradient_source, "llm")
        self.assertIn("route and neighbor evidence", skill.termination_condition)
        self.assertEqual(stats["llm_semantic_gradients"], 1)
        self.assertNotIn("bgp_missing_route_advertisement", prompts[0])
        self.assertNotIn("leaf_router_0_1", prompts[0])

    def test_offline_workflow_writes_skill_update_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp) / "session"
            session_dir.mkdir()
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
