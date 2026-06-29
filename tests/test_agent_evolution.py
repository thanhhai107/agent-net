"""Tests for SIA-style outer-loop agent evolution."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.composition import PolicyOverlayConfig
from agent.langgraph.domain_agents.diagnosis_agent import _load_policy_overlay
from nika.workflows.benchmark.run import _benchmark_row_cli_args
from nika.workflows.evolve.run import (
    AgentEvolutionFeedback,
    EvolutionCaseResult,
    build_feedback_context,
    build_generation_context,
    build_next_policy_update,
    load_generation_results,
    run_agent_evolution,
)


def _write_case(
    root: Path,
    *,
    session_id: str,
    problem: str,
    benchmark_index: int,
    submitted: bool,
    detection: float,
    localization: float,
    rca: float,
) -> None:
    session_dir = root / session_id
    session_dir.mkdir(parents=True)
    (session_dir / "run.json").write_text(
        json.dumps(
            {
                "session_id": session_id,
                "scenario_name": "ospf_enterprise_static",
                "problem_names": [problem],
                "benchmark_index": benchmark_index,
            }
        ),
        encoding="utf-8",
    )
    (session_dir / "eval_metrics.json").write_text(
        json.dumps(
            {
                "detection_score": detection,
                "localization_accuracy": localization,
                "rca_accuracy": rca,
                "steps": 12,
                "tool_calls": 11,
            }
        ),
        encoding="utf-8",
    )
    if submitted:
        (session_dir / "submission.json").write_text(
            json.dumps(
                {
                    "is_anomaly": False,
                    "faulty_devices": [],
                    "root_cause_name": [],
                }
            ),
            encoding="utf-8",
        )


class AgentEvolutionTest(unittest.TestCase):
    def test_load_generation_results_reads_metrics_and_submission_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(
                root,
                session_id="s1",
                problem="host_incorrect_gateway",
                benchmark_index=0,
                submitted=True,
                detection=1.0,
                localization=0.0,
                rca=0.0,
            )

            rows = load_generation_results(root)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].problem, "host_incorrect_gateway")
        self.assertTrue(rows[0].submitted)
        self.assertTrue(rows[0].detection_hit)
        self.assertFalse(rows[0].rca_hit)

    def test_run_agent_evolution_creates_policy_for_next_generation(self) -> None:
        calls = []

        def fake_benchmark(**kwargs):
            calls.append(kwargs)
            root = Path(kwargs["result_root"])
            _write_case(
                root,
                session_id=f"g{len(calls)}-evolution",
                problem="host_incorrect_gateway",
                benchmark_index=0,
                submitted=True,
                detection=1.0,
                localization=0.0,
                rca=0.0,
            )
            _write_case(
                root,
                session_id=f"g{len(calls)}-transfer",
                problem="link_down",
                benchmark_index=1,
                submitted=False,
                detection=-1.0,
                localization=-1.0,
                rca=-1.0,
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with patch(
                "nika.workflows.evolve.run.run_benchmark_from_csv",
                side_effect=fake_benchmark,
            ):
                summaries = run_agent_evolution(
                    benchmark_file=tmp_path / "bench.csv",
                    max_generations=2,
                    run_id="unit",
                    agent_type="mock",
                    llm_backend="openai",
                    model="test-model",
                    feedback_mode="deterministic",
                    runtime_root=tmp_path / "runtime",
                    results_root=tmp_path / "results",
                )

            policy = tmp_path / "runtime" / "unit" / "gen_2" / "policy_overlay.md"
            improvement = tmp_path / "runtime" / "unit" / "gen_2" / "improvement.md"

            self.assertEqual(len(calls), 2)
            self.assertIsNone(calls[0]["policy_overlay"].path)
            self.assertEqual(Path(calls[1]["policy_overlay"].path), policy)
            self.assertTrue(policy.is_file())
            self.assertTrue(improvement.is_file())
            self.assertEqual(len(summaries), 2)
            policy_text = policy.read_text(encoding="utf-8")
            self.assertIn("Agent Evolution Policy Overlay", policy_text)
            self.assertIn("RCA Guardrail", policy_text)
            self.assertNotIn("host_incorrect_gateway", policy_text)
            self.assertNotIn("link_down", policy_text)

    def test_benchmark_row_cli_args_forward_policy_overlay(self) -> None:
        args = _benchmark_row_cli_args(
            {"scenario": "simple_bgp", "problem": "link_down"},
            agent_type="react",
            llm_backend="netmind",
            model="openai/gpt-oss-120b",
            max_steps=50,
            max_attempts=3,
            policy_overlay=PolicyOverlayConfig(path="/tmp/policy.md"),
        )

        self.assertIn("--policy-overlay", args)
        self.assertEqual(args[args.index("--policy-overlay") + 1], "/tmp/policy.md")

    def test_policy_overlay_loader_wraps_content_for_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.md"
            path.write_text("- keep an evidence ledger\n", encoding="utf-8")
            text = _load_policy_overlay(str(path))

        self.assertIn("Agent-evolution policy overlay", text)
        self.assertIn("evidence ledger", text)

    def test_feedback_context_is_sanitized_for_llm_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(
                root,
                session_id="20260629-120332-1a7287",
                problem="host_incorrect_gateway",
                benchmark_index=0,
                submitted=True,
                detection=1.0,
                localization=0.0,
                rca=0.0,
            )
            rows = load_generation_results(root)
            context = build_feedback_context(
                generation=1,
                rows=rows,
            )

        self.assertNotIn("20260629-120332-1a7287", context)
        self.assertNotIn("host_incorrect_gateway", context)
        self.assertIn("Feedback Cases", context)
        self.assertIn("RCA", context)

    def test_generation_context_uses_all_rows_as_feedback(self) -> None:
        rows = [
            EvolutionCaseResult(
                session_id="s1",
                scenario="ospf_enterprise_static",
                problem="host_incorrect_gateway",
                benchmark_index=0,
                submitted=True,
                metrics={"detection_score": 1.0},
                submission={},
            )
        ]

        context = build_generation_context(
            run_id="unit",
            generation=1,
            benchmark_file="benchmark/benchmark_test.csv",
            benchmark_root="results/unit",
            policy_overlay_path=None,
            rows=rows,
        )

        self.assertIn("**Feedback Scope**: all benchmark rows", context)
        self.assertIn("Feedback cases: submitted=1/1", context)

    def test_llm_feedback_agent_output_can_drive_policy(self) -> None:
        feedback = AgentEvolutionFeedback(
            observations=["RCA is weak on the feedback timeline."],
            improvement_plan=["Require evidence-to-RCA mapping before submission."],
            policy_rules=[
                "Write an evidence ledger before the final answer.",
                "Map symptoms to a known root-cause class only after direct evidence.",
                "Stop repeating equivalent checks once a hypothesis is supported.",
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(
                root,
                session_id="s1",
                problem="host_incorrect_gateway",
                benchmark_index=0,
                submitted=True,
                detection=1.0,
                localization=0.0,
                rca=0.0,
            )
            rows = load_generation_results(root)
            with patch(
                "nika.workflows.evolve.run._run_llm_feedback_agent",
                return_value=(
                    "# Improvement Plan\n\n- tighten RCA\n",
                    "# Agent Evolution Policy Overlay\n\n- Write an evidence ledger.\n",
                ),
            ) as feedback_agent:
                improvement, policy, source = build_next_policy_update(
                    generation=1,
                    max_generations=2,
                    rows=rows,
                    feedback_mode="llm",
                    previous_policy_path=None,
                    feedback_llm_backend="openai",
                    feedback_model="test-model",
                )

        self.assertEqual(source, "llm")
        self.assertIn("tighten RCA", improvement)
        self.assertIn("evidence ledger", policy)
        feedback_agent.assert_called_once()
        self.assertTrue(feedback.policy_rules)

    def test_auto_feedback_falls_back_when_llm_feedback_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(
                root,
                session_id="s1",
                problem="host_incorrect_gateway",
                benchmark_index=0,
                submitted=True,
                detection=1.0,
                localization=0.0,
                rca=0.0,
            )
            rows = load_generation_results(root)
            with patch(
                "nika.workflows.evolve.run._run_llm_feedback_agent",
                side_effect=RuntimeError("backend unavailable"),
            ):
                improvement, policy, source = build_next_policy_update(
                    generation=1,
                    max_generations=2,
                    rows=rows,
                    feedback_mode="auto",
                    previous_policy_path=None,
                    feedback_llm_backend="openai",
                    feedback_model="test-model",
                )

        self.assertEqual(source, "deterministic-fallback")
        self.assertIn("Feedback-Agent Fallback", improvement)
        self.assertIn("Agent Evolution Policy Overlay", policy)
