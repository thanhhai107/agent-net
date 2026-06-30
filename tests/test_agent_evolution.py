"""Tests for SIA-H style executable harness evolution."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.composition import HarnessConfig
from agent.harness.runner import validate_target_agent_source
from nika.workflows.benchmark.run import _benchmark_row_cli_args
from nika.workflows.benchmark import run as benchmark_run_module
from nika.workflows.evolve.run import (
    EvolutionCaseResult,
    TargetAgentArtifact,
    _reference_target_agent_source,
    build_feedback_context,
    build_generation_context,
    build_next_target_update,
    load_generation_results,
    run_harness_evolution,
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
    (session_dir / "agent_execution.json").write_text(
        json.dumps(
            {
                "case": {"benchmark_index": benchmark_index},
                "diagnosis_report": (
                    f"Found {problem} evidence from 10.0.0.1 and session {session_id}."
                ),
                "submission_result": f"submitted root cause {problem}",
                "error": None,
                "messages": [
                    {
                        "event": "tool_end",
                        "content": f"{problem} observed at 10.0.0.2",
                    }
                ],
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
                    "root_cause_name": [problem],
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
        self.assertTrue(rows[0].session_dir.endswith("s1"))
        self.assertTrue(rows[0].submitted)
        self.assertTrue(rows[0].detection_hit)
        self.assertFalse(rows[0].rca_hit)

    def test_run_harness_evolution_creates_target_agent_for_next_generation(self) -> None:
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
            with (
                patch("nika.workflows.evolve.run.ensure_kathara_clean"),
                patch(
                    "nika.workflows.evolve.run.run_benchmark_from_csv",
                    side_effect=fake_benchmark,
                ),
            ):
                summaries = run_harness_evolution(
                    benchmark_file=tmp_path / "bench.csv",
                    max_generations=2,
                    run_id="unit",
                    llm_backend="openai",
                    model="test-model",
                    feedback_mode="deterministic",
                    runtime_root=tmp_path / "runtime",
                    results_root=tmp_path / "results",
                )

            gen1_target = tmp_path / "runtime" / "unit" / "gen_1" / "target_agent.py"
            gen2_target = tmp_path / "runtime" / "unit" / "gen_2" / "target_agent.py"
            improvement = tmp_path / "runtime" / "unit" / "gen_2" / "improvement.md"
            execution = (
                tmp_path
                / "runtime"
                / "unit"
                / "gen_1"
                / "agent_execution"
                / "execution_0.json"
            )

            self.assertEqual(len(calls), 2)
            self.assertEqual(Path(calls[0]["harness"].target_agent_path), gen1_target)
            self.assertEqual(Path(calls[1]["harness"].target_agent_path), gen2_target)
            self.assertTrue(calls[0]["harness_allow_failure"])
            self.assertTrue(gen1_target.is_file())
            self.assertTrue(gen2_target.is_file())
            self.assertTrue(improvement.is_file())
            self.assertTrue(execution.is_file())
            self.assertEqual(len(summaries), 2)
            self.assertIn("Deterministic feedback mode", improvement.read_text())

    def test_benchmark_row_cli_args_forward_harness(self) -> None:
        args = _benchmark_row_cli_args(
            {"scenario": "simple_bgp", "problem": "link_down"},
            agent_type="harness",
            llm_backend="netmind",
            model="openai/gpt-oss-120b",
            max_steps=100,
            max_attempts=3,
            harness=HarnessConfig(target_agent_path="/tmp/target_agent.py"),
            harness_allow_failure=True,
        )

        self.assertIn("--harness", args)
        self.assertEqual(args[args.index("--harness") + 1], "/tmp/target_agent.py")
        self.assertIn("--harness-allow-failure", args)

    def test_harness_agent_type_requires_target_path(self) -> None:
        with (
            patch.object(benchmark_run_module, "ensure_kathara_clean"),
            self.assertRaisesRegex(ValueError, "requires the internal --harness"),
        ):
            benchmark_run_module.run_benchmark_from_csv(
                benchmark_file="/tmp/does-not-matter.csv",
                agent_type="harness",
                llm_backend="openai",
                model="model",
                max_steps=100,
            )

    def test_reference_target_agent_passes_harness_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "target_agent.py"
            path.write_text(_reference_target_agent_source(), encoding="utf-8")

            validate_target_agent_source(path)

    def test_feedback_context_is_sanitized_for_meta_agent(self) -> None:
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
                target_agent_path=root / "target_agent.py",
            )

        self.assertNotIn("20260629-120332-1a7287", context)
        self.assertNotIn("host_incorrect_gateway", context)
        self.assertNotIn("10.0.0.1", context)
        self.assertNotIn("10.0.0.2", context)
        self.assertIn("Case Metrics", context)
        self.assertIn("Execution Samples", context)

    def test_generation_context_uses_all_rows_as_feedback(self) -> None:
        rows = [
            EvolutionCaseResult(
                session_id="s1",
                session_dir="/tmp/s1",
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
            target_agent_path="runtime/unit/gen_1/target_agent.py",
            rows=rows,
        )

        self.assertIn("**Feedback Scope**: all benchmark rows", context)
        self.assertIn("Harness Evolution Context", context)
        self.assertIn("submitted=1/1", context)

    def test_llm_feedback_agent_output_can_drive_next_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            current = tmp_path / "current.py"
            current.write_text(_reference_target_agent_source(), encoding="utf-8")
            next_target = tmp_path / "next.py"
            root = tmp_path / "results"
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
                "nika.workflows.evolve.run._invoke_target_meta_agent",
                return_value=TargetAgentArtifact(
                    improvement_md="# Improvement\n\n- tighten RCA mapping\n",
                    target_agent_py=_reference_target_agent_source(),
                ),
            ) as meta_agent:
                improvement, source = build_next_target_update(
                    generation=1,
                    max_generations=2,
                    rows=rows,
                    feedback_mode="llm",
                    current_target_path=current,
                    next_target_path=next_target,
                    feedback_llm_backend="openai",
                    feedback_model="test-model",
                )

            self.assertEqual(source, "llm")
            self.assertIn("tighten RCA", improvement)
            self.assertTrue(next_target.is_file())
            meta_agent.assert_called_once()

    def test_auto_feedback_falls_back_when_meta_agent_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            current = tmp_path / "current.py"
            current.write_text(_reference_target_agent_source(), encoding="utf-8")
            next_target = tmp_path / "next.py"
            root = tmp_path / "results"
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
                "nika.workflows.evolve.run._invoke_target_meta_agent",
                side_effect=RuntimeError("backend unavailable"),
            ):
                improvement, source = build_next_target_update(
                    generation=1,
                    max_generations=2,
                    rows=rows,
                    feedback_mode="auto",
                    current_target_path=current,
                    next_target_path=next_target,
                    feedback_llm_backend="openai",
                    feedback_model="test-model",
                )

            self.assertEqual(source, "deterministic-fallback")
            self.assertIn("carrying forward", improvement)
            self.assertEqual(next_target.read_text(), current.read_text())
