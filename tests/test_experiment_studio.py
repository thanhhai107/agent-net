from __future__ import annotations

import sys
import json
from pathlib import Path

from agent.extensions.config import (
    DEFAULT_LLM_PROVIDER as DEFAULT_LLM_BACKEND,
    DEFAULT_MODEL,
)
import nika.visualization.experiment_runner as experiment_runner
from nika.visualization.experiment_runner import (
    build_experiment_command,
    build_command_plan,
    parse_progress_events,
    prepare_experiment_config,
    resume_run,
    run_status,
)


def _config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "benchmark_file": "benchmark/benchmark_test.yaml",
        "modules": [],
        "agent_type": "react",
        "llm_backend": "custom",
        "model": "openai/gpt-oss-20b",
        "max_steps": 100,
        "tool_library_id": "tools-test",
        "tool_doc_chars": 640,
        "tool_convergence_threshold": 0.8,
        "procedural_memory_bank": "procedural-memory-test",
        "procedural_memory_k": 5,
        "procedural_memory_tokens": 1500,
        "procedural_memory_max_skill_age": 6,
        "procedural_memory_pool_size": 24,
        "procedural_memory_update_threshold": 2,
        "procedural_memory_best_of_n": 5,
        "procedural_memory_ppo_epsilon": 0.15,
    }
    config.update(overrides)
    return config


def test_baseline_command_uses_default_sequential_execution() -> None:
    command = build_experiment_command(_config())

    assert command[:3] == [sys.executable, "-m", "nika.extensions.benchmark"]
    assert command[3] == "--config"
    assert "benchmark/benchmark_test.yaml" in command
    assert command[command.index("--max-steps") + 1] == "100"
    assert "-j" not in command
    assert "--parallel" not in command


def test_result_summary_uses_selected_benchmark_metrics() -> None:
    from nika.visualization.experiment_dashboard import RESULT_SUMMARY_COLUMNS

    metrics = [
        "detection_score",
        "localization_f1",
        "rca_f1",
        "localization_precision",
        "rca_precision",
        "tool_calls",
        "tool_errors",
    ]
    assert [column for column in RESULT_SUMMARY_COLUMNS if column in metrics] == metrics
    assert "progress" not in RESULT_SUMMARY_COLUMNS
    assert "total_tokens" not in RESULT_SUMMARY_COLUMNS


def test_clean_control_composite_requires_false_detection_and_empty_sets() -> None:
    from nika.visualization.experiment_dashboard import _metric_total

    correct = {
        "detection_score": 1.0,
        "localization_f1": 1.0,
        "rca_f1": 1.0,
    }
    inconsistent = {**correct, "localization_f1": 0.0}

    assert _metric_total(correct, is_anomaly=False) == 1.0
    assert _metric_total(inconsistent, is_anomaly=False) < 1.0


def test_studio_counts_benchmark_with_clean_controls(tmp_path: Path) -> None:
    from nika.visualization.experiment_dashboard import _count_rows

    benchmark = tmp_path / "clean-controls.yaml"
    benchmark.write_text(
        "cases:\n  - scenario: simple_bgp\n    problem: no_fault\n    inject: {}\n",
        encoding="utf-8",
    )

    assert _count_rows(benchmark) == 1


def test_command_fallbacks_match_extension_defaults(monkeypatch) -> None:
    monkeypatch.setenv("NIKA_MAX_STEPS", "20")
    config = _config()
    for key in ("llm_backend", "model", "max_steps"):
        config.pop(key)

    command = build_experiment_command(config)

    assert command[command.index("--provider") + 1] == DEFAULT_LLM_BACKEND
    assert command[command.index("--model") + 1] == DEFAULT_MODEL
    assert command[command.index("--max-steps") + 1] == "20"


def test_tool_and_memory_modules_share_one_sequential_command() -> None:
    command = build_experiment_command(
        _config(modules=["tool_refinement", "procedural_memory"])
    )

    assert command[:3] == [sys.executable, "-m", "nika.extensions.benchmark"]
    assert "-j" not in command
    assert "--parallel" not in command

    assert command[command.index("--tool-refinement") + 1] == "tools-test"
    assert command[command.index("--tool-refinement-doc-chars") + 1] == "640"
    assert (
        command[command.index("--tool-refinement-convergence-threshold") + 1] == "0.8"
    )
    assert command[command.index("--procedural-memory") + 1] == "procedural-memory-test"
    assert command[command.index("--procedural-memory-max-skill-age") + 1] == "6"
    assert command[command.index("--procedural-memory-pool-size") + 1] == "24"
    assert command[command.index("--procedural-memory-update-threshold") + 1] == "2"
    assert command[command.index("--procedural-memory-best-of-n") + 1] == "5"
    assert command[command.index("--procedural-memory-ppo-epsilon") + 1] == "0.15"


def test_command_plan_for_memory_has_no_service_prerequisite() -> None:
    plan = build_command_plan(
        _config(
            modules=["procedural_memory"],
        )
    )

    assert len(plan) == 1
    assert plan[0].variant == "benchmark"
    assert plan[0].name == "ReAct + Procedural Memory"


def test_prepare_experiment_config_uses_one_sequential_name_for_outputs(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        experiment_runner,
        "next_experiment_id",
        lambda _benchmark: "benchmark_test-0007",
    )
    monkeypatch.setattr(experiment_runner, "RESULTS_DIR", tmp_path / "results")

    prepared = prepare_experiment_config(
        _config(
            modules=["tool_refinement", "procedural_memory"],
            tool_library_id="",
            procedural_memory_bank="",
        )
    )
    command = build_experiment_command(prepared)

    assert prepared["experiment_id"] == "benchmark_test-0007"
    assert prepared["result_root"] == str(tmp_path / "results" / "benchmark_test-0007")
    assert command[command.index("--result-dir") + 1].endswith("benchmark_test-0007")
    assert command[command.index("--tool-refinement") + 1] == "benchmark_test-0007"
    assert command[command.index("--procedural-memory") + 1] == "benchmark_test-0007"


def test_resume_command_uses_existing_result_root() -> None:
    command = build_experiment_command(
        _config(
            experiment_id="benchmark_evaluate-0001",
            result_root="/tmp/results/benchmark_evaluate-0001",
            resume=True,
        )
    )

    assert (
        command[command.index("--result-dir") + 1]
        == "/tmp/results/benchmark_evaluate-0001"
    )
    assert "--resume" in command


def test_resume_run_reuses_selected_run_directory(monkeypatch, tmp_path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "benchmark_evaluate-0001"
    run_dir.mkdir(parents=True)
    result_root = tmp_path / "results" / "benchmark_evaluate-0001"
    spec = {
        "run_id": "benchmark_evaluate-0001",
        "created_at": "2026-07-03T22:18:00+00:00",
        "config": _config(
            experiment_id="benchmark_evaluate-0001",
            result_root=str(result_root),
        ),
        "commands": [],
    }
    (run_dir / "spec.json").write_text(
        experiment_runner.json.dumps(spec), encoding="utf-8"
    )
    (run_dir / "meta.json").write_text(
        experiment_runner.json.dumps(
            {"run_id": "benchmark_evaluate-0001", "status": "running", "pid": 1}
        ),
        encoding="utf-8",
    )
    (run_dir / "run.log").write_text(
        "\n".join(
            [
                'ui_step_start {"index": 1, "name": "ReAct"}',
                'ui_step_done {"index": 1, "returncode": 1}',
                'ui_run_stopped {"reason": "user_stop"}',
                'ui_run_done {"exit_code": 1}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeProc:
        pid = 4242

    popen_calls: list[list[str]] = []

    def fake_popen(command, **kwargs):
        popen_calls.append(command)
        return FakeProc()

    monkeypatch.setattr(experiment_runner, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(experiment_runner.subprocess, "Popen", fake_popen)

    resumed = resume_run(run_dir)

    assert resumed == run_dir
    assert not (runs_dir / "benchmark_evaluate-0001-resume").exists()
    updated_spec = experiment_runner.read_run_spec(run_dir)
    command = updated_spec["commands"][0]["command"]
    assert updated_spec["run_id"] == "benchmark_evaluate-0001"
    assert updated_spec["config"]["experiment_id"] == "benchmark_evaluate-0001"
    assert updated_spec["config"]["result_root"] == str(result_root)
    assert "--resume" in command
    assert command[command.index("--result-dir") + 1] == str(result_root)
    assert popen_calls
    log_text = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "ui_run_resumed" in log_text
    assert "ui_step_done" not in log_text
    assert "ui_run_stopped" not in log_text
    assert "ui_run_done" not in log_text


def test_run_status_ignores_old_done_after_resume_marker(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "runs" / "benchmark_evaluate-0001"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(
        experiment_runner.json.dumps(
            {"run_id": "benchmark_evaluate-0001", "status": "running", "pid": 4242}
        ),
        encoding="utf-8",
    )
    (run_dir / "run.log").write_text(
        'ui_run_done {"exit_code": 1}\nui_run_resumed {"run_id": "benchmark_evaluate-0001"}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(experiment_runner, "_pid_running", lambda pid: pid == 4242)

    assert run_status(run_dir)["status"] == "running"

    with (run_dir / "run.log").open("a", encoding="utf-8") as handle:
        handle.write('ui_run_done {"exit_code": 0}\n')

    status = run_status(run_dir)
    assert status["status"] == "finished"
    assert status["exit_code"] == 0


def test_parse_progress_events_reads_benchmark_and_ui_events() -> None:
    rows = parse_progress_events(
        "\n".join(
            [
                'ui_step_start {"index": 1, "name": "Baseline"}',
                'ui_run_resumed {"run_id": "benchmark_test-0001"}',
                "benchmark_progress index=1/30 completed=1 failed=0 session_id=s1",
                "benchmark_skip index=2/30 completed=2 failed=0 skipped=1",
                'ui_run_stopped {"reason": "user_stop"}',
                'ui_run_done {"exit_code": 0}',
            ]
        )
    )

    assert [row["event"] for row in rows] == [
        "ui_step_start",
        "ui_run_resumed",
        "benchmark_progress",
        "benchmark_skip",
        "ui_run_stopped",
        "ui_run_done",
    ]
    assert rows[2]["completed"] == "1"
    assert rows[3]["skipped"] == "1"
    assert rows[4]["reason"] == "user_stop"
    assert rows[5]["exit_code"] == "0"


def test_result_aggregation_excludes_clean_controls_from_localization_and_rca(
    monkeypatch, tmp_path: Path
) -> None:
    from nika.visualization import experiment_dashboard as dashboard

    results = tmp_path / "results"
    root = results / "benchmark_test-0001"
    for name, is_anomaly, localization_f1, rca_f1 in (
        ("fault", True, 1.0, 0.8),
        ("clean", False, 0.0, 0.0),
    ):
        session = root / name
        session.mkdir(parents=True)
        (session / "run.json").write_text(
            json.dumps(
                {
                    "session_id": name,
                    "status": "finished",
                    "agent_type": "byo.langgraph",
                    "model": "openai/gpt-oss-20b",
                    "problem_names": ["link_down" if is_anomaly else "no_fault"],
                }
            ),
            encoding="utf-8",
        )
        (session / "ground_truth.json").write_text(
            json.dumps({"is_anomaly": is_anomaly}),
            encoding="utf-8",
        )
        (session / "eval_metrics.json").write_text(
            json.dumps(
                {
                    "detection_score": 1.0,
                    "localization_f1": localization_f1,
                    "rca_f1": rca_f1,
                    "localization_precision": localization_f1,
                    "rca_precision": rca_f1,
                    "tool_calls": 2,
                    "tool_errors": 0,
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(dashboard, "RESULTS_DIR", results)
    row = dashboard._result_rows(benchmark_name="benchmark_test")[0]

    assert row["detection_score"] == "1.00"
    assert row["localization_f1"] == "1.00"
    assert row["rca_f1"] == "0.80"
