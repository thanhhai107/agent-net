from __future__ import annotations

import sys
import json
from pathlib import Path
from unittest.mock import Mock

from agent.module_config import load_module_defaults
from agent.extensions.config import (
    DEFAULT_LLM_PROVIDER as DEFAULT_LLM_BACKEND,
    DEFAULT_MODEL,
)
import nika.visualization.experiment_runner as experiment_runner
from nika.cli.commands import studio as studio_command_module
from nika.visualization.experiment_runner import (
    DEFAULT_STUDIO_BENCHMARK,
    DEFAULT_STUDIO_MAX_STEPS,
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
        "model": "openai/gpt-oss-120b",
        "max_steps": 100,
        "tool_library_id": "tools-test",
        "tool_doc_chars": 640,
        "tool_convergence_threshold": 0.8,
        "tool_exploration_similarity_threshold": 0.85,
        "tool_explorer_reflection_limit": 4,
        "tool_explorer_model": "explorer-model",
        "tool_analyzer_model": "analyzer-model",
        "tool_rewriter_model": "rewriter-model",
        "procedural_memory_bank": "procedural-memory-test",
        "procedural_memory_mode": "evolve",
        "evolve_until": 12,
        "procedural_memory_tokens": 1500,
        "procedural_memory_max_skill_age": 6,
        "procedural_memory_pool_size": 24,
        "procedural_memory_update_threshold": 2,
        "procedural_memory_best_of_n": 5,
        "procedural_memory_ppo_epsilon": 0.15,
        "procedural_memory_selection_epsilon": 0.25,
        "procedural_memory_experience_pool_size": 900,
        "procedural_memory_baseline_ema_alpha": 0.2,
        "procedural_memory_selection_epsilon_decay_cases": 300,
        "procedural_memory_acceptance_margin": 0.005,
        "procedural_memory_evolver_model": "evolver-model",
        "procedural_memory_policy_scorer_model": "policy-model",
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


def test_empty_studio_config_uses_yaml_defaults() -> None:
    command = build_experiment_command(
        {
            "modules": ["tool_refinement", "procedural_memory"],
            "evolve_until": 75,
        }
    )

    assert command[command.index("--config") + 1] == DEFAULT_STUDIO_BENCHMARK
    assert command[command.index("--max-steps") + 1] == "50"
    assert command[command.index("--tool-refinement-explorer-model") + 1] == (
        "openai/gpt-oss-120b"
    )
    assert command[command.index("--procedural-memory-evolver-model") + 1] == (
        "openai/gpt-oss-120b"
    )
    assert command[command.index("--evolve-until") + 1] == "75"
    assert command[command.index("--tool-refinement-update-interval") + 1] == "6"
    assert command[command.index("--procedural-memory-verifier") + 1] == (
        "behavioral_replay"
    )
    assert "--procedural-memory-k" not in command


def test_create_run_snapshots_module_config(monkeypatch, tmp_path: Path) -> None:
    class Process:
        pid = 123

    monkeypatch.setattr(experiment_runner, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(experiment_runner, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(experiment_runner, "list_runs", lambda: [])
    monkeypatch.setattr(
        experiment_runner.subprocess, "Popen", lambda *_a, **_k: Process()
    )
    monkeypatch.setattr(
        experiment_runner,
        "next_experiment_id",
        lambda _benchmark: "experiment-01",
    )

    run_dir = experiment_runner.create_run(
        {"benchmark_file": DEFAULT_STUDIO_BENCHMARK, "modules": []}
    )
    spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
    snapshot = Path(spec["config"]["module_config_snapshot"])

    assert snapshot == run_dir / "modules.yaml"
    assert load_module_defaults(snapshot) == experiment_runner.RESOLVED_MODULE_DEFAULTS


def test_studio_launcher_forces_light_theme(monkeypatch) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(
        studio_command_module.subprocess,
        "run",
        lambda command, **_kwargs: commands.append(command),
    )

    studio_command_module.studio_command(
        host="127.0.0.1",
        port=8502,
        no_browser=True,
    )

    command = commands[0]
    assert command[command.index("--theme.base") + 1] == "light"
    assert command[command.index("--server.fileWatcherType") + 1] == "none"


def test_stop_run_sends_one_sigterm_and_waits_for_cleanup(
    monkeypatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "experiment-01"
    run_dir.mkdir()
    signals: list[int] = []
    states = iter((True, False, False))

    monkeypatch.setattr(
        experiment_runner,
        "_read_json",
        lambda _path: {"pid": 123, "status": "running"},
    )
    monkeypatch.setattr(experiment_runner, "_write_run_meta", lambda *_a: None)
    monkeypatch.setattr(experiment_runner, "_append_run_log", lambda *_a: None)
    monkeypatch.setattr(
        experiment_runner.os,
        "killpg",
        lambda _pid, sig: signals.append(sig),
    )
    monkeypatch.setattr(
        experiment_runner,
        "_process_group_running",
        lambda _pid: next(states),
    )
    monkeypatch.setattr(experiment_runner.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(experiment_runner, "clean_emulation_environment", lambda: None)
    monkeypatch.setattr(experiment_runner, "check_and_start_next_queued", lambda: None)

    experiment_runner.stop_run(run_dir)

    assert signals == [experiment_runner.signal.SIGTERM]


def test_stop_run_does_not_start_queue_when_cleanup_fails(
    monkeypatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "experiment-01"
    run_dir.mkdir()
    written: list[dict] = []
    start_next = Mock()

    monkeypatch.setattr(
        experiment_runner,
        "_read_json",
        lambda _path: {"pid": 123, "status": "running"},
    )
    monkeypatch.setattr(
        experiment_runner,
        "_write_run_meta",
        lambda _path, meta: written.append(dict(meta)),
    )
    monkeypatch.setattr(experiment_runner, "_append_run_log", lambda *_a: None)
    monkeypatch.setattr(experiment_runner.os, "killpg", lambda *_a: None)
    monkeypatch.setattr(experiment_runner, "_process_group_running", lambda _pid: False)
    monkeypatch.setattr(experiment_runner, "_pid_running", lambda _pid: False)
    monkeypatch.setattr(
        experiment_runner,
        "clean_emulation_environment",
        Mock(side_effect=RuntimeError("cleanup failed")),
    )
    monkeypatch.setattr(experiment_runner, "check_and_start_next_queued", start_next)

    experiment_runner.stop_run(run_dir)

    assert written[-1]["status"] == "failed"
    assert "cleanup failed" in written[-1]["cleanup_error"]
    start_next.assert_not_called()


def test_result_summary_uses_selected_benchmark_metrics() -> None:
    from nika.visualization.experiment_dashboard import RESULT_SUMMARY_COLUMNS

    metrics = [
        "incident_success",
        "detection_score",
        "localization_f1",
        "rca_f1",
        "tool_calls",
        "tool_errors",
    ]
    assert [column for column in RESULT_SUMMARY_COLUMNS if column in metrics] == metrics
    assert "progress" not in RESULT_SUMMARY_COLUMNS
    assert "localization_precision" not in RESULT_SUMMARY_COLUMNS
    assert "rca_precision" not in RESULT_SUMMARY_COLUMNS
    assert "total_tokens" not in RESULT_SUMMARY_COLUMNS


def test_result_html_table_escapes_result_values() -> None:
    from nika.visualization.experiment_dashboard import _results_table_html

    rendered = _results_table_html(
        [{"result_root": "<unsafe>", "detection_score": 0.9}],
        ["result_root", "detection_score"],
    )

    assert "<table" in rendered
    assert "&lt;unsafe&gt;" in rendered
    assert ">0.90<" in rendered


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
    from nika.visualization.experiment_dashboard import (
        _count_rows,
        _default_evolve_cases,
    )

    benchmark = tmp_path / "clean-controls.yaml"
    benchmark.write_text(
        "cases:\n  - scenario: simple_bgp\n    problem: no_fault\n    inject: {}\n",
        encoding="utf-8",
    )

    assert _count_rows(benchmark) == 1
    assert _default_evolve_cases(benchmark, row_count=1) == 1


def test_evolve_benchmark_declares_curriculum_cutoff() -> None:
    from nika.visualization.experiment_dashboard import (
        DEFAULT_STUDIO_BENCHMARK,
        _count_rows,
        _default_evolve_cases,
    )

    benchmark = Path(DEFAULT_STUDIO_BENCHMARK)
    row_count = _count_rows(benchmark)

    assert row_count == 125
    assert benchmark.name == "benchmark_evolve.yaml"
    assert _default_evolve_cases(benchmark, row_count=row_count) == 75


def test_command_fallbacks_match_studio_defaults(monkeypatch) -> None:
    monkeypatch.setenv("NIKA_MAX_STEPS", "20")
    config = _config()
    for key in ("benchmark_file", "llm_backend", "model", "max_steps"):
        config.pop(key)

    command = build_experiment_command(config)

    assert command[command.index("--provider") + 1] == DEFAULT_LLM_BACKEND
    assert command[command.index("--model") + 1] == DEFAULT_MODEL
    assert command[command.index("--max-steps") + 1] == str(DEFAULT_STUDIO_MAX_STEPS)
    assert command[command.index("--config") + 1] == DEFAULT_STUDIO_BENCHMARK


def test_learning_role_models_default_to_agent_model() -> None:
    config = _config(modules=["tool_refinement", "procedural_memory"])
    config.pop("model")
    for key in (
        "tool_explorer_model",
        "tool_analyzer_model",
        "tool_rewriter_model",
        "procedural_memory_evolver_model",
        "procedural_memory_policy_scorer_model",
    ):
        config.pop(key)

    command = build_experiment_command(config)

    for flag in (
        "--model",
        "--tool-refinement-explorer-model",
        "--tool-refinement-analyzer-model",
        "--tool-refinement-rewriter-model",
        "--procedural-memory-evolver-model",
        "--procedural-memory-policy-scorer-model",
    ):
        assert command[command.index(flag) + 1] == DEFAULT_MODEL


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
    assert (
        command[command.index("--tool-refinement-exploration-similarity-threshold") + 1]
        == "0.85"
    )
    assert (
        command[command.index("--tool-refinement-explorer-reflection-limit") + 1] == "4"
    )
    assert command[command.index("--tool-refinement-explorer-model") + 1] == (
        "explorer-model"
    )
    assert command[command.index("--tool-refinement-analyzer-model") + 1] == (
        "analyzer-model"
    )
    assert command[command.index("--tool-refinement-rewriter-model") + 1] == (
        "rewriter-model"
    )
    assert command[command.index("--procedural-memory") + 1] == "procedural-memory-test"
    assert command[command.index("--procedural-memory-max-skill-age") + 1] == "6"
    assert command[command.index("--procedural-memory-pool-size") + 1] == "24"
    assert command[command.index("--procedural-memory-update-threshold") + 1] == "2"
    assert command[command.index("--procedural-memory-best-of-n") + 1] == "5"
    assert command[command.index("--procedural-memory-ppo-epsilon") + 1] == "0.15"
    assert command[command.index("--procedural-memory-selection-epsilon") + 1] == "0.25"
    assert (
        command[command.index("--procedural-memory-experience-pool-size") + 1] == "900"
    )
    assert "--procedural-memory-golden-pool-size" not in command
    assert command[command.index("--procedural-memory-baseline-ema-alpha") + 1] == "0.2"
    assert (
        command[command.index("--procedural-memory-selection-epsilon-decay-cases") + 1]
        == "300"
    )
    assert (
        command[command.index("--procedural-memory-acceptance-margin") + 1] == "0.005"
    )
    assert command[command.index("--procedural-memory-evolver-model") + 1] == (
        "evolver-model"
    )
    assert command[command.index("--procedural-memory-policy-scorer-model") + 1] == (
        "policy-model"
    )
    assert command[command.index("--evolve-until") + 1] == "12"


def test_procedural_memory_read_mode_uses_read_only_flag() -> None:
    command = build_experiment_command(
        _config(
            modules=["procedural_memory"],
            procedural_memory_mode="read",
        )
    )

    assert "--procedural-memory-read" in command
    assert "--procedural-memory" not in command
    assert "--evolve-until" not in command


def test_command_plan_for_memory_has_no_service_prerequisite() -> None:
    plan = build_command_plan(
        _config(
            modules=["procedural_memory"],
        )
    )

    assert len(plan) == 1
    assert plan[0].variant == "benchmark"
    assert plan[0].name == "ReAct + Procedural Memory"
    assert "--evolve-until" in plan[0].command
    assert plan[0].command[plan[0].command.index("--evolve-until") + 1] == "12"


def test_prepare_experiment_config_uses_one_sequential_name_for_outputs(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        experiment_runner,
        "next_experiment_id",
        lambda _benchmark: "experiment-07",
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

    assert prepared["experiment_id"] == "experiment-07"
    assert prepared["result_root"] == str(tmp_path / "results" / "experiment-07")
    assert command[command.index("--result-dir") + 1].endswith("experiment-07")
    assert command[command.index("--tool-refinement") + 1] == "experiment-07"
    assert command[command.index("--procedural-memory") + 1] == "experiment-07"


def test_prepare_experiment_config_normalizes_legacy_evolve_until(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        experiment_runner,
        "next_experiment_id",
        lambda _benchmark: "experiment-07",
    )

    legacy = _config(
        procedural_memory_evolve_until=12,
        learning_evolve_until=10,
    )
    legacy.pop("evolve_until")

    prepared = prepare_experiment_config(legacy)

    assert prepared["evolve_until"] == 10
    assert "procedural_memory_evolve_until" not in prepared
    assert "learning_evolve_until" not in prepared


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
                "benchmark_done session_id=s3 scenario=ospf_enterprise_dhcp problem=dns_record_error session_dir=/tmp/s3",
                "benchmark_done index=3/30 scenario=ospf_enterprise_dhcp topo_size=m problem=dns_record_error session_id=s3 session_dir=/tmp/s3",
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
        "benchmark_done",
        "ui_run_stopped",
        "ui_run_done",
    ]
    assert rows[2]["completed"] == "1"
    assert rows[3]["skipped"] == "1"
    assert rows[4]["index"] == "3/30"
    assert rows[4]["topo_size"] == "m"
    assert rows[5]["reason"] == "user_stop"
    assert rows[6]["exit_code"] == "0"


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
                    "agent_type": "react",
                    "model": "openai/gpt-oss-120b",
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
