from __future__ import annotations

import sys

from agent.llm.model_factory import DEFAULT_LLM_BACKEND, DEFAULT_MODEL
import nika.visualization.experiment_runner as experiment_runner
from nika.utils.agent_config import resolve_max_steps
from nika.visualization.experiment_runner import (
    build_experiment_command,
    build_command_plan,
    parse_progress_events,
    prepare_experiment_config,
)


def _config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "benchmark_file": "benchmark/benchmark_test.yaml",
        "modules": [],
        "agent_type": "react",
        "llm_backend": "custom",
        "model": "openai/gpt-oss-20b",
        "max_steps": 100,
        "max_attempts": 3,
        "tool_library_id": "tools-test",
        "tool_doc_chars": 640,
        "tool_prompt_doc_limit": 5,
        "tool_scoped_prompt_doc_limit": 3,
        "tool_planned_checks": 2,
        "tool_next_checks": 1,
        "tool_convergence_threshold": 0.8,
        "memory_bank": "memory-test",
        "memory_k": 5,
        "memory_tokens": 1500,
        "memory_selector": "lcb",
        "memory_meta_controller": "heuristic",
        "memory_max_skill_age": 6,
        "memory_selector_min_lcb": -0.02,
        "memory_selector_nominee_k": 4,
        "memory_pool_size": 24,
        "memory_evolution_threshold": 2,
        "memory_best_of_n": 5,
        "memory_ppo_epsilon": 0.15,
    }
    config.update(overrides)
    return config


def test_baseline_command_uses_default_sequential_execution() -> None:
    command = build_experiment_command(_config())

    assert command[:3] == [sys.executable, "-m", "nika.cli.main"]
    assert command[3:6] == ["benchmark", "run", "--file"]
    assert "benchmark/benchmark_test.yaml" in command
    assert command[command.index("-n") + 1] == "100"
    assert "-j" not in command
    assert "--parallel" not in command


def test_standard_agent_label_uses_selected_baseline_name() -> None:
    plan = build_command_plan(
        _config(agent_type="reflexion", modules=["memory_evolution"])
    )

    assert plan[0].variant == "benchmark"
    assert plan[0].name == "Reflexion + Memory Evolution"


def test_command_fallbacks_match_env_agent_config() -> None:
    config = _config()
    for key in ("llm_backend", "model", "max_steps"):
        config.pop(key)

    command = build_experiment_command(config)

    assert command[command.index("-b") + 1] == DEFAULT_LLM_BACKEND
    assert command[command.index("-m", command.index("-b")) + 1] == DEFAULT_MODEL
    assert command[command.index("-n") + 1] == str(resolve_max_steps(None))


def test_tool_and_memory_modules_share_one_sequential_command() -> None:
    command = build_experiment_command(
        _config(modules=["tool_evolution", "memory_evolution"])
    )

    assert command[3:6] == ["benchmark", "run", "--file"]
    assert "-j" not in command
    assert "--parallel" not in command
    assert command[command.index("--tools") + 1] == "tools-test"
    assert command[command.index("--tool-doc-chars") + 1] == "640"
    assert command[command.index("--tool-prompt-doc-limit") + 1] == "5"
    assert command[command.index("--tool-scoped-prompt-doc-limit") + 1] == "3"
    assert command[command.index("--tool-planned-checks") + 1] == "2"
    assert command[command.index("--tool-next-checks") + 1] == "1"
    assert command[command.index("--tool-convergence-threshold") + 1] == "0.8"
    assert command[command.index("--memory") + 1] == "memory-test"
    assert command[command.index("--memory-selector") + 1] == "lcb"
    assert command[command.index("--memory-meta-controller") + 1] == "heuristic"
    assert command[command.index("--memory-max-skill-age") + 1] == "6"
    assert command[command.index("--memory-selector-min-lcb") + 1] == "-0.02"
    assert command[command.index("--memory-selector-nominee-k") + 1] == "4"
    assert command[command.index("--memory-pool-size") + 1] == "24"
    assert command[command.index("--memory-evolution-threshold") + 1] == "2"
    assert command[command.index("--memory-best-of-n") + 1] == "5"
    assert command[command.index("--memory-ppo-epsilon") + 1] == "0.15"


def test_memory_command_passes_skill_pro_selector_and_controller() -> None:
    command = build_experiment_command(
        _config(
            modules=["memory_evolution"],
            memory_selector="llm_topk_lcb",
            memory_meta_controller="llm",
        )
    )

    assert command[command.index("--memory-selector") + 1] == "llm_topk_lcb"
    assert command[command.index("--memory-meta-controller") + 1] == "llm"


def test_command_plan_for_memory_has_no_service_prerequisite() -> None:
    plan = build_command_plan(
        _config(
            modules=["memory_evolution"],
        )
    )

    assert len(plan) == 1
    assert plan[0].variant == "benchmark"
    assert plan[0].name == "ReAct + Memory Evolution"


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
            modules=["tool_evolution", "memory_evolution"],
            tool_library_id="",
            memory_bank="",
        )
    )
    command = build_experiment_command(prepared)

    assert prepared["experiment_id"] == "benchmark_test-0007"
    assert prepared["result_root"] == str(tmp_path / "results" / "benchmark_test-0007")
    assert command[command.index("--result-root") + 1].endswith("benchmark_test-0007")
    assert command[command.index("--tools") + 1] == "benchmark_test-0007"
    assert command[command.index("--memory") + 1] == "benchmark_test-0007"


def test_parse_progress_events_reads_benchmark_and_ui_events() -> None:
    rows = parse_progress_events(
        '\n'.join(
            [
                'ui_step_start {"index": 1, "name": "Baseline"}',
                "benchmark_progress index=1/30 completed=1 failed=0 session_id=s1",
                'ui_run_stopped {"reason": "user_stop"}',
                'ui_run_done {"exit_code": 0}',
            ]
        )
    )

    assert [row["event"] for row in rows] == [
        "ui_step_start",
        "benchmark_progress",
        "ui_run_stopped",
        "ui_run_done",
    ]
    assert rows[1]["completed"] == "1"
    assert rows[2]["reason"] == "user_stop"
    assert rows[3]["exit_code"] == "0"
