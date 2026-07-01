from __future__ import annotations

import sys

from agent.defaults import DEFAULT_MAX_STEPS
from agent.llm.model_factory import DEFAULT_LLM_BACKEND, DEFAULT_MODEL
from nika.visualization.experiment_runner import (
    build_experiment_command,
    build_command_plan,
    parse_progress_events,
)


def _config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "benchmark_file": "benchmark/benchmark_test.csv",
        "modules": [],
        "agent_type": "react",
        "llm_backend": "netmind",
        "model": "openai/gpt-oss-120b",
        "max_steps": 100,
        "max_attempts": 3,
        "tool_library_id": "tools-test",
        "memory_bank": "memory-test",
        "memory_k": 5,
        "memory_tokens": 1500,
        "max_generations": 3,
        "feedback_backend": "netmind",
        "feedback_model": "openai/gpt-oss-120b",
    }
    config.update(overrides)
    return config


def test_baseline_command_uses_default_sequential_execution() -> None:
    command = build_experiment_command(_config())

    assert command[:3] == [sys.executable, "-m", "nika.codex_cli.main"]
    assert command[3:6] == ["benchmark", "run", "--file"]
    assert "benchmark/benchmark_test.csv" in command
    assert command[command.index("-n") + 1] == "100"
    assert "-j" not in command
    assert "--parallel" not in command


def test_standard_agent_label_uses_selected_baseline_name() -> None:
    plan = build_command_plan(
        _config(agent_type="reflexion", modules=["memory_evolution"])
    )

    assert plan[0].variant == "benchmark"
    assert plan[0].name == "Reflexion + Memory Evolution"


def test_command_fallbacks_match_shared_agent_defaults() -> None:
    config = _config()
    for key in ("llm_backend", "model", "max_steps"):
        config.pop(key)

    command = build_experiment_command(config)

    assert command[command.index("-b") + 1] == DEFAULT_LLM_BACKEND
    assert command[command.index("-m", command.index("-b")) + 1] == DEFAULT_MODEL
    assert command[command.index("-n") + 1] == str(DEFAULT_MAX_STEPS)


def test_tool_and_memory_modules_share_one_sequential_command() -> None:
    command = build_experiment_command(
        _config(modules=["tool_evolution", "memory_evolution"])
    )

    assert command[3:6] == ["benchmark", "run", "--file"]
    assert "-j" not in command
    assert "--parallel" not in command
    assert command[command.index("--tools") + 1] == "tools-test"
    assert command[command.index("--memory") + 1] == "memory-test"


def test_sia_h_agent_uses_one_evolve_command_with_optional_modules() -> None:
    command = build_experiment_command(
        _config(agent_type="sia-h", modules=["tool_evolution", "memory_evolution"])
    )

    assert command[3:6] == ["evolve", "run", "--file"]
    assert command[command.index("--max-gen") + 1] == "3"
    assert "-a" not in command
    assert "-r" not in command
    assert "-j" not in command
    assert "--parallel" not in command
    assert command[command.index("--tools") + 1] == "tools-test"
    assert command[command.index("--memory") + 1] == "memory-test"


def test_sia_h_agent_is_labeled_as_agent_not_module() -> None:
    plan = build_command_plan(
        _config(agent_type="sia-h", modules=["tool_evolution"])
    )

    assert plan[0].variant == "sia_h"
    assert plan[0].name == "SIA-H + Tool Evolution"


def test_command_plan_for_memory_has_no_service_prerequisite() -> None:
    plan = build_command_plan(
        _config(
            modules=["memory_evolution"],
        )
    )

    assert len(plan) == 1
    assert plan[0].variant == "benchmark"
    assert plan[0].name == "ReAct + Memory Evolution"


def test_parse_progress_events_reads_benchmark_and_ui_events() -> None:
    rows = parse_progress_events(
        '\n'.join(
            [
                'ui_step_start {"index": 1, "name": "Baseline"}',
                "benchmark_progress index=1/30 completed=1 failed=0 session_id=s1",
                "kathara_cleanup_done context=studio_run",
                'ui_run_stopped {"reason": "user_stop"}',
                'ui_run_done {"exit_code": 0}',
            ]
        )
    )

    assert [row["event"] for row in rows] == [
        "ui_step_start",
        "benchmark_progress",
        "kathara_cleanup_done",
        "ui_run_stopped",
        "ui_run_done",
    ]
    assert rows[1]["completed"] == "1"
    assert rows[2]["context"] == "studio_run"
    assert rows[3]["reason"] == "user_stop"
    assert rows[4]["exit_code"] == "0"
