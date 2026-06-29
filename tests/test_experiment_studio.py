from __future__ import annotations

import sys

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
        "parallel": 4,
        "tool_library_id": "tools-test",
        "tool_mode": "dual",
        "memory_bank": "memory-test",
        "memory_k": 5,
        "memory_tokens": 1500,
        "max_generations": 3,
        "feedback_mode": "deterministic",
        "feedback_backend": "netmind",
        "feedback_model": "openai/gpt-oss-120b",
    }
    config.update(overrides)
    return config


def test_baseline_command_uses_requested_parallel() -> None:
    command = build_experiment_command(_config())

    assert command[:3] == [sys.executable, "-m", "nika.codex_cli.main"]
    assert command[3:6] == ["benchmark", "run", "--file"]
    assert "benchmark/benchmark_test.csv" in command
    assert command[command.index("-n") + 1] == "100"
    assert command[command.index("-j") + 1] == "4"


def test_tool_and_memory_modules_share_one_sequential_command() -> None:
    command = build_experiment_command(
        _config(modules=["tool_evolution", "memory_evolution"])
    )

    assert command[3:6] == ["benchmark", "run", "--file"]
    assert command[command.index("-j") + 1] == "1"
    assert command[command.index("--tools") + 1] == "tools-test"
    assert command[command.index("--memory") + 1] == "memory-test"


def test_agent_evolution_modules_share_one_evolve_command() -> None:
    command = build_experiment_command(
        _config(modules=["agent_evolution", "tool_evolution", "memory_evolution"])
    )

    assert command[3:6] == ["evolve", "run", "--file"]
    assert command[command.index("--max-gen") + 1] == "3"
    assert command[command.index("--feedback-mode") + 1] == "deterministic"
    assert command[command.index("-j") + 1] == "1"
    assert command[command.index("--tools") + 1] == "tools-test"
    assert command[command.index("--memory") + 1] == "memory-test"


def test_command_plan_can_start_memory_services() -> None:
    plan = build_command_plan(
        _config(
            modules=["memory_evolution"],
            ensure_memory_services=True,
        )
    )

    assert plan[0].name == "Memory services"
    assert plan[0].command == ["docker", "compose", "up", "-d", "postgres", "qdrant"]
    assert plan[1].variant == "benchmark"
    assert plan[1].name == "Memory Evolution"


def test_parse_progress_events_reads_benchmark_and_ui_events() -> None:
    rows = parse_progress_events(
        '\n'.join(
            [
                'ui_step_start {"index": 1, "name": "Baseline"}',
                "benchmark_progress index=1/30 completed=1 failed=0 session_id=s1",
                'ui_run_done {"exit_code": 0}',
            ]
        )
    )

    assert [row["event"] for row in rows] == [
        "ui_step_start",
        "benchmark_progress",
        "ui_run_done",
    ]
    assert rows[1]["completed"] == "1"
    assert rows[2]["exit_code"] == "0"
