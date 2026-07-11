from __future__ import annotations

from pathlib import Path
from argparse import Namespace

import pytest

from agent.byo.langgraph.react_agent import BasicReActAgent
from agent.composition import AgentRunConfig, MemoryConfig
from agent.extensions import react_agent as react_extension
from agent.extensions import run as extension_run
from agent.extensions.react_agent import LearningReActAgent
from nika.extensions import benchmark as benchmark_extension
from nika.extensions.benchmark import load_custom_benchmark
from nika.cli.commands import memory as memory_command


def _config(*, memory_mode: str = "off") -> AgentRunConfig:
    return AgentRunConfig(
        agent_type="byo.langgraph",
        llm_provider="custom",
        model="openai/gpt-oss-20b",
        max_steps=20,
        memory=MemoryConfig(mode=memory_mode),
    )


def test_learning_agent_inherits_original_execution_and_submission_methods() -> None:
    assert LearningReActAgent._run_diagnosis is BasicReActAgent._run_diagnosis
    assert LearningReActAgent._run_submission is BasicReActAgent._run_submission


def test_baseline_factory_constructs_original_agent(monkeypatch) -> None:
    sentinel = object()
    calls: list[dict] = []

    def fake_agent(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(react_extension, "BasicReActAgent", fake_agent)

    assert react_extension.create_react_agent(_config()) is sentinel
    assert calls == [
        {
            "session_id": "",
            "llm_provider": "custom",
            "model": "openai/gpt-oss-20b",
            "max_steps": 20,
        }
    ]


def test_modules_off_delegates_to_original_nika_runner(monkeypatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        extension_run,
        "start_nika_agent",
        lambda **kwargs: calls.append(kwargs),
    )

    extension_run.start_agent(_config(), session_id="session-1")

    assert calls == [
        {
            "agent_type": "byo.langgraph",
            "llm_provider": "custom",
            "model": "openai/gpt-oss-20b",
            "max_steps": 20,
            "session_id": "session-1",
            "stream_output": False,
        }
    ]


def test_custom_url_is_mapped_without_changing_core_factory(monkeypatch) -> None:
    monkeypatch.setenv(
        "CUSTOM_API_URL", "https://stream-netmind.viettel.vn/gateway/v1/"
    )
    monkeypatch.delenv("CUSTOM_API_BASE", raising=False)

    react_extension.configure_custom_provider_environment()

    assert (
        react_extension.os.environ["CUSTOM_API_BASE"]
        == "https://stream-netmind.viettel.vn/gateway/v1"
    )


def test_custom_benchmark_allows_empty_inject_only_for_clean_control(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.yaml"
    valid.write_text(
        "cases:\n"
        "  - scenario: simple_bgp\n"
        "    problem: no_fault\n"
        "    inject: {}\n",
        encoding="utf-8",
    )
    assert load_custom_benchmark(valid)[0]["inject"] == {}

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        "cases:\n"
        "  - scenario: simple_bgp\n"
        "    problem: link_down\n"
        "    inject: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-empty inject"):
        load_custom_benchmark(invalid)


def test_memory_command_routes_through_extension_benchmark(
    monkeypatch, tmp_path: Path
) -> None:
    benchmark = tmp_path / "benchmark.yaml"
    benchmark.write_text(
        "cases:\n"
        "  - scenario: simple_bgp\n"
        "    problem: link_down\n"
        "    inject:\n"
        "      host_name: router1\n"
        "      intf_name: eth0\n",
        encoding="utf-8",
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        memory_command.subprocess,
        "run",
        lambda command, check: calls.append(command),
    )

    memory_command.memory_run(
        file=benchmark,
        limit=None,
        bank="test-bank",
        read=False,
        reset_bank=False,
        llm_backend="custom",
        model="openai/gpt-oss-20b",
        max_steps=20,
        k=4,
        tokens=1200,
    )

    command = calls[0]
    assert command[1:3] == ["-m", "nika.extensions.benchmark"]
    assert command[command.index("--memory") + 1] == "test-bank"
    assert command[command.index("--provider") + 1] == "custom"


def test_baseline_fault_row_routes_to_upstream_single_case(
    monkeypatch, tmp_path: Path
) -> None:
    benchmark = tmp_path / "benchmark.yaml"
    benchmark.write_text(
        "cases:\n"
        "  - scenario: simple_bgp\n"
        "    problem: link_down\n"
        "    inject:\n"
        "      host_name: router1\n"
        "      intf_name: eth0\n",
        encoding="utf-8",
    )
    calls: list[dict] = []
    monkeypatch.setattr(
        benchmark_extension,
        "scan_benchmark_cases",
        lambda **_kwargs: (tmp_path, [0]),
    )
    monkeypatch.setattr(
        benchmark_extension,
        "run_single_case",
        lambda **kwargs: (calls.append(kwargs) or ("session-1", tmp_path / "session-1")),
    )

    result = benchmark_extension.run_batch(
        Namespace(
            config=str(benchmark),
            provider="custom",
            model="openai/gpt-oss-20b",
            max_steps=20,
            result_dir=str(tmp_path / "results"),
            resume=False,
            judge=False,
            judge_provider=None,
            judge_model=None,
            tools=None,
            tool_doc_chars=500,
            tool_convergence_threshold=0.75,
            memory=None,
            memory_read=None,
            memory_k=5,
            memory_tokens=1500,
            memory_max_skill_age=4,
            memory_pool_size=32,
            memory_evolution_threshold=3,
            memory_best_of_n=3,
            memory_ppo_epsilon=0.2,
        )
    )

    assert result == 0
    assert calls[0]["agent_type"] == "byo.langgraph"
    assert calls[0]["llm_provider"] == "custom"
