from __future__ import annotations

from pathlib import Path

import pytest

from agent.byo.langgraph.react_agent import BasicReActAgent
from agent.composition import AgentRunConfig, MemoryConfig
from agent.extensions import react_agent as react_extension
from agent.extensions import run as extension_run
from agent.extensions.react_agent import LearningReActAgent
from nika.extensions.benchmark import load_custom_benchmark


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

