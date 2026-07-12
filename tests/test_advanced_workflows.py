from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from agent.composition import AgentRunConfig, validate_agent_extensions
from agent.extensions import factory
from agent.extensions.plan_execute_agent import PlanExecuteAgent
from agent.extensions.reflexion_agent import ReflexionAgent
from agent.extensions.workflow_base import ExtensionWorkflowBase
from agent.extensions.workflow_models import ReflexionEvaluation
from nika.visualization.experiment_runner import build_experiment_command


def _config(agent_type: str) -> AgentRunConfig:
    return AgentRunConfig(
        agent_type=agent_type,
        llm_provider="custom",
        model="openai/gpt-oss-20b",
        max_steps=20,
        session_id="session-1",
    )


@pytest.mark.parametrize(
    ("configured", "normalized"),
    [
        ("react", "react"),
        ("plan_execute", "plan-execute"),
        ("plan-and-execute", "plan-execute"),
        ("reflexion", "reflexion"),
    ],
)
def test_local_workflow_names_are_normalized(configured: str, normalized: str) -> None:
    config = _config(configured)
    validate_agent_extensions(config)
    assert config.normalized_agent_type == normalized


def test_factory_routes_advanced_workflows(monkeypatch) -> None:
    calls: list[tuple[str, AgentRunConfig]] = []

    monkeypatch.setattr(
        factory,
        "PlanExecuteAgent",
        lambda config: calls.append(("plan", config)) or object(),
    )
    monkeypatch.setattr(
        factory,
        "ReflexionAgent",
        lambda config: calls.append(("reflexion", config)) or object(),
    )

    factory.create_extension_agent(_config("plan-execute"))
    factory.create_extension_agent(_config("reflexion"))

    assert [name for name, _config_value in calls] == ["plan", "reflexion"]


def test_reflexion_success_requires_supported_high_score() -> None:
    with pytest.raises(ValueError, match="score >= 0.8"):
        ReflexionEvaluation(
            success=True,
            quality_score=0.7,
            evidence_sufficient=True,
        )


def test_studio_command_carries_workflow_and_attempt_budget() -> None:
    command = build_experiment_command(
        {
            "benchmark_file": "benchmark/benchmark_test.yaml",
            "agent_type": "reflexion",
            "max_steps": 30,
            "max_attempts": 4,
        }
    )

    assert command[command.index("--agent") + 1] == "reflexion"
    assert command[command.index("--max-attempts") + 1] == "4"


def test_workflow_callback_preserves_diagnosis_agent_and_phase(tmp_path) -> None:
    agent = object.__new__(ExtensionWorkflowBase)
    agent.session_dir = str(tmp_path)

    callback = agent.callback("planner")
    callback._logger.log("test", {"value": 1})

    event = json.loads((tmp_path / "messages.jsonl").read_text(encoding="utf-8"))
    assert event["agent"] == "diagnosis"
    assert event["phase"] == "planner"


class _FakeRunnable:
    def __init__(self, values):
        self.values = list(values)

    async def ainvoke(self, *_args, **_kwargs):
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def test_plan_execute_replans_and_submits_supported_report() -> None:
    agent = object.__new__(PlanExecuteAgent)
    agent.max_steps = 5
    agent.planner = _FakeRunnable(
        [
            {
                "objective": "Find the network fault",
                "steps": [
                    {
                        "step_id": "one",
                        "action": "Inspect reachability",
                        "expected_evidence": "Ping output",
                    }
                ],
            }
        ]
    )
    agent.replanner = _FakeRunnable(
        [
            {
                "completed": True,
                "diagnosis_report": "The observed link is down.",
            }
        ]
    )
    agent.llm = _FakeRunnable([])
    diagnosis = _FakeRunnable(
        [{"messages": [SimpleNamespace(content="The ping failed.")]}]
    )
    agent.prepare_diagnosis = lambda _task: diagnosis
    agent.callback = lambda _phase: None
    agent.write_extension_snapshots = lambda: None

    async def submit(report: str):
        return {"diagnosis_report": report, "submitted": True}

    agent.submit = submit
    result = asyncio.run(agent.run("Diagnose the network"))

    assert result["submitted"] is True
    assert result["diagnosis_report"] == "The observed link is down."


def test_reflexion_uses_feedback_and_submits_best_attempt() -> None:
    agent = object.__new__(ReflexionAgent)
    agent.max_steps = 10
    agent.max_attempts = 2
    agent.evaluator = _FakeRunnable(
        [
            {
                "success": False,
                "quality_score": 0.4,
                "evidence_sufficient": False,
                "feedback": ["Inspect the faulty interface directly."],
            },
            {
                "success": True,
                "quality_score": 0.9,
                "evidence_sufficient": True,
            },
        ]
    )
    agent.reflector = _FakeRunnable(
        [
            {
                "lessons": ["The first attempt used broad checks."],
                "next_strategy": ["Inspect interface state."],
            }
        ]
    )
    diagnosis = _FakeRunnable(
        [
            {"messages": [SimpleNamespace(content="Weak report")]},
            {"messages": [SimpleNamespace(content="Supported report")]},
        ]
    )
    agent.prepare_diagnosis = lambda _task: diagnosis
    agent.callback = lambda _phase: None
    agent.write_extension_snapshots = lambda: None

    async def submit(report: str):
        return {"diagnosis_report": report, "submitted": True}

    agent.submit = submit
    result = asyncio.run(agent.run("Diagnose the network"))

    assert result["submitted"] is True
    assert result["diagnosis_report"] == "Supported report"
