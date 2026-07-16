from __future__ import annotations

import json
from pathlib import Path
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain.agents.middleware import ModelRequest
from langchain_core.messages import HumanMessage

from agent.byo.langgraph.react_agent import BasicReActAgent
from agent.composition import (
    AgentRunConfig,
    ProceduralMemoryConfig,
    ToolRefinementConfig,
)
from agent.extensions import react_agent as react_extension
from agent.extensions import run as extension_run
from agent.extensions.react_agent import LearningReActAgent
from nika.extensions import benchmark as benchmark_extension
from nika.extensions.benchmark import load_custom_benchmark
from nika.cli.commands import procedural_memory as procedural_memory_command
from nika.workflows.benchmark import run as benchmark_run
from nika.workflows.benchmark.run import validate_inject_params


def _config(*, procedural_memory_mode: str = "off") -> AgentRunConfig:
    return AgentRunConfig(
        agent_type="react",
        llm_provider="custom",
        model="openai/gpt-oss-120b",
        max_steps=20,
        procedural_memory=ProceduralMemoryConfig(mode=procedural_memory_mode),
    )


def test_learning_agent_inherits_original_execution_and_submission_methods() -> None:
    assert LearningReActAgent._run_diagnosis is BasicReActAgent._run_diagnosis
    assert LearningReActAgent._run_submission is BasicReActAgent._run_submission


def test_learning_phase_rebuilds_skill_prompt_for_every_model_call() -> None:
    class Runtime:
        calls = 0

        def prompt_suffix(
            self,
            *,
            activate_skill: bool = True,
            decision_context: str = "",
        ) -> str:
            assert activate_skill
            assert decision_context == "Assigned diagnostic step"
            self.calls += 1
            return f"\nDynamic skill context {self.calls}"

    phase = react_extension.LearningDiagnosisPhase.__new__(
        react_extension.LearningDiagnosisPhase
    )
    phase.llm = object()
    phase.tools = []
    phase.skill_tool_runtime = Runtime()

    with patch.object(
        react_extension,
        "create_agent",
        side_effect=lambda **kwargs: SimpleNamespace(config=kwargs),
    ):
        agent = phase.get_agent()

    middleware = agent.config["middleware"][0]
    prompts: list[str] = []
    request = ModelRequest(
        model=phase.llm,
        messages=[HumanMessage(content="Assigned diagnostic step")],
    )

    def handler(current):
        prompts.append(current.system_prompt)
        return object()

    middleware.wrap_model_call(request, handler)
    middleware.wrap_model_call(request, handler)

    assert agent.config["system_prompt"] is None
    assert prompts[0].endswith("Dynamic skill context 1")
    assert prompts[1].endswith("Dynamic skill context 2")


def test_learning_module_failure_does_not_block_other_update(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "eval_metrics.json").write_text("{}", encoding="utf-8")
    (tmp_path / "run.json").write_text("{}", encoding="utf-8")
    session = SimpleNamespace(session_dir=str(tmp_path))
    session_loader = SimpleNamespace(
        load_closed_session=lambda **_kwargs: session,
    )
    memory_updates: list[str] = []

    async def update_memory(**_kwargs):
        memory_updates.append("updated")
        return {"status": "deferred", "episode_reward": 0.5}

    def fail_refinement(**_kwargs):
        raise RuntimeError("refinement unavailable")

    monkeypatch.setattr(benchmark_extension, "Session", lambda: session_loader)
    monkeypatch.setattr(
        benchmark_extension,
        "finalize_tool_refinement_session",
        fail_refinement,
    )
    monkeypatch.setattr(
        benchmark_extension,
        "update_procedural_memory_from_session",
        update_memory,
    )
    monkeypatch.setattr(benchmark_extension, "log_event", lambda *_a, **_k: None)
    config = AgentRunConfig(
        agent_type="react",
        llm_provider="custom",
        model="openai/gpt-oss-120b",
        max_steps=20,
        tool_refinement=ToolRefinementConfig(enabled=True),
        procedural_memory=ProceduralMemoryConfig(mode="evolve"),
    )

    benchmark_extension._update_learning("session-1", config)

    errors = json.loads((tmp_path / "learning_errors.json").read_text(encoding="utf-8"))
    assert memory_updates == ["updated"]
    saved_metrics = json.loads(
        (tmp_path / "eval_metrics.json").read_text(encoding="utf-8")
    )
    assert saved_metrics["procedural_memory"]["episode_reward"] == 0.5
    assert errors[0]["module"] == "tool_refinement"
    assert "refinement unavailable" in errors[0]["error"]

    monkeypatch.setattr(
        benchmark_extension,
        "finalize_tool_refinement_session",
        lambda **_kwargs: {},
    )
    benchmark_extension._update_learning("session-1", config)

    assert memory_updates == ["updated", "updated"]
    assert not (tmp_path / "learning_errors.json").exists()


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
            "model": "openai/gpt-oss-120b",
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
            "agent_type": "react",
            "llm_provider": "custom",
            "model": "openai/gpt-oss-120b",
            "max_steps": 20,
            "session_id": "session-1",
        }
    ]


def test_legacy_langgraph_alias_persists_react_metadata(monkeypatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        extension_run,
        "start_nika_agent",
        lambda **kwargs: calls.append(kwargs),
    )

    extension_run.start_agent(
        AgentRunConfig(
            agent_type="byo.langgraph",
            llm_provider="custom",
            model="openai/gpt-oss-120b",
            max_steps=20,
        ),
        session_id="session-1",
    )

    assert calls[0]["agent_type"] == "react"


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
        "cases:\n  - scenario: simple_bgp\n    problem: no_fault\n    inject: {}\n",
        encoding="utf-8",
    )
    assert load_custom_benchmark(valid)[0]["inject"] == {}
    validate_inject_params("no_fault", "simple_bgp", "", {})
    with pytest.raises(ValueError, match="empty inject map"):
        validate_inject_params("no_fault", "simple_bgp", "", {"host_name": "pc1"})

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        "cases:\n  - scenario: simple_bgp\n    problem: link_down\n    inject: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-empty inject"):
        load_custom_benchmark(invalid)


def test_extension_parser_uses_canonical_feature_terms() -> None:
    parser = benchmark_extension.build_parser()
    canonical = parser.parse_args(
        [
            "--config",
            "benchmark.yaml",
            "--tool-refinement",
            "refinement-bank",
            "--procedural-memory",
            "procedural-bank",
        ]
    )

    assert (canonical.tool_refinement, canonical.procedural_memory) == (
        "refinement-bank",
        "procedural-bank",
    )
    assert canonical.procedural_memory_max_skill_age == 8
    assert canonical.procedural_memory_update_threshold == 6
    assert canonical.procedural_memory_selection_epsilon == 0.25
    assert canonical.tool_refinement_exploration_similarity_threshold == 0.9
    assert canonical.tool_refinement_explorer_reflection_limit == 3
    assert canonical.procedural_memory_experience_pool_size == 256
    assert canonical.procedural_memory_baseline_ema_alpha == 0.1
    assert canonical.procedural_memory_selection_epsilon_decay_cases == 75
    assert canonical.procedural_memory_acceptance_margin == 0.01
    assert canonical.tool_refinement_explorer_model == ""
    assert canonical.tool_refinement_analyzer_model == ""
    assert canonical.tool_refinement_rewriter_model == ""
    assert canonical.procedural_memory_evolver_model == "openai/gpt-oss-120b"
    assert canonical.procedural_memory_policy_scorer_model == "openai/gpt-oss-120b"
    help_text = parser.format_help()
    assert "--tool-refinement" in help_text
    assert "--procedural-memory" in help_text
    assert "--procedural-memory-read" not in help_text


def test_clean_control_scores_false_and_empty_submission_as_fully_correct() -> None:
    from agent.procedural_memory.service import _metric_success

    scores = benchmark_run.no_fault_scores(
        {
            "is_anomaly": 0,
            "faulty_devices": [],
            "root_cause_name": [],
        }
    )

    assert set(scores.values()) == {1.0}
    assert _metric_success(scores) is True


def test_clean_control_scores_each_component_independently() -> None:
    scores = benchmark_run.no_fault_scores(
        {
            "is_anomaly": True,
            "faulty_devices": ["router1"],
            "root_cause_name": [],
        }
    )

    assert scores["detection_score"] == 0.0
    assert scores["localization_f1"] == 0.0
    assert scores["rca_f1"] == 1.0


def test_clean_control_ground_truth_uses_complete_empty_schema(monkeypatch) -> None:
    updates: dict[str, object] = {}
    ground_truth: list[dict] = []

    class FakeSession:
        def update_session(self, key, value):
            updates[key] = value

        def write_gt(self, value):
            ground_truth.append(value)

    monkeypatch.setattr(
        benchmark_run,
        "_no_fault_task_description",
        lambda _session: "Inspect the clean network.",
    )

    benchmark_run.prepare_no_fault_case(FakeSession())

    assert updates == {
        "problem_names": ["no_fault"],
        "root_cause_category": "none",
        "task_description": "Inspect the clean network.",
    }
    assert ground_truth == [
        {
            "is_anomaly": False,
            "faulty_devices": [],
            "root_cause_category": "none",
            "root_cause_name": [],
            "detailed_cause": "",
        }
    ]


def test_clean_control_normalization_updates_artifact_before_learning(
    monkeypatch, tmp_path: Path
) -> None:
    submission = {
        "is_anomaly": False,
        "faulty_devices": [],
        "root_cause_name": [],
    }
    metrics = {
        "detection_score": 1.0,
        "localization_f1": 0.0,
        "rca_f1": 0.0,
        "tool_calls": 4,
    }
    (tmp_path / "submission.json").write_text(
        benchmark_extension.json.dumps(submission), encoding="utf-8"
    )
    (tmp_path / "eval_metrics.json").write_text(
        benchmark_extension.json.dumps(metrics), encoding="utf-8"
    )
    persisted: list[dict] = []

    class FakeSession:
        def load_closed_session(self, *, session_id):
            assert session_id == "clean-session"
            return self

        def update_run_meta(self, key, value):
            assert key == "eval_metrics"
            persisted.append(value)

    monkeypatch.setattr(benchmark_run, "Session", FakeSession)
    monkeypatch.setattr(benchmark_run, "log_event", lambda *_args, **_kwargs: None)

    benchmark_run.normalize_no_fault_metrics("clean-session", tmp_path)

    normalized = benchmark_extension._read_json(tmp_path / "eval_metrics.json")
    assert normalized["tool_calls"] == 4
    assert normalized["detection_score"] == 1.0
    assert normalized["localization_f1"] == 1.0
    assert normalized["rca_f1"] == 1.0
    assert persisted == [normalized]


def test_clean_control_without_submission_keeps_upstream_missing_scores(
    monkeypatch, tmp_path: Path
) -> None:
    metrics = {
        "detection_score": -1.0,
        "localization_f1": -1.0,
        "rca_f1": -1.0,
    }
    metrics_path = tmp_path / "eval_metrics.json"
    metrics_path.write_text(benchmark_extension.json.dumps(metrics), encoding="utf-8")
    monkeypatch.setattr(
        benchmark_run,
        "Session",
        lambda: pytest.fail("a missing submission must not update run metadata"),
    )

    benchmark_run.normalize_no_fault_metrics("missing", tmp_path)

    assert benchmark_extension._read_json(metrics_path) == metrics


def test_clean_control_path_never_injects_and_normalizes_before_learning(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    class FakeSession:
        session_id = "clean-session"

        def load_running_session(self, *, session_id):
            assert session_id == "clean-session"
            return self

        def update_session(self, _key, _value):
            calls.append("fingerprint")

    class FakeStore:
        def get_session(self, session_id):
            assert session_id == "clean-session"
            return {"session_dir": str(tmp_path)}

    monkeypatch.setattr(
        benchmark_extension,
        "start_net_env",
        lambda *_args, **_kwargs: "clean-session",
    )
    monkeypatch.setattr(benchmark_extension, "Session", FakeSession)
    monkeypatch.setattr(benchmark_extension, "SessionStore", FakeStore)
    monkeypatch.setattr(
        benchmark_extension,
        "prepare_no_fault_case",
        lambda _session: calls.append("prepare_clean"),
    )
    monkeypatch.setattr(
        benchmark_extension,
        "inject_failure",
        lambda **_kwargs: pytest.fail("clean controls must not inject a fault"),
    )
    monkeypatch.setattr(
        benchmark_extension,
        "start_agent",
        lambda *_args, **_kwargs: calls.append("agent"),
    )
    monkeypatch.setattr(
        benchmark_extension,
        "eval_results",
        lambda **_kwargs: calls.append("eval"),
    )
    monkeypatch.setattr(
        benchmark_extension,
        "normalize_no_fault_metrics",
        lambda *_args: calls.append("normalize"),
    )
    monkeypatch.setattr(
        benchmark_extension,
        "_update_learning",
        lambda *_args: calls.append("learn"),
    )

    benchmark_extension.run_extended_case(
        {
            "scenario": "simple_bgp",
            "topo_size": "",
            "problem": "no_fault",
            "inject": {},
        },
        config=_config(),
        result_dir=str(tmp_path),
    )

    assert calls == [
        "prepare_clean",
        "fingerprint",
        "agent",
        "eval",
        "normalize",
        "learn",
    ]


def test_procedural_memory_command_routes_through_extension_benchmark(
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
        procedural_memory_command.subprocess,
        "run",
        lambda command, check: calls.append(command),
    )

    procedural_memory_command.procedural_memory_run(
        file=benchmark,
        limit=None,
        evolve_until=2,
        bank="test-bank",
        reset_bank=False,
        llm_backend="custom",
        model="openai/gpt-oss-120b",
        max_steps=20,
        tokens=1200,
        max_skill_age=6,
        pool_size=24,
        update_threshold=2,
        best_of_n=5,
        ppo_epsilon=0.15,
        selection_epsilon=0.25,
        experience_pool_size=900,
        baseline_ema_alpha=0.2,
        selection_epsilon_decay_cases=300,
        acceptance_margin=0.005,
        verifier="structured_replay",
        holdout_size=1,
        min_positive_advantage=1,
        evolver_model="evolver-model",
        policy_scorer_model="policy-model",
    )

    command = calls[0]
    assert command[1:3] == ["-m", "nika.extensions.benchmark"]
    assert command[command.index("--procedural-memory") + 1] == "test-bank"
    assert "--procedural-memory-read" not in command
    assert command[command.index("--provider") + 1] == "custom"
    assert command[command.index("--procedural-memory-token-budget") + 1] == "1200"
    assert command[command.index("--procedural-memory-max-skill-age") + 1] == "6"
    assert command[command.index("--procedural-memory-pool-size") + 1] == "24"
    assert command[command.index("--procedural-memory-update-threshold") + 1] == "2"
    assert command[command.index("--procedural-memory-best-of-n") + 1] == "5"
    assert command[command.index("--procedural-memory-ppo-epsilon") + 1] == "0.15"
    assert command[command.index("--procedural-memory-selection-epsilon") + 1] == "0.25"
    assert (
        command[command.index("--procedural-memory-experience-pool-size") + 1] == "900"
    )
    assert command[command.index("--procedural-memory-baseline-ema-alpha") + 1] == "0.2"
    assert (
        command[command.index("--procedural-memory-selection-epsilon-decay-cases") + 1]
        == "300"
    )
    assert command[command.index("--procedural-memory-verifier") + 1] == (
        "structured_replay"
    )
    assert command[command.index("--procedural-memory-holdout-size") + 1] == "1"
    assert command[command.index("--procedural-memory-acceptance-margin") + 1] == (
        "0.005"
    )
    assert (
        command[command.index("--procedural-memory-min-positive-advantage") + 1] == "1"
    )
    assert command[command.index("--procedural-memory-evolver-model") + 1] == (
        "evolver-model"
    )
    assert command[command.index("--procedural-memory-policy-scorer-model") + 1] == (
        "policy-model"
    )
    assert command[command.index("--evolve-until") + 1] == "2"


def test_baseline_fault_row_routes_to_upstream_single_case(
    monkeypatch, tmp_path: Path
) -> None:
    benchmark = tmp_path / "benchmark.yaml"
    benchmark.write_text(
        "evolve_first_cases: 1\n"
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
        lambda **kwargs: calls.append(kwargs) or ("session-1", tmp_path / "session-1"),
    )

    result = benchmark_extension.run_batch(
        Namespace(
            config=str(benchmark),
            provider="custom",
            model="openai/gpt-oss-120b",
            max_steps=20,
            result_dir=str(tmp_path / "results"),
            resume=False,
            judge=False,
            judge_provider=None,
            judge_model=None,
            tool_refinement=None,
            tool_refinement_doc_chars=500,
            tool_refinement_convergence_threshold=0.75,
            procedural_memory=None,
            procedural_memory_tokens=1500,
            procedural_memory_max_skill_age=4,
            procedural_memory_pool_size=32,
            procedural_memory_update_threshold=3,
            procedural_memory_best_of_n=3,
            procedural_memory_ppo_epsilon=0.2,
        )
    )

    assert result == 0
    assert calls[0]["agent_type"] == "react"
    assert calls[0]["llm_provider"] == "custom"
    assert calls[0]["benchmark_index"] == 1
    assert calls[0]["benchmark_phase"] == "evolve"


def test_benchmark_switches_both_modules_to_read_after_evolve_cutoff(
    monkeypatch, tmp_path: Path
) -> None:
    benchmark = tmp_path / "benchmark.yaml"
    benchmark.write_text(
        "evolve_first_cases: 2\n"
        "cases:\n"
        "  - {scenario: simple_bgp, problem: no_fault, inject: {}}\n"
        "  - {scenario: simple_bgp, problem: no_fault, inject: {}}\n"
        "  - {scenario: simple_bgp, problem: no_fault, inject: {}}\n",
        encoding="utf-8",
    )
    modes: list[tuple[str, str, bool]] = []
    monkeypatch.setattr(
        benchmark_extension,
        "scan_benchmark_cases",
        lambda **_kwargs: (tmp_path, [1, 2]),
    )

    def run_case(_row, *, config, **_kwargs):
        modes.append(
            (
                config.procedural_memory.mode,
                config.tool_refinement.learning_mode,
                config.tool_refinement.update_due,
            )
        )
        index = len(modes)
        return f"session-{index}", tmp_path / f"session-{index}"

    monkeypatch.setattr(benchmark_extension, "run_extended_case", run_case)

    class FakeMemory:
        def __init__(self, *, bank_id):
            self.bank_id = bank_id

        def freeze_for_evaluation(self, *, output_path):
            output_path.write_text("snapshot\n", encoding="utf-8")
            return {
                "bank_id": self.bank_id,
                "iteration": 2,
                "state_hash": "frozen-hash",
                "snapshot_path": str(output_path),
                "retired_probationary_skill_ids": [],
                "validated_skill_ids": [],
            }

        def bank_state_hash(self):
            return "frozen-hash"

    monkeypatch.setattr(benchmark_extension, "ProceduralMemoryModule", FakeMemory)

    result = benchmark_extension.run_batch(
        Namespace(
            config=str(benchmark),
            agent="react",
            provider="custom",
            model="openai/gpt-oss-120b",
            max_steps=20,
            max_attempts=3,
            result_dir=str(tmp_path / "results"),
            resume=True,
            judge=False,
            judge_provider=None,
            judge_model=None,
            tool_refinement="shared-tools",
            tool_refinement_doc_chars=500,
            tool_refinement_convergence_threshold=0.75,
            procedural_memory="shared-bank",
            evolve_until=None,
            procedural_memory_tokens=1500,
            procedural_memory_max_skill_age=8,
            procedural_memory_pool_size=32,
            procedural_memory_update_threshold=6,
            procedural_memory_best_of_n=3,
            procedural_memory_ppo_epsilon=0.2,
            procedural_memory_selection_epsilon=0.3,
        )
    )

    assert result == 0
    assert modes == [
        ("evolve", "evolve", True),
        ("read", "read", False),
    ]
