from __future__ import annotations

import json
from pathlib import Path
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
from agent.extensions.react_agent import TrainingReActAgent
from nika.extensions import benchmark as benchmark_extension
from nika.extensions.benchmark import load_custom_benchmark
from nika.cli.commands import procedural_memory as procedural_memory_command
from nika.workflows.benchmark import run as benchmark_run
from nika.workflows.benchmark.run import validate_inject_params


def _config(*, procedural_memory_enabled: bool = False) -> AgentRunConfig:
    return AgentRunConfig(
        agent_type="react",
        llm_provider="custom",
        model="openai/gpt-oss-120b",
        max_steps=20,
        procedural_memory=ProceduralMemoryConfig(enabled=procedural_memory_enabled),
    )


def test_training_agent_inherits_original_execution_and_submission_methods() -> None:
    assert TrainingReActAgent._run_diagnosis is BasicReActAgent._run_diagnosis
    assert TrainingReActAgent._run_submission is BasicReActAgent._run_submission


def test_training_phase_rebuilds_skill_prompt_for_every_model_call() -> None:
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

    phase = react_extension.TrainingDiagnosisPhase.__new__(
        react_extension.TrainingDiagnosisPhase
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


def test_training_module_failure_does_not_block_other_update(
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
        allow_training_updates=True,
        tool_refinement=ToolRefinementConfig(enabled=True),
        procedural_memory=ProceduralMemoryConfig(enabled=True),
    )

    benchmark_extension._update_training("session-1", config)

    errors = json.loads((tmp_path / "training_errors.json").read_text(encoding="utf-8"))
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
    benchmark_extension._update_training("session-1", config)

    assert memory_updates == ["updated", "updated"]
    assert not (tmp_path / "training_errors.json").exists()


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
            "--training-benchmark",
            "training.yaml",
            "--evaluate-benchmark",
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
    assert canonical.procedural_memory_update_threshold == 3
    assert canonical.procedural_memory_selection_epsilon == 0.25
    assert canonical.tool_refinement_exploration_similarity_threshold == 0.9
    assert canonical.tool_refinement_explorer_reflection_limit == 3
    assert canonical.procedural_memory_experience_pool_size == 256
    assert canonical.procedural_memory_baseline_ema_alpha == 0.1
    assert canonical.procedural_memory_selection_epsilon_decay_cases == 100
    assert canonical.procedural_memory_acceptance_margin == 0.01
    assert canonical.tool_refinement_explorer_model == ""
    assert canonical.tool_refinement_analyzer_model == ""
    assert canonical.tool_refinement_rewriter_model == ""
    assert canonical.procedural_memory_evolver_model == "openai/gpt-oss-120b"
    assert canonical.procedural_memory_policy_scorer_model == "openai/gpt-oss-120b"
    help_text = parser.format_help()
    assert "--tool-refinement" in help_text
    assert "--procedural-memory" in help_text
    assert "--training-benchmark" in help_text
    assert "--evaluate-benchmark" in help_text
    assert "--config" not in help_text
    assert "--evolve-until" not in help_text


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


def test_clean_control_normalization_updates_artifact_before_training(
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


def test_clean_control_path_never_injects_and_normalizes_before_training(
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
        "_update_training",
        lambda *_args: calls.append("learn") or {},
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
        training_benchmark=benchmark,
        evaluate_benchmark=benchmark,
        bank="test-bank",
        result_dir=tmp_path / "results",
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
    assert command[command.index("--training-benchmark") + 1] == str(benchmark)
    assert command[command.index("--evaluate-benchmark") + 1] == str(benchmark)
    assert command[command.index("--result-dir") + 1] == str(tmp_path / "results")
    assert "--resume" in command
    assert command[command.index("--procedural-memory") + 1] == "test-bank"
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
    assert "--evolve-until" not in command


def test_baseline_fault_row_routes_to_upstream_single_case(
    monkeypatch, tmp_path: Path
) -> None:
    benchmark = tmp_path / "benchmark.yaml"
    benchmark.write_text(
        "benchmark_role: evaluation\n"
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

    args = benchmark_extension.build_parser().parse_args(
        [
            "--evaluate-benchmark",
            str(benchmark),
            "--provider",
            "custom",
            "--model",
            "openai/gpt-oss-120b",
            "--max-steps",
            "20",
            "--result-dir",
            str(tmp_path / "results"),
        ]
    )
    result = benchmark_extension.run_batch(args)

    assert result == 0
    assert calls[0]["agent_type"] == "react"
    assert calls[0]["llm_provider"] == "custom"
    assert calls[0]["benchmark_index"] == 1
    assert calls[0]["benchmark_role"] == "evaluation"


def test_training_barrier_contains_loadable_immutable_snapshots(
    monkeypatch, tmp_path: Path
) -> None:
    from agent.procedural_memory import store as memory_store_module
    from agent.tool_refinement import store as tool_store_module

    monkeypatch.setattr(
        memory_store_module,
        "PROCEDURAL_MEMORY_DIR",
        tmp_path / "live-memory",
    )
    monkeypatch.setattr(
        tool_store_module,
        "TOOL_REFINEMENT_DIR",
        tmp_path / "live-tools",
    )
    memory = benchmark_extension.ProceduralMemoryModule(bank_id="barrier-bank")
    memory_state = memory.store.load()
    memory_state.iteration = 3
    memory.store.save(memory_state)
    tool_store = benchmark_extension.ToolRefinementStore("barrier-tools")
    tool_store.save(tool_store.load())

    training_path = tmp_path / "training.yaml"
    training_path.write_text(
        "benchmark_role: training\n"
        "cases:\n"
        "  - {scenario: simple_bgp, problem: no_fault, inject: {}}\n",
        encoding="utf-8",
    )
    evaluation_path = tmp_path / "evaluation.yaml"
    evaluation_path.write_text(
        "benchmark_role: evaluation\n"
        "cases:\n"
        "  - {scenario: p4_mpls, problem: no_fault, inject: {}}\n",
        encoding="utf-8",
    )
    training_manifest = benchmark_extension.load_benchmark_manifest(
        training_path,
        expected_role="training",
    )
    evaluation_manifest = benchmark_extension.load_benchmark_manifest(
        evaluation_path,
        expected_role="evaluation",
    )
    config = AgentRunConfig(
        agent_type="react",
        llm_provider="custom",
        model="openai/gpt-oss-120b",
        max_steps=20,
        allow_training_updates=True,
        procedural_memory=ProceduralMemoryConfig(
            enabled=True,
            bank="barrier-bank",
        ),
        tool_refinement=ToolRefinementConfig(
            enabled=True,
            library_id="barrier-tools",
        ),
    )

    barrier = benchmark_extension._freeze_training_modules(
        config=config,
        results_root=tmp_path / "results",
        training_manifest=training_manifest,
        evaluation_manifest=evaluation_manifest,
        config_fingerprint="config-hash",
    )

    memory_snapshot = Path(barrier["procedural_memory"]["snapshot_path"])
    tool_snapshot = Path(barrier["tool_refinement"]["snapshot_path"])
    frozen_memory = benchmark_extension.ProceduralMemoryModule(
        bank_id="barrier-bank",
        store_path=memory_snapshot,
        read_only=True,
    )
    frozen_tools = benchmark_extension.ToolRefinementStore(
        "barrier-tools",
        state_path=tool_snapshot,
        read_only=True,
    )

    assert frozen_memory.store.load().iteration == 3
    assert frozen_tools.load().library_id == "barrier-tools"
    benchmark_extension._validate_training_barrier(
        barrier,
        training_manifest=training_manifest,
        evaluation_manifest=evaluation_manifest,
        config_fingerprint="config-hash",
        config=config,
    )
    benchmark_extension._verify_frozen_modules(barrier=barrier, config=config)
    with pytest.raises(PermissionError):
        frozen_memory.store.save(frozen_memory.store.load())
    with pytest.raises(PermissionError):
        frozen_tools.save(frozen_tools.load())

    memory_snapshot.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="frozen snapshot changed"):
        benchmark_extension._verify_frozen_modules(barrier=barrier, config=config)


def test_benchmark_pipeline_enables_updates_only_for_training_stage(
    monkeypatch, tmp_path: Path
) -> None:
    training = tmp_path / "training.yaml"
    training.write_text(
        "benchmark_role: training\n"
        "cases:\n"
        "  - {scenario: simple_bgp, problem: no_fault, inject: {}}\n"
        "  - {scenario: ospf_enterprise_dhcp, topo_size: s, problem: no_fault, inject: {}}\n",
        encoding="utf-8",
    )
    evaluation = tmp_path / "evaluation.yaml"
    evaluation.write_text(
        "benchmark_role: evaluation\n"
        "cases:\n"
        "  - {scenario: simple_bgp, problem: no_fault, inject: {}}\n",
        encoding="utf-8",
    )
    stages: list[tuple[bool, bool, bool, bool, str]] = []
    monkeypatch.setattr(
        benchmark_extension,
        "scan_benchmark_cases",
        lambda **kwargs: (tmp_path, list(range(len(kwargs["rows"])))),
    )

    def run_case(_row, *, config, benchmark_role, **_kwargs):
        stages.append(
            (
                config.allow_training_updates,
                config.procedural_memory.enabled,
                config.tool_refinement.enabled,
                config.tool_refinement.update_due,
                benchmark_role,
            )
        )
        index = len(stages)
        return f"session-{index}", tmp_path / f"session-{index}"

    monkeypatch.setattr(benchmark_extension, "run_extended_case", run_case)
    monkeypatch.setattr(
        benchmark_extension,
        "_freeze_training_modules",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        benchmark_extension,
        "_verify_frozen_modules",
        lambda **_kwargs: None,
    )
    args = benchmark_extension.build_parser().parse_args(
        [
            "--training-benchmark",
            str(training),
            "--evaluate-benchmark",
            str(evaluation),
            "--result-dir",
            str(tmp_path / "results"),
            "--tool-refinement",
            "shared-tools",
            "--procedural-memory",
            "shared-bank",
        ]
    )
    result = benchmark_extension.run_batch(args)

    assert result == 0
    assert stages == [
        (True, True, True, False, "training"),
        (True, True, True, True, "training"),
        (False, True, True, False, "evaluation"),
    ]
