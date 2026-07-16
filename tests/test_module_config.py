from __future__ import annotations

from pathlib import Path

import pytest

from agent.composition import ProceduralMemoryConfig, ToolRefinementConfig
from agent.module_config import load_module_defaults
from nika.extensions.benchmark import build_parser


def test_module_defaults_are_shared_by_composition_and_cli() -> None:
    defaults = load_module_defaults()
    tool = ToolRefinementConfig()
    memory = ProceduralMemoryConfig()
    parser = build_parser()
    args = parser.parse_args(["--evaluate-benchmark", "benchmark/example.yaml"])

    assert tool.tool_doc_chars == defaults.tool_refinement.tool_doc_chars
    assert tool.convergence_threshold == defaults.tool_refinement.convergence_threshold
    assert tool.update_interval == defaults.tool_refinement.update_interval
    assert tool.publish_min_utility == defaults.tool_refinement.publish_min_utility
    assert memory.token_budget == defaults.procedural_memory.token_budget
    assert memory.pool_size == defaults.procedural_memory.pool_size
    assert memory.verifier == "behavioral_replay"
    assert memory.holdout_size == defaults.procedural_memory.holdout_size
    assert memory.evolver_model == defaults.procedural_memory.llm_model
    assert memory.policy_scorer_model == defaults.procedural_memory.skill_logprob_model
    assert args.tool_refinement_doc_chars == defaults.tool_refinement.tool_doc_chars
    assert args.procedural_memory_tokens == defaults.procedural_memory.token_budget
    assert args.procedural_memory_pool_size == defaults.procedural_memory.pool_size
    assert (
        args.tool_refinement_update_interval == defaults.tool_refinement.update_interval
    )
    assert args.procedural_memory_verifier == defaults.procedural_memory.verifier
    assert args.procedural_memory_evolver_model == defaults.procedural_memory.llm_model
    assert (
        args.procedural_memory_policy_scorer_model
        == defaults.procedural_memory.skill_logprob_model
    )
    assert args.agent == defaults.baseline.agent_type
    assert args.max_steps == defaults.baseline.max_steps
    assert args.max_attempts == defaults.baseline.max_attempts

    canonical_budget = parser.parse_args(
        [
            "--evaluate-benchmark",
            "benchmark.yaml",
            "--procedural-memory-token-budget",
            "777",
        ]
    )
    legacy_budget = parser.parse_args(
        [
            "--evaluate-benchmark",
            "benchmark.yaml",
            "--procedural-memory-tokens",
            "888",
        ]
    )
    assert canonical_budget.procedural_memory_tokens == 777
    assert legacy_budget.procedural_memory_tokens == 888
    assert args.judge == defaults.baseline.judge_evaluation
    assert args.judge_provider == defaults.baseline.judge_provider
    assert args.judge_model == defaults.baseline.judge_model
    assert defaults.baseline.benchmark == "benchmark_selected.yaml"
    assert defaults.baseline.training_benchmark == "benchmark_training.yaml"
    assert defaults.baseline.evaluate_benchmark == "benchmark_selected.yaml"
    assert defaults.procedural_memory.selection_epsilon_decay_cases == 100
    assert defaults.procedural_memory.evolution_threshold == 3
    assert defaults.procedural_memory.best_of_n == 2
    assert defaults.procedural_memory.holdout_size == 1
    assert defaults.baseline.max_steps == 50
    assert defaults.baseline.judge_provider == "custom"
    assert defaults.baseline.judge_model == "openai/gpt-oss-120b"


def test_module_config_rejects_missing_sections(tmp_path) -> None:
    path = tmp_path / "modules.yaml"
    path.write_text("tool_refinement: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="procedural_memory"):
        load_module_defaults(path)


def test_module_config_rejects_unsafe_runtime_values(tmp_path) -> None:
    config_path = tmp_path / "modules.yaml"
    config_path.write_text(
        Path("config/modules.yaml")
        .read_text(encoding="utf-8")
        .replace("policy_token_budget_divisor: 2", "policy_token_budget_divisor: 0"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sizes and budgets"):
        load_module_defaults(config_path)
