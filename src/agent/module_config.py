"""Versioned defaults shared by the Procedural Memory and Tool Refinement modules."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import yaml

from nika.config import _REPO_ROOT


MODULE_CONFIG_PATH = _REPO_ROOT / "config" / "modules.yaml"
ENV_MODULE_CONFIG_PATH = "NIKA_MODULE_CONFIG_PATH"


@dataclass(frozen=True)
class ToolRefinementDefaults:
    llm_backend: str
    llm_model: str
    timeout_seconds: float
    max_retries: int
    tool_doc_chars: int
    convergence_threshold: float
    exploration_similarity_threshold: float
    explorer_reflection_limit: int
    update_interval: int
    min_new_trials: int
    max_tools_per_update: int
    publish_min_utility: float
    guidance_total_token_budget: int
    guidance_min_token_budget: int
    guidance_per_tool_token_budget: int


@dataclass(frozen=True)
class ProceduralMemoryDefaults:
    llm_backend: str
    llm_model: str
    skill_logprob_model: str
    timeout_seconds: float
    max_retries: int
    token_budget: int
    max_skill_age: int
    pool_size: int
    evolution_threshold: int
    best_of_n: int
    ppo_epsilon: float
    selection_epsilon: float
    experience_pool_size: int
    baseline_ema_alpha: float
    selection_epsilon_decay_cases: int
    acceptance_margin: float
    verifier: str
    holdout_size: int
    min_positive_advantage: int
    policy_token_budget_min: int
    policy_token_budget_max: int
    policy_token_budget_divisor: int
    tool_guidance_char_budget: int


@dataclass(frozen=True)
class BaselineDefaults:
    benchmark: str
    learning_benchmark: str
    evaluate_benchmark: str
    agent_type: str
    llm_provider: str
    model: str
    max_steps: int
    max_attempts: int
    judge_evaluation: bool
    judge_provider: str
    judge_model: str


@dataclass(frozen=True)
class ModuleDefaults:
    tool_refinement: ToolRefinementDefaults
    procedural_memory: ProceduralMemoryDefaults
    baseline: BaselineDefaults


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"module config section '{name}' must be a mapping")
    return value


def _int(section: dict[str, Any], key: str) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"module config '{key}' must be an integer")
    return value


def _float(section: dict[str, Any], key: str) -> float:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"module config '{key}' must be a number")
    return float(value)


def _str(section: dict[str, Any], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"module config '{key}' must be a non-empty string")
    return value.strip()


def _bool(section: dict[str, Any], key: str) -> bool:
    value = section.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"module config '{key}' must be a boolean")
    return value


def module_config_path() -> Path:
    raw = os.getenv(ENV_MODULE_CONFIG_PATH, "").strip()
    if not raw:
        return MODULE_CONFIG_PATH
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def load_module_defaults(path: str | Path | None = None) -> ModuleDefaults:
    """Load module defaults from a versioned YAML file with basic validation."""
    config_path = Path(path) if path is not None else module_config_path()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read module config: {config_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("module config must be a mapping")
    tool = _section(raw, "tool_refinement")
    memory = _section(raw, "procedural_memory")
    baseline = _section(raw, "baseline")
    defaults = ModuleDefaults(
        tool_refinement=ToolRefinementDefaults(
            llm_backend=_str(tool, "llm_backend"),
            llm_model=_str(tool, "llm_model"),
            timeout_seconds=_float(tool, "timeout_seconds"),
            max_retries=_int(tool, "max_retries"),
            tool_doc_chars=_int(tool, "tool_doc_chars"),
            convergence_threshold=_float(tool, "convergence_threshold"),
            exploration_similarity_threshold=_float(
                tool, "exploration_similarity_threshold"
            ),
            explorer_reflection_limit=_int(tool, "explorer_reflection_limit"),
            update_interval=_int(tool, "update_interval"),
            min_new_trials=_int(tool, "min_new_trials"),
            max_tools_per_update=_int(tool, "max_tools_per_update"),
            publish_min_utility=_float(tool, "publish_min_utility"),
            guidance_total_token_budget=_int(tool, "guidance_total_token_budget"),
            guidance_min_token_budget=_int(tool, "guidance_min_token_budget"),
            guidance_per_tool_token_budget=_int(tool, "guidance_per_tool_token_budget"),
        ),
        procedural_memory=ProceduralMemoryDefaults(
            llm_backend=_str(memory, "llm_backend"),
            llm_model=_str(memory, "llm_model"),
            skill_logprob_model=_str(memory, "skill_logprob_model"),
            timeout_seconds=_float(memory, "timeout_seconds"),
            max_retries=_int(memory, "max_retries"),
            token_budget=_int(memory, "token_budget"),
            max_skill_age=_int(memory, "max_skill_age"),
            pool_size=_int(memory, "pool_size"),
            evolution_threshold=_int(memory, "evolution_threshold"),
            best_of_n=_int(memory, "best_of_n"),
            ppo_epsilon=_float(memory, "ppo_epsilon"),
            selection_epsilon=_float(memory, "selection_epsilon"),
            experience_pool_size=_int(memory, "experience_pool_size"),
            baseline_ema_alpha=_float(memory, "baseline_ema_alpha"),
            selection_epsilon_decay_cases=_int(memory, "selection_epsilon_decay_cases"),
            acceptance_margin=_float(memory, "acceptance_margin"),
            verifier=_str(memory, "verifier"),
            holdout_size=_int(memory, "holdout_size"),
            min_positive_advantage=_int(memory, "min_positive_advantage"),
            policy_token_budget_min=_int(memory, "policy_token_budget_min"),
            policy_token_budget_max=_int(memory, "policy_token_budget_max"),
            policy_token_budget_divisor=_int(memory, "policy_token_budget_divisor"),
            tool_guidance_char_budget=_int(memory, "tool_guidance_char_budget"),
        ),
        baseline=BaselineDefaults(
            benchmark=_str(baseline, "benchmark"),
            learning_benchmark=_str(baseline, "learning_benchmark"),
            evaluate_benchmark=_str(baseline, "evaluate_benchmark"),
            agent_type=_str(baseline, "agent_type"),
            llm_provider=_str(baseline, "llm_provider"),
            model=_str(baseline, "model"),
            max_steps=_int(baseline, "max_steps"),
            max_attempts=_int(baseline, "max_attempts"),
            judge_evaluation=_bool(baseline, "judge_evaluation"),
            judge_provider=_str(baseline, "judge_provider"),
            judge_model=_str(baseline, "judge_model"),
        ),
    )
    if defaults.baseline.agent_type not in {"react", "plan-execute", "reflexion"}:
        raise ValueError(
            "baseline agent_type must be react, plan-execute, or reflexion"
        )
    if defaults.baseline.max_steps < 1 or defaults.baseline.max_attempts < 1:
        raise ValueError("baseline steps and attempts must be positive")
    tool_defaults = defaults.tool_refinement
    if tool_defaults.timeout_seconds <= 0 or tool_defaults.max_retries < 0:
        raise ValueError("tool refinement timeout/retries are invalid")
    if tool_defaults.tool_doc_chars < 100:
        raise ValueError("tool refinement tool_doc_chars must be at least 100")
    if not 0 <= tool_defaults.convergence_threshold <= 1:
        raise ValueError("tool refinement convergence_threshold must be in [0, 1]")
    if not 0 <= tool_defaults.exploration_similarity_threshold <= 1:
        raise ValueError("tool refinement similarity threshold must be in [0, 1]")
    if tool_defaults.explorer_reflection_limit < 0:
        raise ValueError("tool refinement reflection limit must not be negative")
    if (
        min(
            tool_defaults.update_interval,
            tool_defaults.min_new_trials,
            tool_defaults.max_tools_per_update,
        )
        < 1
    ):
        raise ValueError("tool refinement update controls must be positive")
    if not 0 <= tool_defaults.publish_min_utility <= 1:
        raise ValueError("tool refinement publication utility must be in [0, 1]")
    if (
        min(
            tool_defaults.guidance_total_token_budget,
            tool_defaults.guidance_min_token_budget,
            tool_defaults.guidance_per_tool_token_budget,
        )
        < 1
    ):
        raise ValueError("tool refinement guidance budgets must be positive")
    if (
        tool_defaults.guidance_min_token_budget
        > tool_defaults.guidance_total_token_budget
    ):
        raise ValueError("tool refinement guidance minimum cannot exceed total budget")
    memory_defaults = defaults.procedural_memory
    if memory_defaults.timeout_seconds <= 0 or memory_defaults.max_retries < 0:
        raise ValueError("procedural memory timeout/retries are invalid")
    positive_memory_values = (
        memory_defaults.token_budget,
        memory_defaults.max_skill_age,
        memory_defaults.pool_size,
        memory_defaults.evolution_threshold,
        memory_defaults.best_of_n,
        memory_defaults.experience_pool_size,
        memory_defaults.selection_epsilon_decay_cases,
        memory_defaults.holdout_size,
        memory_defaults.policy_token_budget_min,
        memory_defaults.policy_token_budget_max,
        memory_defaults.policy_token_budget_divisor,
        memory_defaults.tool_guidance_char_budget,
    )
    if min(positive_memory_values) < 1:
        raise ValueError("procedural memory sizes and budgets must be positive")
    if not 0 <= memory_defaults.ppo_epsilon <= 1:
        raise ValueError("procedural memory ppo_epsilon must be in [0, 1]")
    if not 0 <= memory_defaults.selection_epsilon <= 1:
        raise ValueError("procedural memory selection_epsilon must be in [0, 1]")
    if not 0 < memory_defaults.baseline_ema_alpha <= 1:
        raise ValueError("procedural memory baseline_ema_alpha must be in (0, 1]")
    if memory_defaults.acceptance_margin < 0:
        raise ValueError("procedural memory acceptance_margin must not be negative")
    if memory_defaults.verifier not in {
        "behavioral_replay",
        "structured_replay",
        "policy_logprob",
    }:
        raise ValueError("procedural memory verifier is invalid")
    if memory_defaults.min_positive_advantage < 0:
        raise ValueError("procedural memory positive-advantage support is invalid")
    if memory_defaults.min_positive_advantage > memory_defaults.holdout_size:
        raise ValueError(
            "procedural memory positive-advantage support exceeds holdout size"
        )
    if memory_defaults.evolution_threshold < 2:
        raise ValueError("procedural memory evolution batch size must be at least 2")
    if memory_defaults.holdout_size >= memory_defaults.evolution_threshold:
        raise ValueError(
            "procedural memory holdout must leave at least one generation trajectory"
        )
    if (
        memory_defaults.policy_token_budget_min
        > memory_defaults.policy_token_budget_max
    ):
        raise ValueError("procedural memory policy token minimum cannot exceed maximum")
    return defaults


@lru_cache(maxsize=1)
def module_defaults() -> ModuleDefaults:
    return load_module_defaults()
