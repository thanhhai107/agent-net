"""Offline Skill-Pro update hook for closed diagnosis sessions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.module_config import module_defaults
from agent.procedural_memory.models import EvaluationEvidence, SkillStep
from agent.procedural_memory.runtime import (
    strip_integrated_learning_guidance,
)
from agent.procedural_memory.service import ProceduralMemoryModule, _metric_success
from nika.evaluator.result_log import MESSAGES_FILENAME

PROCEDURAL_MEMORY_AGENT_NAME = "procedural_memory_agent"


def extract_skill_steps(trace_path: str | Path) -> list[SkillStep]:
    path = Path(trace_path)
    if not path.exists():
        return []
    entries = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return _extract_runtime_skill_steps(entries)


def _normalize_args(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value in (None, "", [], {}):
        return {}
    return {"_value": value}


def _extract_runtime_skill_steps(entries: list[dict[str, Any]]) -> list[SkillStep]:
    steps: list[SkillStep] = []
    for entry in entries:
        if entry.get("agent") != PROCEDURAL_MEMORY_AGENT_NAME or entry.get(
            "event"
        ) not in {"skill_transition", "skill_terminal_transition"}:
            continue
        terminal = entry.get("event") == "skill_terminal_transition"
        tool_name = str(entry.get("tool") or "")
        action = str(entry.get("action") or "") if terminal else ""
        if not tool_name and not action:
            continue
        skill_id = str(entry.get("active_skill_id") or "")
        steps.append(
            SkillStep(
                order=len(steps) + 1,
                action=action
                or (
                    f"Use active Skill-Pro option `{skill_id or 'none'}` while "
                    f"calling `{tool_name}` and interpreting its observation."
                ),
                skill_id=skill_id,
                tool_name=tool_name,
                arguments_hint=_normalize_args(entry.get("tool_input")),
                observation_summary=_short_text(entry.get("observation_summary")),
                status=str(entry.get("status") or "unknown")
                if entry.get("status") in {"success", "error", "unknown"}
                else "unknown",
                rationale=(
                    "Observed terminal diagnosis action."
                    if terminal
                    else "Observed Skill-Pro online runtime transition."
                ),
                # This is the canonical replay context, not a byte-equivalent
                # serialization of provider-specific chat/tool payloads.
                policy_state=str(entry.get("policy_state") or ""),
                policy_context=str(entry.get("policy_context") or ""),
                policy_token_budget=int(entry.get("policy_token_budget") or 0),
                selection_probability=float(
                    entry.get("selection_probability") or 0.0
                ),
                activation_id=str(entry.get("activation_id") or ""),
            )
        )
    return steps


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime_overhead_metrics(runtime_snapshot: dict[str, Any]) -> dict[str, int]:
    fields = (
        "prompt_added_tokens",
        "tool_description_added_tokens",
        "total_added_tokens",
        "prompt_injection_count",
        "tool_description_injection_count",
    )
    metrics: dict[str, int] = {}
    for field in fields:
        try:
            metrics[f"procedural_memory_{field}"] = int(
                runtime_snapshot.get(field) or 0
            )
        except (TypeError, ValueError):
            metrics[f"procedural_memory_{field}"] = 0
    return metrics


def _int_meta(run_meta: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(run_meta.get(key) if run_meta.get(key) is not None else default)
    except (TypeError, ValueError):
        return default


def _float_meta(run_meta: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(run_meta.get(key) if run_meta.get(key) is not None else default)
    except (TypeError, ValueError):
        return default


def _strip_integrated_guidance(value: Any) -> str:
    return strip_integrated_learning_guidance(value)


def _short_text(value: Any, *, limit: int = 900) -> str:
    text = _strip_integrated_guidance(value)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


async def update_procedural_memory_from_session(
    *,
    run_meta: dict[str, Any],
    metrics: dict[str, Any],
    session_dir: str | Path,
) -> dict[str, Any]:
    defaults = module_defaults().procedural_memory
    if run_meta.get("procedural_memory_mode", "off") != "evolve":
        return {"status": "skipped", "reason": "procedural_memory_mode is not evolve"}
    session_path = Path(session_dir)
    bank_id = str(run_meta.get("procedural_memory_bank") or "default")
    gt = _load_json(session_path / "ground_truth.json")
    runtime_snapshot = _load_json(
        session_path / "procedural_memory_runtime_session.json"
    )
    metrics_with_runtime = dict(metrics)
    metrics_with_runtime.update(_runtime_overhead_metrics(runtime_snapshot))
    module = ProceduralMemoryModule(
        bank_id=bank_id,
        llm_backend=run_meta.get("llm_backend"),
        model=run_meta.get("model"),
        pool_size=_int_meta(
            run_meta, "procedural_memory_pool_size", defaults.pool_size
        ),
        evolution_threshold=_int_meta(
            run_meta, "procedural_memory_update_threshold", defaults.evolution_threshold
        ),
        best_of_n=_int_meta(
            run_meta, "procedural_memory_best_of_n", defaults.best_of_n
        ),
        ppo_epsilon=_float_meta(
            run_meta, "procedural_memory_ppo_epsilon", defaults.ppo_epsilon
        ),
        experience_pool_size=_int_meta(
            run_meta,
            "procedural_memory_experience_pool_size",
            defaults.experience_pool_size,
        ),
        baseline_ema_alpha=_float_meta(
            run_meta,
            "procedural_memory_baseline_ema_alpha",
            defaults.baseline_ema_alpha,
        ),
        selection_epsilon_decay_cases=_int_meta(
            run_meta,
            "procedural_memory_selection_epsilon_decay_cases",
            defaults.selection_epsilon_decay_cases,
        ),
        acceptance_margin=_float_meta(
            run_meta, "procedural_memory_acceptance_margin", defaults.acceptance_margin
        ),
        verifier=str(run_meta.get("procedural_memory_verifier") or defaults.verifier),
        holdout_size=_int_meta(
            run_meta, "procedural_memory_holdout_size", defaults.holdout_size
        ),
        min_positive_advantage=_int_meta(
            run_meta,
            "procedural_memory_min_positive_advantage",
            defaults.min_positive_advantage,
        ),
        evolver_model=str(
            run_meta.get("procedural_memory_evolver_model") or defaults.llm_model
        ),
        policy_scorer_model=str(
            run_meta.get("procedural_memory_policy_scorer_model")
            or defaults.skill_logprob_model
        ),
    )
    evidence = EvaluationEvidence(
        session_id=str(run_meta.get("session_id") or session_path.name),
        task_description=str(run_meta.get("task_description") or ""),
        scenario=str(run_meta.get("scenario_name") or ""),
        topology_class=str(run_meta.get("scenario_topo_size") or ""),
        root_cause=list(gt.get("root_cause_name") or []),
        faulty_devices=list(gt.get("faulty_devices") or []),
        ground_truth_is_anomaly=(
            bool(gt.get("is_anomaly")) if "is_anomaly" in gt else None
        ),
        metrics=metrics_with_runtime,
        steps=int(metrics_with_runtime.get("steps") or 0),
        tool_calls=int(metrics_with_runtime.get("tool_calls") or 0),
        success=_metric_success(
            metrics_with_runtime,
            bool(gt.get("is_anomaly")) if "is_anomaly" in gt else None,
        ),
    )
    report = module.learn_from_episode(
        evidence=evidence,
        tool_steps=extract_skill_steps(session_path / MESSAGES_FILENAME),
    )
    report.update(
        {
            "method": "Skill-Pro",
            "bank_id": bank_id,
            "runtime_controller": {
                key: int(runtime_snapshot.get(key) or 0)
                for key in (
                    "selector_calls",
                    "selector_errors",
                    "selector_none",
                    "termination_calls",
                    "termination_errors",
                )
            },
            "procedural_memory_config": {
                "token_budget": _int_meta(
                    run_meta, "procedural_memory_token_budget", defaults.token_budget
                ),
                "selection_policy": "llm_direct_epsilon_greedy",
                "selection_epsilon": _float_meta(
                    run_meta,
                    "procedural_memory_selection_epsilon",
                    defaults.selection_epsilon,
                ),
                "meta_controller": "llm_with_runtime_guards",
                "max_skill_age": _int_meta(
                    run_meta,
                    "procedural_memory_max_skill_age",
                    defaults.max_skill_age,
                ),
                "pool_size": module.pool_size,
                "evolution_threshold": module.evolution_threshold,
                "best_of_n": module.best_of_n,
                "ppo_epsilon": module.ppo_epsilon,
                "experience_pool_size": module.experience_pool_size,
                "baseline_ema_alpha": module.baseline_ema_alpha,
                "selection_epsilon_decay_cases": (module.selection_epsilon_decay_cases),
                "acceptance_margin": module.acceptance_margin,
                "verifier": module.verifier,
                "verification_role": "offline_admissibility_prescreen",
                "publication_policy": "online_probation_lcb",
                "holdout_size": module.holdout_size,
                "min_positive_advantage": module.min_positive_advantage,
                "evolver_model": module._selected_evolver_model(),
                "policy_scorer_model": module._selected_policy_scorer_model(),
            },
        }
    )
    (session_path / "procedural_memory_update.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
