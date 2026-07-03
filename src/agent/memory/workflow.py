"""Offline Skill-Pro update hook for closed diagnosis sessions."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from agent.memory.models import EvaluationEvidence, SkillStep
from agent.memory.runtime import (
    INTEGRATED_GUIDANCE_MARKER,
    strip_integrated_learning_guidance,
)
from agent.memory.service import ProceduralMemoryModule, _metric_success
from agent.utils.phases import DIAGNOSIS
from nika.evaluator.result_log import MESSAGES_FILENAME

DIAGNOSIS_AGENT_NAMES = frozenset({DIAGNOSIS, "diagnosis_agent"})
MEMORY_AGENT_NAME = "memory_agent"


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return {"_raw": text}
    return parsed if isinstance(parsed, dict) else {"_value": parsed}


def extract_skill_steps(trace_path: str | Path) -> list[SkillStep]:
    path = Path(trace_path)
    if not path.exists():
        return []
    entries = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    runtime_steps = _extract_runtime_skill_steps(entries)
    if runtime_steps:
        return runtime_steps
    starts: dict[str, dict[str, Any]] = {}
    anonymous_starts: list[tuple[str, dict[str, Any]]] = []
    steps: list[SkillStep] = []
    unnamed_index = 0
    for entry in entries:
        if entry.get("agent") not in DIAGNOSIS_AGENT_NAMES:
            continue
        event = entry.get("event")
        raw_run_id = str(entry.get("run_id") or "")
        run_id = raw_run_id or f"anon-{unnamed_index}"
        if event == "tool_start":
            unnamed_index += 1
            tool = entry.get("tool") or {}
            name = str(tool.get("name") or "")
            if not name:
                continue
            start = {
                "tool_name": name,
                "arguments": _parse_args(entry.get("input")),
            }
            starts[run_id] = start
            if not raw_run_id:
                anonymous_starts.append((run_id, start))
            continue
        if event not in {"tool_end", "tool_error"}:
            continue
        start = starts.get(run_id)
        if start is None and not raw_run_id and anonymous_starts:
            run_id, start = anonymous_starts.pop(0)
        if start is None:
            continue
        status = "success" if event == "tool_end" else "error"
        output = entry.get("output") or entry.get("error") or ""
        name = start["tool_name"]
        steps.append(
            SkillStep(
                order=len(steps) + 1,
                action=(
                    f"Call `{name}` to collect diagnostic evidence and interpret "
                    "whether the observation supports or contradicts the active hypothesis."
                ),
                tool_name=name,
                arguments_hint=start["arguments"],
                observation_summary=_short_text(output),
                status=status,
                rationale="Observed in diagnosis trajectory with tool feedback.",
            )
        )
        starts.pop(run_id, None)
    for start in starts.values():
        name = start["tool_name"]
        steps.append(
            SkillStep(
                order=len(steps) + 1,
                action=(
                    f"Call `{name}` to collect diagnostic evidence; no tool output "
                    "was captured in the trace."
                ),
                tool_name=name,
                arguments_hint=start["arguments"],
                status="unknown",
                rationale="Observed tool start without a matched tool result.",
            )
        )
    return steps


def _normalize_args(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value in (None, "", [], {}):
        return {}
    return {"_value": value}


def _extract_runtime_skill_steps(entries: list[dict[str, Any]]) -> list[SkillStep]:
    steps: list[SkillStep] = []
    for entry in entries:
        if (
            entry.get("agent") != MEMORY_AGENT_NAME
            or entry.get("event") != "skill_transition"
        ):
            continue
        tool_name = str(entry.get("tool") or "")
        if not tool_name:
            continue
        skill_id = str(entry.get("active_skill_id") or "")
        steps.append(
            SkillStep(
                order=len(steps) + 1,
                action=(
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
                rationale="Observed Skill-Pro online runtime transition.",
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
        "followup_added_tokens",
        "total_added_tokens",
        "prompt_injection_count",
        "tool_description_injection_count",
        "followup_guidance_count",
    )
    metrics: dict[str, int] = {}
    for field in fields:
        try:
            metrics[f"memory_{field}"] = int(runtime_snapshot.get(field) or 0)
        except (TypeError, ValueError):
            metrics[f"memory_{field}"] = 0
    return metrics


def _strip_integrated_guidance(value: Any) -> str:
    return strip_integrated_learning_guidance(value)


def _short_text(value: Any, *, limit: int = 900) -> str:
    text = _strip_integrated_guidance(value)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


async def evolve_session_memory(
    *,
    run_meta: dict[str, Any],
    metrics: dict[str, Any],
    session_dir: str | Path,
) -> dict[str, Any]:
    if run_meta.get("memory_mode", "off") != "evolve":
        return {"status": "skipped", "reason": "memory_mode is not evolve"}
    session_path = Path(session_dir)
    bank_id = str(run_meta.get("memory_bank") or "default")
    gt = _load_json(session_path / "ground_truth.json")
    runtime_snapshot = _load_json(session_path / "memory_runtime_session.json")
    metrics_with_runtime = dict(metrics)
    metrics_with_runtime.update(_runtime_overhead_metrics(runtime_snapshot))
    module = ProceduralMemoryModule(
        bank_id=bank_id,
        llm_backend=run_meta.get("llm_backend"),
        model=run_meta.get("model"),
    )
    evidence = EvaluationEvidence(
        session_id=str(run_meta.get("session_id") or session_path.name),
        task_description=str(run_meta.get("task_description") or ""),
        scenario=str(run_meta.get("scenario_name") or ""),
        topology_class=str(run_meta.get("scenario_topo_size") or ""),
        root_cause=list(gt.get("root_cause_name") or []),
        faulty_devices=list(gt.get("faulty_devices") or []),
        metrics=metrics_with_runtime,
        steps=int(metrics_with_runtime.get("steps") or 0),
        tool_calls=int(metrics_with_runtime.get("tool_calls") or 0),
        success=_metric_success(metrics_with_runtime),
    )
    report = module.learn_from_episode(
        evidence=evidence,
        tool_steps=extract_skill_steps(session_path / MESSAGES_FILENAME),
    )
    report.update({"method": "Skill-Pro", "bank_id": bank_id})
    (session_path / "memory_update.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
