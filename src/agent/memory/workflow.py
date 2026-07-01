"""Offline Skill-Pro update hook for closed diagnosis sessions."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from agent.memory.models import EvaluationEvidence, SkillStep
from agent.memory.service import ProceduralMemoryModule, _metric_success
from nika.evaluator.result_log import MESSAGES_FILENAME


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
    steps: list[SkillStep] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("agent") != "diagnosis_agent" or entry.get("event") != "tool_start":
            continue
        tool = entry.get("tool") or {}
        name = str(tool.get("name") or "")
        if not name:
            continue
        steps.append(
            SkillStep(
                order=len(steps) + 1,
                action=f"Call `{name}` to collect diagnostic evidence.",
                tool_name=name,
                arguments_hint=_parse_args(entry.get("input")),
                rationale="Observed in diagnosis trajectory.",
            )
        )
    return steps


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
        metrics=metrics,
        steps=int(metrics.get("steps") or 0),
        tool_calls=int(metrics.get("tool_calls") or 0),
        success=_metric_success(metrics),
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
