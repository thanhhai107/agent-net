"""Post-evaluation lifecycle for online memory evolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.memory.models import EvaluationEvidence
from agent.memory.service import ProceduralMemoryModule


async def evolve_session_memory(
    *,
    run_meta: dict[str, Any],
    metrics: dict[str, Any],
    session_dir: str | Path,
) -> dict[str, Any]:
    """Extract and persist memory after numeric evaluation is available."""
    if run_meta.get("memory_mode", "off") != "evolve":
        return {"status": "skipped", "reason": "memory mode is not evolve"}

    bank_id = str(run_meta.get("memory_bank") or "default")
    session_id = str(run_meta["session_id"])
    module = ProceduralMemoryModule(
        bank_id=bank_id,
        llm_backend=str(run_meta.get("llm_backend") or "openai"),
        model=str(run_meta.get("model") or "gpt-5-mini"),
    )
    if module.store.episode_is_evaluated(bank_id, session_id):
        return {"status": "skipped", "reason": "episode already evolved"}

    session_path = Path(session_dir)
    trace = module.compact_trace(session_path / "messages.jsonl")
    candidates = await module.extract(
        task_description=str(run_meta.get("task_description") or ""),
        trace=trace,
        scenario=str(run_meta.get("scenario_name") or ""),
        topology_class=str(run_meta.get("scenario_topo_size") or ""),
    )
    evidence = EvaluationEvidence(
        detection_score=float(metrics.get("detection_score", -1.0)),
        localization_f1=float(metrics.get("localization_f1", -1.0)),
        rca_f1=float(metrics.get("rca_f1", -1.0)),
    )
    gated = module.validate(candidates, evidence)
    memories = await module.consolidate(
        source_session_id=session_id,
        validated=gated,
        successful_episode=evidence.fully_successful,
    )
    snapshot_path = module.snapshot(
        session_id=session_id,
        output_path=session_path / "memory_snapshot.jsonl",
    )
    module.store.record_episode_evaluation(
        bank_id,
        session_id,
        metrics,
        str(snapshot_path),
    )
    report = {
        "status": "completed",
        "bank_id": bank_id,
        "candidate_count": len(candidates),
        "accepted_count": len(gated),
        "memory_ids": [memory.memory_id for memory in memories],
        "fully_successful": evidence.fully_successful,
        "snapshot_path": str(snapshot_path),
    }
    (session_path / "memory_update.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
