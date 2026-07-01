"""JSON persistence for DRAFT tool documentation state."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Iterable

from nika.config import TOOL_EVOLUTION_DIR

from agent.tool_evolution.models import (
    ComprehensionGap,
    DocumentationRevision,
    DraftToolState,
    ToolDocumentation,
    ToolTrial,
    utc_now,
)


def safe_library_id(library_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", library_id).strip("._")
    return cleaned or "default"


class ToolEvolutionStore:
    """Persistent DRAFT library under ``runtime/tool_evolution/<library_id>``."""

    def __init__(self, library_id: str = "default", root: str | Path | None = None) -> None:
        self.library_id = safe_library_id(library_id)
        self.root = Path(root) if root is not None else TOOL_EVOLUTION_DIR
        self.library_dir = self.root / self.library_id
        self.state_path = self.library_dir / "state.json"

    def load(self) -> DraftToolState:
        if not self.state_path.exists():
            return DraftToolState(library_id=self.library_id)
        return DraftToolState.model_validate_json(
            self.state_path.read_text(encoding="utf-8")
        )

    def save(self, state: DraftToolState) -> DraftToolState:
        state.library_id = self.library_id
        state.updated_at = utc_now()
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            state.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return state

    def clear(self) -> None:
        if self.library_dir.exists():
            shutil.rmtree(self.library_dir)

    def upsert_document(self, doc: ToolDocumentation) -> ToolDocumentation:
        state = self.load()
        state.documents[doc.name] = doc
        self.save(state)
        return doc

    def get_document(self, tool_name: str) -> ToolDocumentation | None:
        return self.load().documents.get(tool_name)

    def record_trials(self, trials: Iterable[ToolTrial]) -> int:
        incoming = list(trials)
        if not incoming:
            return 0
        state = self.load()
        seen = {trial.trial_id for trial in state.trials}
        added = 0
        for trial in incoming:
            if trial.trial_id in seen:
                continue
            state.trials.append(trial)
            seen.add(trial.trial_id)
            added += 1
        if added:
            self.save(state)
        return added

    def record_gap(self, gap: ComprehensionGap) -> None:
        state = self.load()
        if not any(item.gap_id == gap.gap_id for item in state.gaps):
            state.gaps.append(gap)
            self.save(state)

    def record_revision(self, revision: DocumentationRevision) -> None:
        state = self.load()
        if not any(item.revision_id == revision.revision_id for item in state.revisions):
            state.revisions.append(revision)
            self.save(state)

    def stats(self) -> dict:
        state = self.load()
        total = len(state.trials)
        successes = sum(trial.status == "success" for trial in state.trials)
        errors = sum(trial.status == "error" for trial in state.trials)
        frozen = sum(doc.frozen for doc in state.documents.values())
        return {
            "library_id": state.library_id,
            "documents": len(state.documents),
            "trials": total,
            "successful_trials": successes,
            "error_trials": errors,
            "gaps": len(state.gaps),
            "revisions": len(state.revisions),
            "frozen_documents": frozen,
        }

    def as_json(self) -> str:
        return json.dumps(self.load().model_dump(), ensure_ascii=False, indent=2)
