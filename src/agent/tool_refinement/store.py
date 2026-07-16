"""JSON persistence for DRAFT tool documentation state."""

from __future__ import annotations

import hashlib
import json
import fcntl
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

from agent.extensions.config import TOOL_REFINEMENT_DIR
from agent.utils.atomic import atomic_write_text

from agent.tool_refinement.models import (
    DraftToolState,
    ToolDocumentation,
    ToolTrial,
    utc_now,
)


def safe_library_id(library_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", library_id).strip("._")
    return cleaned or "default"


class ToolRefinementStore:
    """Persistent DRAFT library under ``runtime/tool_refinement/<library_id>``."""

    def __init__(
        self,
        library_id: str = "default",
        root: str | Path | None = None,
        state_path: str | Path | None = None,
        read_only: bool = False,
    ) -> None:
        self.library_id = safe_library_id(library_id)
        self.read_only = bool(read_only)
        if state_path is not None:
            self.state_path = Path(state_path)
            self.library_dir = self.state_path.parent
            self.root = self.library_dir
            self.lock_path = self.library_dir / f".{self.state_path.name}.lock"
        else:
            self.root = Path(root) if root is not None else TOOL_REFINEMENT_DIR
            self.library_dir = self.root / self.library_id
            self.state_path = self.library_dir / "state.json"
            self.lock_path = self.library_dir / ".lock"

    def as_read_only(self) -> "ToolRefinementStore":
        """Return a read-only view over the exact same persisted state."""

        return ToolRefinementStore(
            self.library_id,
            state_path=self.state_path,
            read_only=True,
        )

    def _require_writable(self) -> None:
        if self.read_only:
            raise PermissionError("Tool Refinement store is read-only")

    @contextmanager
    def exclusive(self):
        """Serialize read-modify-write training cycles for one library."""

        self._require_writable()
        self.library_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def load(self) -> DraftToolState:
        if not self.state_path.exists():
            return DraftToolState(library_id=self.library_id)
        state = DraftToolState.model_validate_json(
            self.state_path.read_text(encoding="utf-8")
        )
        trials_by_id = {trial.trial_id: trial for trial in state.trials}
        normalized = []
        seen_trial_ids: set[str] = set()
        changed = False
        for exploration in state.explorations:
            if exploration.status == "invalidated":
                changed = True
                continue
            trial_id = exploration.trial_id
            if not trial_id:
                matches = [
                    trial
                    for trial in state.trials
                    if trial.session_id == exploration.session_id
                    and trial.tool_name == exploration.tool_name
                    and trial.arguments == exploration.parameters
                    and (
                        trial.output_summary
                        if trial.status == "success"
                        else trial.error_summary
                    )
                    == exploration.observation
                ]
                if len(matches) == 1:
                    trial_id = matches[0].trial_id
                    exploration.trial_id = trial_id
                    exploration.status = matches[0].status
                    changed = True
            if (
                not trial_id
                or trial_id not in trials_by_id
                or trial_id in seen_trial_ids
            ):
                changed = True
                continue
            normalized.append(exploration)
            seen_trial_ids.add(trial_id)
        if changed:
            state.explorations = normalized
        return state

    def save(self, state: DraftToolState) -> DraftToolState:
        self._require_writable()
        state.library_id = self.library_id
        state.updated_at = utc_now()
        atomic_write_text(self.state_path, state.model_dump_json(indent=2))
        return state

    def clear(self) -> None:
        self._require_writable()
        with self.exclusive():
            for path in self.library_dir.iterdir():
                if path == self.lock_path:
                    continue
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)

    def upsert_document(self, doc: ToolDocumentation) -> ToolDocumentation:
        self._require_writable()
        with self.exclusive():
            state = self.load()
            state.documents[doc.name] = doc
            self.save(state)
        return doc

    def get_document(self, tool_name: str) -> ToolDocumentation | None:
        return self.load().documents.get(tool_name)

    def record_trials(self, trials: Iterable[ToolTrial]) -> int:
        self._require_writable()
        incoming = list(trials)
        if not incoming:
            return 0
        with self.exclusive():
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

    def stats(self) -> dict:
        state = self.load()
        total = len(state.trials)
        successes = sum(trial.status == "success" for trial in state.trials)
        errors = sum(trial.status == "error" for trial in state.trials)
        frozen = sum(doc.frozen for doc in state.documents.values())
        mastered = sum(stat.mastered for stat in state.tool_stats.values())
        documented_rates = [
            stat.documented_path_rate for stat in state.tool_stats.values()
        ]
        success_rates = [stat.success_path_rate for stat in state.tool_stats.values()]
        return {
            "library_id": state.library_id,
            "documents": len(state.documents),
            "library_usage_description": state.library_usage_description,
            "trials": total,
            "successful_trials": successes,
            "error_trials": errors,
            "explorations": len(state.explorations),
            "analyzer_suggestions": len(state.analyzer_suggestions),
            "gaps": len(state.gaps),
            "revisions": len(state.revisions),
            "llm_attempts": sum(
                revision.metrics.get("llm_attempted") == 1.0
                for revision in state.revisions
            ),
            "llm_failures": sum(
                bool(revision.llm_error) for revision in state.revisions
            ),
            "llm_rewrites": sum(
                revision.metrics.get("llm_rewrite") == 1.0
                for revision in state.revisions
            ),
            "frozen_documents": frozen,
            "mastered_tools": mastered,
            "avg_documented_path_rate": (
                sum(documented_rates) / len(documented_rates)
                if documented_rates
                else 0.0
            ),
            "avg_success_path_rate": (
                sum(success_rates) / len(success_rates) if success_rates else 0.0
            ),
            "tool_stats": {
                name: stat.model_dump(mode="json")
                for name, stat in sorted(state.tool_stats.items())
            },
        }

    def as_json(self) -> str:
        return json.dumps(self.load().model_dump(), ensure_ascii=False, indent=2)

    def state_hash(self) -> str:
        """Hash the persisted state bytes for barrier/invariance checks."""

        if not self.state_path.exists():
            return ""
        return hashlib.sha256(self.state_path.read_bytes()).hexdigest()

    def snapshot(self, output_path: str | Path) -> Path:
        """Copy the current state atomically to a standalone frozen artifact."""

        destination = Path(output_path)
        payload = (
            self.state_path.read_text(encoding="utf-8")
            if self.state_path.exists()
            else DraftToolState(library_id=self.library_id).model_dump_json(indent=2)
        )
        atomic_write_text(destination, payload)
        return destination
