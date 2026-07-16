"""JSON store for Skill-Pro procedural skills."""

from __future__ import annotations

import hashlib
import json
import fcntl
import re
import shutil
from contextlib import contextmanager
from pathlib import Path

from agent.extensions.config import PROCEDURAL_MEMORY_DIR
from agent.utils.atomic import atomic_write_text

from agent.procedural_memory.models import (
    EvaluationEvidence,
    ProceduralMemoryState,
    utc_now,
)


def public_episode_evidence(evidence: EvaluationEvidence) -> EvaluationEvidence:
    """Return the persistable episode view without hidden answer labels."""
    return evidence.model_copy(update={"root_cause": [], "faulty_devices": []})


def safe_bank_id(bank_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", bank_id).strip("._")
    return cleaned or "default"


class ProceduralMemoryStore:
    def __init__(
        self,
        bank_id: str = "default",
        root: str | Path | None = None,
        state_path: str | Path | None = None,
        read_only: bool = False,
    ) -> None:
        self.bank_id = safe_bank_id(bank_id)
        self.read_only = bool(read_only)
        self._explicit_state_path = Path(state_path) if state_path is not None else None
        self._legacy_state_path: Path | None = None
        if self._explicit_state_path is not None:
            self.state_path = self._explicit_state_path
            self.bank_dir = self.state_path.parent
            self.root = self.bank_dir
            self.lock_path = self.bank_dir / f".{self.state_path.name}.lock"
            legacy_path = self.state_path.parent / self.bank_id / "skills.json"
            if legacy_path != self.state_path:
                self._legacy_state_path = legacy_path
        else:
            self.root = Path(root) if root is not None else PROCEDURAL_MEMORY_DIR
            self.bank_dir = self.root / self.bank_id
            self.state_path = self.bank_dir / "skills.json"
            self.lock_path = self.bank_dir / ".lock"

    def as_read_only(self) -> "ProceduralMemoryStore":
        """Return a read-only view over the exact same persisted state."""

        return ProceduralMemoryStore(
            bank_id=self.bank_id,
            state_path=self.state_path,
            read_only=True,
        )

    def _require_writable(self) -> None:
        if self.read_only:
            raise PermissionError("Procedural Memory store is read-only")

    @contextmanager
    def exclusive(self):
        """Serialize one bank's read-modify-write training cycle."""

        self._require_writable()
        self.bank_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def load(self) -> ProceduralMemoryState:
        source_path = self.state_path
        if (
            not source_path.exists()
            and self._legacy_state_path is not None
            and self._legacy_state_path.exists()
        ):
            source_path = self._legacy_state_path
        if not source_path.exists():
            return ProceduralMemoryState(bank_id=self.bank_id)
        return ProceduralMemoryState.model_validate_json(
            source_path.read_text(encoding="utf-8")
        )

    @property
    def needs_migration(self) -> bool:
        return bool(
            self._legacy_state_path is not None
            and self._legacy_state_path.exists()
            and not self.state_path.exists()
        )

    def save(self, state: ProceduralMemoryState) -> ProceduralMemoryState:
        self._require_writable()
        state.bank_id = self.bank_id
        state.updated_at = utc_now()
        atomic_write_text(self.state_path, state.model_dump_json(indent=2))
        return state

    def clear(self) -> None:
        self._require_writable()
        with self.exclusive():
            if self._explicit_state_path is not None:
                self.state_path.unlink(missing_ok=True)
                if self._legacy_state_path is not None:
                    self._legacy_state_path.unlink(missing_ok=True)
                return
            for path in self.bank_dir.iterdir():
                if path == self.lock_path:
                    continue
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)

    def bank_stats(self) -> dict:
        state = self.load()
        skills = list(state.skills.values())
        gradients = [
            gradient for skill in skills for gradient in skill.semantic_gradients
        ]
        last_decision = state.ppo_decisions[-1] if state.ppo_decisions else None
        completed_evolution = [
            event
            for event in state.evolution_log
            if event.get("action") in {"accepted", "rejected"}
        ]
        parent_attempt_counts = dict(state.evolution_parent_attempt_counts)
        derived_parent_attempts: dict[str, int] = {}
        for event in completed_evolution:
            parent_id = str(event.get("parent") or "")
            if parent_id:
                derived_parent_attempts[parent_id] = (
                    derived_parent_attempts.get(parent_id, 0) + 1
                )
        for parent_id, count in derived_parent_attempts.items():
            parent_attempt_counts.setdefault(parent_id, count)
        candidate_attempts = sum(
            int(event.get("generated_candidate_count") or 0)
            for event in state.evolution_log
        )
        gate_failures: dict[str, int] = {}
        recorded_gate_checks = [
            attempt.get("gate_checks") or {}
            for event in state.evolution_log
            for attempt in event.get("candidate_attempts", [])
        ]
        if not recorded_gate_checks:
            recorded_gate_checks = [
                decision.gate_checks for decision in state.ppo_decisions
            ]
        for checks in recorded_gate_checks:
            for check, passed in checks.items():
                if not passed:
                    gate_failures[check] = gate_failures.get(check, 0) + 1
        return {
            "bank_id": state.bank_id,
            "skills": len(skills),
            "validated_skills": sum(skill.status == "validated" for skill in skills),
            "candidate_skills": sum(skill.status == "candidate" for skill in skills),
            "probationary_skills": sum(
                skill.status == "probationary" for skill in skills
            ),
            "retired_skills": sum(skill.status == "retired" for skill in skills),
            "episodes": len(state.episodes),
            "experiences": len(state.experiences),
            "ppo_decisions": len(state.ppo_decisions),
            "ppo_accepted": sum(decision.accepted for decision in state.ppo_decisions),
            "ppo_rejected": sum(
                not decision.accepted for decision in state.ppo_decisions
            ),
            "last_ppo_j_score": last_decision.j_score if last_decision else None,
            "last_candidate_alignment": (
                last_decision.candidate_alignment if last_decision else None
            ),
            "last_baseline_alignment": (
                last_decision.baseline_alignment if last_decision else None
            ),
            "last_verification_method": (
                last_decision.verification_method if last_decision else None
            ),
            "last_verified_success_count": (
                last_decision.verified_success_count if last_decision else None
            ),
            "iteration": state.iteration,
            "evolution_events": len(state.evolution_log),
            "completed_evolution_attempts": len(completed_evolution),
            "candidate_attempts_generated": candidate_attempts,
            "gate_failure_counts": gate_failures,
            "evolution_parent_attempt_counts": parent_attempt_counts,
            "maintenance_events": len(state.maintenance_log),
            "semantic_gradients": len(gradients),
            "llm_semantic_gradients": sum(
                gradient.gradient_source == "llm" for gradient in gradients
            ),
            "total_skill_frequency": sum(skill.frequency for skill in skills),
        }

    def snapshot_jsonl(self) -> list[str]:
        state = self.load()
        rows = [
            {
                "kind": "snapshot",
                "bank_id": state.bank_id,
                "skills": len(state.skills),
                "episodes": len(state.episodes),
            }
        ]
        rows.extend(
            {"kind": "skill", **skill.model_dump()} for skill in state.skills.values()
        )
        rows.extend(
            {"kind": "episode", **episode.model_dump()} for episode in state.episodes
        )
        rows.extend(
            {"kind": "experience", **experience.model_dump()}
            for experience in state.experiences
        )
        return [json.dumps(row, ensure_ascii=False, default=str) for row in rows]

    def _active_state_path(self) -> Path | None:
        if self.state_path.exists():
            return self.state_path
        if self._legacy_state_path is not None and self._legacy_state_path.exists():
            return self._legacy_state_path
        return None

    def state_hash(self) -> str:
        """Hash the persisted bank bytes for barrier/invariance checks."""

        source = self._active_state_path()
        if source is None:
            return ""
        return hashlib.sha256(source.read_bytes()).hexdigest()

    def snapshot(self, output_path: str | Path) -> Path:
        """Copy the current bank atomically to a standalone frozen artifact."""

        destination = Path(output_path)
        source = self._active_state_path()
        payload = (
            source.read_text(encoding="utf-8")
            if source is not None
            else ProceduralMemoryState(bank_id=self.bank_id).model_dump_json(indent=2)
        )
        atomic_write_text(destination, payload)
        return destination
