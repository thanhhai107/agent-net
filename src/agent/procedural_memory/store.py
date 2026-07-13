"""JSON store for Skill-Pro procedural skills."""

from __future__ import annotations

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
    PPOGateDecision,
    ProceduralSkill,
    SkillExperience,
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
        self, bank_id: str = "default", root: str | Path | None = None
    ) -> None:
        self.bank_id = safe_bank_id(bank_id)
        self.root = Path(root) if root is not None else PROCEDURAL_MEMORY_DIR
        self.bank_dir = self.root / self.bank_id
        self.state_path = self.bank_dir / "skills.json"
        self.lock_path = self.bank_dir / ".lock"

    @contextmanager
    def exclusive(self):
        """Serialize one bank's read-modify-write learning cycle."""

        self.bank_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def load(self) -> ProceduralMemoryState:
        if not self.state_path.exists():
            return ProceduralMemoryState(bank_id=self.bank_id)
        return ProceduralMemoryState.model_validate_json(
            self.state_path.read_text(encoding="utf-8")
        )

    def save(self, state: ProceduralMemoryState) -> ProceduralMemoryState:
        state.bank_id = self.bank_id
        state.updated_at = utc_now()
        atomic_write_text(self.state_path, state.model_dump_json(indent=2))
        return state

    def clear(self) -> None:
        with self.exclusive():
            for path in self.bank_dir.iterdir():
                if path == self.lock_path:
                    continue
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)

    def upsert_skill(self, skill: ProceduralSkill) -> ProceduralSkill:
        with self.exclusive():
            state = self.load()
            skill.updated_at = utc_now()
            state.skills[skill.skill_id] = skill
            self.save(state)
        return skill

    def record_episode(self, evidence: EvaluationEvidence) -> None:
        with self.exclusive():
            state = self.load()
            if not any(
                item.session_id == evidence.session_id for item in state.episodes
            ):
                state.episodes.append(public_episode_evidence(evidence))
                self.save(state)

    def record_experience(
        self,
        experience: SkillExperience,
        *,
        max_experiences: int = 1000,
        golden_size: int = 20,
    ) -> None:
        with self.exclusive():
            state = self.load()
            if not any(
                item.experience_id == experience.experience_id
                for item in state.experiences
            ):
                state.experiences.append(experience)
                state.experiences = state.experiences[-max_experiences:]
            if experience.transitions:
                gold = {item.experience_id: item for item in state.golden_experiences}
                gold[experience.experience_id] = experience
                state.golden_experiences = sorted(
                    gold.values(),
                    key=lambda item: item.reward,
                    reverse=True,
                )[:golden_size]
            self.save(state)

    def record_decision(self, decision: PPOGateDecision) -> None:
        with self.exclusive():
            state = self.load()
            state.ppo_decisions.append(decision)
            self.save(state)

    def bank_stats(self, bank_id: str | None = None) -> dict:
        state = self.load()
        skills = list(state.skills.values())
        gradients = [
            gradient for skill in skills for gradient in skill.semantic_gradients
        ]
        last_decision = state.ppo_decisions[-1] if state.ppo_decisions else None
        return {
            "bank_id": state.bank_id,
            "skills": len(skills),
            "validated_skills": sum(skill.status == "validated" for skill in skills),
            "candidate_skills": sum(skill.status == "candidate" for skill in skills),
            "retired_skills": sum(skill.status == "retired" for skill in skills),
            "episodes": len(state.episodes),
            "experiences": len(state.experiences),
            "golden_experiences": len(state.golden_experiences),
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
            "maintenance_events": len(state.maintenance_log),
            "semantic_gradients": len(gradients),
            "llm_semantic_gradients": sum(
                gradient.gradient_source == "llm" for gradient in gradients
            ),
            "semantic_gradient_llm_failures": sum(
                bool(gradient.llm_error) for gradient in gradients
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
        rows.extend(
            {"kind": "golden_experience", **experience.model_dump()}
            for experience in state.golden_experiences
        )
        return [json.dumps(row, ensure_ascii=False, default=str) for row in rows]
