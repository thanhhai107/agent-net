"""JSON store for Skill-Pro procedural skills."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from nika.config import MEMORY_DIR

from agent.memory.models import (
    EvaluationEvidence,
    PPOGateDecision,
    ProceduralSkill,
    SkillExperience,
    SkillMemoryState,
    utc_now,
)


def public_episode_evidence(evidence: EvaluationEvidence) -> EvaluationEvidence:
    """Return the persistable episode view without hidden answer labels."""
    return evidence.model_copy(update={"root_cause": [], "faulty_devices": []})


def safe_bank_id(bank_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", bank_id).strip("._")
    return cleaned or "default"


class SkillMemoryStore:
    def __init__(self, bank_id: str = "default", root: str | Path | None = None) -> None:
        self.bank_id = safe_bank_id(bank_id)
        self.root = Path(root) if root is not None else MEMORY_DIR
        self.bank_dir = self.root / self.bank_id
        self.state_path = self.bank_dir / "skills.json"

    def load(self) -> SkillMemoryState:
        if not self.state_path.exists():
            return SkillMemoryState(bank_id=self.bank_id)
        return SkillMemoryState.model_validate_json(
            self.state_path.read_text(encoding="utf-8")
        )

    def save(self, state: SkillMemoryState) -> SkillMemoryState:
        state.bank_id = self.bank_id
        state.updated_at = utc_now()
        self.bank_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        return state

    def clear(self) -> None:
        if self.bank_dir.exists():
            shutil.rmtree(self.bank_dir)

    def upsert_skill(self, skill: ProceduralSkill) -> ProceduralSkill:
        state = self.load()
        skill.updated_at = utc_now()
        state.skills[skill.skill_id] = skill
        self.save(state)
        return skill

    def record_episode(self, evidence: EvaluationEvidence) -> None:
        state = self.load()
        if not any(item.session_id == evidence.session_id for item in state.episodes):
            state.episodes.append(public_episode_evidence(evidence))
            self.save(state)

    def record_experience(
        self,
        experience: SkillExperience,
        *,
        max_experiences: int = 1000,
        golden_size: int = 20,
    ) -> None:
        state = self.load()
        if not any(item.experience_id == experience.experience_id for item in state.experiences):
            state.experiences.append(experience)
            state.experiences = state.experiences[-max_experiences:]
        if experience.transitions and experience.success:
            gold = {
                item.experience_id: item
                for item in state.golden_experiences
            }
            gold[experience.experience_id] = experience
            state.golden_experiences = sorted(
                gold.values(),
                key=lambda item: item.reward,
                reverse=True,
            )[:golden_size]
        self.save(state)

    def record_decision(self, decision: PPOGateDecision) -> None:
        state = self.load()
        state.ppo_decisions.append(decision)
        self.save(state)

    def bank_stats(self, bank_id: str | None = None) -> dict:
        state = self.load()
        skills = list(state.skills.values())
        gradients = [
            gradient
            for skill in skills
            for gradient in skill.semantic_gradients
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
            "ppo_rejected": sum(not decision.accepted for decision in state.ppo_decisions),
            "last_ppo_j_score": last_decision.j_score if last_decision else None,
            "last_candidate_alignment": (
                last_decision.candidate_alignment if last_decision else None
            ),
            "last_baseline_alignment": (
                last_decision.baseline_alignment if last_decision else None
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
        rows.extend({"kind": "skill", **skill.model_dump()} for skill in state.skills.values())
        rows.extend({"kind": "episode", **episode.model_dump()} for episode in state.episodes)
        rows.extend({"kind": "experience", **experience.model_dump()} for experience in state.experiences)
        rows.extend({"kind": "golden_experience", **experience.model_dump()} for experience in state.golden_experiences)
        return [json.dumps(row, ensure_ascii=False, default=str) for row in rows]
