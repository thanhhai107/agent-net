"""Composable wrapper that injects Skill-Pro procedural context."""

from __future__ import annotations

from typing import Any

from agent.memory.attributes import infer_memory_attributes
from agent.memory.models import MemoryQuery
from agent.memory.service import ProceduralMemoryModule
from agent.utils.loggers import MessageLogger


class MemoryAugmentedAgent:
    def __init__(
        self,
        agent: Any,
        memory: ProceduralMemoryModule,
        *,
        memory_mode: str,
        memory_top_k: int = 5,
        memory_token_budget: int = 1500,
    ) -> None:
        if memory_mode not in {"read", "evolve"}:
            raise ValueError("memory_mode must be read or evolve for the adapter")
        self.agent = agent
        self.memory = memory
        self.memory_mode = memory_mode
        self.memory_top_k = memory_top_k
        self.memory_token_budget = memory_token_budget

    def __getattr__(self, name: str) -> Any:
        return getattr(self.agent, name)

    async def run(self, task_description: str) -> Any:
        session = getattr(self.agent, "session", None)
        session_id = str(getattr(session, "session_id", "") or getattr(self.agent, "session_id", ""))
        tools = list(getattr(self.agent, "diagnosis_tool_names", []) or [])
        attrs = infer_memory_attributes(
            task_description,
            scenario=str(getattr(session, "scenario_name", "") or ""),
            topology_class=str(getattr(session, "scenario_topo_size", "") or ""),
            task_stage="diagnosis",
            tools=tools,
        )
        query = MemoryQuery(
            text=task_description,
            scenario=str(getattr(session, "scenario_name", "") or ""),
            topology_class=str(getattr(session, "scenario_topo_size", "") or ""),
            protocols=attrs.protocols,
            services=attrs.services,
            symptoms=attrs.symptoms,
            task_stage="diagnosis",
            tools=tools,
            top_k=self.memory_top_k,
            token_budget=self.memory_token_budget,
        )
        active = self.memory.select_skill(query=query, session_id=session_id)
        retrieved = self.memory.retrieve(query=query, session_id=session_id)
        if active is not None and all(item.skill.skill_id != active.skill.skill_id for item in retrieved):
            retrieved.insert(0, active)
        context = self.memory.format_context(retrieved)
        session_dir = getattr(self.agent, "session_dir", "")
        if session_dir:
            MessageLogger(
                agent="memory_agent",
                session_dir=session_dir,
                extra_fields={"phase": "retrieval"},
            ).log(
                "skill_retrieval",
                {
                    "bank_id": self.memory.bank_id,
                    "memory_mode": self.memory_mode,
                    "active_skill_id": active.skill.skill_id if active else "",
                    "skill_ids": [item.skill.skill_id for item in retrieved],
                    "scores": [round(item.score, 6) for item in retrieved],
                },
            )
        augmented = (
            f"{context}\n\nOriginal diagnosis task:\n{task_description}"
            if context
            else task_description
        )
        return await self.agent.run(augmented)
