"""Composable wrapper that installs integrated Skill-Pro runtime context."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.memory.service import ProceduralMemoryModule


class MemoryAugmentedAgent:
    def __init__(
        self,
        agent: Any,
        memory: ProceduralMemoryModule,
        *,
        memory_mode: str,
        memory_top_k: int = 5,
        memory_token_budget: int = 1500,
        memory_max_skill_age: int = 4,
    ) -> None:
        if memory_mode not in {"read", "evolve"}:
            raise ValueError("memory_mode must be read or evolve for the adapter")
        self.agent = agent
        self.memory = memory
        self.memory_mode = memory_mode
        self.memory_top_k = memory_top_k
        self.memory_token_budget = memory_token_budget
        self.memory_max_skill_age = memory_max_skill_age

    def __getattr__(self, name: str) -> Any:
        return getattr(self.agent, name)

    async def run(self, task_description: str) -> Any:
        install_runtime = getattr(self.agent, "install_memory_runtime", None)
        if not callable(install_runtime):
            raise RuntimeError(
                "Skill-Pro memory requires an agent with integrated "
                "install_memory_runtime support; prompt-only memory injection is disabled."
            )
        install_runtime(
            memory=self.memory,
            memory_mode=self.memory_mode,
            task_description=task_description,
            top_k=self.memory_top_k,
            token_budget=self.memory_token_budget,
            max_skill_age=self.memory_max_skill_age,
        )
        try:
            return await self.agent.run(task_description)
        finally:
            self._write_runtime_snapshot()

    def _write_runtime_snapshot(self) -> None:
        runtime = getattr(self.agent, "skill_tool_runtime", None)
        session_dir = getattr(self.agent, "session_dir", "")
        if runtime is None or not session_dir:
            return
        path = Path(session_dir) / "memory_runtime_session.json"
        path.write_text(
            json.dumps(runtime.snapshot(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
