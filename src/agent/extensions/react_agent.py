"""Skill-Pro and DRAFT composition around NIKA's original ReAct agent."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from langchain.agents import create_agent

from agent.byo.langgraph.phases.diagnosis import DiagnosisPhase
from agent.byo.langgraph.react_agent import BasicReActAgent
from agent.composition import AgentRunConfig
from agent.memory.runtime import SkillToolRuntime
from agent.memory.service import ProceduralMemoryModule
from agent.tool_evolution.integration import write_tool_evolution_session
from agent.tool_evolution.runtime import ToolEvolutionRuntime
from agent.utils.phases import DIAGNOSIS
from agent.utils.template import OVERALL_DIAGNOSIS_PROMPT
from nika.utils.session import Session


def configure_custom_provider_environment() -> None:
    """Translate the local URL name to the variable expected by NIKA core."""
    custom_url = os.getenv("CUSTOM_API_URL", "").strip()
    if custom_url:
        os.environ["CUSTOM_API_BASE"] = custom_url.rstrip("/")


class LearningDiagnosisPhase(DiagnosisPhase):
    """Original diagnosis phase with optional learned tool wrappers."""

    def __init__(self, config: AgentRunConfig) -> None:
        configure_custom_provider_environment()
        super().__init__(
            session_id=config.session_id,
            llm_provider=config.llm_provider,
            model=config.model,
            scenario_name=Session()
            .load_running_session(session_id=config.session_id)
            .scenario_name,
        )
        self.config = config
        self.session = Session().load_running_session(session_id=config.session_id)
        self.tool_evolution_runtime: ToolEvolutionRuntime | None = None
        self.skill_tool_runtime: SkillToolRuntime | None = None
        self._base_tools = []

    async def load_tools(self) -> None:
        await super().load_tools()
        if self.config.tool_evolution.enabled:
            self.tool_evolution_runtime = ToolEvolutionRuntime(
                session=self.session,
                primitive_tools=self.tools or [],
                library_id=self.config.tool_evolution.library_id,
                tool_doc_chars=self.config.tool_evolution.tool_doc_chars,
            )
            self.tools = self.tool_evolution_runtime.build_tools()
        self._base_tools = list(self.tools or [])

    def install_memory(self, task_description: str, session_dir: str) -> None:
        memory_config = self.config.memory
        if not memory_config.enabled:
            return
        memory = ProceduralMemoryModule(
            bank_id=memory_config.bank,
            llm_backend=self.config.llm_provider,
            model=self.config.model,
            pool_size=memory_config.pool_size,
            evolution_threshold=memory_config.evolution_threshold,
            best_of_n=memory_config.best_of_n,
            ppo_epsilon=memory_config.ppo_epsilon,
        )
        self.skill_tool_runtime = SkillToolRuntime(
            memory=memory,
            memory_mode=memory_config.mode,
            session=self.session,
            task_description=task_description,
            tools=list(self._base_tools),
            session_dir=session_dir,
            tool_evolution_runtime=self.tool_evolution_runtime,
            top_k=memory_config.top_k,
            token_budget=memory_config.token_budget,
            max_skill_age=memory_config.max_skill_age,
            meta_controller_llm=self.llm,
        )
        self.tools = self.skill_tool_runtime.wrap_tools(list(self._base_tools))

    def get_agent(self):
        system_prompt = OVERALL_DIAGNOSIS_PROMPT
        if self.skill_tool_runtime is not None:
            system_prompt += self.skill_tool_runtime.prompt_suffix(activate_skill=True)
        return create_agent(
            model=self.llm,
            system_prompt=system_prompt,
            tools=self.tools,
            name=DIAGNOSIS,
        )


class LearningReActAgent(BasicReActAgent):
    """NIKA ReAct with diagnosis-only learning extensions enabled."""

    def __init__(self, config: AgentRunConfig) -> None:
        configure_custom_provider_environment()
        super().__init__(
            session_id=config.session_id,
            llm_provider=config.llm_provider,
            model=config.model,
            max_steps=config.max_steps,
        )
        self.extension_config = config
        self._learning_phase = LearningDiagnosisPhase(config)
        asyncio.run(self._learning_phase.load_tools())
        if config.tool_evolution.enabled and not config.memory.enabled:
            self._diagnosis_runner = self._learning_phase.get_agent()

    async def run(self, task_description: str):
        if self.extension_config.memory.enabled:
            self._learning_phase.install_memory(task_description, self.session_dir)
            self._diagnosis_runner = self._learning_phase.get_agent()
        try:
            return await super().run(task_description)
        finally:
            self._write_extension_snapshots()

    def _write_extension_snapshots(self) -> None:
        write_tool_evolution_session(
            self._learning_phase.tool_evolution_runtime,
            self.session_dir,
        )
        runtime = self._learning_phase.skill_tool_runtime
        if runtime is None:
            return
        path = Path(self.session_dir) / "memory_runtime_session.json"
        path.write_text(
            json.dumps(runtime.snapshot(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def create_react_agent(config: AgentRunConfig):
    """Return original NIKA for baseline, extension subclass otherwise."""
    if not config.extensions_enabled:
        configure_custom_provider_environment()
        return BasicReActAgent(
            session_id=config.session_id,
            llm_provider=config.llm_provider,
            model=config.model,
            max_steps=config.max_steps,
        )
    return LearningReActAgent(config)

