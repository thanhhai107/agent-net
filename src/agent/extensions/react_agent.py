"""Skill-Pro and DRAFT composition around NIKA's original ReAct agent."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt

from agent.byo.langgraph.phases.diagnosis import DiagnosisPhase
from agent.byo.langgraph.react_agent import BasicReActAgent
from agent.composition import AgentRunConfig
from agent.extensions.llm import load_extension_model
from agent.procedural_memory.runtime import SkillToolRuntime
from agent.procedural_memory.service import ProceduralMemoryModule
from agent.tool_refinement.integration import write_tool_refinement_session
from agent.tool_refinement.runtime import ToolRefinementRuntime
from agent.utils.phases import DIAGNOSIS
from agent.utils.template import OVERALL_DIAGNOSIS_PROMPT
from nika.utils.session import Session


def _decision_context(request: ModelRequest) -> str:
    for message in reversed(request.messages):
        if getattr(message, "type", "") == "human":
            return str(getattr(message, "content", "") or "")
    return ""


def configure_custom_provider_environment() -> None:
    """Translate the local URL name to the variable expected by NIKA core."""
    custom_url = os.getenv("CUSTOM_API_URL", "").strip()
    if custom_url:
        os.environ["CUSTOM_API_BASE"] = custom_url.rstrip("/")


class _ExploringDiagnosisRunner:
    """Run DRAFT exploration after one complete ReAct diagnosis invocation."""

    def __init__(self, runner, runtime: ToolRefinementRuntime) -> None:
        self.runner = runner
        self.runtime = runtime

    async def ainvoke(self, inputs, *args, **kwargs):
        result = await self.runner.ainvoke(inputs, *args, **kwargs)
        messages = inputs.get("messages") or [] if isinstance(inputs, dict) else []
        task_description = str(getattr(messages[0], "content", "")) if messages else ""
        await self.runtime.explore(task_description)
        return result


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
        self.tool_refinement_runtime: ToolRefinementRuntime | None = None
        self.skill_tool_runtime: SkillToolRuntime | None = None
        self._base_tools = []

    async def load_tools(self) -> None:
        await super().load_tools()
        if self.config.tool_refinement.enabled:
            explorer_model = self.config.tool_refinement.explorer_model.strip()
            explorer_llm = (
                self.llm
                if self.config.tool_refinement.learning_mode == "evolve"
                and self.config.tool_refinement.update_due
                else None
            )
            if (
                explorer_llm is not None
                and explorer_model
                and explorer_model != self.config.model
            ):
                explorer_llm = load_extension_model(
                    self.config.llm_provider,
                    explorer_model,
                )
            self.tool_refinement_runtime = ToolRefinementRuntime(
                session=self.session,
                primitive_tools=self.tools or [],
                library_id=self.config.tool_refinement.library_id,
                tool_doc_chars=self.config.tool_refinement.tool_doc_chars,
                explorer_llm=explorer_llm,
                llm_backend=self.config.llm_provider,
                model=self.config.model,
                convergence_threshold=self.config.tool_refinement.convergence_threshold,
                exploration_similarity_threshold=(
                    self.config.tool_refinement.exploration_similarity_threshold
                ),
                explorer_reflection_limit=(
                    self.config.tool_refinement.explorer_reflection_limit
                ),
                max_tools_per_update=(self.config.tool_refinement.max_tools_per_update),
                explorer_model=self.config.tool_refinement.explorer_model,
                analyzer_model=self.config.tool_refinement.analyzer_model,
                rewriter_model=self.config.tool_refinement.rewriter_model,
            )
            self.tools = self.tool_refinement_runtime.build_tools(
                append_docs=not self.config.procedural_memory.enabled
            )
        self._base_tools = list(self.tools or [])

    def install_procedural_memory(
        self, task_description: str, session_dir: str
    ) -> None:
        procedural_memory_config = self.config.procedural_memory
        if not procedural_memory_config.enabled:
            return
        procedural_memory = ProceduralMemoryModule(
            bank_id=procedural_memory_config.bank,
            llm_backend=self.config.llm_provider,
            model=self.config.model,
            pool_size=procedural_memory_config.pool_size,
            evolution_threshold=procedural_memory_config.evolution_threshold,
            best_of_n=procedural_memory_config.best_of_n,
            ppo_epsilon=procedural_memory_config.ppo_epsilon,
            experience_pool_size=procedural_memory_config.experience_pool_size,
            baseline_ema_alpha=procedural_memory_config.baseline_ema_alpha,
            selection_epsilon_decay_cases=(
                procedural_memory_config.selection_epsilon_decay_cases
            ),
            acceptance_margin=procedural_memory_config.acceptance_margin,
            verifier=procedural_memory_config.verifier,
            holdout_size=procedural_memory_config.holdout_size,
            min_positive_advantage=(procedural_memory_config.min_positive_advantage),
            evolver_model=procedural_memory_config.evolver_model,
            policy_scorer_model=procedural_memory_config.policy_scorer_model,
        )
        self.skill_tool_runtime = SkillToolRuntime(
            procedural_memory=procedural_memory,
            procedural_memory_mode=procedural_memory_config.mode,
            session=self.session,
            task_description=task_description,
            tools=list(self._base_tools),
            session_dir=session_dir,
            tool_refinement_runtime=self.tool_refinement_runtime,
            token_budget=procedural_memory_config.token_budget,
            max_skill_age=procedural_memory_config.max_skill_age,
            selection_epsilon=procedural_memory_config.selection_epsilon,
            meta_controller_llm=self.llm,
        )
        self.tools = self.skill_tool_runtime.wrap_tools(list(self._base_tools))

    def get_agent(self):
        middleware = []
        system_prompt = OVERALL_DIAGNOSIS_PROMPT
        if self.skill_tool_runtime is not None:
            runtime = self.skill_tool_runtime

            @dynamic_prompt
            def skill_policy_prompt(request: ModelRequest) -> str:
                return OVERALL_DIAGNOSIS_PROMPT + runtime.prompt_suffix(
                    activate_skill=True,
                    decision_context=_decision_context(request),
                )

            middleware.append(skill_policy_prompt)
            system_prompt = None
        return create_agent(
            model=self.llm,
            system_prompt=system_prompt,
            tools=self.tools,
            middleware=middleware,
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
        if config.tool_refinement.enabled and not config.procedural_memory.enabled:
            self._diagnosis_runner = self._learning_phase.get_agent()
            self._install_exploring_runner()

    async def run(self, task_description: str):
        if self.extension_config.procedural_memory.enabled:
            self._learning_phase.install_procedural_memory(
                task_description, self.session_dir
            )
            self._diagnosis_runner = self._learning_phase.get_agent()
            self._install_exploring_runner()
        try:
            result = await super().run(task_description)
            reports = result.get("diagnosis_report") or []
            report = reports[-1] if isinstance(reports, list) and reports else reports
            runtime = self._learning_phase.skill_tool_runtime
            if runtime is not None:
                runtime.record_terminal_diagnosis(str(report or ""))
            return result
        finally:
            self._write_extension_snapshots()

    def _install_exploring_runner(self) -> None:
        runtime = self._learning_phase.tool_refinement_runtime
        if runtime is not None and not isinstance(
            self._diagnosis_runner, _ExploringDiagnosisRunner
        ):
            self._diagnosis_runner = _ExploringDiagnosisRunner(
                self._diagnosis_runner,
                runtime,
            )

    def _write_extension_snapshots(self) -> None:
        write_tool_refinement_session(
            self._learning_phase.tool_refinement_runtime,
            self.session_dir,
        )
        runtime = self._learning_phase.skill_tool_runtime
        if runtime is None:
            return
        path = Path(self.session_dir) / "procedural_memory_runtime_session.json"
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
