import asyncio
import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from pydantic import Field, ValidationError
from typing_extensions import TypedDict

from agent.langgraph.langfuse_tracing import callback_config, create_langfuse_callbacks
from agent.langgraph.evidence_gate import (
    EvidenceGateResult,
    ToolObservation,
    evaluate_fault_family_evidence,
    observations_from_messages,
    observations_from_runtime_snapshot,
)
from agent.langgraph.phases.diagnosis import DiagnosisPhase
from agent.langgraph.phases.submission import SubmissionPhase
from agent.llm.model_factory import DEFAULT_LLM_BACKEND, DEFAULT_MODEL
from agent.tool_evolution.integration import write_tool_evolution_session
from agent.utils.loggers import AgentCallbackLogger
from agent.utils.phases import DIAGNOSIS, SUBMISSION
from agent.utils.tracing import langsmith_tracing_context, session_problem_label
from nika.utils.session import Session

load_dotenv()


logging.basicConfig(level=logging.INFO)


_DIAGNOSIS_MESSAGE_BUDGET_CHARS = 120_000
_DIAGNOSIS_MESSAGE_CONTENT_LIMIT_CHARS = 8_000
_DIAGNOSIS_RECENT_MESSAGES = 40


class AgentState(TypedDict):
    """The state of the agent."""

    task_description: str = Field(
        default="",
        description="The original user-visible diagnosis task.",
    )
    messages: list[BaseMessage]
    diagnosis_report: str = Field(
        default="",
        description="The diagnosis report of the network state after analysis.",
    )
    is_max_steps_reached: bool = Field(
        default=False,
        description="Indicates whether the agent has reached the maximum number of steps allowed.",
    )


class BasicReActAgent:
    def __init__(
        self,
        session_id: str,
        max_steps: int,
        llm_backend: str = DEFAULT_LLM_BACKEND,
        model: str = DEFAULT_MODEL,
        tool_evolution_enabled: bool = False,
        tool_library_id: str = "default",
        tool_doc_chars: int = 500,
        tool_prompt_doc_limit: int = 6,
        tool_scoped_prompt_doc_limit: int = 4,
        tool_planned_checks: int = 4,
        tool_next_checks: int = 2,
        use_problem_tool_hints: bool = True,
        evidence_gate_enabled: bool = True,
        evidence_gate_retries: int = 1,
    ):
        self.session_id = session_id
        self.model = model
        self.max_steps = max_steps
        self.evidence_gate_enabled = evidence_gate_enabled
        self.evidence_gate_retries = self._resolve_evidence_gate_retries(
            evidence_gate_retries
        )
        self.session = Session()
        self.session.load_running_session(session_id=session_id)
        self.session_dir = self.session.session_dir

        self.langfuse_callbacks = create_langfuse_callbacks()

        diagnosis_phase = DiagnosisPhase(
            session_id=session_id,
            llm_backend=llm_backend,
            model=model,
            scenario_name=self.session.scenario_name,
            load_all_tools=not use_problem_tool_hints,
            tool_evolution_enabled=tool_evolution_enabled,
            tool_library_id=tool_library_id,
            tool_doc_chars=tool_doc_chars,
            tool_prompt_doc_limit=tool_prompt_doc_limit,
            tool_scoped_prompt_doc_limit=tool_scoped_prompt_doc_limit,
            tool_planned_checks=tool_planned_checks,
            tool_next_checks=tool_next_checks,
        )
        asyncio.run(diagnosis_phase.load_tools())
        self._diagnosis_phase = diagnosis_phase
        self.llm = diagnosis_phase.llm
        self.tool_evolution_runtime = diagnosis_phase.tool_evolution_runtime
        self.skill_tool_runtime = diagnosis_phase.skill_tool_runtime
        self._refresh_diagnosis_agent()

        submission_phase = SubmissionPhase(
            session_id=session_id,
            llm_backend=llm_backend,
            model=model,
        )
        asyncio.run(submission_phase.load_tools())
        self.submission_agent = submission_phase.get_agent()

        worker_builder = StateGraph(AgentState)
        worker_builder.add_node(DIAGNOSIS, self.diagnosis_agent_builder)
        worker_builder.add_node(SUBMISSION, self.submission_agent_builder)

        worker_builder.add_edge(START, DIAGNOSIS)
        worker_builder.add_conditional_edges(
            DIAGNOSIS,
            lambda state: state.get("is_max_steps_reached", False),
            {
                True: END,
                False: SUBMISSION,
            },
        )

        worker_builder.add_edge(SUBMISSION, END)
        self.graph = worker_builder.compile()

    def _refresh_diagnosis_agent(self) -> None:
        self.tool_evolution_runtime = self._diagnosis_phase.tool_evolution_runtime
        self.skill_tool_runtime = self._diagnosis_phase.skill_tool_runtime
        self.diagnosis_tool_names = [
            tool.name for tool in (self._diagnosis_phase.tools or [])
        ]
        self.diagnosis_agent = self._diagnosis_phase.get_agent()

    def _learning_prompt_suffix(self) -> str:
        suffix = self._diagnosis_phase.prompt_suffix(activate_skill=True)
        if not suffix:
            return ""
        return (
            "\n\nCurrent integrated learning context for this diagnosis run:\n"
            "Use the following Skill-Pro/DRAFT guidance to choose diagnostic "
            "tools and interpret tool outputs. It is not evidence; only current "
            "tool outputs can support the final diagnosis."
            f"{suffix}"
        )

    @staticmethod
    def _message_content_text(message: Any) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        return str(content)

    @classmethod
    def _clip_message_content(cls, message: Any) -> Any:
        content = cls._message_content_text(message)
        if len(content) <= _DIAGNOSIS_MESSAGE_CONTENT_LIMIT_CHARS:
            return message
        clipped = (
            content[: _DIAGNOSIS_MESSAGE_CONTENT_LIMIT_CHARS]
            + "\n...[truncated previous observation to keep diagnosis context bounded]"
        )
        if hasattr(message, "model_copy"):
            return message.model_copy(update={"content": clipped})
        if hasattr(message, "copy"):
            return message.copy(update={"content": clipped})
        return HumanMessage(content=clipped)

    @classmethod
    def _bounded_messages(cls, messages: list[Any]) -> list[Any]:
        """Keep ReAct context below model limits while preserving recent evidence."""
        if not messages:
            return []
        first = cls._clip_message_content(messages[0])
        recent = [
            cls._clip_message_content(message)
            for message in messages[1:][-_DIAGNOSIS_RECENT_MESSAGES:]
        ]
        kept: list[Any] = []
        total = len(cls._message_content_text(first))
        for message in reversed(recent):
            size = len(cls._message_content_text(message))
            if kept and total + size > _DIAGNOSIS_MESSAGE_BUDGET_CHARS:
                break
            kept.append(message)
            total += size
        kept.reverse()
        if len(kept) < len(messages) - 1:
            omitted = len(messages) - 1 - len(kept)
            marker = HumanMessage(
                content=(
                    f"[Context bounded: omitted {omitted} older diagnosis messages. "
                    "Use the remaining recent tool observations as current evidence.]"
                )
            )
            return [first, marker, *kept]
        return [first, *kept]

    @staticmethod
    def _resolve_evidence_gate_retries(default: int) -> int:
        raw_value = os.getenv("NIKA_EVIDENCE_GATE_RETRIES")
        if raw_value is None:
            return max(0, int(default))
        try:
            return max(0, int(raw_value))
        except ValueError:
            return max(0, int(default))

    def _diagnosis_tool_names(self) -> list[str]:
        if getattr(self, "diagnosis_tool_names", None):
            return list(self.diagnosis_tool_names)
        phase = getattr(self, "_diagnosis_phase", None)
        return [tool.name for tool in getattr(phase, "tools", []) or []]

    def _state_task_description(self, state: AgentState) -> str:
        task_description = str(state.get("task_description") or "").strip()
        if task_description:
            return task_description
        for message in state.get("messages", []):
            content = str(getattr(message, "content", "") or "").strip()
            if content:
                return content
        session_task = getattr(getattr(self, "session", None), "task_description", "")
        return str(session_task or "")

    def _current_tool_observations(
        self,
        messages: list[Any],
    ) -> list[ToolObservation]:
        observations: list[ToolObservation] = []
        runtime = getattr(self, "skill_tool_runtime", None)
        if runtime is not None and hasattr(runtime, "snapshot"):
            observations.extend(observations_from_runtime_snapshot(runtime.snapshot()))
        observations.extend(observations_from_messages(messages))
        return observations

    def _evaluate_evidence_gate(
        self,
        *,
        task_description: str,
        diagnosis_report: str,
        messages: list[Any],
    ) -> EvidenceGateResult:
        return evaluate_fault_family_evidence(
            task_description=task_description,
            diagnosis_report=diagnosis_report,
            observations=self._current_tool_observations(messages),
            available_tools=self._diagnosis_tool_names(),
        )

    def _is_evidence_gate_enabled(self) -> bool:
        return bool(getattr(self, "evidence_gate_enabled", True))

    def install_memory_runtime(
        self,
        *,
        memory,
        memory_mode: str,
        task_description: str,
        top_k: int = 5,
        token_budget: int = 1500,
        skill_selector_mode: str = "lcb",
        meta_controller_mode: str = "heuristic",
        max_skill_age: int = 4,
        selector_min_lcb: float = -0.05,
        selector_nominee_k: int = 3,
    ) -> None:
        self._diagnosis_phase.install_memory_runtime(
            memory=memory,
            memory_mode=memory_mode,
            task_description=task_description,
            top_k=top_k,
            token_budget=token_budget,
            session_dir=self.session_dir,
            skill_selector_mode=skill_selector_mode,
            meta_controller_mode=meta_controller_mode,
            max_skill_age=max_skill_age,
            selector_min_lcb=selector_min_lcb,
            selector_nominee_k=selector_nominee_k,
        )
        self._refresh_diagnosis_agent()

    async def run(self, task_description: str):
        with langsmith_tracing_context(
            project_name=os.getenv("LANGSMITH_PROJECT", "NIKA"),
            metadata={
                "scenario": self.session.scenario_name,
                "problem": session_problem_label(self.session),
                "topo_size": self.session.scenario_topo_size,
                "model": self.model,
            },
        ):
            try:
                return await self.graph.ainvoke(
                    {
                        "task_description": task_description,
                        "messages": [HumanMessage(content=task_description)],
                    },
                    config=callback_config(self.langfuse_callbacks),
                )
            finally:
                write_tool_evolution_session(
                    self.tool_evolution_runtime,
                    self.session_dir,
                )

    async def diagnosis_agent_builder(self, state: AgentState):
        try:
            cb = AgentCallbackLogger(agent=DIAGNOSIS, session_dir=self.session_dir)
            messages = list(state["messages"])
            task_description = self._state_task_description(state)
            learning_context = self._learning_prompt_suffix()
            if learning_context:
                messages.append(HumanMessage(content=learning_context))
            messages = self._bounded_messages(messages)
            diagnosis_report = await self.diagnosis_agent.ainvoke(
                {"messages": messages},
                config={
                    "callbacks": [cb],
                    "recursion_limit": self.max_steps,
                },
                debug=True,
            )
            report_messages = list(diagnosis_report.get("messages", []))
            report_text = str(report_messages[-1].content)
            gate = None
            if self._is_evidence_gate_enabled():
                gate = self._evaluate_evidence_gate(
                    task_description=task_description,
                    diagnosis_report=report_text,
                    messages=report_messages,
                )
            if (
                gate is not None
                and not gate.sufficient
                and getattr(self, "evidence_gate_retries", 1) > 0
                and gate.prompt
            ):
                cb._log(
                    "evidence_gate_blocked",
                    {
                        "attempt": 1,
                        **gate.to_log_payload(),
                    },
                )
                retry_messages = [
                    *report_messages,
                    HumanMessage(content=gate.prompt),
                ]
                retry_messages = self._bounded_messages(retry_messages)
                diagnosis_report = await self.diagnosis_agent.ainvoke(
                    {"messages": retry_messages},
                    config={
                        "callbacks": [cb],
                        "recursion_limit": self.max_steps,
                    },
                    debug=True,
                )
                report_messages = list(diagnosis_report.get("messages", []))
                report_text = str(report_messages[-1].content)
                retry_gate = self._evaluate_evidence_gate(
                    task_description=task_description,
                    diagnosis_report=report_text,
                    messages=report_messages,
                )
                cb._log(
                    "evidence_gate_retry_result",
                    {
                        "attempt": 1,
                        **retry_gate.to_log_payload(),
                    },
                )
            return {
                "diagnosis_report": report_text,
                "is_max_steps_reached": False,
            }
        except ValidationError as e:
            AgentCallbackLogger(
                agent=DIAGNOSIS, session_dir=self.session_dir
            )._log("error", {"message": f"Validation error: {e}"})
            return {
                "messages": [HumanMessage(content=f"Error: {e}")],
                "diagnosis_report": "ERROR_VALIDATION",
                "is_max_steps_reached": False,
            }
        except GraphRecursionError:
            AgentCallbackLogger(
                agent=DIAGNOSIS, session_dir=self.session_dir
            )._log(
                "error",
                {"message": "Diagnosis phase reached max recursion limit."},
            )
            return {
                "messages": [
                    HumanMessage(
                        content="Error: diagnosis did not finish within max steps."
                    )
                ],
                "diagnosis_report": "ERROR_MAX_STEPS_REACHED",
                "is_max_steps_reached": True,
            }

    async def submission_agent_builder(self, state: AgentState):
        diag_text = state["diagnosis_report"]
        try:
            result = await self.submission_agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content=f"Based on the diagnosis report: {diag_text}, please provide the submission. Do not submit if no report available."
                        ),
                    ]
                },
                config={
                    "callbacks": [
                        AgentCallbackLogger(
                            agent=SUBMISSION, session_dir=self.session_dir
                        )
                    ],
                    "recursion_limit": self.max_steps,
                },
                debug=True,
            )
            return {
                "messages": result["messages"],
            }
        except GraphRecursionError:
            AgentCallbackLogger(
                agent=SUBMISSION, session_dir=self.session_dir
            )._log(
                "max_recursion_reached",
                {"message": "Submission phase reached max recursion limit."},
            )
            return {
                "messages": [
                    HumanMessage(
                        content="Submission was not produced before the max step limit."
                    )
                ],
            }
