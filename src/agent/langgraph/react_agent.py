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
from agent.langgraph.evidence import (
    ToolObservation,
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
    diagnosis_observations: list[ToolObservation]
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
        use_problem_tool_hints: bool = True,
    ):
        self.session_id = session_id
        self.model = model
        self.max_steps = max_steps
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
        self.submission_phase = submission_phase

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

    def install_memory_runtime(
        self,
        *,
        memory,
        memory_mode: str,
        task_description: str,
        top_k: int = 5,
        token_budget: int = 1500,
        max_skill_age: int = 4,
    ) -> None:
        self._diagnosis_phase.install_memory_runtime(
            memory=memory,
            memory_mode=memory_mode,
            task_description=task_description,
            top_k=top_k,
            token_budget=token_budget,
            session_dir=self.session_dir,
            max_skill_age=max_skill_age,
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
                        "diagnosis_observations": [],
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
            return {
                "diagnosis_report": report_text,
                "diagnosis_observations": self._current_tool_observations(
                    report_messages
                ),
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
            result = await self.submission_phase.submit_report(
                task_description=self._state_task_description(state),
                diagnosis_report=diag_text,
                observations=state.get("diagnosis_observations", []),
                session_dir=self.session_dir,
            )
            if result is None:
                return {}
            return {
                "messages": result["messages"],
            }
        except Exception as exc:
            AgentCallbackLogger(
                agent=SUBMISSION, session_dir=self.session_dir
            )._log(
                "error",
                {"message": f"Evidence-bound submission failed: {exc}"},
            )
            return {}
