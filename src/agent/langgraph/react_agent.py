import asyncio
import json
import logging
import os

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from pydantic import Field, ValidationError
from typing_extensions import TypedDict

from agent.langgraph.langfuse_tracing import callback_config, create_langfuse_callbacks
from agent.langgraph.phases.diagnosis import DiagnosisPhase
from agent.langgraph.phases.submission import SubmissionPhase
from agent.llm.model_factory import DEFAULT_LLM_BACKEND, DEFAULT_MODEL
from agent.memory.runtime import strip_integrated_learning_guidance
from agent.tool_evolution.integration import write_tool_evolution_session
from agent.utils.evidence import extract_link_down_devices
from agent.utils.loggers import AgentCallbackLogger
from agent.utils.phases import DIAGNOSIS, SUBMISSION
from agent.utils.tracing import langsmith_tracing_context
from nika.utils.session import Session

load_dotenv()


logging.basicConfig(level=logging.INFO)


class AgentState(TypedDict):
    """The state of the agent."""

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
        worker_builder.add_edge(DIAGNOSIS, SUBMISSION)

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
        )
        self._refresh_diagnosis_agent()

    async def run(self, task_description: str):
        with langsmith_tracing_context(
            project_name=os.getenv("LANGSMITH_PROJECT", "NIKA"),
            metadata={
                "scenario": self.session.scenario_name,
                "problem": self.session.problem_names[0],
                "topo_size": self.session.scenario_topo_size,
                "model": self.model,
            },
        ):
            try:
                return await self.graph.ainvoke(
                    {
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
            learning_context = self._learning_prompt_suffix()
            if learning_context:
                messages.append(HumanMessage(content=learning_context))
            diagnosis_report = await self.diagnosis_agent.ainvoke(
                {"messages": messages},
                config={
                    "callbacks": [cb],
                    "recursion_limit": self.max_steps,
                },
                debug=True,
            )
            return {
                "diagnosis_report": diagnosis_report["messages"][-1].content,
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
                "diagnosis_report": self._fallback_diagnosis_report(
                    "max recursion limit"
                ),
                "is_max_steps_reached": True,
            }

    def _fallback_diagnosis_report(self, reason: str) -> str:
        observations = self._recent_tool_observations()
        if not observations:
            return (
                "No evidence-backed diagnosis report is available because diagnosis "
                f"reached {reason} before preserving any current tool observation."
            )
        evidence = "\n".join(f"- {item}" for item in observations)
        assessment = self._fallback_assessment(observations)
        return (
            "Current evidence-backed diagnosis report generated from preserved "
            f"tool observations after diagnosis reached {reason}.\n"
            "Assessment:\n"
            f"{assessment}\n"
            "Current tool evidence:\n"
            f"{evidence}\n"
            "Use this as the diagnosis report for final submission. Memory, DRAFT "
            "guidance, and the fallback condition are not evidence; the current "
            "tool observations above are the evidence."
        )

    @staticmethod
    def _fallback_assessment(observations: list[str]) -> str:
        text = "\n".join(observations)
        lower = text.lower()
        link_down_devices = extract_link_down_devices(text, window=700)
        if link_down_devices:
            devices = ", ".join(link_down_devices)
            return (
                f"The observations support an anomaly localized to {devices}. "
                "An endpoint-facing interface or attached link is down while the "
                "routing/control-plane observations remain healthy. The supported "
                "root-cause class is an interface/link-down fault; use an exact "
                "root-cause id only after checking the available submission options."
            )
        if (
            "destination host unreachable" in lower
            or "100% packet loss" in lower
            or '"status":"unknown"' in lower
        ):
            return (
                "The observations support a reachability anomaly. Localize only "
                "to devices or interfaces directly named by the failing ping, "
                "reachability, or interface observations, and choose the most "
                "specific available root cause supported by those observations."
            )
        return (
            "The observations below are the preserved current evidence. Submit "
            "only anomaly status, localization, and root cause conclusions that "
            "are directly supported by them."
        )

    def _recent_tool_observations(self) -> list[str]:
        runtime = getattr(self, "skill_tool_runtime", None)
        if runtime is None or not hasattr(runtime, "snapshot"):
            return []
        try:
            snapshot = runtime.snapshot()
        except Exception:
            return []
        observations: list[str] = []
        for transition in snapshot.get("recent_transitions") or []:
            if not isinstance(transition, dict):
                continue
            tool = str(transition.get("tool") or "tool")
            tool_input = transition.get("tool_input", {})
            summary = strip_integrated_learning_guidance(
                transition.get("observation_summary", "")
            )
            if not summary:
                continue
            try:
                args = json.dumps(tool_input, ensure_ascii=False, default=str)
            except TypeError:
                args = str(tool_input)
            observations.append(f"{tool}({args}) -> {self._clip_report_text(summary)}")
        if observations:
            return self._select_report_observations(observations)
        for item in snapshot.get("recent_observations") or []:
            text = strip_integrated_learning_guidance(item)
            if text:
                observations.append(self._clip_report_text(text))
        return self._select_report_observations(observations)

    @staticmethod
    def _select_report_observations(observations: list[str]) -> list[str]:
        if len(observations) <= 8:
            return observations
        priority_terms = (
            "state down",
            "ip_route is empty",
            "flags=4098",
            "destination host unreachable",
            "100% packet loss",
            '"status":"unknown"',
            "bgp",
            "state up",
        )
        scored: list[tuple[int, int, str]] = []
        for index, observation in enumerate(observations):
            lower = observation.lower()
            score = sum(
                (len(priority_terms) - rank)
                for rank, term in enumerate(priority_terms)
                if term in lower
            )
            scored.append((score, index, observation))
        selected = sorted(
            sorted(scored, key=lambda row: (row[0], row[1]), reverse=True)[:8],
            key=lambda row: row[1],
        )
        return [observation for _, _, observation in selected]

    @staticmethod
    def _clip_report_text(text: str, limit: int = 1200) -> str:
        clean = " ".join(str(text or "").split())
        if len(clean) <= limit:
            return clean
        return clean[: limit - 3] + "..."

    async def submission_agent_builder(self, state: AgentState):
        diag_text = state["diagnosis_report"]
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
