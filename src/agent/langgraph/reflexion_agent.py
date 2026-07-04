"""Iterative Reflexion workflow for tool-assisted network diagnosis."""

import asyncio
import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from agent.langgraph.phases.diagnosis import DiagnosisPhase as DiagnosisAgent
from agent.langgraph.phases.submission import SubmissionPhase as SubmissionAgent
from agent.langgraph.langfuse_tracing import callback_config, create_langfuse_callbacks
from agent.langgraph.evidence_gate import (
    ToolObservation,
    evaluate_fault_family_evidence,
    observations_from_runtime_snapshot,
)
from agent.langgraph.workflow_models import ReflexionEvaluation, ReflexionMemory
from agent.llm.model_factory import DEFAULT_LLM_BACKEND, DEFAULT_MODEL, load_model
from agent.memory.runtime import strip_integrated_learning_guidance
from agent.tool_evolution.integration import write_tool_evolution_session
from agent.utils.loggers import AgentCallbackLogger
from agent.utils.template import EVIDENCE_CONTRACT_PROMPT
from agent.utils.tracing import langsmith_tracing_context, session_problem_label
from nika.utils.session import Session

load_dotenv()
logging.basicConfig(level=logging.INFO)

ACTOR_PROMPT = """\
You are a network troubleshooting expert operating inside an iterative Reflexion
workflow. Perform a fresh evidence-driven investigation using the available tools.
Focus on anomaly detection, faulty-device localization, and root-cause analysis.

Treat prior memories as lessons and investigation guidance, not as facts. Verify
their hypotheses with current tool observations. Prefer direct, fault-specific
evidence over broad health checks. Before the tool budget is exhausted, return a
complete diagnosis report that explicitly states:
1. whether an anomaly exists,
2. the exact faulty device or devices,
3. the root cause,
4. concrete supporting observations, and
5. remaining uncertainty.

Do not propose mitigation.
""" + "\n\n" + EVIDENCE_CONTRACT_PROMPT

EVALUATOR_PROMPT = """\
You are the strict evaluator in a Reflexion loop. Judge one network-diagnosis
attempt against the task using its tool trajectory and final report. A successful
attempt must explicitly decide anomaly status, localize the exact faulty device or
devices, identify a supported root cause, and cite concrete tool observations
without unresolved contradictions.

Set success=false when evidence is missing, conclusions are speculative, the
attempt ended before producing a report, or any of detection/localization/root
cause remains unsupported. Return concise, actionable failure feedback. Do not
call tools, rewrite the report, or assume hidden ground truth.

Assign quality_score from 0.0 to 1.0. Set success=true only when quality_score is
at least 0.8 and the evidence is sufficient.
""" + "\n\n" + EVIDENCE_CONTRACT_PROMPT

REFLEXION_PROMPT = """\
You are the self-review component of a Reflexion agent. Given a failed
troubleshooting attempt and its evaluator feedback, produce compact episodic
memory for the next attempt. Identify what reasoning or tool-selection mistakes
caused failure, what evidence is still needed, which hypotheses need verification,
and a materially different next strategy.

Do not solve the task, call tools, or repeat the full report. The memory must be
actionable and must not promote unverified conclusions to facts.
""" + "\n\n" + EVIDENCE_CONTRACT_PROMPT


class ReflexionState(TypedDict, total=False):
    task_description: str
    attempt_count: int
    attempt_report: str
    attempt_error: str
    attempt_trace: list[dict[str, Any]]
    diagnosis_report: str
    is_max_steps_reached: bool
    best_score: float
    evaluation: ReflexionEvaluation | None
    evaluation_failed: bool
    memories: list[ReflexionMemory]


class ReflexionAgent:
    """Attempt → evaluate → reflect → retry with persistent episodic memory."""

    def __init__(
        self,
        session_id: str,
        max_steps: int,
        llm_backend: str = DEFAULT_LLM_BACKEND,
        model: str = DEFAULT_MODEL,
        max_attempts: int = 3,
        tool_evolution_enabled: bool = False,
        tool_library_id: str = "default",
        tool_doc_chars: int = 500,
        tool_prompt_doc_limit: int = 6,
        tool_scoped_prompt_doc_limit: int = 4,
        tool_planned_checks: int = 4,
        tool_next_checks: int = 2,
        use_problem_tool_hints: bool = True,
        evidence_gate_enabled: bool = True,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

        self.session_id = session_id
        self.max_steps = max_steps
        self.max_attempts = max_attempts
        self.evidence_gate_enabled = evidence_gate_enabled
        self.session = Session()
        self.session.load_running_session(session_id=session_id)
        self.session_dir = self.session.session_dir
        self.llm = load_model(llm_backend=llm_backend, model=model)
        self.evaluator = self.llm.with_structured_output(ReflexionEvaluation)
        self.reflector = self.llm.with_structured_output(ReflexionMemory)

        diagnosis = DiagnosisAgent(
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
        asyncio.run(diagnosis.load_tools())
        self._diagnosis_phase = diagnosis
        self.tool_evolution_runtime = diagnosis.tool_evolution_runtime
        self.skill_tool_runtime = diagnosis.skill_tool_runtime
        self._refresh_actor()

        submission = SubmissionAgent(
            session_id=session_id,
            llm_backend=llm_backend,
            model=model,
        )
        asyncio.run(submission.load_tools())
        self.submission_agent = submission.get_agent()

        self.langfuse_callbacks = create_langfuse_callbacks()

        builder = StateGraph(ReflexionState)
        builder.add_node("attempt", self._attempt)
        builder.add_node("evaluator", self._evaluate)
        builder.add_node("reflect", self._reflect)
        builder.add_node("submission", self._submit)
        builder.add_edge(START, "attempt")
        builder.add_edge("attempt", "evaluator")
        builder.add_conditional_edges(
            "evaluator",
            self._route_after_evaluation,
            {
                "reflect": "reflect",
                "submission": "submission",
                "end": END,
            },
        )
        builder.add_edge("reflect", "attempt")
        builder.add_edge("submission", END)
        self.graph = builder.compile()

    def _refresh_actor(self) -> None:
        self.tool_evolution_runtime = self._diagnosis_phase.tool_evolution_runtime
        self.skill_tool_runtime = self._diagnosis_phase.skill_tool_runtime
        self.diagnosis_tool_names = [
            tool.name for tool in (self._diagnosis_phase.tools or [])
        ]
        self.actor = create_agent(
            model=self.llm,
            system_prompt=ACTOR_PROMPT,
            tools=self._diagnosis_phase.tools,
            name="ReflexionActor",
        )

    def _learning_prompt_suffix(self) -> str:
        suffix = self._diagnosis_phase.prompt_suffix()
        if not suffix:
            return ""
        return (
            "\n\nIntegrated learning context for this Reflexion attempt:\n"
            "Use the following Skill-Pro/DRAFT guidance to choose diagnostic "
            "checks and avoid repeating completed options. It is not evidence; "
            "only current tool outputs can support the final diagnosis."
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
        self._refresh_actor()

    def _callback(self, phase: str) -> AgentCallbackLogger:
        return AgentCallbackLogger(
            agent="diagnosis_agent",
            session_dir=self.session_dir,
            extra_fields={"phase": phase},
        )

    def _diagnosis_tool_names(self) -> list[str]:
        if getattr(self, "diagnosis_tool_names", None):
            return list(self.diagnosis_tool_names)
        phase = getattr(self, "_diagnosis_phase", None)
        return [tool.name for tool in getattr(phase, "tools", []) or []]

    def _current_tool_observations(
        self,
        state: ReflexionState,
    ) -> list[ToolObservation]:
        observations: list[ToolObservation] = []
        runtime = getattr(self, "skill_tool_runtime", None)
        if runtime is not None and hasattr(runtime, "snapshot"):
            observations.extend(observations_from_runtime_snapshot(runtime.snapshot()))
        for item in state.get("attempt_trace", []):
            if not isinstance(item, dict):
                continue
            message_type = str(item.get("type") or "")
            name = str(item.get("name") or "")
            content = str(item.get("content") or "")
            if "ToolMessage" not in message_type and not name:
                continue
            observations.append(
                ToolObservation(
                    tool=name,
                    summary=content,
                )
            )
        return observations

    def _evidence_gate(
        self,
        state: ReflexionState,
    ):
        return evaluate_fault_family_evidence(
            task_description=state.get("task_description", ""),
            diagnosis_report=state.get("attempt_report", ""),
            observations=self._current_tool_observations(state),
            available_tools=self._diagnosis_tool_names(),
        )

    def _is_evidence_gate_enabled(self) -> bool:
        return bool(getattr(self, "evidence_gate_enabled", True))

    def _route_after_evaluation(self, state: ReflexionState) -> str:
        if state.get("is_max_steps_reached"):
            return "end"
        evaluation = state.get("evaluation")
        if evaluation is not None and evaluation.success:
            return "submission"
        if state.get("attempt_count", 0) >= self.max_attempts:
            return "submission" if state.get("diagnosis_report", "").strip() else "end"
        return "reflect"

    @staticmethod
    def _compact_trace(messages: list[Any]) -> list[dict[str, Any]]:
        """Keep evaluator-visible trajectory evidence within a bounded context."""
        trace: list[dict[str, Any]] = []
        for message in messages[-24:]:
            content = strip_integrated_learning_guidance(
                getattr(message, "content", "")
            )
            entry: dict[str, Any] = {
                "type": message.__class__.__name__,
                "content": content[:2500],
            }
            name = getattr(message, "name", None)
            if name:
                entry["name"] = name
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                entry["tool_calls"] = tool_calls
            trace.append(entry)
        return trace

    async def run(self, task_description: str) -> dict[str, Any]:
        with langsmith_tracing_context(
            project_name=os.getenv("LANGSMITH_PROJECT", "NIKA"),
            metadata={
                "scenario": self.session.scenario_name,
                "problem": session_problem_label(self.session),
                "topo_size": self.session.scenario_topo_size,
                "model": getattr(self.session, "model", ""),
                "agent": "reflexion",
                "max_attempts": self.max_attempts,
            },
        ):
            try:
                try:
                    return await self.graph.ainvoke(
                        {
                            "task_description": task_description,
                            "attempt_count": 0,
                            "attempt_report": "",
                            "attempt_error": "",
                            "attempt_trace": [],
                            "diagnosis_report": "",
                            "is_max_steps_reached": False,
                            "best_score": -1.0,
                            "evaluation_failed": False,
                            "memories": [],
                        },
                        config={
                            **callback_config(self.langfuse_callbacks),
                            "recursion_limit": self.max_attempts * 4 + 4,
                        },
                    )
                except GraphRecursionError:
                    self._callback("workflow")._log(
                        "max_recursion_reached",
                        {
                            "message": (
                                "Reflexion workflow reached max recursion limit "
                                "before producing a submission."
                            )
                        },
                    )
                    return {
                        "task_description": task_description,
                        "diagnosis_report": "ERROR_MAX_STEPS_REACHED",
                        "is_max_steps_reached": True,
                        "attempt_error": "ERROR_MAX_STEPS_REACHED",
                        "messages": [
                            HumanMessage(
                                content=(
                                    "Error: reflexion did not finish within max steps."
                                )
                            )
                        ],
                    }
            finally:
                write_tool_evolution_session(
                    self.tool_evolution_runtime,
                    self.session_dir,
                )

    async def _attempt(self, state: ReflexionState) -> dict[str, Any]:
        if state.get("is_max_steps_reached"):
            return {}
        attempt_count = state.get("attempt_count", 0) + 1
        callback = self._callback(f"attempt_{attempt_count}")
        task_description = state.get("task_description", "")
        prompt = json.dumps(
            {
                "task": task_description,
                "attempt": attempt_count,
                "max_attempts": self.max_attempts,
                "tool_step_limit": self.max_steps,
                "episodic_memory": [
                    memory.model_dump() for memory in state.get("memories", [])
                ],
                "instruction": (
                    "Run a fresh investigation. Use the episodic memory to avoid "
                    "repeating failed strategies, verify every hypothesis with tools, "
                    "and return a complete report before the step limit."
                ),
            },
            ensure_ascii=False,
            default=str,
        )
        learning_context = self._learning_prompt_suffix()
        if learning_context:
            prompt += learning_context
        try:
            result = await self.actor.ainvoke(
                {"messages": [HumanMessage(content=prompt)]},
                config={
                    "callbacks": [callback],
                    "recursion_limit": self.max_steps,
                },
            )
            report = str(result["messages"][-1].content).strip()
            return {
                "attempt_count": attempt_count,
                "attempt_report": report,
                "attempt_error": "",
                "attempt_trace": self._compact_trace(result["messages"]),
                "evaluation": None,
                "evaluation_failed": False,
            }
        except GraphRecursionError:
            callback._log(
                "error",
                {
                    "message": (
                        f"Reflexion attempt {attempt_count} reached max recursion limit."
                    )
                },
            )
            return {
                "attempt_count": attempt_count,
                "attempt_report": "",
                "attempt_error": "ERROR_MAX_STEPS_REACHED",
                "attempt_trace": [],
                "diagnosis_report": "ERROR_MAX_STEPS_REACHED",
                "is_max_steps_reached": True,
                "evaluation": None,
                "evaluation_failed": False,
            }
        except Exception as exc:
            error = str(exc).strip() or exc.__class__.__name__
            callback._log(
                "error",
                {"message": f"Reflexion attempt {attempt_count} failed: {error}"},
            )
            return {
                "attempt_count": attempt_count,
                "attempt_report": "",
                "attempt_error": error,
                "attempt_trace": [],
                "evaluation": None,
                "evaluation_failed": False,
            }

    async def _evaluate(self, state: ReflexionState) -> dict[str, Any]:
        if state.get("is_max_steps_reached"):
            return {"evaluation": None, "evaluation_failed": True}
        attempt_count = state.get("attempt_count", 0)
        callback = self._callback(f"evaluator_{attempt_count}")
        task_description = state.get("task_description", "")
        payload = json.dumps(
            {
                "task": task_description,
                "attempt": attempt_count,
                "attempt_report": state.get("attempt_report", ""),
                "attempt_error": state.get("attempt_error", ""),
                "tool_trajectory": state.get("attempt_trace", []),
            },
            ensure_ascii=False,
            default=str,
        )
        try:
            raw_evaluation = await self.evaluator.ainvoke(
                [
                    SystemMessage(content=EVALUATOR_PROMPT),
                    HumanMessage(content=payload),
                ],
                config={"callbacks": [callback]},
            )
            evaluation = ReflexionEvaluation.model_validate(raw_evaluation)
            gate = (
                self._evidence_gate(state)
                if self._is_evidence_gate_enabled()
                else None
            )
            if gate is not None and not gate.sufficient:
                callback._log("evidence_gate_blocked", gate.to_log_payload())
                evaluation = ReflexionEvaluation(
                    success=False,
                    quality_score=min(evaluation.quality_score, 0.79),
                    evidence_sufficient=False,
                    anomaly_assessment=evaluation.anomaly_assessment,
                    localization_assessment=evaluation.localization_assessment,
                    root_cause_assessment=evaluation.root_cause_assessment,
                    contradictions=list(evaluation.contradictions),
                    missing_evidence=[
                        *evaluation.missing_evidence,
                        *gate.missing_evidence,
                    ],
                    failure_reasons=[
                        *evaluation.failure_reasons,
                        "Fault-family evidence gate did not pass.",
                    ],
                )
            update: dict[str, Any] = {
                "evaluation": evaluation,
                "evaluation_failed": False,
            }
            report = state.get("attempt_report", "").strip()
            best_score = state.get("best_score", -1.0)
            if report and (
                evaluation.success or evaluation.quality_score >= best_score
            ):
                update["diagnosis_report"] = report
                update["best_score"] = evaluation.quality_score
            return update
        except Exception as exc:
            callback._log(
                "error",
                {"message": f"Evaluator failed on attempt {attempt_count}: {exc}"},
            )
            update = {"evaluation_failed": True}
            report = state.get("attempt_report", "").strip()
            if report and not state.get("diagnosis_report", "").strip():
                update["diagnosis_report"] = report
                update["best_score"] = 0.0
            return update

    async def _reflect(self, state: ReflexionState) -> dict[str, Any]:
        attempt_count = state.get("attempt_count", 0)
        callback = self._callback(f"reflexion_{attempt_count}")
        evaluation = state.get("evaluation")
        task_description = state.get("task_description", "")
        payload = json.dumps(
            {
                "task": task_description,
                "attempt": attempt_count,
                "attempt_report": state.get("attempt_report", ""),
                "attempt_error": state.get("attempt_error", ""),
                "tool_trajectory": state.get("attempt_trace", []),
                "evaluation": evaluation.model_dump() if evaluation else None,
                "existing_memory": [
                    memory.model_dump() for memory in state.get("memories", [])
                ],
            },
            ensure_ascii=False,
            default=str,
        )
        try:
            raw_memory = await self.reflector.ainvoke(
                [
                    SystemMessage(content=REFLEXION_PROMPT),
                    HumanMessage(content=payload),
                ],
                config={"callbacks": [callback]},
            )
            memory = ReflexionMemory.model_validate(raw_memory)
        except Exception as exc:
            callback._log(
                "error",
                {
                    "message": (
                        f"Reflexion memory generation failed on attempt {attempt_count}; "
                        f"no memory was appended: {exc}"
                    )
                },
            )
            return {"memories": state.get("memories", [])}
        return {"memories": [*state.get("memories", []), memory]}

    async def _submit(self, state: ReflexionState) -> dict[str, Any]:
        report = state.get("diagnosis_report", "").strip()
        if not report:
            self._callback("submission")._log(
                "error",
                {
                    "message": "Submission skipped because no valid diagnosis report is available."
                },
            )
            return {}
        try:
            result = await self.submission_agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content=(
                                f"Based on the diagnosis report: {report}, please provide "
                                "the submission. Do not submit if no report is available."
                            )
                        )
                    ]
                },
                config={
                    "callbacks": [
                        AgentCallbackLogger(
                            agent="submission_agent",
                            session_dir=self.session_dir,
                            extra_fields={"phase": "submission"},
                        )
                    ],
                    "recursion_limit": self.max_steps,
                },
            )
            return {"messages": result["messages"]}
        except GraphRecursionError:
            self._callback("submission")._log(
                "error",
                {"message": "Submission agent reached max recursion limit."},
            )
            return {}
        except Exception as exc:
            self._callback("submission")._log(
                "error",
                {"message": f"Submission agent failed: {exc}"},
            )
            return {}
