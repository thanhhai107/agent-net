"""Iterative Reflexion workflow for tool-assisted network diagnosis."""

import asyncio
import json
import logging
import os
from typing import Any

import langsmith as ls
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from agent.langgraph.domain_agents.diagnosis_agent import DiagnosisAgent
from agent.langgraph.domain_agents.submission_agent import SubmissionAgent
from agent.langgraph.workflow_models import ReflexionEvaluation, ReflexionMemory
from agent.llm.model_factory import load_model
from agent.utils.loggers import AgentCallbackLogger
from nika.utils.logger import system_logger
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
"""

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
"""

REFLEXION_PROMPT = """\
You are the self-review component of a Reflexion agent. Given a failed
troubleshooting attempt and its evaluator feedback, produce compact episodic
memory for the next attempt. Identify what reasoning or tool-selection mistakes
caused failure, what evidence is still needed, which hypotheses need verification,
and a materially different next strategy.

Do not solve the task, call tools, or repeat the full report. The memory must be
actionable and must not promote unverified conclusions to facts.
"""


class ReflexionState(TypedDict, total=False):
    task_description: str
    attempt_count: int
    attempt_report: str
    attempt_error: str
    attempt_trace: list[dict[str, Any]]
    diagnosis_report: str
    best_score: float
    evaluation: ReflexionEvaluation | None
    evaluation_failed: bool
    memories: list[ReflexionMemory]


class ReflexionAgent:
    """Attempt → evaluate → reflect → retry with persistent episodic memory."""

    def __init__(
        self,
        session_id: str,
        llm_backend: str = "openai",
        model: str = "gpt-5-mini",
        max_steps: int = 20,
        max_attempts: int = 3,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

        self.session_id = session_id
        self.max_steps = max_steps
        self.max_attempts = max_attempts
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
            problem_names=self.session.problem_names,
        )
        asyncio.run(diagnosis.load_tools())
        self.actor = create_agent(
            model=self.llm,
            system_prompt=ACTOR_PROMPT,
            tools=diagnosis.tools,
            name="ReflexionActor",
        )

        submission = SubmissionAgent(
            session_id=session_id,
            llm_backend=llm_backend,
            model=model,
        )
        asyncio.run(submission.load_tools())
        self.submission_agent = submission.get_agent()

        self.langfuse_handler = CallbackHandler()
        if get_client().auth_check():
            system_logger.info("Authentication to Langfuse successful.")
        else:
            system_logger.warning(
                "Authentication to Langfuse failed. Please check your LANGFUSE_API_KEY."
            )

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

    def _callback(self, phase: str) -> AgentCallbackLogger:
        return AgentCallbackLogger(
            agent="diagnosis_agent",
            session_dir=self.session_dir,
            extra_fields={"phase": phase},
        )

    def _route_after_evaluation(self, state: ReflexionState) -> str:
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
            content = str(getattr(message, "content", ""))
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
        with ls.tracing_context(
            project_name=os.getenv("LANGSMITH_PROJECT", "NIKA"),
            metadata={
                "scenario": self.session.scenario_name,
                "problem": self.session.problem_names[0],
                "topo_size": self.session.scenario_topo_size,
                "model": self.session.model,
                "agent": "reflexion",
                "max_attempts": self.max_attempts,
            },
        ):
            return await self.graph.ainvoke(
                {
                    "task_description": task_description,
                    "attempt_count": 0,
                    "attempt_report": "",
                    "attempt_error": "",
                    "attempt_trace": [],
                    "diagnosis_report": "",
                    "best_score": -1.0,
                    "evaluation_failed": False,
                    "memories": [],
                },
                config={
                    "callbacks": [self.langfuse_handler],
                    "recursion_limit": self.max_attempts * 4 + 4,
                },
            )

    async def _attempt(self, state: ReflexionState) -> dict[str, Any]:
        attempt_count = state.get("attempt_count", 0) + 1
        callback = self._callback(f"attempt_{attempt_count}")
        prompt = json.dumps(
            {
                "task": state["task_description"],
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
        attempt_count = state["attempt_count"]
        callback = self._callback(f"evaluator_{attempt_count}")
        payload = json.dumps(
            {
                "task": state["task_description"],
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
            update: dict[str, Any] = {
                "evaluation": evaluation,
                "evaluation_failed": False,
            }
            report = state.get("attempt_report", "").strip()
            best_score = state.get("best_score", -1.0)
            if report and (evaluation.success or evaluation.quality_score >= best_score):
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

    def _fallback_memory(self, state: ReflexionState) -> ReflexionMemory:
        evaluation = state.get("evaluation")
        if evaluation is not None:
            feedback = (
                evaluation.failure_reasons
                + evaluation.contradictions
                + evaluation.missing_evidence
            )
        else:
            feedback = []
        attempt_error = state.get("attempt_error", "").strip()
        if attempt_error:
            feedback.append(attempt_error)
        if not feedback:
            feedback.append("The prior attempt could not be reliably evaluated.")
        return ReflexionMemory(
            summary="The previous attempt did not produce a verified complete diagnosis.",
            lessons=feedback,
            next_strategy=[
                "Start from the strongest direct anomaly signal.",
                "Use fault-specific tools and finish with an evidence-backed report.",
            ],
            evidence_to_collect=(
                evaluation.missing_evidence if evaluation is not None else []
            ),
            avoid_repeating=[
                "Do not infer whole-network health from one healthy subsystem."
            ],
        )

    async def _reflect(self, state: ReflexionState) -> dict[str, Any]:
        attempt_count = state["attempt_count"]
        callback = self._callback(f"reflexion_{attempt_count}")
        evaluation = state.get("evaluation")
        payload = json.dumps(
            {
                "task": state["task_description"],
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
                        f"using deterministic fallback: {exc}"
                    )
                },
            )
            memory = self._fallback_memory(state)
        return {"memories": [*state.get("memories", []), memory]}

    async def _submit(self, state: ReflexionState) -> dict[str, Any]:
        report = state.get("diagnosis_report", "").strip()
        if not report:
            self._callback("submission")._log(
                "error",
                {"message": "Submission skipped because no valid diagnosis report is available."},
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
