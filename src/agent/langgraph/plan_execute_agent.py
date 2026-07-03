"""Plan-and-execute troubleshooting workflow."""

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
from agent.langgraph.workflow_models import (
    InvestigationPlan,
    PlanStep,
    ReplanDecision,
    StepResult,
)
from agent.llm.model_factory import DEFAULT_LLM_BACKEND, DEFAULT_MODEL, load_model
from agent.memory.runtime import strip_integrated_learning_guidance
from agent.tool_evolution.integration import write_tool_evolution_session
from agent.utils.loggers import AgentCallbackLogger
from agent.utils.template import EVIDENCE_CONTRACT_PROMPT
from agent.utils.tracing import langsmith_tracing_context
from nika.utils.session import Session

load_dotenv()
logging.basicConfig(level=logging.INFO)

PLANNER_PROMPT = """\
You are a network troubleshooting planner. Create a concise, ordered investigation
plan for the task. Each step must be independently executable with network
diagnostic tools and must name the evidence it expects to collect. Do not diagnose
the fault yet and do not propose mitigation.
""" + "\n\n" + EVIDENCE_CONTRACT_PROMPT

EXECUTOR_PROMPT = """\
You are a network troubleshooting executor. Perform only the assigned investigation
step using the available tools. Report the commands or checks performed, the
observations, and what those observations imply. Do not submit a final answer and
do not perform unrelated plan steps.
""" + "\n\n" + EVIDENCE_CONTRACT_PROMPT

REPLANNER_PROMPT = """\
You are coordinating a network investigation. Review the original objective,
completed evidence, and remaining plan. If anomaly detection, faulty-device
localization, and root-cause identification are sufficiently supported, finish with
a concise diagnosis report. Otherwise return a revised ordered list of only the
remaining investigation steps. Do not include already completed work.
""" + "\n\n" + EVIDENCE_CONTRACT_PROMPT

SYNTHESIS_PROMPT = """\
You are a network troubleshooting expert. Produce the best possible final diagnosis
from the collected evidence. Explicitly state whether an anomaly exists, the faulty
devices, the likely root cause, and the supporting evidence. Acknowledge uncertainty
when evidence is incomplete. Do not propose mitigation.
""" + "\n\n" + EVIDENCE_CONTRACT_PROMPT


class PlanExecuteState(TypedDict, total=False):
    task_description: str
    objective: str
    plan: list[PlanStep]
    completed_steps: list[StepResult]
    executed_steps: int
    diagnosis_report: str
    planning_failed: bool
    is_max_steps_reached: bool
    messages: list[Any]


class PlanExecuteAgent:
    """Planner → executor → replanner troubleshooting workflow."""

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
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")

        self.session_id = session_id
        self.max_steps = max_steps
        self.session = Session()
        self.session.load_running_session(session_id=session_id)
        self.session_dir = self.session.session_dir
        self.llm = load_model(llm_backend=llm_backend, model=model)
        self.planner = self.llm.with_structured_output(InvestigationPlan)
        self.replanner = self.llm.with_structured_output(ReplanDecision)

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
        self._refresh_executor()

        submission = SubmissionAgent(
            session_id=session_id,
            llm_backend=llm_backend,
            model=model,
        )
        asyncio.run(submission.load_tools())
        self.submission_agent = submission.get_agent()

        self.langfuse_callbacks = create_langfuse_callbacks()

        builder = StateGraph(PlanExecuteState)
        builder.add_node("planner", self._plan)
        builder.add_node("executor", self._execute)
        builder.add_node("replanner", self._replan)
        builder.add_node("submission", self._submit)
        builder.add_edge(START, "planner")
        builder.add_conditional_edges(
            "planner",
            self._route_after_plan,
            {"executor": "executor", "end": END},
        )
        builder.add_edge("executor", "replanner")
        builder.add_conditional_edges(
            "replanner",
            self._route_after_replan,
            {
                "executor": "executor",
                "submission": "submission",
                "end": END,
            },
        )
        builder.add_edge("submission", END)
        self.graph = builder.compile()

    def _refresh_executor(self) -> None:
        self.tool_evolution_runtime = self._diagnosis_phase.tool_evolution_runtime
        self.skill_tool_runtime = self._diagnosis_phase.skill_tool_runtime
        self.diagnosis_tool_names = [
            tool.name for tool in (self._diagnosis_phase.tools or [])
        ]
        self.executor = create_agent(
            model=self.llm,
            system_prompt=EXECUTOR_PROMPT,
            tools=self._diagnosis_phase.tools,
            name="PlanExecutor",
        )

    def _learning_prompt_suffix(self, *, activate_skill: bool = True) -> str:
        suffix = self._diagnosis_phase.prompt_suffix(activate_skill=activate_skill)
        if not suffix:
            return ""
        return (
            "\n\nIntegrated learning context for this workflow:\n"
            "Use the following Skill-Pro/DRAFT guidance to choose and sequence "
            "diagnostic checks. It is not evidence; only current tool outputs "
            "can support the final diagnosis."
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
        self._refresh_executor()

    def _callback(self, phase: str) -> AgentCallbackLogger:
        return AgentCallbackLogger(
            agent="diagnosis_agent",
            session_dir=self.session_dir,
            extra_fields={"phase": phase},
        )

    async def run(self, task_description: str) -> dict[str, Any]:
        with langsmith_tracing_context(
            project_name=os.getenv("LANGSMITH_PROJECT", "NIKA"),
            metadata={
                "scenario": self.session.scenario_name,
                "problem": self.session.problem_names[0],
                "topo_size": self.session.scenario_topo_size,
                "model": self.session.model,
                "agent": "plan-execute",
            },
        ):
            try:
                return await self.graph.ainvoke(
                    {
                        "task_description": task_description,
                        "plan": [],
                        "completed_steps": [],
                        "executed_steps": 0,
                        "diagnosis_report": "",
                        "planning_failed": False,
                        "is_max_steps_reached": False,
                        "messages": [HumanMessage(content=task_description)],
                    },
                    config={
                        **callback_config(self.langfuse_callbacks),
                        "recursion_limit": self.max_steps * 3 + 10,
                    },
                )
            finally:
                write_tool_evolution_session(
                    self.tool_evolution_runtime,
                    self.session_dir,
                )

    async def _plan(self, state: PlanExecuteState) -> dict[str, Any]:
        callback = self._callback("planner")
        try:
            raw_plan = await self.planner.ainvoke(
                [
                    SystemMessage(
                        content=PLANNER_PROMPT
                        + self._learning_prompt_suffix(activate_skill=False)
                    ),
                    HumanMessage(content=state["task_description"]),
                ],
                config={"callbacks": [callback]},
            )
            plan = InvestigationPlan.model_validate(raw_plan)
            return {
                "objective": plan.objective,
                "plan": plan.steps,
                "planning_failed": False,
            }
        except Exception as exc:
            callback._log("error", {"message": f"Planner failed: {exc}"})
            return {"plan": [], "planning_failed": True}

    @staticmethod
    def _route_after_plan(state: PlanExecuteState) -> str:
        return (
            "executor"
            if state.get("plan") and not state.get("planning_failed")
            else "end"
        )

    async def _execute(self, state: PlanExecuteState) -> dict[str, Any]:
        step = state["plan"][0]
        callback = self._callback("executor")
        prior_evidence = [
            item.model_dump() for item in state.get("completed_steps", [])
        ]
        prompt = (
            f"Original task: {state['task_description']}\n"
            f"Investigation objective: {state['objective']}\n"
            f"Step ID: {step.step_id}\n"
            f"Action: {step.action}\n"
            f"Expected evidence: {step.expected_evidence}\n"
            f"Evidence from completed steps: "
            f"{json.dumps(prior_evidence, ensure_ascii=False)}"
        )
        learning_context = self._learning_prompt_suffix(activate_skill=True)
        if learning_context:
            prompt += (
                "\n\nCurrent integrated learning context for this execution step:"
                f"{learning_context}"
            )
        try:
            result = await self.executor.ainvoke(
                {"messages": [HumanMessage(content=prompt)]},
                config={
                    "callbacks": [callback],
                    "recursion_limit": self.max_steps,
                },
            )
            observation = strip_integrated_learning_guidance(
                result["messages"][-1].content
            )
            step_result = StepResult(step=step, observation=observation)
        except GraphRecursionError:
            callback._log(
                "error",
                {"message": f"Executor reached max recursion limit for {step.step_id}."},
            )
            return {
                "plan": state["plan"][1:],
                "completed_steps": state.get("completed_steps", []),
                "executed_steps": state.get("executed_steps", 0),
                "diagnosis_report": "ERROR_MAX_STEPS_REACHED",
                "is_max_steps_reached": True,
            }
        except Exception as exc:
            callback._log(
                "error",
                {"message": f"Executor failed for {step.step_id}: {exc}"},
            )
            step_result = StepResult(
                step=step,
                observation=f"ERROR: {exc}",
                succeeded=False,
            )

        return {
            "plan": state["plan"][1:],
            "completed_steps": [*state.get("completed_steps", []), step_result],
            "executed_steps": state.get("executed_steps", 0) + 1,
        }

    async def _replan(self, state: PlanExecuteState) -> dict[str, Any]:
        callback = self._callback("replanner")
        evidence = [item.model_dump() for item in state.get("completed_steps", [])]
        remaining = [item.model_dump() for item in state.get("plan", [])]
        prompt = json.dumps(
            {
                "objective": state["objective"],
                "task": state["task_description"],
                "completed_evidence": evidence,
                "remaining_plan": remaining,
            },
            ensure_ascii=False,
        )
        try:
            raw_decision = await self.replanner.ainvoke(
                [
                    SystemMessage(
                        content=REPLANNER_PROMPT
                        + self._learning_prompt_suffix(activate_skill=False)
                    ),
                    HumanMessage(content=prompt),
                ],
                config={"callbacks": [callback]},
            )
            decision = ReplanDecision.model_validate(raw_decision)
            if decision.completed:
                return {
                    "diagnosis_report": decision.diagnosis_report,
                    "plan": [],
                }
            return {"plan": decision.remaining_steps}
        except Exception as exc:
            callback._log("error", {"message": f"Replanner failed: {exc}"})
            return {"plan": state.get("plan", [])}

    def _route_after_replan(self, state: PlanExecuteState) -> str:
        if state.get("is_max_steps_reached"):
            return "end"
        if state.get("diagnosis_report", "").strip():
            return "submission"
        if state.get("executed_steps", 0) >= self.max_steps:
            return "end"
        if state.get("plan"):
            return "executor"
        return "end"

    async def _submit(self, state: PlanExecuteState) -> dict[str, Any]:
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
