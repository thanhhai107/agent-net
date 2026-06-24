"""Diagnosis workflow with a single critique-and-revision pass."""

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
from agent.langgraph.workflow_models import DiagnosisCritique
from agent.llm.model_factory import load_model
from agent.utils.loggers import AgentCallbackLogger
from nika.utils.logger import system_logger
from nika.utils.session import Session

load_dotenv()
logging.basicConfig(level=logging.INFO)

CRITIC_PROMPT = """\
You are a rigorous reviewer of a network diagnosis. Evaluate whether the report's
anomaly decision, faulty-device localization, and root-cause conclusion are
supported by concrete observations. Identify contradictions and missing evidence,
then provide actionable revision instructions. Do not call tools and do not rewrite
the report.
"""

REVISER_PROMPT = """\
You are revising a network diagnosis after expert critique. Use the available tools
when the critique identifies missing or weak evidence. Return one final diagnosis
that explicitly states anomaly status, faulty devices, root cause, supporting
evidence, and any remaining uncertainty. Do not propose mitigation.
"""


class ReflectionState(TypedDict, total=False):
    task_description: str
    messages: list[Any]
    initial_report: str
    diagnosis_report: str
    critique: DiagnosisCritique
    critic_failed: bool


class ReflectionAgent:
    """ReAct diagnosis → structured critic → tool-enabled reviser."""

    def __init__(
        self,
        session_id: str,
        llm_backend: str = "openai",
        model: str = "gpt-5-mini",
        max_steps: int = 20,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")

        self.session_id = session_id
        self.max_steps = max_steps
        self.session = Session()
        self.session.load_running_session(session_id=session_id)
        self.session_dir = self.session.session_dir
        self.llm = load_model(llm_backend=llm_backend, model=model)
        self.critic = self.llm.with_structured_output(DiagnosisCritique)

        diagnosis = DiagnosisAgent(
            session_id=session_id,
            llm_backend=llm_backend,
            model=model,
            scenario_name=self.session.scenario_name,
            problem_names=self.session.problem_names,
        )
        asyncio.run(diagnosis.load_tools())
        self.diagnosis_agent = diagnosis.get_agent()
        self.reviser = create_agent(
            model=self.llm,
            system_prompt=REVISER_PROMPT,
            tools=diagnosis.tools,
            name="DiagnosisReviser",
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

        builder = StateGraph(ReflectionState)
        builder.add_node("diagnosis", self._diagnose)
        builder.add_node("critic", self._critique)
        builder.add_node("reviser", self._revise)
        builder.add_node("submission", self._submit)
        builder.add_edge(START, "diagnosis")
        builder.add_conditional_edges(
            "diagnosis",
            self._route_after_diagnosis,
            {"critic": "critic", "end": END},
        )
        builder.add_conditional_edges(
            "critic",
            self._route_after_critique,
            {"submission": "submission", "reviser": "reviser"},
        )
        builder.add_edge("reviser", "submission")
        builder.add_edge("submission", END)
        self.graph = builder.compile()

    def _callback(self, phase: str) -> AgentCallbackLogger:
        return AgentCallbackLogger(
            agent="diagnosis_agent",
            session_dir=self.session_dir,
            extra_fields={"phase": phase},
        )

    @staticmethod
    def _route_after_diagnosis(state: ReflectionState) -> str:
        return "critic" if state.get("initial_report", "").strip() else "end"

    @staticmethod
    def _route_after_critique(state: ReflectionState) -> str:
        return "submission" if state.get("critic_failed") else "reviser"

    async def run(self, task_description: str) -> dict[str, Any]:
        with ls.tracing_context(
            project_name=os.getenv("LANGSMITH_PROJECT", "NIKA"),
            metadata={
                "scenario": self.session.scenario_name,
                "problem": self.session.problem_names[0],
                "topo_size": self.session.scenario_topo_size,
                "model": self.session.model,
                "agent": "reflection",
            },
        ):
            return await self.graph.ainvoke(
                {
                    "task_description": task_description,
                    "messages": [HumanMessage(content=task_description)],
                    "initial_report": "",
                    "diagnosis_report": "",
                    "critic_failed": False,
                },
                config={"callbacks": [self.langfuse_handler]},
            )

    async def _diagnose(self, state: ReflectionState) -> dict[str, Any]:
        callback = self._callback("diagnosis")
        try:
            result = await self.diagnosis_agent.ainvoke(
                {"messages": state["messages"]},
                config={
                    "callbacks": [callback],
                    "recursion_limit": self.max_steps,
                },
            )
            report = str(result["messages"][-1].content)
            return {"initial_report": report, "diagnosis_report": report}
        except Exception as exc:
            callback._log("error", {"message": f"Initial diagnosis failed: {exc}"})
            return {"initial_report": "", "diagnosis_report": ""}

    async def _critique(self, state: ReflectionState) -> dict[str, Any]:
        callback = self._callback("critic")
        try:
            raw_critique = await self.critic.ainvoke(
                [
                    SystemMessage(content=CRITIC_PROMPT),
                    HumanMessage(
                        content=(
                            f"Task:\n{state['task_description']}\n\n"
                            f"Diagnosis report:\n{state['initial_report']}"
                        )
                    ),
                ],
                config={"callbacks": [callback]},
            )
            critique = DiagnosisCritique.model_validate(raw_critique)
            return {"critique": critique, "critic_failed": False}
        except Exception as exc:
            callback._log("error", {"message": f"Critic failed: {exc}"})
            return {
                "diagnosis_report": state["initial_report"],
                "critic_failed": True,
            }

    async def _revise(self, state: ReflectionState) -> dict[str, Any]:
        callback = self._callback("reviser")
        prompt = json.dumps(
            {
                "task": state["task_description"],
                "initial_report": state["initial_report"],
                "critique": state["critique"].model_dump(),
            },
            ensure_ascii=False,
        )
        try:
            result = await self.reviser.ainvoke(
                {"messages": [HumanMessage(content=prompt)]},
                config={
                    "callbacks": [callback],
                    "recursion_limit": self.max_steps,
                },
            )
            return {"diagnosis_report": str(result["messages"][-1].content)}
        except Exception as exc:
            callback._log(
                "error",
                {"message": f"Reviser failed; using initial report: {exc}"},
            )
            return {"diagnosis_report": state["initial_report"]}

    async def _submit(self, state: ReflectionState) -> dict[str, Any]:
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
