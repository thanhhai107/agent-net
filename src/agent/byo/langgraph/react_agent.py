import asyncio
import logging
import os

import langsmith as ls
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from pydantic import Field, ValidationError
from typing_extensions import TypedDict

from agent.byo.langgraph.phases.diagnosis import DiagnosisPhase
from agent.byo.langgraph.phases.submission import SubmissionPhase
from agent.utils.loggers import AgentCallbackLogger, MessageLogger
from agent.utils.phases import DIAGNOSIS, SUBMISSION
from nika.utils.logger import system_logger
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
        llm_provider: str = "openai",
        model: str = "gpt-5-mini",
        max_steps: int = 20,
    ):
        self.session_id = session_id
        self.max_steps = max_steps
        self.session = Session()
        self.session.load_running_session(session_id=session_id)
        self.session_dir = self.session.session_dir

        # Set up Langfuse callback handler
        # Initialize Langfuse client
        langfuse = get_client()

        # Initialize Langfuse CallbackHandler for Langchain (tracing)
        self.langfuse_handler = CallbackHandler()

        if langfuse.auth_check():
            system_logger.info("Authentication to Langfuse successful.")
        else:
            system_logger.warning(
                "Authentication to Langfuse failed. Please check your LANGFUSE_API_KEY."
            )

        diagnosis_phase = DiagnosisPhase(
            session_id=session_id,
            llm_provider=llm_provider,
            model=model,
            scenario_name=self.session.scenario_name,
            problem_names=self.session.problem_names,
        )
        asyncio.run(diagnosis_phase.load_tools())
        self._diagnosis_runner = diagnosis_phase.get_agent()

        submission_phase = SubmissionPhase(
            session_id=session_id, llm_provider=llm_provider, model=model
        )
        asyncio.run(submission_phase.load_tools())
        self._submission_runner = submission_phase.get_agent()

        worker_builder = StateGraph(AgentState)
        worker_builder.add_node(DIAGNOSIS, self._run_diagnosis)
        worker_builder.add_node(SUBMISSION, self._run_submission)

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

        # compile the graph
        self.graph = worker_builder.compile()

    async def run(self, task_description: str):
        with ls.tracing_context(
            project_name=os.getenv("LANGSMITH_PROJECT", "NIKA"),
            metadata={
                "scenario": self.session.scenario_name,
                "problem": self.session.problem_names[0],
                "topo_size": self.session.scenario_topo_size,
                "model": self.session.model,
            },
        ):
            result = await self.graph.ainvoke(
                {
                    "messages": [HumanMessage(content=task_description)],
                },
                config={"callbacks": [self.langfuse_handler]},
            )
            return result

    async def _run_diagnosis(self, state: AgentState):
        try:
            cb = AgentCallbackLogger(agent=DIAGNOSIS, session_dir=self.session_dir)
            diagnosis_report = await self._diagnosis_runner.ainvoke(
                {"messages": state["messages"]},
                config={
                    "callbacks": [cb],
                    "recursion_limit": self.max_steps,
                },
                debug=True,
            )
            return {
                "diagnosis_report": [diagnosis_report["messages"][-1].content],
                "is_max_steps_reached": False,
            }
        except ValidationError as e:
            MessageLogger(agent=DIAGNOSIS, session_dir=self.session_dir).log(
                "error", {"message": f"Validation error: {e}"}
            )
            return {
                "messages": [HumanMessage(content=f"Error: {e}")],
                "diagnosis_report": ["ERROR_VALIDATION"],
                "is_max_steps_reached": False,
            }
        except GraphRecursionError:
            MessageLogger(agent=DIAGNOSIS, session_dir=self.session_dir).log(
                "error",
                {"message": "Diagnosis phase reached max recursion limit."},
            )
            return {
                "messages": [
                    HumanMessage(
                        content="Error: diagnosis did not finish within max steps."
                    )
                ],
                "diagnosis_report": ["ERROR_MAX_STEPS_REACHED"],
                "is_max_steps_reached": True,
            }

    async def _run_submission(self, state: AgentState):
        diag_text = state["diagnosis_report"][-1]
        result = await self._submission_runner.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=f"Based on the diagnosis report: {diag_text}, please provide the submission. Do not submit if no report available."
                    ),
                ]
            },
            config={
                "callbacks": [
                    AgentCallbackLogger(agent=SUBMISSION, session_dir=self.session_dir)
                ],
                "recursion_limit": self.max_steps,
            },
            debug=True,
        )
        return {
            "messages": result["messages"],
        }
