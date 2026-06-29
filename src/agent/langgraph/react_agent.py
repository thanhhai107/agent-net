import asyncio
import logging
import os

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from pydantic import Field, ValidationError
from typing_extensions import TypedDict

from agent.defaults import DEFAULT_MAX_STEPS
from agent.langgraph.domain_agents.diagnosis_agent import DiagnosisAgent
from agent.langgraph.domain_agents.submission_agent import SubmissionAgent
from agent.llm.model_factory import DEFAULT_LLM_BACKEND, DEFAULT_MODEL
from agent.tool_evolution.integration import write_tool_evolution_session
from agent.utils.loggers import AgentCallbackLogger
from agent.utils.tracing import langsmith_tracing_context
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
        llm_backend: str = DEFAULT_LLM_BACKEND,
        model: str = DEFAULT_MODEL,
        max_steps: int = DEFAULT_MAX_STEPS,
        oracle_routing: bool = False,
        tool_evolution_enabled: bool = False,
        tool_library_id: str = "default",
        tool_evolution_mode: str = "dual",
        use_problem_tool_hints: bool = True,
        policy_overlay_path: str | None = None,
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

        # load agent and tools
        diagnosis_agent = DiagnosisAgent(
            session_id=session_id,
            llm_backend=llm_backend,
            model=model,
            scenario_name=self.session.scenario_name,
            problem_names=(
                self.session.problem_names if use_problem_tool_hints else []
            ),
            oracle_routing=oracle_routing,
            load_all_tools=not use_problem_tool_hints,
            tool_evolution_enabled=tool_evolution_enabled,
            tool_library_id=tool_library_id,
            tool_evolution_mode=tool_evolution_mode,
            policy_overlay_path=policy_overlay_path,
        )
        asyncio.run(diagnosis_agent.load_tools())
        self.llm = diagnosis_agent.llm
        self.tool_evolution_runtime = diagnosis_agent.tool_evolution_runtime
        self.diagnosis_tool_names = [
            tool.name for tool in (diagnosis_agent.tools or [])
        ]
        self.diagnosis_agent = diagnosis_agent.get_agent()

        submission_agent = SubmissionAgent(
            session_id=session_id, llm_backend=llm_backend, model=model
        )
        asyncio.run(submission_agent.load_tools())
        self.submission_agent = submission_agent.get_agent()

        # build the state graph
        worker_builder = StateGraph(AgentState)
        worker_builder.add_node("diagnosis_agent", self.diagnosis_agent_builder)
        worker_builder.add_node("submission_agent", self.submission_agent_builder)

        worker_builder.add_edge(START, "diagnosis_agent")
        worker_builder.add_conditional_edges(
            "diagnosis_agent",
            lambda state: state.get("is_max_steps_reached", False),
            {
                True: END,
                False: "submission_agent",
            },
        )

        worker_builder.add_edge("submission_agent", END)

        # compile the graph
        self.graph = worker_builder.compile()

    async def run(self, task_description: str):
        with langsmith_tracing_context(
            project_name=os.getenv("LANGSMITH_PROJECT", "NIKA"),
            metadata={
                "scenario": self.session.scenario_name,
                "problem": self.session.problem_names[0],
                "topo_size": self.session.scenario_topo_size,
                "model": self.session.model,
            },
        ):
            try:
                return await self.graph.ainvoke(
                    {
                        "messages": [HumanMessage(content=task_description)],
                    },
                    config={"callbacks": [self.langfuse_handler]},
                )
            finally:
                write_tool_evolution_session(
                    self.tool_evolution_runtime,
                    self.session_dir,
                )

    async def diagnosis_agent_builder(self, state: AgentState):
        try:
            cb = AgentCallbackLogger(
                agent="diagnosis_agent", session_dir=self.session_dir
            )
            diagnosis_report = await self.diagnosis_agent.ainvoke(
                {"messages": state["messages"]},
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
                agent="diagnosis_agent", session_dir=self.session_dir
            )._log("error", {"message": f"Validation error: {e}"})
            return {
                "messages": [HumanMessage(content=f"Error: {e}")],
                "diagnosis_report": "ERROR_VALIDATION",
                "is_max_steps_reached": False,
            }
        except GraphRecursionError:
            AgentCallbackLogger(
                agent="diagnosis_agent", session_dir=self.session_dir
            )._log(
                "error",
                {"message": "Diagnosis agent reached max recursion limit."},
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
                        agent="submission_agent", session_dir=self.session_dir
                    )
                ],
                "recursion_limit": self.max_steps,
            },
            debug=True,
        )
        return {
            "messages": result["messages"],
        }
