from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.llm.model_factory import DEFAULT_LLM_BACKEND, DEFAULT_MODEL, load_model
from agent.tool_evolution.models import ToolEvolutionMode
from agent.tool_evolution.runtime import ToolEvolutionRuntime
from agent.utils.mcp_servers import MCPServerConfig, select_diagnosis_servers
from nika.utils.session import Session

load_dotenv()

OVERALL_DIAGNOSIS_PROMPT = """\
    You are a network troubleshooting expert.
    Focus on (1) detecting if there is an anomaly, (2) localizing the faulty devices, and (3) identifying the root cause.

    Basic requirements:
    - Use the provided tools to gather necessary information.
    - Do not provide mitigation unless explicitly required.
"""


class DiagnosisAgent:
    """An agent that performs the total process of network diagnosis using the ReAct framework."""

    def __init__(
        self,
        session_id: str,
        llm_backend: str = DEFAULT_LLM_BACKEND,
        model: str = DEFAULT_MODEL,
        scenario_name: str = "",
        problem_names: list[str] | None = None,
        oracle_routing: bool = False,
        load_all_tools: bool = False,
        tool_evolution_enabled: bool = False,
        tool_library_id: str = "default",
        tool_evolution_mode: str = "dual",
    ):
        mcp_cfg = MCPServerConfig(session_id=session_id)
        if load_all_tools:
            mcp_server_config = mcp_cfg.load_config(if_submit=False)
        else:
            server_names = select_diagnosis_servers(
                scenario_name,
                problem_names or [],
                oracle=oracle_routing,
            )
            mcp_server_config = mcp_cfg.load_filtered_config(server_names)
        self.client = MultiServerMCPClient(connections=mcp_server_config)
        self.tools: list[BaseTool] = []
        self.llm = load_model(llm_backend=llm_backend, model=model)
        self.session_id = session_id
        self.model = model
        self.tool_evolution_enabled = tool_evolution_enabled
        self.tool_library_id = tool_library_id
        self.tool_evolution_mode = (
            ToolEvolutionMode(tool_evolution_mode)
            if tool_evolution_enabled
            else None
        )
        self.tool_evolution_runtime: ToolEvolutionRuntime | None = None

    async def load_tools(self):
        self.tools = await self.client.get_tools()
        for tool in self.tools:
            tool.handle_tool_error = True
            tool.handle_validation_error = True
        if self.tool_evolution_enabled:
            session = Session().load_running_session(session_id=self.session_id)
            assert self.tool_evolution_mode is not None
            self.tool_evolution_runtime = ToolEvolutionRuntime(
                session=session,
                primitive_tools=self.tools,
                library_id=self.tool_library_id,
                mode=self.tool_evolution_mode,
                model=self.model,
                task_description=getattr(session, "task_description", ""),
            )
            self.tools = self.tool_evolution_runtime.build_tools()

    def prompt_suffix(self) -> str:
        parts: list[str] = []
        if self.tool_evolution_runtime is not None:
            parts.append(self.tool_evolution_runtime.prompt_suffix())
        return "\n".join(parts)

    def get_agent(self):
        agent = create_agent(
            model=self.llm,
            system_prompt=OVERALL_DIAGNOSIS_PROMPT + self.prompt_suffix(),
            tools=self.tools,
            name="DiagnosisAgent",
        )
        return agent
