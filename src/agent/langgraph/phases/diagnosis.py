from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.llm.model_factory import DEFAULT_LLM_BACKEND, DEFAULT_MODEL, load_model
from agent.tool_evolution.runtime import ToolEvolutionRuntime
from agent.utils.mcp_servers import MCPServerConfig, select_diagnosis_servers
from agent.utils.phases import DIAGNOSIS
from agent.utils.template import OVERALL_DIAGNOSIS_PROMPT
from nika.utils.session import Session

load_dotenv()


class DiagnosisPhase:
    """LangChain ReAct worker for the diagnosis phase."""

    def __init__(
        self,
        session_id: str,
        llm_backend: str = DEFAULT_LLM_BACKEND,
        model: str = DEFAULT_MODEL,
        scenario_name: str = "",
        load_all_tools: bool = False,
        tool_evolution_enabled: bool = False,
        tool_library_id: str = "default",
    ):
        mcp_cfg = MCPServerConfig(session_id=session_id)
        if load_all_tools:
            mcp_server_config = mcp_cfg.load_config(if_submit=False)
        else:
            server_names = select_diagnosis_servers(scenario_name)
            mcp_server_config = mcp_cfg.load_filtered_config(server_names)
        self.client = MultiServerMCPClient(connections=mcp_server_config)
        self.tools: list[BaseTool] = []
        self.llm = load_model(llm_backend=llm_backend, model=model)
        self.session_id = session_id
        self.model = model
        self.tool_evolution_enabled = tool_evolution_enabled
        self.tool_library_id = tool_library_id
        self.tool_evolution_runtime: ToolEvolutionRuntime | None = None

    async def load_tools(self):
        self.tools = await self.client.get_tools()
        for tool in self.tools:
            tool.handle_tool_error = True
            tool.handle_validation_error = True
        if self.tool_evolution_enabled:
            session = Session().load_running_session(session_id=self.session_id)
            self.tool_evolution_runtime = ToolEvolutionRuntime(
                session=session,
                primitive_tools=self.tools,
                library_id=self.tool_library_id,
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
        return create_agent(
            model=self.llm,
            system_prompt=OVERALL_DIAGNOSIS_PROMPT + self.prompt_suffix(),
            tools=self.tools,
            name=DIAGNOSIS,
        )
