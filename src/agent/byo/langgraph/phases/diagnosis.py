from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools.structured import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.utils.template import OVERALL_DIAGNOSIS_PROMPT
from agent.llm.model_factory import load_model
from agent.utils.mcp_servers import MCPServerConfig, select_diagnosis_servers
from agent.utils.phases import DIAGNOSIS

load_dotenv()


class DiagnosisPhase:
    """LangChain ReAct worker for the diagnosis phase."""

    def __init__(
        self,
        session_id: str,
        llm_provider: str = "openai",
        model: str = "gpt-5-mini",
        scenario_name: str = "",
        problem_names: list[str] | None = None,
    ):
        mcp_cfg = MCPServerConfig(session_id=session_id)
        server_names = select_diagnosis_servers(scenario_name, problem_names or [])
        mcp_server_config = mcp_cfg.load_filtered_config(server_names)
        self.client = MultiServerMCPClient(connections=mcp_server_config)
        self.tools = None
        self.llm = load_model(llm_provider=llm_provider, model=model)

    async def load_tools(self):
        self.tools: list[StructuredTool] = await self.client.get_tools()
        for tool in self.tools:
            tool.handle_tool_error = True
            tool.handle_validation_error = True

    def get_agent(self):
        agent = create_agent(
            model=self.llm, system_prompt=OVERALL_DIAGNOSIS_PROMPT, tools=self.tools, name=DIAGNOSIS
        )
        return agent
