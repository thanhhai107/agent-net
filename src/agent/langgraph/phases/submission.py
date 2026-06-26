from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.llm.model_factory import DEFAULT_LLM_BACKEND, DEFAULT_MODEL, load_model
from agent.utils.mcp_servers import MCPServerConfig
from agent.utils.phases import SUBMISSION
from agent.utils.template import SUBMIT_PROMPT_TEMPLATE

load_dotenv()


class SubmissionPhase:
    """LangChain ReAct worker for the submission phase."""

    def __init__(
        self,
        session_id: str,
        llm_backend: str = DEFAULT_LLM_BACKEND,
        model: str = DEFAULT_MODEL,
    ):
        mcp_server_config = MCPServerConfig(session_id=session_id).load_config(if_submit=True)
        self.client = MultiServerMCPClient(connections=mcp_server_config)
        self.tools: list[BaseTool] = []
        self.llm = load_model(llm_backend=llm_backend, model=model)

    async def load_tools(self):
        self.tools = await self.client.get_tools()
        for tool in self.tools:
            tool.handle_tool_error = True
            tool.handle_validation_error = True

    def get_agent(self):
        return create_agent(
            model=self.llm,
            system_prompt=SUBMIT_PROMPT_TEMPLATE,
            tools=self.tools,
            name=SUBMISSION,
        )
