from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.llm.model_factory import DEFAULT_LLM_BACKEND, DEFAULT_MODEL, load_model
from agent.memory.runtime import SkillToolRuntime
from agent.memory.service import ProceduralMemoryModule
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
        tool_doc_chars: int = 500,
        tool_prompt_doc_limit: int = 6,
        tool_scoped_prompt_doc_limit: int = 4,
        tool_planned_checks: int = 4,
        tool_next_checks: int = 2,
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
        self.tool_doc_chars = tool_doc_chars
        self.tool_prompt_doc_limit = tool_prompt_doc_limit
        self.tool_scoped_prompt_doc_limit = tool_scoped_prompt_doc_limit
        self.tool_planned_checks = tool_planned_checks
        self.tool_next_checks = tool_next_checks
        self.tool_evolution_runtime: ToolEvolutionRuntime | None = None
        self.skill_tool_runtime: SkillToolRuntime | None = None
        self._memory_runtime_base_tools: list[BaseTool] = []
        self.session = Session().load_running_session(session_id=session_id)

    async def load_tools(self):
        self.tools = await self.client.get_tools()
        for tool in self.tools:
            tool.handle_tool_error = True
            tool.handle_validation_error = True
        if self.tool_evolution_enabled:
            self.tool_evolution_runtime = ToolEvolutionRuntime(
                session=self.session,
                primitive_tools=self.tools,
                library_id=self.tool_library_id,
                model=self.model,
                task_description=getattr(self.session, "task_description", ""),
                tool_doc_chars=self.tool_doc_chars,
                prompt_doc_limit=self.tool_prompt_doc_limit,
                scoped_prompt_doc_limit=self.tool_scoped_prompt_doc_limit,
                planned_checks=self.tool_planned_checks,
                next_checks=self.tool_next_checks,
            )
            self.tools = self.tool_evolution_runtime.build_tools()
        self._memory_runtime_base_tools = list(self.tools)

    def install_memory_runtime(
        self,
        *,
        memory: ProceduralMemoryModule,
        memory_mode: str,
        task_description: str,
        top_k: int = 5,
        token_budget: int = 1500,
        session_dir: str = "",
        skill_selector_mode: str = "lcb",
        meta_controller_mode: str = "heuristic",
        max_skill_age: int = 4,
        selector_min_lcb: float = -0.05,
        selector_nominee_k: int = 3,
    ) -> None:
        if not self.tools:
            raise RuntimeError("Diagnosis tools must be loaded before installing memory")
        if self.tool_evolution_runtime is not None:
            self._memory_runtime_base_tools = self.tool_evolution_runtime.build_tools(
                append_docs=False
            )
        self.tools = list(self._memory_runtime_base_tools or self.tools)
        self.skill_tool_runtime = SkillToolRuntime(
            memory=memory,
            memory_mode=memory_mode,
            session=self.session,
            task_description=task_description,
            tools=self.tools,
            session_dir=session_dir,
            tool_evolution_runtime=self.tool_evolution_runtime,
            top_k=top_k,
            token_budget=token_budget,
            meta_controller_llm=self.llm,
            meta_controller_mode=meta_controller_mode,
            skill_selector_mode=skill_selector_mode,
            max_skill_age=max_skill_age,
            selector_min_lcb=selector_min_lcb,
            selector_nominee_k=selector_nominee_k,
        )
        self.tools = self.skill_tool_runtime.wrap_tools(self.tools)

    def prompt_suffix(self, *, activate_skill: bool = True) -> str:
        parts: list[str] = []
        if (
            self.tool_evolution_runtime is not None
            and self.skill_tool_runtime is None
        ):
            parts.append(self.tool_evolution_runtime.prompt_suffix())
        if self.skill_tool_runtime is not None:
            parts.append(
                self.skill_tool_runtime.prompt_suffix(
                    activate_skill=activate_skill
                )
            )
        return "\n".join(parts)

    def get_agent(self, *, include_learning_context: bool = False):
        system_prompt = OVERALL_DIAGNOSIS_PROMPT
        if include_learning_context:
            system_prompt += self.prompt_suffix()
        return create_agent(
            model=self.llm,
            system_prompt=system_prompt,
            tools=self.tools,
            name=DIAGNOSIS,
        )
