"""Public per-case dataset builder for executable target agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.composition import MemoryConfig, ToolEvolutionConfig
from agent.memory.attributes import infer_memory_attributes
from agent.memory.models import MemoryQuery
from agent.memory.service import ProceduralMemoryModule
from agent.utils.loggers import MessageLogger
from agent.utils.mcp_servers import select_diagnosis_servers


def _retrieve_memory_context(
    *,
    session: Any,
    memory: MemoryConfig,
    llm_backend: str,
    model: str,
    diagnosis_servers: list[str],
) -> str:
    if not memory.enabled:
        return ""
    module = ProceduralMemoryModule(
        bank_id=memory.bank,
        llm_backend=llm_backend,
        model=model,
    )
    tools = list(diagnosis_servers)
    task_description = str(getattr(session, "task_description", "") or "")
    attrs = infer_memory_attributes(
        task_description,
        scenario=str(getattr(session, "scenario_name", "") or ""),
        topology_class=str(getattr(session, "scenario_topo_size", "") or ""),
        task_stage="diagnosis",
        tools=tools,
    )
    retrieved = module.retrieve(
        query=MemoryQuery(
            text=task_description,
            scenario=str(getattr(session, "scenario_name", "") or ""),
            topology_class=str(getattr(session, "scenario_topo_size", "") or ""),
            protocols=attrs.protocols,
            services=attrs.services,
            symptoms=attrs.symptoms,
            task_stage="diagnosis",
            tools=tools,
            top_k=memory.top_k,
            token_budget=memory.token_budget,
        ),
        session_id=str(getattr(session, "session_id")),
    )
    MessageLogger(
        agent="memory_agent",
        session_dir=str(getattr(session, "session_dir")),
        extra_fields={"phase": "retrieval"},
    ).log(
        "memory_retrieval",
        {
            "bank_id": memory.bank,
            "memory_mode": memory.mode,
            "workflow": "HarnessTargetAgent",
            "memory_ids": [item.memory.memory_id for item in retrieved],
            "scores": [round(item.score, 6) for item in retrieved],
        },
    )
    return module.format_context(retrieved)


def build_public_case_dataset(
    *,
    session: Any,
    output_dir: str | Path,
    memory: MemoryConfig,
    tool_evolution: ToolEvolutionConfig,
    llm_backend: str,
    model: str,
) -> Path:
    """Write the public files a generated target agent is allowed to read."""
    dataset_dir = Path(output_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    scenario_name = str(getattr(session, "scenario_name", "") or "")
    diagnosis_servers = select_diagnosis_servers(scenario_name)
    memory_context = _retrieve_memory_context(
        session=session,
        memory=memory,
        llm_backend=llm_backend,
        model=model,
        diagnosis_servers=diagnosis_servers,
    )
    context = {
        "schema_version": 1,
        "session_id": str(getattr(session, "session_id", "")),
        "benchmark_index": getattr(session, "benchmark_index", None),
        "scenario_name": scenario_name,
        "scenario_topo_size": getattr(session, "scenario_topo_size", None),
        "topology": getattr(session, "topology", []) or [],
        "task_description": str(getattr(session, "task_description", "") or ""),
        "diagnosis_mcp_servers": diagnosis_servers,
        "tool_contract": {
            "surface": "fixed_mcp_tools",
            "allowed_mcp_servers": diagnosis_servers
            + (["nika_tool_docs"] if tool_evolution.enabled else []),
            "may_define_local_helpers": True,
            "may_create_new_primitive_tools": False,
            "may_create_new_mcp_servers": False,
        },
        "submission_contract": {
            "must_call": ["list_avail_problems", "submit"],
            "submission_file": "submission.json",
        },
        "memory_mode": memory.mode,
        "memory_context": memory_context,
        "tool_library_id": (
            tool_evolution.library_id if tool_evolution.enabled else None
        ),
        "tool_evolution_note": (
            "The nika_tool_docs MCP server exposes DRAFT-refined "
            "documentation for fixed primitive tools. It does not create new "
            "benchmark primitive tools."
            if tool_evolution.enabled
            else ""
        ),
    }
    (dataset_dir / "case_context.json").write_text(
        json.dumps(context, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (dataset_dir / "task.md").write_text(
        "\n".join(
            [
                "# NIKA public diagnosis task",
                "",
                context["task_description"],
                "",
                "Use only the MCP tools configured for this session. "
                "Do not create new primitive tools or MCP servers. "
                "Submit via the task MCP server.",
            ]
        ),
        encoding="utf-8",
    )
    return dataset_dir
