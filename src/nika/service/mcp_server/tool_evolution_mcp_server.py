"""FastMCP adapter for persistent evolved diagnostic workflows and helpers."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP

from agent.tool_evolution.generated_tools import run_generated_tool
from agent.tool_evolution.runtime import (
    COMPOSABLE_PRIMITIVE_TOOLS,
    _resolve_template,
    _tool_output_is_error,
    _validate_argument_safety,
    _validate_composite_arguments,
    _validate_generated_arguments,
    _validate_step_argument_policy,
)
from agent.tool_evolution.store import ToolEvolutionStore
from nika.service.mcp_server import (
    kathara_base_mcp_server as base,
    kathara_bmv2_mcp_server as bmv2,
    kathara_frr_mcp_server as frr,
    kathara_telemetry_mcp_server as telemetry,
)
from nika.utils.errors import safe_tool


mcp = FastMCP("nika_diagnostic_toolbox")

_PRIMITIVES: dict[str, Callable[..., Any]] = {
    name: getattr(module, name)
    for module in (base, frr, bmv2, telemetry)
    for name in COMPOSABLE_PRIMITIVE_TOOLS
    if hasattr(module, name)
}


def _store() -> ToolEvolutionStore:
    library_id = os.environ.get("NIKA_TOOL_LIBRARY_ID", "").strip()
    if not library_id:
        raise ValueError("NIKA_TOOL_LIBRARY_ID is required")
    return ToolEvolutionStore(library_id)


@safe_tool
@mcp.tool()
def list_evolved_tools(include_candidates: bool = False) -> list[dict[str, Any]]:
    """List promoted evolved diagnostic tools; candidates are opt-in only."""
    state = _store().load()
    return [
        {
            "kind": "composite",
            "name": item.name,
            "description": item.description,
            "status": item.status,
            "version": item.version,
            "parameters": [parameter.model_dump() for parameter in item.parameters],
            "output_contract": item.output_contract,
            "utility": item.utility_score(),
        }
        for item in state.composites.values()
        if item.status == "promoted"
        or (include_candidates and item.status == "candidate")
    ] + [
        {
            "kind": "generated_python",
            "name": item.name,
            "description": item.description,
            "status": item.status,
            "version": item.version,
            "parameters": [parameter.model_dump() for parameter in item.parameters],
            "output_description": item.output_description,
            "utility": item.utility_score(),
        }
        for item in state.generated_tools.values()
        if item.status == "promoted"
        or (include_candidates and item.status == "candidate")
    ]


@safe_tool
@mcp.tool()
async def execute_evolved_tool(
    name: str,
    arguments_json: str,
) -> str:
    """Execute one evolved tool from the configured library."""
    store = _store()
    composite = store.get_composite(name)
    generated = store.get_generated_tool(name)
    arguments = json.loads(arguments_json)
    if generated is not None:
        if generated.status != "promoted":
            raise ValueError(f"Unknown or unpromoted evolved tool: {name}")
        arguments = _validate_generated_arguments(generated, arguments)
        output = await asyncio.to_thread(run_generated_tool, generated, arguments)
        if not output.get("success"):
            raise RuntimeError(output.get("stderr") or "generated tool failed")
        return json.dumps(
            {
                "tool": generated.name,
                "kind": "generated_python",
                "status": "success",
                "result": output.get("result"),
            },
            ensure_ascii=False,
            default=str,
        )

    if composite is None or composite.status != "promoted":
        raise ValueError(f"Unknown or unpromoted evolved tool: {name}")
    arguments = _validate_composite_arguments(composite, arguments)

    outputs: list[dict[str, Any]] = []
    for index, step in enumerate(composite.steps):
        if step.tool not in COMPOSABLE_PRIMITIVE_TOOLS or step.tool not in _PRIMITIVES:
            raise ValueError(f"Unsafe or unavailable primitive: {step.tool}")
        resolved = _resolve_template(step.arguments, arguments)
        _validate_argument_safety(resolved, allow_placeholders=False)
        _validate_step_argument_policy(
            step.tool,
            resolved,
            allow_placeholders=False,
        )
        output = _PRIMITIVES[step.tool](**resolved)
        if inspect.isawaitable(output):
            output = await output
        if _tool_output_is_error(output):
            raise RuntimeError(f"Primitive '{step.tool}' returned an error: {output}")
        outputs.append(
            {
                "step": index,
                "tool": step.tool,
                "label": step.label,
                "output": output,
            }
        )
    return json.dumps(
        {"tool": composite.name, "status": "success", "observations": outputs},
        ensure_ascii=False,
        default=str,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
