"""Build the NIKA MCP HTTP gateway ASGI application."""

from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager
from importlib import import_module

from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount

from nika.service.mcp_gateway.middleware import PhaseGateMiddleware, _empty_mcp
from nika.service.mcp_server.registry import MCP_SERVER_SPECS

_MCP_MODULE_ATTRS: dict[str, tuple[str, str]] = {
    "kathara_base_mcp_server": (
        "nika.service.mcp_server.common.host_server",
        "mcp",
    ),
    "pingmesh_mcp_server": (
        "nika.service.mcp_server.common.pingmesh_server",
        "mcp",
    ),
    "task_mcp_server": ("nika.service.mcp_server.common.task_server", "mcp"),
    "kathara_frr_mcp_server": (
        "nika.service.mcp_server.kathara.frr_server",
        "mcp",
    ),
    "kathara_bmv2_mcp_server": (
        "nika.service.mcp_server.kathara.bmv2_server",
        "mcp",
    ),
    "kathara_telemetry_mcp_server": (
        "nika.service.mcp_server.kathara.telemetry_server",
        "mcp",
    ),
    "containerlab_srl_mcp_server": (
        "nika.service.mcp_server.containerlab.srl_server",
        "mcp",
    ),
}


def _load_mcp(name: str) -> FastMCP:
    module_path, attr = _MCP_MODULE_ATTRS[name]
    module = import_module(module_path)
    mcp: FastMCP = getattr(module, attr)
    return mcp


def reset_gateway_mcp_state() -> None:
    """Allow a fresh gateway process to attach new HTTP session managers."""
    for name in MCP_SERVER_SPECS:
        _load_mcp(name)._session_manager = None  # type: ignore[attr-defined]
    _empty_mcp._session_manager = None  # type: ignore[attr-defined]


def create_gateway_app() -> Starlette:
    """Return a Starlette app exposing all registered MCP servers over HTTP."""
    reset_gateway_mcp_state()
    routes = []
    session_managers = []

    blocked_app = _empty_mcp.streamable_http_app()
    session_managers.append(_empty_mcp.session_manager)

    for name in MCP_SERVER_SPECS:
        mcp = _load_mcp(name)
        starlette_app = mcp.streamable_http_app()
        session_managers.append(mcp.session_manager)
        inner = PhaseGateMiddleware(
            starlette_app,
            server_name=name,
            blocked_app=blocked_app,
        )
        routes.append(Mount(f"/mcp/{name}", app=inner))

    @asynccontextmanager
    async def lifespan(_app: Starlette):
        async with AsyncExitStack() as stack:
            for manager in session_managers:
                await stack.enter_async_context(manager.run())
            yield

    return Starlette(routes=routes, lifespan=lifespan)
