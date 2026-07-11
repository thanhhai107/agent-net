from mcp.server.fastmcp import FastMCP

from nika.service.mcp_server.session_context import get_lab_api
from nika.service.pingmesh.engine import (
    run_pingmesh_snapshot as execute_pingmesh_snapshot,
    snapshot_to_json,
)
from nika.utils.errors import safe_tool

mcp = FastMCP("pingmesh_mcp_server")


@safe_tool
@mcp.tool()
async def run_pingmesh_snapshot(
    sources: list[str] | None = None,
    targets: list[str] | None = None,
    count: int = 4,
    high_latency_ms: float = 100.0,
    max_pairs: int = 64,
) -> str:
    """Run an on-demand PingMesh snapshot across endpoint hosts in the current lab.

    Probes reachability, packet loss, and RTT from each source endpoint to each
    target endpoint, returning a matrix plus anomaly pairs and a summary. Endpoint
    hosts are business nodes such as client, pc, host, and server — routers and
    switches are excluded by default.

    Args:
        sources: Optional subset of endpoint names to use as probe sources.
            Defaults to all discovered endpoints.
        targets: Optional subset of endpoint names to use as probe targets.
            Defaults to all discovered endpoints.
        count: Number of ping packets per pair (1-20). Defaults to 4.
        high_latency_ms: RTT average threshold for high-latency anomalies.
            Defaults to 100.0.
        max_pairs: Maximum number of source-target pairs to probe. Defaults to 64.

    Returns:
        str: JSON snapshot with timestamp, endpoints, results, anomalies, and summary.
    """
    lab_api = get_lab_api()
    snapshot = await execute_pingmesh_snapshot(
        lab_api,
        sources=sources,
        targets=targets,
        count=count,
        high_latency_ms=high_latency_ms,
        max_pairs=max_pairs,
    )
    return snapshot_to_json(snapshot)


if __name__ == "__main__":
    mcp.run(transport="stdio")
