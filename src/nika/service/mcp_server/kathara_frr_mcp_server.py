from mcp.server.fastmcp import FastMCP

from nika.service.kathara import KatharaFRRAPI
from nika.service.mcp_server.mcp_session_context import get_lab_name
from nika.utils.errors import safe_tool

# Initialize FastMCP server
mcp = FastMCP("kathara_bmv2_mcp_server")


@safe_tool
@mcp.tool()
def frr_get_bgp_conf(router_name: str) -> str:
    """Get the BGP configuration from the FRR router.

    Args:
        router_name (str): The name of the router.

    Returns:
        str: The BGP configuration from the FRR router.
    """
    kathara_api = KatharaFRRAPI(lab_name=get_lab_name())
    return kathara_api.frr_get_bgp_conf(router_name)


@safe_tool
@mcp.tool()
def frr_show_bgp_summary(router_name: str) -> str:
    """Get BGP neighbor and session-state summary from an FRR router.

    Args:
        router_name (str): The name of the router.

    Returns:
        str: BGP peers, session states, and prefix counters.
    """
    kathara_api = KatharaFRRAPI(lab_name=get_lab_name())
    return kathara_api.frr_show_bgp_summary(router_name)


@safe_tool
@mcp.tool()
def frr_show_running_config(router_name: str) -> str:
    """Get the running configuration from the FRR router.

    Args:
        router_name (str): The name of the router.
    Returns:
        str: The running configuration from the FRR router.
    """
    kathara_api = KatharaFRRAPI(lab_name=get_lab_name())
    return kathara_api.frr_show_running_config(router_name)


@safe_tool
@mcp.tool()
def frr_show_ip_route(router_name: str) -> str:
    """Get the IP routing table from the FRR router.

    Args:
        router_name (str): The name of the router.
    Returns:
        str: The IP routing table from the FRR router.
    """
    kathara_api = KatharaFRRAPI(lab_name=get_lab_name())
    return kathara_api.frr_show_route(router_name)


@safe_tool
@mcp.tool()
def frr_get_ospf_conf(router_name: str) -> str:
    """Get the OSPF configuration from the FRR router.

    Args:
        router_name (str): The name of the router.

    Returns:
        str: The OSPF configuration from the FRR router.
    """
    kathara_api = KatharaFRRAPI(lab_name=get_lab_name())
    return kathara_api.frr_get_ospf_conf(router_name)


@safe_tool
@mcp.tool()
def frr_exec(router_name: str, command: str) -> str:
    """Execute a vtysh command on a FRR router."""
    kathara_api = KatharaFRRAPI(lab_name=get_lab_name())
    return kathara_api.frr_exec(router_name, command)


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")
