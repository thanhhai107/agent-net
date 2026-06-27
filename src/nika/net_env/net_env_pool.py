from typing import Dict

from nika.net_env.base import NetworkEnvBase
from nika.net_env.data_center_routing.dc_clos_bgp.lab_services import DCClosService
from nika.net_env.data_center_routing.dc_clos_bgp.lab_workers import DCClosBGP
from nika.net_env.interdomain_routing.simple_bgp.lab import SimpleBGP
from nika.net_env.intradomain_routing.ospf_enterprise.lab_dhcp import OSPFEnterpriseDHCP
from nika.net_env.intradomain_routing.ospf_enterprise.lab_static import OSPFEnterpriseStatic
from nika.net_env.intradomain_routing.rip_vpn.lab import RIPSmallInternetVPN
from nika.net_env.p4.p4_bloom_filter.lab import P4BloomFilter
from nika.net_env.p4.p4_counter.lab import P4Counter
from nika.net_env.p4.p4_int.lab import P4INT
from nika.net_env.p4.p4_mpls.lab import P4_MPLS
from nika.net_env.sdn.clos_topo import SDNClos
from nika.net_env.sdn.star_topo import SDNStar

_NET_ENVS: Dict[str, NetworkEnvBase] = {
    DCClosBGP.LAB_NAME: DCClosBGP,
    DCClosService.LAB_NAME: DCClosService,
    OSPFEnterpriseDHCP.LAB_NAME: OSPFEnterpriseDHCP,
    OSPFEnterpriseStatic.LAB_NAME: OSPFEnterpriseStatic,
    RIPSmallInternetVPN.LAB_NAME: RIPSmallInternetVPN,
    SDNStar.LAB_NAME: SDNStar,
    SDNClos.LAB_NAME: SDNClos,
    P4BloomFilter.LAB_NAME: P4BloomFilter,
    P4Counter.LAB_NAME: P4Counter,
    P4INT.LAB_NAME: P4INT,
    P4_MPLS.LAB_NAME: P4_MPLS,
    SimpleBGP.LAB_NAME: SimpleBGP,
}


def get_net_env_instance(scenario_name: str, **kwargs) -> NetworkEnvBase:
    """Get an instance of the specified network environment.

    Args:
        scenario_name: The name of the network environment.

    Returns:
        An instance of the specified network environment.

    Raises:
        ValueError: If the specified network environment is not found.
    """
    if scenario_name not in _NET_ENVS:
        raise ValueError(f"Network environment '{scenario_name}' not found in the pool.")
    lab_name = kwargs.pop("lab_name", None)
    instance = _NET_ENVS[scenario_name](**kwargs)
    if lab_name:
        instance.name = lab_name
        instance.lab.name = lab_name
    return instance


def list_all_net_envs() -> dict[str, NetworkEnvBase]:
    """List all available network environment names."""
    return _NET_ENVS


def scenario_requires_topo_size(scenario_name: str) -> bool:
    """Return True if this scenario's lab expects an explicit topo size (s/m/l)."""
    if scenario_name not in _NET_ENVS:
        raise ValueError(f"Network environment '{scenario_name}' not found in the pool.")
    topo_size = getattr(_NET_ENVS[scenario_name], "TOPO_SIZE", None)
    return isinstance(topo_size, list)
