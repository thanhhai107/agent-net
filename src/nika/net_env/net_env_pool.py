from pathlib import Path
from typing import Dict, Type

from nika.net_env.base import NetworkEnvBase
from nika.net_env.containerlab.min3clos.lab import ContainerlabMin3Clos
from nika.net_env.kathara.data_center_routing.dc_clos_bgp.lab_services import (
    DCClosService,
)
from nika.net_env.kathara.data_center_routing.dc_clos_bgp.lab_workers import DCClosBGP
from nika.net_env.kathara.interdomain_routing.simple_bgp.lab import SimpleBGP
from nika.net_env.kathara.intradomain_routing.ospf_enterprise.lab_dhcp import (
    OSPFEnterpriseDHCP,
)
from nika.net_env.kathara.intradomain_routing.ospf_enterprise.lab_static import (
    OSPFEnterpriseStatic,
)
from nika.net_env.kathara.intradomain_routing.rip_vpn.lab import RIPSmallInternetVPN
from nika.net_env.kathara.kubernetes.k8s_lab.lab import K8sFatTreeBGP
from nika.net_env.kathara.kubernetes.llmd_lab.lab import LLMDInferenceCluster
from nika.net_env.kathara.p4.p4_bloom_filter.lab import P4BloomFilter
from nika.net_env.kathara.p4.p4_counter.lab import P4Counter
from nika.net_env.kathara.p4.p4_int.lab import P4INT
from nika.net_env.kathara.p4.p4_mpls.lab import P4_MPLS
from nika.net_env.kathara.sdn.clos_topo import SDNClos
from nika.net_env.kathara.sdn.star_topo import SDNStar

_NET_ENVS: Dict[str, Type[NetworkEnvBase]] = {
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
    ContainerlabMin3Clos.LAB_NAME: ContainerlabMin3Clos,
    K8sFatTreeBGP.LAB_NAME: K8sFatTreeBGP,
    LLMDInferenceCluster.LAB_NAME: LLMDInferenceCluster,
}


def scenario_tags(scenario_name: str) -> list[str]:
    """Return metadata tags declared by the network environment class."""
    if scenario_name not in _NET_ENVS:
        raise ValueError(
            f"Network environment '{scenario_name}' not found in the pool."
        )
    return list(getattr(_NET_ENVS[scenario_name], "TAGS", []) or [])


def scenario_supported_backends(scenario_name: str) -> list[str]:
    """Return backends supported by ``scenario_name``."""
    if scenario_name not in _NET_ENVS:
        raise ValueError(
            f"Network environment '{scenario_name}' not found in the pool."
        )
    return list(_NET_ENVS[scenario_name].SUPPORTED_BACKENDS)


def scenario_backend(scenario_name: str) -> str:
    """Return the lab backend bound to ``scenario_name``."""
    supported = scenario_supported_backends(scenario_name)
    if len(supported) != 1:
        raise ValueError(
            f"Scenario '{scenario_name}' must declare exactly one backend; "
            f"found: {', '.join(supported)}"
        )
    return supported[0]


def get_net_env_instance(
    scenario_name: str, *, backend: str = "kathara", **kwargs
) -> NetworkEnvBase:
    """Get an instance of the specified network environment.

    Args:
        scenario_name: The name of the network environment.
        backend: Lab runtime backend (``kathara`` or ``containerlab``).

    Returns:
        An instance of the specified network environment.

    Raises:
        ValueError: If the specified network environment is not found or backend unsupported.
    """
    if scenario_name not in _NET_ENVS:
        raise ValueError(
            f"Network environment '{scenario_name}' not found in the pool."
        )
    cls = _NET_ENVS[scenario_name]
    if backend not in cls.SUPPORTED_BACKENDS:
        raise ValueError(
            f"Scenario '{scenario_name}' does not support backend '{backend}'. "
            f"Supported: {', '.join(cls.SUPPORTED_BACKENDS)}"
        )
    lab_name = kwargs.pop("lab_name", None)
    topology_file = kwargs.pop("topology_file", None)
    runtime_workdir = kwargs.pop("runtime_workdir", None)
    instance = cls(**kwargs)
    instance.backend = backend
    if lab_name:
        instance.name = lab_name
        if instance.lab is not None:
            instance.lab.name = lab_name
    if topology_file is not None:
        instance.topology_file = Path(topology_file)
    if runtime_workdir is not None:
        instance.runtime_workdir = Path(runtime_workdir)
    return instance


def list_all_net_envs(*, backend: str | None = None) -> dict[str, Type[NetworkEnvBase]]:
    """List available network environment classes, optionally filtered by backend."""
    if backend is None:
        return _NET_ENVS
    return {
        name: cls
        for name, cls in _NET_ENVS.items()
        if backend in cls.SUPPORTED_BACKENDS
    }


def scenario_requires_topo_size(scenario_name: str) -> bool:
    """Return True if this scenario's lab expects an explicit topo size (s/m/l)."""
    if scenario_name not in _NET_ENVS:
        raise ValueError(
            f"Network environment '{scenario_name}' not found in the pool."
        )
    topo_size = getattr(_NET_ENVS[scenario_name], "TOPO_SIZE", None)
    return isinstance(topo_size, list)
