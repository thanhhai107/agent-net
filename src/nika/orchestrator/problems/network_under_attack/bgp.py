import ipaddress
from typing import Optional

from pydantic import BaseModel, Field

from nika.orchestrator.problems.problem_base import (
    RootCauseCategory,
    build_verify_result,
    ProblemBase,
)
from nika.utils.logger import system_logger

# ==================================================================
# Problem: BGP hijacking problem.
# ==================================================================


class BGPHijackingParams(BaseModel):
    """Parameters for injecting a BGP hijacking fault."""

    host_name: str = Field(description="Target router host name.")
    target_network: Optional[str] = Field(
        default=None,
        description="Network prefix to advertise. Defaults to runtime selection.",
    )


class BGPHijacking(ProblemBase):
    root_cause_category: RootCauseCategory = RootCauseCategory.NETWORK_UNDER_ATTACK
    root_cause_name: str = "bgp_hijacking"
    TAGS: str = ["bgp"]

    Params = BGPHijackingParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__(scenario_name, **kwargs)
        self.logger = system_logger

    def _default_target_network(self) -> str:
        web_servers = self.net_env.servers.get("web", [])
        target_host = web_servers[-1] if web_servers else self.net_env.hosts[-1]
        target_network = self.runtime.get_host_ip(target_host, with_prefix=True)
        return str(
            ipaddress.ip_network(target_network, strict=False)
            .subnets(new_prefix=25)
            .__next__()
        )

    def inject_fault(self, params: BGPHijackingParams):
        self.set_faulty_devices([params.host_name])
        target_network = (
            params.target_network
            if params.target_network is not None
            else self._default_target_network()
        )
        asn_number = self.runtime.frr_get_bgp_asn_number(params.host_name)
        self.runtime.exec(
            params.host_name,
            f"vtysh -c 'configure terminal' -c 'interface lo' -c 'ip address {target_network}' ",
        )
        self.runtime.exec(
            params.host_name,
            f"vtysh -c 'configure terminal' -c 'router bgp {asn_number}' -c 'network {target_network}' -c 'end' -c 'write memory' ",
        )
        self.runtime.exec(params.host_name, "systemctl restart frr")
        self.logger.info(
            f"Injected BGP hijacking on {params.host_name}: {target_network}."
        )

    def verify_fault(self, params: BGPHijackingParams) -> dict:
        """Verify the router is advertising the hijacked network via BGP."""
        target_network = (
            params.target_network
            if params.target_network is not None
            else self._default_target_network()
        )
        running_config = self.runtime.exec(
            params.host_name, "vtysh -c 'show running-config' 2>/dev/null"
        ).strip()
        network_prefix = target_network.split("/")[0]
        has_advertisement = (
            f"network {target_network}" in running_config
            or f"network {network_prefix}" in running_config
        )
        lo_output = self.runtime.exec(
            params.host_name, "ip addr show lo 2>/dev/null"
        ).strip()
        has_lo_ip = network_prefix in lo_output
        verified = has_advertisement and has_lo_ip
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={
                "host": params.host_name,
                "target_network": target_network,
                "has_advertisement": has_advertisement,
                "has_lo_ip": has_lo_ip,
            },
        )
