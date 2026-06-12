import ipaddress
import random
from typing import Optional

from pydantic import BaseModel, Field

from nika.generator.fault.injector_base import FaultInjectorBase
from nika.net_env.net_env_pool import get_net_env_instance
from nika.orchestrator.problems.problem_base import ProblemMeta, RootCauseCategory, TaskDescription, TaskLevel, build_verify_result
from nika.orchestrator.tasks.detection import DetectionTask
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask
from nika.service.kathara import KatharaAPIALL

# ==================================================================
# Problem: BGP hijacking problem.
# ==================================================================


class BGPHijackingParams(BaseModel):
    """Parameters for injecting a BGP hijacking fault."""

    host_name: Optional[str] = Field(default=None, description="Target router host name. Defaults to a randomly selected router.")
    target_network: Optional[str] = Field(default=None, description="Network prefix to advertise. Defaults to runtime selection.")


class BGPHijackingBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.NETWORK_UNDER_ATTACK
    root_cause_name: str = "bgp_hijacking"
    TAGS: str = ["bgp", "http"]

    Params = BGPHijackingParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaAPIALL(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorBase(lab_name=self.net_env.lab.name)
        self.faulty_devices = [random.choice(self.net_env.routers)]
        web_server = self.net_env.servers["web"][-1]
        self.target_network = self.kathara_api.get_host_ip(web_server, with_prefix=True)
        self.target_network = str(
            ipaddress.ip_network(self.target_network, strict=False).subnets(new_prefix=25).__next__()
        )

    def inject_fault(self, params: BGPHijackingParams | None = None):
        if params is None:
            params = BGPHijackingParams()
        host = params.host_name if params.host_name is not None else self.faulty_devices[0]
        target_network = params.target_network if params.target_network is not None else self.target_network
        asn_number = self.kathara_api.frr_get_bgp_asn_number(self.faulty_devices[0])
        self.injector.inject_bgp_add_interface(host_name=host, intf_name="lo", ip_address=target_network)
        self.injector.inject_bgp_add_advertisement(host_name=host, network=target_network, AS=asn_number)

    def verify_fault(self, params: BGPHijackingParams | None = None) -> dict:
        """Verify the router is advertising the hijacked network via BGP."""
        if params is None:
            params = BGPHijackingParams()
        host = params.host_name if params.host_name is not None else self.faulty_devices[0]
        target_network = params.target_network if params.target_network is not None else self.target_network
        running_config = self.kathara_api.exec_cmd(
            host, "vtysh -c 'show running-config' 2>/dev/null"
        ).strip()
        network_prefix = target_network.split("/")[0]
        has_advertisement = f"network {target_network}" in running_config or f"network {network_prefix}" in running_config
        lo_output = self.kathara_api.exec_cmd(host, "ip addr show lo 2>/dev/null").strip()
        has_lo_ip = network_prefix in lo_output
        verified = has_advertisement and has_lo_ip
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={
                "host": host,
                "target_network": target_network,
                "has_advertisement": has_advertisement,
                "has_lo_ip": has_lo_ip,
            },
        )


class BGPHijackingDetection(BGPHijackingBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=BGPHijackingBase.root_cause_category,
        root_cause_name=BGPHijackingBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class BGPHijackingLocalization(BGPHijackingBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=BGPHijackingBase.root_cause_category,
        root_cause_name=BGPHijackingBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class BGPHijackingRCA(BGPHijackingBase, RCATask):
    META = ProblemMeta(
        root_cause_category=BGPHijackingBase.root_cause_category,
        root_cause_name=BGPHijackingBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )
