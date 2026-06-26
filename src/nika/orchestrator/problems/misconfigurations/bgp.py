import ipaddress
import logging
import random
import re
from typing import Optional

from pydantic import BaseModel, Field

from nika.net_env.net_env_pool import get_net_env_instance
from nika.orchestrator.problems.problem_base import ProblemMeta, RootCauseCategory, TaskDescription, TaskLevel, build_verify_result
from nika.orchestrator.tasks.detection import DetectionTask
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask
from nika.service.kathara import KatharaAPIALL
from nika.utils.logger import system_logger

# ==================================================================
""" Problem: Base class for a BGP ASN misconfiguration problem. """
# ==================================================================


class BGPAsnMisconfigParams(BaseModel):
    """Parameters for injecting a BGP ASN misconfiguration fault."""

    host_name: Optional[str] = Field(default=None, description="Target router host name. Defaults to a randomly selected router.")


class BGPAsnMisconfigBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.MISCONFIGURATION
    root_cause_name: str = "bgp_asn_misconfig"
    TAGS: str = ["bgp"]

    Params = BGPAsnMisconfigParams

    symptom_desc = "Some hosts are experiencing connectivity issues."

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaAPIALL(lab_name=self.net_env.lab.name)
        self.logger = system_logger
        self.faulty_devices = [random.choice(self.net_env.routers)]

    def inject_fault(self, params: BGPAsnMisconfigParams | None = None):
        if params is None:
            params = BGPAsnMisconfigParams()
        host = params.host_name if params.host_name is not None else self.faulty_devices[0]
        asn = self.kathara_api.exec_cmd(host, "vtysh -c 'show bgp summary' | grep 'BGP router identifier'")
        match = re.search(r"local AS number\s+(\d+)", asn)
        if match:
            as_number = int(match.group(1))
        else:
            raise ValueError("Could not find AS number in BGP summary output")
        wrong_asn = as_number + 600
        self.kathara_api.exec_cmd(
            host,
            f"sed -i.bak 's/^router bgp {as_number}$/router bgp {wrong_asn}/' /etc/frr/frr.conf && service frr restart 2>/dev/null || true",
        )
        self.logger.info(f"Injected BGP ASN misconfiguration on {host} from ASN {as_number} to {wrong_asn}.")

    def verify_fault(self, params: BGPAsnMisconfigParams | None = None) -> dict:
        """Verify the ASN in frr.conf and in the running daemon was changed."""
        if params is None:
            params = BGPAsnMisconfigParams()
        host = params.host_name if params.host_name is not None else self.faulty_devices[0]
        file_asn_raw = self.kathara_api.exec_cmd(
            host,
            "grep -E '^router bgp' /etc/frr/frr.conf 2>/dev/null | awk '{print $3}'",
        ).strip()
        orig_asn_raw = self.kathara_api.exec_cmd(
            host,
            "grep -E '^router bgp' /etc/frr/frr.conf.bak 2>/dev/null | awk '{print $3}'",
        ).strip()
        running_asn_raw = self.kathara_api.exec_cmd(
            host,
            "vtysh -c 'show running-config' 2>/dev/null | grep -E '^router bgp' | awk '{print $3}'",
        ).strip()
        file_changed = bool(file_asn_raw) and bool(orig_asn_raw) and file_asn_raw != orig_asn_raw
        daemon_changed = bool(running_asn_raw) and running_asn_raw != orig_asn_raw
        verified = file_changed and daemon_changed
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={
                "host": host,
                "file_asn": file_asn_raw,
                "orig_asn": orig_asn_raw,
                "running_asn": running_asn_raw,
            },
        )


class BGPAsnMisconfigDetection(BGPAsnMisconfigBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=BGPAsnMisconfigBase.root_cause_category,
        root_cause_name=BGPAsnMisconfigBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class BGPAsnMisconfigLocalization(BGPAsnMisconfigBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=BGPAsnMisconfigBase.root_cause_category,
        root_cause_name=BGPAsnMisconfigBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class BGPAsnMisconfigRCA(BGPAsnMisconfigBase, RCATask):
    META = ProblemMeta(
        root_cause_category=BGPAsnMisconfigBase.root_cause_category,
        root_cause_name=BGPAsnMisconfigBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==================================================================
""" Problem: Base class for a BGP missing route advertisement problem. """
# ==================================================================


class BGPMissingAdvertiseParams(BaseModel):
    """Parameters for injecting a BGP missing route advertisement fault."""

    host_name: Optional[str] = Field(default=None, description="Target router host name. Defaults to a randomly selected router.")


class BGPMissingAdvertiseBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.MISCONFIGURATION
    root_cause_name: str = "bgp_missing_route_advertisement"
    TAGS: str = ["bgp"]

    Params = BGPMissingAdvertiseParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaAPIALL(lab_name=self.net_env.lab.name)
        self.logger = system_logger
        self.faulty_devices = [random.choice(self.net_env.routers)]

    def inject_fault(self, params: BGPMissingAdvertiseParams | None = None):
        if params is None:
            params = BGPMissingAdvertiseParams()
        host = params.host_name if params.host_name is not None else self.faulty_devices[0]
        self.kathara_api.exec_cmd(
            host,
            "sed -i.bak -E 's/^([[:space:]]*)network /\\1# network /' /etc/frr/frr.conf && service frr restart 2>/dev/null || true",
        )
        self.logger.info(f"Injected BGP missing route on {host}.")

    def verify_fault(self, params: BGPMissingAdvertiseParams | None = None) -> dict:
        """Verify frr.conf has commented-out network lines and running daemon has no network advertisements."""
        if params is None:
            params = BGPMissingAdvertiseParams()
        host = params.host_name if params.host_name is not None else self.faulty_devices[0]
        count_raw = self.kathara_api.exec_cmd(
            host,
            "grep -c '^[[:space:]]*# network' /etc/frr/frr.conf 2>/dev/null || echo 0",
        ).strip()
        try:
            count = int(count_raw)
        except ValueError:
            count = 0
        running_count_raw = self.kathara_api.exec_cmd(
            host,
            "vtysh -c 'show running-config' 2>/dev/null | grep -c '^[[:space:]]*network' || echo 0",
        ).strip()
        try:
            running_count = int(running_count_raw)
        except ValueError:
            running_count = 0
        verified = count > 0 and running_count == 0
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={
                "host": host,
                "commented_network_count": count,
                "running_network_count": running_count,
            },
        )


class BGPMissingAdvertiseDetection(BGPMissingAdvertiseBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=BGPMissingAdvertiseBase.root_cause_category,
        root_cause_name=BGPMissingAdvertiseBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class BGPMissingAdvertiseLocalization(BGPMissingAdvertiseBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=BGPMissingAdvertiseBase.root_cause_category,
        root_cause_name=BGPMissingAdvertiseBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class BGPMissingAdvertiseRCA(BGPMissingAdvertiseBase, RCATask):
    META = ProblemMeta(
        root_cause_category=BGPMissingAdvertiseBase.root_cause_category,
        root_cause_name=BGPMissingAdvertiseBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==================================================================
""" Problem: BGP static blackhole route misconfiguration problem. """
# ==================================================================


class StaticBlackHoleParams(BaseModel):
    """Parameters for injecting a static blackhole route fault."""

    host_name: Optional[str] = Field(default=None, description="Target router host name. Defaults to a randomly selected router with connected hosts.")


class StaticBlackHoleBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.MISCONFIGURATION
    root_cause_name: str = "host_static_blackhole"
    TAGS: str = ["bgp"]

    Params = StaticBlackHoleParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaAPIALL(lab_name=self.net_env.lab.name)
        self.logger = system_logger
        for router in random.sample(self.net_env.routers, len(self.net_env.routers)):
            connected_devices = self.kathara_api.get_connected_devices(router)
            connected_hosts = [dev for dev in connected_devices if "switch" not in dev and "router" not in dev]
            if connected_hosts:
                self.faulty_devices = [router]
                self.victim_device = connected_hosts[0]
                self.victim_ip = self.kathara_api.get_host_ip(self.victim_device, with_prefix=False)
                break

    def inject_fault(self, params: StaticBlackHoleParams | None = None):
        if params is None:
            params = StaticBlackHoleParams()
        host = params.host_name if params.host_name is not None else self.faulty_devices[0]
        host_network = ipaddress.ip_network(
            self.kathara_api.get_host_ip(self.victim_device, with_prefix=True), strict=False
        )
        self.kathara_api.exec_cmd(host, f"ip route replace blackhole {host_network}")
        self.logger.info(f"Injected addition of blackhole route {host_network} on {host}.")

    def verify_fault(self, params: StaticBlackHoleParams | None = None) -> dict:
        """Verify a blackhole route for the victim's network exists."""
        if params is None:
            params = StaticBlackHoleParams()
        host = params.host_name if params.host_name is not None else self.faulty_devices[0]
        host_network = str(
            ipaddress.ip_network(
                self.kathara_api.get_host_ip(self.victim_device, with_prefix=True), strict=False
            )
        )
        route_output = self.kathara_api.exec_cmd(host, "ip route show").strip()
        verified = f"blackhole {host_network}" in route_output or "blackhole" in route_output
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"host": host, "network": host_network, "route_output": route_output},
        )


class StaticBlackHoleDetection(StaticBlackHoleBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=StaticBlackHoleBase.root_cause_category,
        root_cause_name=StaticBlackHoleBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class StaticBlackHoleLocalization(StaticBlackHoleBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=StaticBlackHoleBase.root_cause_category,
        root_cause_name=StaticBlackHoleBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class StaticBlackHoleRCA(StaticBlackHoleBase, RCATask):
    META = ProblemMeta(
        root_cause_category=StaticBlackHoleBase.root_cause_category,
        root_cause_name=StaticBlackHoleBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==================================================================
""" Problem: BGP blackhole route advertisement misconfiguration problem. """
# ==================================================================


class BGPBlackholeRouteLeakParams(BaseModel):
    """Parameters for injecting a BGP blackhole route leak fault."""

    host_name: Optional[str] = Field(default=None, description="Target router host name. Defaults to a randomly selected router with connected hosts.")


class BGPBlackholeRouteLeakBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.MISCONFIGURATION
    root_cause_name: str = "bgp_blackhole_route_leak"
    TAGS: str = ["bgp"]

    Params = BGPBlackholeRouteLeakParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaAPIALL(lab_name=self.net_env.lab.name)
        self.logger = system_logger
        for router in random.sample(self.net_env.routers, len(self.net_env.routers)):
            connected_devices = self.kathara_api.get_connected_devices(router)
            connected_hosts = [dev for dev in connected_devices if "switch" not in dev and "router" not in dev]
            if connected_hosts:
                self.faulty_devices = [router]
                self.victim_device = connected_hosts[0]
                self.victim_ip = self.kathara_api.get_host_ip(self.victim_device, with_prefix=False)
                break

    def inject_fault(self, params: BGPBlackholeRouteLeakParams | None = None):
        if params is None:
            params = BGPBlackholeRouteLeakParams()
        host = params.host_name if params.host_name is not None else self.faulty_devices[0]
        network_30 = ipaddress.ip_network(f"{self.victim_ip}/30", strict=False)
        asn_number = self.kathara_api.exec_cmd(host, "vtysh -c 'show bgp summary' | grep 'BGP router identifier'")
        match = re.search(r"local AS number\s+(\d+)", asn_number)
        if match:
            as_number = int(match.group(1))
        else:
            raise ValueError("Could not find AS number in BGP summary output")
        cmd = (
            "vtysh -c 'configure terminal' "
            f"-c 'ip route {network_30} Null0' "
            f"-c 'router bgp {as_number}' "
            f"-c 'network {network_30}' "
            "-c 'end' "
            "-c 'write memory' "
        )
        self.kathara_api.exec_cmd(host, cmd)
        self.logger.info(f"Injected BGP advertise blackhole route on {host}: {network_30}.")

    def verify_fault(self, params: BGPBlackholeRouteLeakParams | None = None) -> dict:
        """Verify vtysh running-config contains the Null0 route advertisement."""
        if params is None:
            params = BGPBlackholeRouteLeakParams()
        host = params.host_name if params.host_name is not None else self.faulty_devices[0]
        network_30 = str(ipaddress.ip_network(f"{self.victim_ip}/30", strict=False))
        running_config = self.kathara_api.exec_cmd(
            host, "vtysh -c 'show running-config' 2>/dev/null"
        ).strip()
        has_null_route = f"ip route {network_30} Null0" in running_config or "Null0" in running_config
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=has_null_route,
            details={"host": host, "network_30": network_30, "has_null_route": has_null_route},
        )


class BGPBlackholeRouteLeakDetection(BGPBlackholeRouteLeakBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=BGPBlackholeRouteLeakBase.root_cause_category,
        root_cause_name=BGPBlackholeRouteLeakBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class BGPBlackholeRouteLeakLocalization(BGPBlackholeRouteLeakBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=BGPBlackholeRouteLeakBase.root_cause_category,
        root_cause_name=BGPBlackholeRouteLeakBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class BGPBlackholeRouteLeakRCA(BGPBlackholeRouteLeakBase, RCATask):
    META = ProblemMeta(
        root_cause_category=BGPBlackholeRouteLeakBase.root_cause_category,
        root_cause_name=BGPBlackholeRouteLeakBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==================================================================
# Problem: BGP hijacking problem.
# ==================================================================


class BGPHijackingParams(BaseModel):
    """Parameters for injecting a BGP hijacking fault."""

    host_name: Optional[str] = Field(default=None, description="Target hijacking router host name. Defaults to a randomly selected router.")
    target_network: Optional[str] = Field(default=None, description="Network prefix to advertise. Defaults to runtime selection.")


class BGPHijackingBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.MISCONFIGURATION
    root_cause_name: str = "bgp_hijacking"
    TAGS: str = ["bgp", "http"]

    Params = BGPHijackingParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaAPIALL(lab_name=self.net_env.lab.name)
        self.logger = system_logger
        self.faulty_devices = [random.choice(self.net_env.routers)]
        web_servers = self.net_env.servers.get("web", [])
        target_host = web_servers[-1] if web_servers else self.net_env.hosts[-1]
        self.target_network = self.kathara_api.get_host_ip(target_host, with_prefix=True)
        self.target_network = str(
            ipaddress.ip_network(self.target_network, strict=False).subnets(new_prefix=25).__next__()
        )

    def inject_fault(self, params: BGPHijackingParams | None = None):
        if params is None:
            params = BGPHijackingParams()
        host = params.host_name if params.host_name is not None else self.faulty_devices[0]
        target_network = params.target_network if params.target_network is not None else self.target_network
        asn_number = self.kathara_api.frr_get_bgp_asn_number(self.faulty_devices[0])
        self.kathara_api.exec_cmd(
            host,
            f"vtysh -c 'configure terminal' -c 'interface lo' -c 'ip address {target_network}' ",
        )
        self.kathara_api.exec_cmd(
            host,
            f"vtysh -c 'configure terminal' -c 'router bgp {asn_number}' -c 'network {target_network}' -c 'end' -c 'write memory' ",
        )
        self.logger.info(f"Injected BGP hijacking on {host}: {target_network}.")

    def verify_fault(self, params: BGPHijackingParams | None = None) -> dict:
        """Verify the router's running-config contains the hijacked network advertisement."""
        if params is None:
            params = BGPHijackingParams()
        host = params.host_name if params.host_name is not None else self.faulty_devices[0]
        target_network = params.target_network if params.target_network is not None else self.target_network
        running_config = self.kathara_api.exec_cmd(
            host, "vtysh -c 'show running-config' 2>/dev/null"
        ).strip()
        has_advertisement = f"network {target_network}" in running_config
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=has_advertisement,
            details={"host": host, "target_network": target_network, "has_advertisement": has_advertisement},
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
