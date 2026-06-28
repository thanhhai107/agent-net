import ipaddress

from pydantic import BaseModel, Field

from nika.generator.fault.injector_service import FaultInjectorService
from nika.net_env.net_env_pool import get_net_env_instance
from nika.orchestrator.problems.problem_base import ProblemMeta, RootCauseCategory, TaskDescription, TaskLevel, build_verify_result
from nika.orchestrator.tasks.detection import DetectionTask
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask
from nika.service.kathara import KatharaBaseAPI

# ==================================================================
# Problem: DHCP distributing spoofed gateway to hosts
# ==================================================================


class DHCPSpoofedGatewayParams(BaseModel):
    """Parameters for injecting a DHCP spoofed gateway fault."""

    host_name: str = Field(description="DHCP server host name.")
    host_name_2: str = Field(description="Affected client host name.")


class DHCPSpoofedGatewayBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.NETWORK_UNDER_ATTACK
    root_cause_name: str = "dhcp_spoofed_gateway"

    TAGS: str = ["dhcp"]

    Params = DHCPSpoofedGatewayParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaBaseAPI(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorService(lab_name=self.net_env.lab.name)
        self.faulty_devices: list[str] = []

    def _client_subnet(self, client_host: str) -> str:
        return str(
            ipaddress.ip_network(
                self.kathara_api.get_host_ip(client_host, with_prefix=True), strict=False
            ).network_address
        )

    def inject_fault(self, params: DHCPSpoofedGatewayParams):
        dhcp_server = params.host_name
        client_host = params.host_name_2
        self.faulty_devices = [dhcp_server, client_host]
        subnet = self._client_subnet(client_host)
        self.injector.inject_wrong_gateway(
            dhcp_server=dhcp_server,
            subnet=subnet,
            wrong_gw=".".join(subnet.split(".")[:3] + ["254"]),
        )

    def verify_fault(self, params: DHCPSpoofedGatewayParams) -> dict:
        """Verify dhcpd.conf has spoofed gateway ending in .254."""
        dhcp_server = params.host_name
        grep_result = self.kathara_api.exec_cmd(
            dhcp_server,
            "grep 'option routers.*\\.254' /etc/dhcp/dhcpd.conf && echo found || echo absent",
        ).strip()
        verified = "found" in grep_result
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"dhcp_server": dhcp_server, "grep_result": grep_result},
        )


class DHCPSpoofedGatewayDetection(DHCPSpoofedGatewayBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=DHCPSpoofedGatewayBase.root_cause_category,
        root_cause_name=DHCPSpoofedGatewayBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class DHCPSpoofedGatewayLocalization(DHCPSpoofedGatewayBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=DHCPSpoofedGatewayBase.root_cause_category,
        root_cause_name=DHCPSpoofedGatewayBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class DHCPSpoofedGatewayRCA(DHCPSpoofedGatewayBase, RCATask):
    META = ProblemMeta(
        root_cause_category=DHCPSpoofedGatewayBase.root_cause_category,
        root_cause_name=DHCPSpoofedGatewayBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==================================================================
# Problem: DHCP distributing spoofed DNS to hosts
# ==================================================================


class DHCPSpoofedDNSParams(BaseModel):
    """Parameters for injecting a DHCP spoofed DNS fault."""

    host_name: str = Field(description="DHCP server host name.")
    host_name_2: str = Field(description="Affected client host name.")
    wrong_dns: str = Field(default="8.8.8.8", description="Spoofed DNS IP.")


class DHCPSpoofedDNSBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.NETWORK_UNDER_ATTACK
    root_cause_name: str = "dhcp_spoofed_dns"

    symptom_desc = "Some hosts can not access webservices."
    TAGS: str = ["dhcp"]

    Params = DHCPSpoofedDNSParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaBaseAPI(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorService(lab_name=self.net_env.lab.name)
        self.faulty_devices: list[str] = []

    def _client_subnet(self, client_host: str) -> str:
        return str(
            ipaddress.ip_network(
                self.kathara_api.get_host_ip(client_host, with_prefix=True), strict=False
            ).network_address
        )

    def inject_fault(self, params: DHCPSpoofedDNSParams):
        dhcp_server = params.host_name
        client_host = params.host_name_2
        self.faulty_devices = [dhcp_server, client_host]
        subnet = self._client_subnet(client_host)
        self.injector.inject_wrong_dns(dhcp_server=dhcp_server, subnet=subnet, wrong_dns=params.wrong_dns)

    def verify_fault(self, params: DHCPSpoofedDNSParams) -> dict:
        """Verify dhcpd.conf has spoofed DNS server 8.8.8.8."""
        dhcp_server = params.host_name
        wrong_dns = params.wrong_dns
        grep_result = self.kathara_api.exec_cmd(
            dhcp_server,
            f"grep 'option domain-name-servers.*{wrong_dns}' /etc/dhcp/dhcpd.conf && echo found || echo absent",
        ).strip()
        verified = "found" in grep_result
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"dhcp_server": dhcp_server, "wrong_dns": wrong_dns, "grep_result": grep_result},
        )


class DHCPSpoofedDNSDetection(DHCPSpoofedDNSBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=DHCPSpoofedDNSBase.root_cause_category,
        root_cause_name=DHCPSpoofedDNSBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class DHCPSpoofedDNSLocalization(DHCPSpoofedDNSBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=DHCPSpoofedDNSBase.root_cause_category,
        root_cause_name=DHCPSpoofedDNSBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class DHCPSpoofedDNSRCA(DHCPSpoofedDNSBase, RCATask):
    META = ProblemMeta(
        root_cause_category=DHCPSpoofedDNSBase.root_cause_category,
        root_cause_name=DHCPSpoofedDNSBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==================================================================
""" Problem: DHCP missing subnet configuration """
# ==================================================================


class DHCPSpoofedSubnetParams(BaseModel):
    """Parameters for injecting a DHCP spoofed subnet fault."""

    host_name: str = Field(description="DHCP server host name.")
    host_name_2: str = Field(description="Affected client host name.")


class DHCPSpoofedSubnetBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.NETWORK_UNDER_ATTACK
    root_cause_name: str = "dhcp_spoofed_subnet"

    TAGS: str = ["dhcp"]

    Params = DHCPSpoofedSubnetParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaBaseAPI(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorService(lab_name=self.net_env.lab.name)
        self.faulty_devices: list[str] = []

    def _client_subnet(self, client_host: str) -> str:
        return str(
            ipaddress.ip_network(
                self.kathara_api.get_host_ip(client_host, with_prefix=True), strict=False
            ).network_address
        )

    def inject_fault(self, params: DHCPSpoofedSubnetParams):
        dhcp_server = params.host_name
        client_host = params.host_name_2
        self.faulty_devices = [dhcp_server, client_host]
        subnet = self._client_subnet(client_host)
        self.deleted_subnet = subnet
        self.injector.inject_delete_subnet(dhcp_server=dhcp_server, subnet=subnet)

    def verify_fault(self, params: DHCPSpoofedSubnetParams) -> dict:
        """Verify the target subnet has been removed from dhcpd.conf."""
        dhcp_server = params.host_name
        subnet = getattr(self, "deleted_subnet", None) or self._client_subnet(params.host_name_2)
        sub_escaped = subnet.replace(".", "\\.")
        match_output = self.kathara_api.exec_cmd(
            dhcp_server,
            f"grep 'subnet {sub_escaped} netmask' /etc/dhcp/dhcpd.conf | wc -l",
        ).strip()
        count_output = self.kathara_api.exec_cmd(
            dhcp_server,
            "grep 'subnet.*netmask' /etc/dhcp/dhcpd.conf | wc -l",
        ).strip()
        try:
            match_count = int(match_output)
        except ValueError:
            match_count = -1
        try:
            subnet_count = int(count_output)
        except ValueError:
            subnet_count = -1
        verified = match_count == 0
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"dhcp_server": dhcp_server, "subnet_count": subnet_count, "deleted_subnet": subnet},
        )


class DHCPSpoofedSubnetDetection(DHCPSpoofedSubnetBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=DHCPSpoofedSubnetBase.root_cause_category,
        root_cause_name=DHCPSpoofedSubnetBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class DHCPSpoofedSubnetLocalization(DHCPSpoofedSubnetBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=DHCPSpoofedSubnetBase.root_cause_category,
        root_cause_name=DHCPSpoofedSubnetBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class DHCPSpoofedSubnetRCA(DHCPSpoofedSubnetBase, RCATask):
    META = ProblemMeta(
        root_cause_category=DHCPSpoofedSubnetBase.root_cause_category,
        root_cause_name=DHCPSpoofedSubnetBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )
