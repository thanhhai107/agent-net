import ipaddress
import random

from nika.generator.fault.injector_service import FaultInjectorService
from nika.net_env.net_env_pool import get_net_env_instance
from nika.orchestrator.problems.problem_base import ProblemMeta, RootCauseCategory, TaskDescription, TaskLevel
from nika.orchestrator.tasks.detection import DetectionTask
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask
from nika.service.kathara import KatharaBaseAPI
from nika.utils.failure_params import FailureParamField, FailureParamSchema

# ==================================================================
# Problem: DHCP distributing spoofed gateway to hosts
# ==================================================================


class DHCPSpoofedGatewayBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.NETWORK_UNDER_ATTACK
    root_cause_name: str = "dhcp_spoofed_gateway"

    TAGS: str = ["dhcp"]
    FAILURE_PARAM_SCHEMA = FailureParamSchema(
        problem_name="dhcp_spoofed_gateway",
        summary="Distribute spoofed gateway via DHCP.",
        fields=(
            FailureParamField("host_name", "str", "DHCP server host name."),
            FailureParamField("host_name_2", "str", "Affected client host name."),
        ),
        example="nika failure inject dhcp_spoofed_gateway --set host_name=dhcp0 --set host_name_2=h1",
    )

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaBaseAPI(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorService(lab_name=self.net_env.lab.name)
        self.faulty_devices = [random.choice(self.net_env.servers["dhcp"])]
        self.faulty_devices.append(random.choice(self.net_env.hosts))

    def inject_fault(self):
        subnet = str(
            ipaddress.ip_network(
                self.kathara_api.get_host_ip(self.faulty_devices[1], with_prefix=True), strict=False
            ).network_address
        )

        self.injector.inject_wrong_gateway(
            dhcp_server=self.faulty_devices[0],
            subnet=subnet,
            wrong_gw=".".join(subnet.split(".")[:3] + ["254"]),
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


class DHCPSpoofedDNSBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.NETWORK_UNDER_ATTACK
    root_cause_name: str = "dhcp_spoofed_dns"

    symptom_desc = "Some hosts can not access webservices."
    TAGS: str = ["dhcp"]
    FAILURE_PARAM_SCHEMA = FailureParamSchema(
        problem_name="dhcp_spoofed_dns",
        summary="Distribute spoofed DNS via DHCP.",
        fields=(
            FailureParamField("host_name", "str", "DHCP server host name."),
            FailureParamField("host_name_2", "str", "Affected client host name."),
            FailureParamField("wrong_dns", "str", "Spoofed DNS IP.", default="8.8.8.8"),
        ),
        example="nika failure inject dhcp_spoofed_dns --set host_name=dhcp0 --set host_name_2=h1",
    )

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaBaseAPI(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorService(lab_name=self.net_env.lab.name)
        self.faulty_devices = [random.choice(self.net_env.servers["dhcp"])]
        self.faulty_devices.append(random.choice(self.net_env.hosts))
        self.wrong_dns = "8.8.8.8"

    def inject_fault(self):
        subnet = str(
            ipaddress.ip_network(
                self.kathara_api.get_host_ip(self.faulty_devices[1], with_prefix=True), strict=False
            ).network_address
        )

        self.injector.inject_wrong_dns(dhcp_server=self.faulty_devices[0], subnet=subnet, wrong_dns=self.wrong_dns)

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


class DHCPSpoofedSubnetBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.NETWORK_UNDER_ATTACK
    root_cause_name: str = "dhcp_spoofed_subnet"

    TAGS: str = ["dhcp"]
    FAILURE_PARAM_SCHEMA = FailureParamSchema(
        problem_name="dhcp_spoofed_subnet",
        summary="Delete DHCP subnet entry for an active client subnet.",
        fields=(
            FailureParamField("host_name", "str", "DHCP server host name."),
            FailureParamField("host_name_2", "str", "Affected client host name."),
        ),
        example="nika failure inject dhcp_spoofed_subnet --set host_name=dhcp0 --set host_name_2=h1",
    )

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaBaseAPI(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorService(lab_name=self.net_env.lab.name)
        self.faulty_devices = [random.choice(self.net_env.servers["dhcp"])]
        self.faulty_devices.append(random.choice(self.net_env.hosts))

    def inject_fault(self):
        subnet = str(
            ipaddress.ip_network(
                self.kathara_api.get_host_ip(self.faulty_devices[1], with_prefix=True), strict=False
            ).network_address
        )
        self.injector.inject_delete_subnet(
            dhcp_server=self.faulty_devices[0],
            subnet=subnet,
        )

