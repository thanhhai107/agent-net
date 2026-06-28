import logging

from pydantic import BaseModel, Field

from nika.generator.fault.injector_base import FaultInjectorBase
from nika.net_env.net_env_pool import get_net_env_instance
from nika.orchestrator.problems.problem_base import ProblemMeta, RootCauseCategory, TaskDescription, TaskLevel, build_verify_result
from nika.orchestrator.tasks.detection import DetectionTask
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask
from nika.service.kathara import KatharaBaseAPI
from nika.utils.logger import system_logger

logger = system_logger


# ==================================================================
# Problem: DNS service down
# ==================================================================


class DNSServiceDownParams(BaseModel):
    """Parameters for injecting a DNS service down fault."""

    host_name: str = Field(description="Target DNS server host name.")
    service_name: str = Field(default="named", description="Service name.")


class DNSServiceDownBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.LINK_FAILURE
    root_cause_name: str = "dns_service_down"

    faulty_devices = "dns_server"
    symptom_desc = "Some hosts cannot access external websites."
    TAGS: str = ["dns"]

    Params = DNSServiceDownParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaBaseAPI(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorBase(lab_name=self.net_env.lab.name)
        self.faulty_devices: list[str] = []
        self.service_name = "named"

    def inject_fault(self, params: DNSServiceDownParams):
        host = params.host_name
        self.faulty_devices = [host]
        self.injector.inject_process_kill(host_name=host, process_name="named")

    def verify_fault(self, params: DNSServiceDownParams) -> dict:
        """Verify named process is not running."""
        host = params.host_name
        service = params.service_name
        pgrep_output = self.kathara_api.exec_cmd(host, "pgrep -a named 2>/dev/null || echo NONE").strip()
        verified = "named" not in pgrep_output or pgrep_output == "NONE"
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"host": host, "service": service, "pgrep_output": pgrep_output},
        )


class DNSServiceDownDetection(DNSServiceDownBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=DNSServiceDownBase.root_cause_category,
        root_cause_name=DNSServiceDownBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class DNSServiceDownLocalization(DNSServiceDownBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=DNSServiceDownBase.root_cause_category,
        root_cause_name=DNSServiceDownBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class DNSServiceDownRCA(DNSServiceDownBase, RCATask):
    META = ProblemMeta(
        root_cause_category=DNSServiceDownBase.root_cause_category,
        root_cause_name=DNSServiceDownBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==================================================================
# Problem: DHCP service down
# ==================================================================


class DHCPServiceDownParams(BaseModel):
    """Parameters for injecting a DHCP service down fault."""

    host_name: str = Field(description="Target DHCP server host name.")
    service_name: str = Field(default="isc-dhcp-server", description="Service name.")


class DHCPServiceDownBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.LINK_FAILURE
    root_cause_name: str = "dhcp_service_down"

    TAGS: str = ["dhcp"]

    Params = DHCPServiceDownParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaBaseAPI(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorBase(lab_name=self.net_env.lab.name)
        self.faulty_devices: list[str] = []
        self.service_name = "isc-dhcp-server"

    def inject_fault(self, params: DHCPServiceDownParams):
        host = params.host_name
        self.faulty_devices = [host]
        self.injector.inject_process_kill(host_name=host, process_name="dhcpd")

    def verify_fault(self, params: DHCPServiceDownParams) -> dict:
        """Verify DHCP server process is not running."""
        host = params.host_name
        service = params.service_name
        pgrep_output = self.kathara_api.exec_cmd(
            host, "pgrep -a dhcpd 2>/dev/null || echo NONE"
        ).strip()
        verified = "dhcpd" not in pgrep_output or pgrep_output == "NONE"
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"host": host, "service": service, "pgrep_output": pgrep_output},
        )


class DHCPServiceDownDetection(DHCPServiceDownBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=DHCPServiceDownBase.root_cause_category,
        root_cause_name=DHCPServiceDownBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class DHCPServiceDownLocalization(DHCPServiceDownBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=DHCPServiceDownBase.root_cause_category,
        root_cause_name=DHCPServiceDownBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class DHCPServiceDownRCA(DHCPServiceDownBase, RCATask):
    META = ProblemMeta(
        root_cause_category=DHCPServiceDownBase.root_cause_category,
        root_cause_name=DHCPServiceDownBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )
