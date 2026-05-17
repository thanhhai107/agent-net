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
from nika.utils.logger import system_logger

# ==================================================================
# Problem: DHCP missing subnet
# ==================================================================


class DHCPMissingSubnetBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.MISCONFIGURATION
    root_cause_name: str = "dhcp_missing_subnet"

    TAGS: str = ["dhcp"]
    FAILURE_PARAM_SCHEMA = FailureParamSchema(
        problem_name="dhcp_missing_subnet",
        summary="Delete DHCP subnet config for one affected host subnet.",
        fields=(
            FailureParamField("host_name", "str", "DHCP server host name."),
            FailureParamField("host_name_2", "str", "Affected client host name."),
        ),
        example="nika failure inject dhcp_missing_subnet --set host_name=dhcp0 --set host_name_2=client1",
    )

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaBaseAPI(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorService(lab_name=self.net_env.lab.name)
        self.faulty_devices = [random.choice(self.net_env.servers["dhcp"])]
        self.faulty_devices.append(random.choice(self.net_env.hosts))

    def inject_fault(self):
        system_logger.info(
            f"Injecting DHCP missing subnet fault: DHCP server {self.faulty_devices[0]}, affected host {self.faulty_devices[1]}"
        )
        subnet = str(
            ipaddress.ip_network(
                self.kathara_api.get_host_ip(self.faulty_devices[1], with_prefix=True), strict=False
            ).network_address
        )
        self.injector.inject_delete_subnet(
            dhcp_server=self.faulty_devices[0],
            subnet=subnet,
        )

class DHCPMissingSubnetDetection(DHCPMissingSubnetBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=DHCPMissingSubnetBase.root_cause_category,
        root_cause_name=DHCPMissingSubnetBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class DHCPMissingSubnetLocalization(DHCPMissingSubnetBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=DHCPMissingSubnetBase.root_cause_category,
        root_cause_name=DHCPMissingSubnetBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class DHCPMissingSubnetRCA(DHCPMissingSubnetBase, RCATask):
    META = ProblemMeta(
        root_cause_category=DHCPMissingSubnetBase.root_cause_category,
        root_cause_name=DHCPMissingSubnetBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


if __name__ == "__main__":
    problem = DHCPMissingSubnetRCA(scenario_name="ospf_enterprise_dhcp")
    # problem.inject_fault()
