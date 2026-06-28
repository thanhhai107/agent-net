from pydantic import BaseModel, Field

from nika.generator.fault.injector_service import FaultInjectorService
from nika.net_env.base import NetworkEnvBase
from nika.net_env.net_env_pool import get_net_env_instance
from nika.orchestrator.problems.problem_base import ProblemMeta, RootCauseCategory, TaskDescription, TaskLevel, build_verify_result
from nika.orchestrator.tasks.detection import DetectionTask
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask
from nika.service.kathara import KatharaAPIALL
from nika.utils.logger import system_logger

# ==================================================================
# Problem: Web service under DoS attack
# ==================================================================


class WebDoSParams(BaseModel):
    """Parameters for injecting a web DoS attack fault."""

    host_name: str = Field(description="Target web server host name.")
    attacker_device: str = Field(description="Attacker host name.")


class WebDoSBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.NETWORK_UNDER_ATTACK
    root_cause_name: str = "web_dos_attack"
    symptom_desc: str = "Users reports high latency when accessing some web services."
    TAGS: str = ["http"]

    Params = WebDoSParams

    def __init__(self, scenario_name: NetworkEnvBase, **kwargs):
        super().__init__()
        self.logger = system_logger
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaAPIALL(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorService(lab_name=self.net_env.lab.name)
        self.faulty_devices: list[str] = []

    def inject_fault(self, params: WebDoSParams):
        web_server = params.host_name
        attacker = params.attacker_device
        self.faulty_devices = [web_server]
        target_ip = self.kathara_api.get_host_ip(web_server, with_prefix=False)
        self.injector.inject_ab_attack(attacker_host=attacker, website=target_ip)

    def verify_fault(self, params: WebDoSParams) -> dict:
        """Verify the ab attack process is running on the attacker device."""
        web_server = params.host_name
        attacker = params.attacker_device
        target_ip = self.kathara_api.get_host_ip(web_server, with_prefix=False)
        pgrep_output = self.kathara_api.exec_cmd(attacker, "pgrep -a ab 2>/dev/null || echo NONE").strip()
        verified = "ab" in pgrep_output and pgrep_output != "NONE"
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"attacker": attacker, "target_ip": target_ip, "pgrep_output": pgrep_output},
        )


class WebDoSDetection(WebDoSBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=WebDoSBase.root_cause_category,
        root_cause_name=WebDoSBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class WebDoSLocalization(WebDoSBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=WebDoSBase.root_cause_category,
        root_cause_name=WebDoSBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class WebDoSRCA(WebDoSBase, RCATask):
    META = ProblemMeta(
        root_cause_category=WebDoSBase.root_cause_category,
        root_cause_name=WebDoSBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )
