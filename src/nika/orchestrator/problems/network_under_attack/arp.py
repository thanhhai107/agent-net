from pydantic import BaseModel, Field

from nika.orchestrator.problems.context import init_problem
from nika.orchestrator.problems.problem_base import ProblemMeta, RootCauseCategory, TaskDescription, TaskLevel, build_verify_result
from nika.orchestrator.tasks.detection import DetectionTask
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask
from nika.utils.logger import system_logger

# ==================================================================
# Problem: Arp cache poisoning causing data plane issues.
# ==================================================================


class ArpCachePoisoningParams(BaseModel):
    """Parameters for injecting an ARP cache poisoning fault."""

    host_name: str = Field(description="Target host name.")
    fake_mac: str = Field(default="00:11:22:33:44:55", description="Forged MAC address.")


class ArpCachePoisoningBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.NETWORK_UNDER_ATTACK
    root_cause_name: str = "arp_cache_poisoning"
    TAGS: str = ["arp"]

    Params = ArpCachePoisoningParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.logger = system_logger
        self.net_env, self.runtime = init_problem(scenario_name, **kwargs)
        self.faulty_devices: list[str] = []

    def inject_fault(self, params: ArpCachePoisoningParams):
        host = params.host_name
        self.faulty_devices = [host]
        default_gateway = self.runtime.get_default_gateway(host)
        self.runtime.exec(host, f"arp -s {default_gateway} {params.fake_mac}")

    def verify_fault(self, params: ArpCachePoisoningParams) -> dict:
        """Verify the ARP cache has the fake MAC for the default gateway."""
        host = params.host_name
        fake_mac = params.fake_mac
        gateway = self.runtime.get_default_gateway(host)
        neigh_output = self.runtime.exec(host, f"ip neigh show | grep '{fake_mac}'").strip()
        verified = bool(neigh_output)
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"host": host, "gateway": gateway, "fake_mac": fake_mac, "neigh_entry": neigh_output},
        )


class ArpCachePoisoningDetection(ArpCachePoisoningBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=ArpCachePoisoningBase.root_cause_category,
        root_cause_name=ArpCachePoisoningBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class ArpCachePoisoningLocalization(ArpCachePoisoningBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=ArpCachePoisoningBase.root_cause_category,
        root_cause_name=ArpCachePoisoningBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class ArpCachePoisoningRCA(ArpCachePoisoningBase, RCATask):
    META = ProblemMeta(
        root_cause_category=ArpCachePoisoningBase.root_cause_category,
        root_cause_name=ArpCachePoisoningBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )
