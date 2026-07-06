import re

from nika.orchestrator.problems.context import init_problem
from pydantic import BaseModel, Field

from nika.orchestrator.problems.problem_base import ProblemMeta, RootCauseCategory, TaskDescription, TaskLevel, build_verify_result
from nika.orchestrator.tasks.detection import DetectionTask
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask
from nika.utils.logger import system_logger

# ==================================================================
# Problem: OSPF Area Misconfiguration
# ==================================================================


class OSPFAreaMisconfigParams(BaseModel):
    """Parameters for injecting an OSPF area misconfiguration fault."""

    host_name: str = Field(description="Target router host name.")


class OSPFAreaMisconfigBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.MISCONFIGURATION
    root_cause_name: str = "ospf_area_misconfiguration"

    TAGS: str = ["ospf"]

    Params = OSPFAreaMisconfigParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env, self.runtime = init_problem(scenario_name, **kwargs)
        self.logger = system_logger
        self.faulty_devices: list[str] = []

    def inject_fault(self, params: OSPFAreaMisconfigParams):
        host = params.host_name
        self.faulty_devices = [host]
        running_cfg = self.runtime.exec(host, "vtysh -c 'show running-config'")
        pattern = re.compile(r"^\s*network\s+\S+\s+area\s+(\S+)", re.MULTILINE)
        m = pattern.search(running_cfg)
        if not m:
            self.logger.error(f"Could not find OSPF area on {host}")
        correct_area = m.group(1)
        wrong_area = "66" if correct_area != "66" else "99"
        self.runtime.exec(
            host,
            f"sed -i.bak -E 's/(area ){correct_area}$/\\1{wrong_area}/g' /etc/frr/frr.conf && service frr restart 2>/dev/null || true",
        )
        self.logger.info(f"Injected OSPF area misconfiguration on {host} from area {correct_area} to {wrong_area}.")

    def verify_fault(self, params: OSPFAreaMisconfigParams) -> dict:
        """Verify the OSPF area in frr.conf and in the running daemon was changed."""
        host = params.host_name
        self.faulty_devices = [host]
        file_areas_raw = self.runtime.exec(
            host,
            "grep -E '^[[:space:]]*network .* area ' /etc/frr/frr.conf 2>/dev/null | awk '{print $NF}' | sort -u",
        ).strip()
        orig_areas_raw = self.runtime.exec(
            host,
            "grep -E '^[[:space:]]*network .* area ' /etc/frr/frr.conf.bak 2>/dev/null | awk '{print $NF}' | sort -u",
        ).strip()
        running_areas_raw = self.runtime.exec(
            host,
            "vtysh -c 'show running-config' 2>/dev/null | grep -E '^[[:space:]]*network .* area ' | awk '{print $NF}' | sort -u",
        ).strip()
        file_areas = set(file_areas_raw.splitlines()) if file_areas_raw else set()
        orig_areas = set(orig_areas_raw.splitlines()) if orig_areas_raw else set()
        running_areas = set(running_areas_raw.splitlines()) if running_areas_raw else set()
        file_changed = bool(file_areas) and bool(orig_areas) and file_areas != orig_areas
        daemon_changed = bool(running_areas) and running_areas != orig_areas
        verified = file_changed and daemon_changed
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={
                "host": host,
                "file_areas": list(file_areas),
                "orig_areas": list(orig_areas),
                "running_areas": list(running_areas),
            },
        )


class OSPFAreaMisconfigDetection(OSPFAreaMisconfigBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=OSPFAreaMisconfigBase.root_cause_category,
        root_cause_name=OSPFAreaMisconfigBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class OSPFAreaMisconfigLocalization(OSPFAreaMisconfigBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=OSPFAreaMisconfigBase.root_cause_category,
        root_cause_name=OSPFAreaMisconfigBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class OSPFAreaMisconfigRCA(OSPFAreaMisconfigBase, RCATask):
    META = ProblemMeta(
        root_cause_category=OSPFAreaMisconfigBase.root_cause_category,
        root_cause_name=OSPFAreaMisconfigBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==================================================================
# Problem: OSPF Neighbor Missing
# ==================================================================


class OSPFNeighborMissingParams(BaseModel):
    """Parameters for injecting an OSPF neighbor missing fault."""

    host_name: str = Field(description="Target router host name.")


class OSPFNeighborMissingBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.MISCONFIGURATION
    root_cause_name: str = "ospf_neighbor_missing"

    TAGS: str = ["ospf"]

    Params = OSPFNeighborMissingParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env, self.runtime = init_problem(scenario_name, **kwargs)
        self.logger = system_logger
        self.faulty_devices: list[str] = []

    def inject_fault(self, params: OSPFNeighborMissingParams):
        host = params.host_name
        self.faulty_devices = [host]
        cmd = (
            "sed -i.bak -E "
            "'s|^([[:space:]]*)network([[:space:]])|\\1# network\\2|' "
            "/etc/frr/frr.conf"
        )
        self.runtime.exec(host, cmd)
        self.runtime.exec(host, "service frr restart 2>/dev/null || true")
        self.logger.info(f"Injected OSPF neighbor missing on {host}.")

    def verify_fault(self, params: OSPFNeighborMissingParams) -> dict:
        """Verify network lines are commented in frr.conf and removed from the running daemon."""
        host = params.host_name
        self.faulty_devices = [host]
        commented_count_raw = self.runtime.exec(
            host,
            "grep -c '^[[:space:]]*# network' /etc/frr/frr.conf 2>/dev/null || echo 0",
        ).strip()
        try:
            commented_count = int(commented_count_raw)
        except ValueError:
            commented_count = 0
        running_network_count_raw = self.runtime.exec(
            host,
            "vtysh -c 'show running-config' 2>/dev/null | grep -c '^[[:space:]]*network' || echo 0",
        ).strip()
        try:
            running_network_count = int(running_network_count_raw)
        except ValueError:
            running_network_count = 0
        verified = commented_count > 0 and running_network_count == 0
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={
                "host": host,
                "commented_network_count": commented_count,
                "running_network_count": running_network_count,
            },
        )


class OSPFNeighborMissingDetection(OSPFNeighborMissingBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=OSPFNeighborMissingBase.root_cause_category,
        root_cause_name=OSPFNeighborMissingBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class OSPFNeighborMissingLocalization(OSPFNeighborMissingBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=OSPFNeighborMissingBase.root_cause_category,
        root_cause_name=OSPFNeighborMissingBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class OSPFNeighborMissingRCA(OSPFNeighborMissingBase, RCATask):
    META = ProblemMeta(
        root_cause_category=OSPFNeighborMissingBase.root_cause_category,
        root_cause_name=OSPFNeighborMissingBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )
