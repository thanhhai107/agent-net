from pydantic import BaseModel, Field

from nika.orchestrator.problems.context import init_problem
from nika.orchestrator.problems.problem_base import ProblemMeta, RootCauseCategory, TaskDescription, TaskLevel, build_verify_result
from nika.orchestrator.tasks.detection import DetectionTask
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask

# ==========================================
# Problem: Host crash simulated by pausing a docker instance
# ==========================================


class HostCrashParams(BaseModel):
    """Parameters for injecting a host-crash fault."""

    host_name: str = Field(description="Target host name.")


class HostCrashBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.END_HOST_FAILURE
    root_cause_name: str = "host_crash"
    TAGS: str = ["pc"]

    Params = HostCrashParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env, self.runtime = init_problem(scenario_name, **kwargs)
        self.faulty_devices: list[str] = []

    def inject_fault(self, params: HostCrashParams):
        host = params.host_name
        self.faulty_devices = [host]
        self.runtime.pause(host)

    def verify_fault(self, params: HostCrashParams) -> dict:
        """Verify the host container is paused (simulated crash)."""
        host = params.host_name
        container_status = self.runtime.node_status(host)
        verified = container_status == "paused"
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"host": host, "container_status": container_status},
        )


class HostCrashDetection(HostCrashBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=HostCrashBase.root_cause_category,
        root_cause_name=HostCrashBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class HostCrashLocalization(HostCrashBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=HostCrashBase.root_cause_category,
        root_cause_name=HostCrashBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class HostCrashRCA(HostCrashBase, RCATask):
    META = ProblemMeta(
        root_cause_category=HostCrashBase.root_cause_category,
        root_cause_name=HostCrashBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )
