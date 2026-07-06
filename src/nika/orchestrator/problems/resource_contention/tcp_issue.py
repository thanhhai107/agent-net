from pathlib import Path

from nika.config import pkg_path
from nika.orchestrator.problems.context import init_problem
from nika.orchestrator.problems.problem_base import ProblemMeta, RootCauseCategory, TaskDescription, TaskLevel, build_verify_result
from nika.orchestrator.tasks.detection import DetectionTask
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask
from nika.utils.logger import system_logger
from pydantic import BaseModel, Field

_STRESS_CMD = (
    "nohup stress-ng --cpu 0 --cpu-load 100 --iomix 0 --sock 0 --hdd 2 "
    "--vm 0 --vm-bytes 75% --timeout {duration} </dev/null >/dev/null 2>&1 &"
)


# ==================================================================
# Problem: sender resource contention. Ref. Dapper: Data Plane Performance Diagnosis of TCP
# ==================================================================


class SenderResourceContentionParams(BaseModel):
    """Parameters for injecting a sender resource contention fault."""

    host_name: str = Field(description="Target sender host name.")
    duration: int = Field(default=600, description="Stress duration in seconds.")


class SenderResourceContentionBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.RESOURCE_CONTENTION
    root_cause_name: str = "sender_resource_contention"
    TAGS: str = ["http"]

    Params = SenderResourceContentionParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env, self.runtime = init_problem(scenario_name, **kwargs)
        self.faulty_devices: list[str] = []

    def inject_fault(self, params: SenderResourceContentionParams):
        host = params.host_name
        self.faulty_devices = [host]
        self.runtime.exec(host, _STRESS_CMD.format(duration=params.duration))
        system_logger.info(f"Injected TCP slow sender issue on host {host}")

    def verify_fault(self, params: SenderResourceContentionParams) -> dict:
        """Verify stress-ng is running on the sender host."""
        host = params.host_name
        pgrep_output = self.runtime.exec(host, "pgrep -a stress-ng 2>/dev/null || echo NONE").strip()
        verified = "stress-ng" in pgrep_output and pgrep_output != "NONE"
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"host": host, "pgrep_output": pgrep_output},
        )


class SenderResourceContentionDetection(SenderResourceContentionBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=SenderResourceContentionBase.root_cause_category,
        root_cause_name=SenderResourceContentionBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class SenderResourceContentionLocalization(SenderResourceContentionBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=SenderResourceContentionBase.root_cause_category,
        root_cause_name=SenderResourceContentionBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class SenderResourceContentionRCA(SenderResourceContentionBase, RCATask):
    META = ProblemMeta(
        root_cause_category=SenderResourceContentionBase.root_cause_category,
        root_cause_name=SenderResourceContentionBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==================================================================
# Problem: Application level delay causing TCP sender issues
# ==================================================================


class SenderApplicationDelayParams(BaseModel):
    """Parameters for injecting a sender application delay fault."""

    host_name: str = Field(description="Target sender host name.")


class SenderApplicationDelayBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.RESOURCE_CONTENTION
    root_cause_name: str = "sender_application_delay"
    TAGS: str = ["http"]

    Params = SenderApplicationDelayParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env, self.runtime = init_problem(scenario_name, **kwargs)
        self.faulty_devices: list[str] = []

    def inject_fault(self, params: SenderApplicationDelayParams):
        host = params.host_name
        self.faulty_devices = [host]
        self.runtime.exec(host, "cp web_server.py web_server.py.bak")
        src_path = Path(pkg_path("net_env/kathara/utils/web/slow_sender_server.py"))
        self.runtime.write_file(host, "/web_server.py", src_path.read_text())
        self.runtime.systemctl(host, "web_server.service", "restart")
        system_logger.info(f"Injected TCP sender application delay issue on host {host}")

    def verify_fault(self, params: SenderApplicationDelayParams) -> dict:
        """Verify the web_server.py has a sleep call injected."""
        host = params.host_name
        has_sleep = self.runtime.exec(
            host, "grep -l 'time.sleep' /web_server.py 2>/dev/null && echo yes || echo no"
        ).strip()
        verified = has_sleep.endswith("yes") or has_sleep == "yes"
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"host": host, "has_sleep": has_sleep},
        )


class SenderApplicationDelayDetection(SenderApplicationDelayBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=SenderApplicationDelayBase.root_cause_category,
        root_cause_name=SenderApplicationDelayBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class SenderApplicationDelayLocalization(SenderApplicationDelayBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=SenderApplicationDelayBase.root_cause_category,
        root_cause_name=SenderApplicationDelayBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class SenderApplicationDelayRCA(SenderApplicationDelayBase, RCATask):
    META = ProblemMeta(
        root_cause_category=SenderApplicationDelayBase.root_cause_category,
        root_cause_name=SenderApplicationDelayBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==================================================================
# Problem: receiver resource contention
# ==================================================================


class ReceiverResourceContentionParams(BaseModel):
    """Parameters for injecting a receiver resource contention fault."""

    host_name: str = Field(description="Target receiver host name.")
    duration: int = Field(default=600, description="Stress duration in seconds.")


class ReceiverResourceContentionBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.RESOURCE_CONTENTION
    root_cause_name: str = "receiver_resource_contention"
    TAGS: str = ["http"]

    Params = ReceiverResourceContentionParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env, self.runtime = init_problem(scenario_name, **kwargs)
        self.faulty_devices: list[str] = []

    def inject_fault(self, params: ReceiverResourceContentionParams):
        host = params.host_name
        self.faulty_devices = [host]
        self.runtime.exec(host, _STRESS_CMD.format(duration=params.duration))
        system_logger.info(f"Injected TCP receiver resource contention on host {host}")

    def verify_fault(self, params: ReceiverResourceContentionParams) -> dict:
        """Verify stress-ng is running on the receiver host."""
        host = params.host_name
        pgrep_output = self.runtime.exec(host, "pgrep -a stress-ng 2>/dev/null || echo NONE").strip()
        verified = "stress-ng" in pgrep_output and pgrep_output != "NONE"
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"host": host, "pgrep_output": pgrep_output},
        )


class ReceiverResourceContentionDetection(ReceiverResourceContentionBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=ReceiverResourceContentionBase.root_cause_category,
        root_cause_name=ReceiverResourceContentionBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class ReceiverResourceContentionLocalization(ReceiverResourceContentionBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=ReceiverResourceContentionBase.root_cause_category,
        root_cause_name=ReceiverResourceContentionBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class ReceiverResourceContentionRCA(ReceiverResourceContentionBase, RCATask):
    META = ProblemMeta(
        root_cause_category=ReceiverResourceContentionBase.root_cause_category,
        root_cause_name=ReceiverResourceContentionBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )
