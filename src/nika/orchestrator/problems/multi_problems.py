import asyncio

from nika.net_env.net_env_pool import get_net_env_instance
from nika.orchestrator.problems.problem_base import ProblemMeta, RootCauseCategory, TaskDescription, TaskLevel, build_verify_result
from nika.orchestrator.tasks.base import TaskBase
from nika.orchestrator.tasks.detection import DetectionTask
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask


class MultiFaultBase(TaskBase):
    root_cause_category = RootCauseCategory.MULTIPLE_FAULTS
    root_cause_name = ""  # can only be get after init

    def __init__(self, sub_faults: list[TaskBase], scenario_name: str, **kwargs):
        super().__init__()
        self.sub_faults = sub_faults
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.root_cause_name = [f.root_cause_name for f in sub_faults]
        self.faulty_devices = []
        for sub_fault in self.sub_faults:
            if isinstance(sub_fault.faulty_devices, list):
                self.faulty_devices.extend(sub_fault.faulty_devices)
            else:
                self.faulty_devices.append(sub_fault.faulty_devices)

    async def _inject_fault_async(self):
        loop = asyncio.get_running_loop()
        tasks = [loop.run_in_executor(None, fault.inject_fault) for fault in self.sub_faults]
        await asyncio.gather(*tasks)

    def inject_fault(self):
        asyncio.run(self._inject_fault_async())

    def verify_fault(self) -> dict:
        """Verify all sub-faults and aggregate results."""
        sub_results = []
        all_verified = True
        for fault in self.sub_faults:
            if hasattr(fault, "verify_fault"):
                r = fault.verify_fault()
                sub_results.append(r)
                if not r.get("verified", False):
                    all_verified = False
            else:
                all_verified = False
                sub_results.append({
                    "verified": False,
                    "root_cause_name": getattr(fault, "root_cause_name", "unknown"),
                    "details": {"error": "no verify_fault method"},
                })
        return build_verify_result(
            root_cause_name=str(self.root_cause_name),
            faulty_devices=self.faulty_devices,
            verified=all_verified,
            details={"sub_results": sub_results},
        )


class MultiFaultDetection(MultiFaultBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=MultiFaultBase.root_cause_category,
        root_cause_name=MultiFaultBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class MultiFaultLocalization(MultiFaultBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=MultiFaultBase.root_cause_category,
        root_cause_name=MultiFaultBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class MultiFaultRCA(MultiFaultBase, RCATask):
    META = ProblemMeta(
        root_cause_category=MultiFaultBase.root_cause_category,
        root_cause_name=MultiFaultBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )
