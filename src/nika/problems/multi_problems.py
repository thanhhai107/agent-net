import asyncio

from nika.problems.problem_base import (
    ProblemBase,
    ProblemGroundTruth,
    RootCauseCategory,
    build_verify_result,
)


class MultiFaultProblem(ProblemBase):
    """Composite problem that injects multiple sub-faults in parallel."""

    root_cause_category = RootCauseCategory.MULTIPLE_FAULTS
    root_cause_name: list[str] = []

    def __init__(self, sub_faults: list[ProblemBase], scenario_name: str, **kwargs):
        super().__init__(scenario_name, **kwargs)
        self.sub_faults = sub_faults
        self._refresh_aggregates()

    def _refresh_aggregates(self) -> None:
        root_names: list[str] = []
        devices: list[str] = []
        for fault in self.sub_faults:
            name = fault.root_cause_name
            if isinstance(name, str):
                if name:
                    root_names.append(name)
            else:
                root_names.extend(name)
            for device in fault.faulty_devices:
                if device not in devices:
                    devices.append(device)
        self.root_cause_name = root_names
        self.set_faulty_devices(devices)

    async def _inject_fault_async(self) -> None:
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(None, fault.inject_fault) for fault in self.sub_faults
        ]
        await asyncio.gather(*tasks)

    def inject_fault(self, params=None) -> None:
        del params
        asyncio.run(self._inject_fault_async())
        self._refresh_aggregates()

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
                sub_results.append(
                    {
                        "verified": False,
                        "root_cause_name": getattr(fault, "root_cause_name", "unknown"),
                        "details": {"error": "no verify_fault method"},
                    }
                )
        self._refresh_aggregates()
        return build_verify_result(
            root_cause_name=str(self.root_cause_name),
            faulty_devices=self.faulty_devices,
            verified=all_verified,
            details={"sub_results": sub_results},
        )

    def get_ground_truth(self) -> ProblemGroundTruth:
        self._refresh_aggregates()
        assert self.faulty_devices, (
            "Faulty devices not set before building ground truth."
        )
        return ProblemGroundTruth(
            is_anomaly=True,
            faulty_devices=list(self.faulty_devices),
            root_cause_category=str(self.root_cause_category),
            root_cause_name=list(self.root_cause_name),
            detailed_cause="",
        )
