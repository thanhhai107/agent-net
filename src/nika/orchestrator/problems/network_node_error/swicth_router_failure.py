from pydantic import BaseModel, Field

from nika.generator.fault.injector_base import FaultInjectorBase
from nika.net_env.net_env_pool import get_net_env_instance
from nika.orchestrator.problems.problem_base import ProblemMeta, RootCauseCategory, TaskDescription, TaskLevel, build_verify_result
from nika.orchestrator.tasks.detection import DetectionTask
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask
from nika.service.kathara import KatharaAPIALL, KatharaBaseAPI

# ==================================================================
# Problem: P4 switch device failure (bmv2 switch down)
# ==================================================================


class Bmv2SwitchDownParams(BaseModel):
    """Parameters for injecting a BMv2 switch down fault."""

    host_name: str = Field(description="Target BMv2 switch name.")


class Bmv2SwitchDownBase:
    root_cause_category = RootCauseCategory.LINK_FAILURE
    root_cause_name = "bmv2_switch_down"
    TAGS: str = ["p4"]

    Params = Bmv2SwitchDownParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaAPIALL(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorBase(lab_name=self.net_env.lab.name)
        self.faulty_devices: list[str] = []

    def inject_fault(self, params: Bmv2SwitchDownParams):
        host = params.host_name
        self.faulty_devices = [host]
        self.injector.inject_bmv2_down(host_name=host)

    def verify_fault(self, params: Bmv2SwitchDownParams) -> dict:
        """Verify simple_switch process is NOT running on the BMv2 switch."""
        host = params.host_name
        pgrep_output = self.kathara_api.exec_cmd(host, "pgrep -a simple_switch 2>/dev/null || echo NONE").strip()
        verified = pgrep_output == "NONE" or "simple_switch" not in pgrep_output
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"host": host, "pgrep_output": pgrep_output},
        )


class Bmv2SwitchDownDetection(Bmv2SwitchDownBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=Bmv2SwitchDownBase.root_cause_category,
        root_cause_name=Bmv2SwitchDownBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class Bmv2SwitchDownLocalization(Bmv2SwitchDownBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=Bmv2SwitchDownBase.root_cause_category,
        root_cause_name=Bmv2SwitchDownBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class Bmv2SwitchDownRCA(Bmv2SwitchDownBase, RCATask):
    META = ProblemMeta(
        root_cause_category=Bmv2SwitchDownBase.root_cause_category,
        root_cause_name=Bmv2SwitchDownBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==================================================================
# Problem: FRR service down on a router device
# ==================================================================


class FrrDownParams(BaseModel):
    """Parameters for injecting an FRR service down fault."""

    host_name: str = Field(description="Target router host name.")
    service_name: str = Field(default="frr", description="Service name.")


class FrrDownBase:
    """Base class for a FRR device down problem."""

    root_cause_category: RootCauseCategory = RootCauseCategory.NETWORK_NODE_ERROR
    root_cause_name: str = "frr_service_down"
    TAGS: str = ["frr"]

    Params = FrrDownParams

    symptom_desc = "Users report connectivity issues to other hosts in the network."

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaBaseAPI(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorBase(lab_name=self.net_env.lab.name)
        self.faulty_devices: list[str] = []

    def inject_fault(self, params: FrrDownParams):
        host = params.host_name
        self.faulty_devices = [host]
        # systemctl is a no-op in Kathara; kill FRR daemons directly with pkill.
        # watchfrr must be killed first so it does not restart the routing daemons.
        for daemon in ("watchfrr", "zebra", "mgmtd", "ospfd", "bgpd", "staticd", "ospf6d", "ripd"):
            self.injector.inject_process_kill(host_name=host, process_name=daemon)

    def verify_fault(self, params: FrrDownParams) -> dict:
        """Verify FRR is down by checking zebra is not running and routing is unavailable."""
        host = params.host_name
        zebra_output = self.kathara_api.exec_cmd(host, "pgrep -a zebra 2>/dev/null || echo NONE").strip()
        # show version still succeeds in FRR 9.x when zebra is down; use show ip route instead.
        vtysh_output = self.kathara_api.exec_cmd(
            host, "vtysh -c 'show ip route' 2>&1 | head -3"
        ).strip()
        zebra_down = zebra_output == "NONE" or "zebra" not in zebra_output
        routing_unavailable = (
            "failed to connect" in vtysh_output.lower() or "not running" in vtysh_output.lower()
        )
        verified = zebra_down and routing_unavailable
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={
                "host": host,
                "zebra_output": zebra_output,
                "vtysh_output": vtysh_output,
                "zebra_down": zebra_down,
                "routing_unavailable": routing_unavailable,
            },
        )


class FrrDownDetection(FrrDownBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=FrrDownBase.root_cause_category,
        root_cause_name=FrrDownBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class FrrDownLocalization(FrrDownBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=FrrDownBase.root_cause_category,
        root_cause_name=FrrDownBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class FrrDownRCA(FrrDownBase, RCATask):
    META = ProblemMeta(
        root_cause_category=FrrDownBase.root_cause_category,
        root_cause_name=FrrDownBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )
