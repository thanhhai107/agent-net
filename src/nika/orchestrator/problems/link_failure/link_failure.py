import random

from nika.generator.fault.injector_base import FaultInjectorBase
from nika.net_env.net_env_pool import get_net_env_instance
from nika.orchestrator.problems.problem_base import ProblemMeta, RootCauseCategory, TaskDescription, TaskLevel
from nika.orchestrator.tasks.detection import DetectionTask
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask
from nika.service.kathara import KatharaBaseAPI
from nika.utils.failure_params import FailureParamField, FailureParamSchema

# ==================================================================
# Problem: Link failure by ip link down on host interface
# ==================================================================


class LinkFailureBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.LINK_FAILURE
    root_cause_name: str = "link_down"
    TAGS: str = ["link"]
    FAILURE_PARAM_SCHEMA = FailureParamSchema(
        problem_name="link_down",
        summary="Bring one host interface down.",
        fields=(
            FailureParamField("host_name", "str", "Target host name."),
            FailureParamField("intf_name", "str", "Target interface name.", default="eth0"),
        ),
        example="nika failure inject link_down --set host_name=pc1 --set intf_name=eth0",
    )

    symptom_desc = "Users report connectivity issues to other hosts."

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaBaseAPI(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorBase(lab_name=self.net_env.lab.name)
        self.faulty_devices = [random.choice(self.net_env.hosts)]
        self.faulty_intf = "eth0"
        self.down_time = 1
        self.up_time = 1

    def inject_fault(self):
        self.injector.inject_intf_down(
            host_name=self.faulty_devices[0],
            intf_name=self.faulty_intf,
        )

class LinkFailureDetection(LinkFailureBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=LinkFailureBase.root_cause_category,
        root_cause_name=LinkFailureBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class LinkFailureLocalization(LinkFailureBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=LinkFailureBase.root_cause_category,
        root_cause_name=LinkFailureBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class LinkFailureRCA(LinkFailureBase, RCATask):
    META = ProblemMeta(
        root_cause_category=LinkFailureBase.root_cause_category,
        root_cause_name=LinkFailureBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==========================================
# Problem: Link flapping by manual script
# ==========================================


class LinkFlapBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.LINK_FAILURE
    root_cause_name: str = "link_flap"
    TAGS: str = ["link"]
    FAILURE_PARAM_SCHEMA = FailureParamSchema(
        problem_name="link_flap",
        summary="Flap one host interface repeatedly.",
        fields=(
            FailureParamField("host_name", "str", "Target host name."),
            FailureParamField("intf_name", "str", "Target interface name.", default="eth0"),
            FailureParamField("down_time", "int", "Down duration in seconds.", default=1),
            FailureParamField("up_time", "int", "Up duration in seconds.", default=1),
        ),
        example="nika failure inject link_flap --set host_name=pc1 --set down_time=2 --set up_time=2",
    )

    symptom_desc = "Users report connectivity issues to other hosts."

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaBaseAPI(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorBase(lab_name=self.net_env.lab.name)
        self.faulty_devices = [random.choice(self.net_env.hosts)]
        self.faulty_intf = "eth0"

    def inject_fault(self):
        self.injector.inject_link_flap(
            host_name=self.faulty_devices[0],
            intf_name=self.faulty_intf,
            down_time=self.down_time,
            up_time=self.up_time,
        )

class LinkFlapDetection(LinkFlapBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=LinkFlapBase.root_cause_category,
        root_cause_name=LinkFlapBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class LinkFlapLocalization(LinkFlapBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=LinkFlapBase.root_cause_category,
        root_cause_name=LinkFlapBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class LinkFlapRCA(LinkFlapBase, RCATask):
    META = ProblemMeta(
        root_cause_category=LinkFlapBase.root_cause_category,
        root_cause_name=LinkFlapBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==========================================
# Problem: Link detached. Note: the recover is not working
# ==========================================


class LinkDetachBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.LINK_FAILURE
    root_cause_name: str = "link_detach"
    TAGS: str = ["link"]
    FAILURE_PARAM_SCHEMA = FailureParamSchema(
        problem_name="link_detach",
        summary="Detach one host interface.",
        fields=(
            FailureParamField("host_name", "str", "Target host name."),
            FailureParamField("intf_name", "str", "Target interface name.", default="eth0"),
        ),
        example="nika failure inject link_detach --set host_name=pc1",
    )

    symptom_desc = "Users report connectivity issues to other hosts."

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaBaseAPI(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorBase(lab_name=self.net_env.lab.name)
        self.faulty_devices = [random.choice(self.net_env.hosts)]
        self.faulty_intf = "eth0"

    def inject_fault(self):
        self.injector.inject_link_detach(
            host_name=self.faulty_devices[0],
            intf_name=self.faulty_intf,
        )

class LinkDetachDetection(LinkDetachBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=LinkDetachBase.root_cause_category,
        root_cause_name=LinkDetachBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class LinkDetachLocalization(LinkDetachBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=LinkDetachBase.root_cause_category,
        root_cause_name=LinkDetachBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class LinkDetachRCA(LinkDetachBase, RCATask):
    META = ProblemMeta(
        root_cause_category=LinkDetachBase.root_cause_category,
        root_cause_name=LinkDetachBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==========================================
# Problem: Link fragmentation disabled, drop large packets
# ==========================================


class LinkFragBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.LINK_FAILURE
    root_cause_name: str = "link_fragmentation_disabled"
    TAGS: str = ["link"]
    FAILURE_PARAM_SCHEMA = FailureParamSchema(
        problem_name="link_fragmentation_disabled",
        summary="Drop oversized packets on a host.",
        fields=(
            FailureParamField("host_name", "str", "Target host name."),
            FailureParamField("mtu", "int", "Packet size threshold.", default=10),
        ),
        example="nika failure inject link_fragmentation_disabled --set host_name=pc1 --set mtu=20",
    )

    symptom_desc = "Users report partial packet loss when communicating with other hosts."

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaBaseAPI(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorBase(lab_name=self.net_env.lab.name)
        self.faulty_devices = [random.choice(self.net_env.hosts)]
        self.mtu = 10

    def inject_fault(self):
        self.injector.inject_fragmentation_disabled(host_name=self.faulty_devices[0], mtu=self.mtu)

class LinkFragDetection(LinkFragBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=LinkFragBase.root_cause_category,
        root_cause_name=LinkFragBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class LinkFragLocalization(LinkFragBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=LinkFragBase.root_cause_category,
        root_cause_name=LinkFragBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class LinkFragRCA(LinkFragBase, RCATask):
    META = ProblemMeta(
        root_cause_category=LinkFragBase.root_cause_category,
        root_cause_name=LinkFragBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


if __name__ == "__main__":
    task = LinkFailureDetection()
    # task.inject_fault()
    # Here you would typically run your detection logic
