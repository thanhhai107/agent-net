import logging
import random

from nika.generator.fault.injector_base import FaultInjectorBase
from nika.net_env.net_env_pool import get_net_env_instance
from nika.orchestrator.problems.problem_base import ProblemMeta, RootCauseCategory, TaskDescription, TaskLevel
from nika.orchestrator.tasks.detection import DetectionTask
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask
from nika.service.kathara import KatharaAPIALL
from nika.utils.failure_params import FailureParamField, FailureParamSchema
from nika.utils.logger import system_logger

# ==================================================================
# Problem: SDN controller crash
# ==================================================================
logger = system_logger


class SDNControllerCrashBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.NETWORK_NODE_ERROR
    root_cause_name: str = "sdn_controller_crash"
    TAGS: str = ["sdn"]
    FAILURE_PARAM_SCHEMA = FailureParamSchema(
        problem_name="sdn_controller_crash",
        summary="Crash one SDN controller by killing the ryu-manager process.",
        fields=(
            FailureParamField("host_name", "str", "Target SDN controller host name."),
        ),
        example="nika failure inject sdn_controller_crash --set host_name=ctrl1",
    )

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaAPIALL(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorBase(lab_name=self.net_env.lab.name)
        self.faulty_devices = [random.choice(self.net_env.sdn_controllers)]

    def inject_fault(self):
        self.kathara_api.exec_cmd(
            self.faulty_devices[0],
            "pkill -f ryu-manager",
        )

class SDNControllerCrashDetection(SDNControllerCrashBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=SDNControllerCrashBase.root_cause_category,
        root_cause_name=SDNControllerCrashBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class SDNControllerCrashLocalization(SDNControllerCrashBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=SDNControllerCrashBase.root_cause_category,
        root_cause_name=SDNControllerCrashBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class SDNControllerCrashRCA(SDNControllerCrashBase, RCATask):
    META = ProblemMeta(
        root_cause_category=SDNControllerCrashBase.root_cause_category,
        root_cause_name=SDNControllerCrashBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==================================================================
# Problem: Southbound port block
# ==================================================================


class SouthboundPortBlockBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.NETWORK_NODE_ERROR
    root_cause_name: str = "southbound_port_block"
    TAGS: str = ["sdn"]
    FAILURE_PARAM_SCHEMA = FailureParamSchema(
        problem_name="southbound_port_block",
        summary="Block SDN southbound port with ACL.",
        fields=(
            FailureParamField("host_name", "str", "Target SDN controller host name."),
            FailureParamField("southbound_port", "int", "Port to block.", default=6633),
        ),
        example="nika failure inject southbound_port_block --set host_name=ctrl1 --set southbound_port=6633",
    )

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaAPIALL(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorBase(lab_name=self.net_env.lab.name)
        self.faulty_devices = [random.choice(self.net_env.sdn_controllers)]
        self.southbound_port: int = 6633  # Default OpenFlow port

    def inject_fault(self):
        self.injector.inject_acl_rule(
            host_name=self.faulty_devices[0],
            rule=f"tcp dport {self.southbound_port} drop",
        )

class SouthboundPortBlockDetection(SouthboundPortBlockBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=SouthboundPortBlockBase.root_cause_category,
        root_cause_name=SouthboundPortBlockBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class SouthboundPortBlockLocalization(SouthboundPortBlockBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=SouthboundPortBlockBase.root_cause_category,
        root_cause_name=SouthboundPortBlockBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class SouthboundPortBlockRCA(SouthboundPortBlockBase, RCATask):
    META = ProblemMeta(
        root_cause_category=SouthboundPortBlockBase.root_cause_category,
        root_cause_name=SouthboundPortBlockBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==================================================================
# Problem: Southbound port mismatch
# ==================================================================


class SouthboundPortMismatchBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.NETWORK_NODE_ERROR
    root_cause_name: str = "southbound_port_mismatch"
    TAGS: str = ["sdn"]
    FAILURE_PARAM_SCHEMA = FailureParamSchema(
        problem_name="southbound_port_mismatch",
        summary="Restart SDN controller with mismatched OpenFlow port.",
        fields=(
            FailureParamField("host_name", "str", "Target SDN controller host name."),
            FailureParamField("mismatched_port", "int", "Port used after restart.", default=6653),
            FailureParamField("original_port", "int", "Expected original OpenFlow port.", default=6633),
        ),
        example="nika failure inject southbound_port_mismatch --set host_name=ctrl1 --set mismatched_port=6653",
    )

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaAPIALL(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorBase(lab_name=self.net_env.lab.name)
        self.faulty_devices = [random.choice(self.net_env.sdn_controllers)]
        self.original_port: int = 6633  # Default OpenFlow port
        self.mismatched_port: int = 6653  # Common alternative OpenFlow port

    def inject_fault(self):
        self.kathara_api.exec_cmd(
            self.faulty_devices[0],
            "pkill -f ryu-manager",
        )
        self.kathara_api.exec_cmd(
            self.faulty_devices[0],
            f"ryu-manager --ofp-tcp-listen-port {self.mismatched_port} ryu.app.simple_switch &",
        )

class SouthboundPortMismatchDetection(SouthboundPortMismatchBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=SouthboundPortMismatchBase.root_cause_category,
        root_cause_name=SouthboundPortMismatchBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class SouthboundPortMismatchLocalization(SouthboundPortMismatchBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=SouthboundPortMismatchBase.root_cause_category,
        root_cause_name=SouthboundPortMismatchBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class SouthboundPortMismatchRCA(SouthboundPortMismatchBase, RCATask):
    META = ProblemMeta(
        root_cause_category=SouthboundPortMismatchBase.root_cause_category,
        root_cause_name=SouthboundPortMismatchBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==================================================================
# Problem: Flow rule shadowing
# ==================================================================


class FlowRuleShadowingBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.NETWORK_NODE_ERROR
    root_cause_name: str = "flow_rule_shadowing"
    TAGS: str = ["sdn"]
    FAILURE_PARAM_SCHEMA = FailureParamSchema(
        problem_name="flow_rule_shadowing",
        summary="Insert a high-priority drop flow on one OVS switch.",
        fields=(FailureParamField("host_name", "str", "Target OVS switch name."),),
        example="nika failure inject flow_rule_shadowing --set host_name=ovs1",
    )

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaAPIALL(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorBase(lab_name=self.net_env.lab.name)
        self.faulty_devices = [random.choice(self.net_env.ovs_switches)]

    def inject_fault(self):
        # Inject a shadowing flow rule that matches all traffic and forwards to a blackhole
        self.kathara_api.exec_cmd(
            self.faulty_devices[0],
            f"ovs-ofctl add-flow {self.faulty_devices[0]} 'priority=100,actions=drop'",
        )

class FlowRuleShadowingDetection(FlowRuleShadowingBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=FlowRuleShadowingBase.root_cause_category,
        root_cause_name=FlowRuleShadowingBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class FlowRuleShadowingLocalization(FlowRuleShadowingBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=FlowRuleShadowingBase.root_cause_category,
        root_cause_name=FlowRuleShadowingBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class FlowRuleShadowingRCA(FlowRuleShadowingBase, RCATask):
    META = ProblemMeta(
        root_cause_category=FlowRuleShadowingBase.root_cause_category,
        root_cause_name=FlowRuleShadowingBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==================================================================
# Problem: Flow rule loop
# ==================================================================


class FlowRuleLoopBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.NETWORK_NODE_ERROR
    root_cause_name: str = "flow_rule_loop"
    TAGS: str = ["sdn"]
    FAILURE_PARAM_SCHEMA = FailureParamSchema(
        problem_name="flow_rule_loop",
        summary="Inject loop-inducing flow rules on two OVS switches.",
        fields=(
            FailureParamField("host_name", "str", "Primary OVS switch name."),
            FailureParamField("host_name_2", "str", "Secondary OVS switch name."),
        ),
        example="nika failure inject flow_rule_loop --set host_name=ovs1 --set host_name_2=ovs2",
    )

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaAPIALL(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorBase(lab_name=self.net_env.lab.name)
        self.faulty_devices = self.net_env.ovs_switches[:2]

    def inject_fault(self):
        # Inject flow rules that create a forwarding loop between two ports
        self.kathara_api.exec_cmd(
            self.faulty_devices[0],
            f"ovs-ofctl add-flow {self.faulty_devices[0]} 'in_port=eth0,actions=output:eth0'",
        )
        self.kathara_api.exec_cmd(
            self.faulty_devices[1],
            f"ovs-ofctl add-flow {self.faulty_devices[1]} 'in_port=eth1,actions=output:eth1'",
        )

class FlowRuleLoopDetection(FlowRuleLoopBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=FlowRuleLoopBase.root_cause_category,
        root_cause_name=FlowRuleLoopBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class FlowRuleLoopLocalization(FlowRuleLoopBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=FlowRuleLoopBase.root_cause_category,
        root_cause_name=FlowRuleLoopBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class FlowRuleLoopRCA(FlowRuleLoopBase, RCATask):
    META = ProblemMeta(
        root_cause_category=FlowRuleLoopBase.root_cause_category,
        root_cause_name=FlowRuleLoopBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


if __name__ == "__main__":
    # For quick test
    logging.basicConfig(level=logging.INFO)
    problem = FlowRuleLoopBase()
    problem.inject_fault()
