import random

from nika.generator.fault.injector_host import FaultInjectorHost
from nika.generator.fault.injector_tc import FaultInjectorTC
from nika.net_env.net_env_pool import get_net_env_instance
from nika.orchestrator.problems.problem_base import ProblemMeta, RootCauseCategory, TaskDescription, TaskLevel
from nika.orchestrator.tasks.detection import DetectionTask
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask
from nika.service.kathara import KatharaAPIALL
from nika.utils.failure_params import FailureParamField, FailureParamSchema

# ==================================================================
# Problem: Web service experiencing high DNS lookup latency causing performance degradation.
# ==================================================================


class DNSLookupLatencyBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.RESOURCE_CONTENTION
    root_cause_name: str = "dns_lookup_latency"
    symptom_desc: str = "Users experience high latency when accessing web services."
    TAGS: str = ["dns", "http"]
    FAILURE_PARAM_SCHEMA = FailureParamSchema(
        problem_name="dns_lookup_latency",
        summary="Inject DNS lookup latency by adding delay on DNS server interface.",
        fields=(
            FailureParamField("host_name", "str", "Target DNS server host name."),
            FailureParamField("intf_name", "str", "Interface name.", default="eth0"),
            FailureParamField("delay_ms", "int", "Delay in milliseconds.", default=1000),
        ),
        example="nika failure inject dns_lookup_latency --set host_name=dns0 --set delay_ms=1000",
    )

    def __init__(self, scenario_name: str = "dc_clos_service", **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaAPIALL(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorTC(lab_name=self.net_env.lab.name)
        self.faulty_devices = [random.choice(self.net_env.servers["dns"])]
        self.intf_name = "eth0"
        self.delay_ms = 1000

    def inject_fault(self):
        self.injector.inject_delay(host_name=self.faulty_devices[0], intf_name=self.intf_name, delay_ms=self.delay_ms)

class DNSLookupLatencyDetection(DNSLookupLatencyBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=DNSLookupLatencyBase.root_cause_category,
        root_cause_name=DNSLookupLatencyBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class DNSLookupLatencyLocalization(DNSLookupLatencyBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=DNSLookupLatencyBase.root_cause_category,
        root_cause_name=DNSLookupLatencyBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class DNSLookupLatencyRCA(DNSLookupLatencyBase, RCATask):
    META = ProblemMeta(
        root_cause_category=DNSLookupLatencyBase.root_cause_category,
        root_cause_name=DNSLookupLatencyBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==================================================================
# Problem: Load balancer overload causing performance degradation.
# ==================================================================


class LoadBalancerOverloadBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.RESOURCE_CONTENTION
    root_cause_name: str = "load_balancer_overload"
    TAGS: str = ["load_balancer", "http"]
    FAILURE_PARAM_SCHEMA = FailureParamSchema(
        problem_name="load_balancer_overload",
        summary="Stress load balancer host resources.",
        fields=(
            FailureParamField("host_name", "str", "Target load balancer host name."),
            FailureParamField("duration", "int", "Stress duration in seconds.", default=300),
        ),
        example="nika failure inject load_balancer_overload --set host_name=lb0 --set duration=300",
    )

    def __init__(self, scenario_name: str = "load_balancer", **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaAPIALL(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorHost(lab_name=self.net_env.lab.name)
        self.faulty_devices = [random.choice(self.net_env.servers["load_balancer"])]
        self.duration = 300

    def inject_fault(self):
        self.injector.inject_stress_all(host_name=self.faulty_devices[0], duration=self.duration)

class LoadBalancerOverloadDetection(LoadBalancerOverloadBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=LoadBalancerOverloadBase.root_cause_category,
        root_cause_name=LoadBalancerOverloadBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class LoadBalancerOverloadLocalization(LoadBalancerOverloadBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=LoadBalancerOverloadBase.root_cause_category,
        root_cause_name=LoadBalancerOverloadBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class LoadBalancerOverloadRCA(LoadBalancerOverloadBase, RCATask):
    META = ProblemMeta(
        root_cause_category=LoadBalancerOverloadBase.root_cause_category,
        root_cause_name=LoadBalancerOverloadBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


if __name__ == "__main__":
    # Test the fault injection and recovery
    problem = LoadBalancerOverloadBase()
    problem.inject_fault()
