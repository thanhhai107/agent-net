from nika.orchestrator.problems.context import init_problem
from pydantic import BaseModel, Field

from nika.orchestrator.problems.problem_base import ProblemMeta, RootCauseCategory, TaskDescription, TaskLevel, build_verify_result
from nika.orchestrator.tasks.detection import DetectionTask
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask

# ==================================================================
# Problem: Web service experiencing high DNS lookup latency causing performance degradation.
# ==================================================================


class DNSLookupLatencyParams(BaseModel):
    """Parameters for injecting a DNS lookup latency fault."""

    host_name: str = Field(description="Target DNS server host name.")
    intf_name: str = Field(default="eth0", description="Interface name.")
    delay_ms: int = Field(default=1000, description="Delay in milliseconds.")


class DNSLookupLatencyBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.RESOURCE_CONTENTION
    root_cause_name: str = "dns_lookup_latency"
    symptom_desc: str = "Users experience high latency when accessing web services."
    TAGS: str = ["dns", "http"]

    Params = DNSLookupLatencyParams

    def __init__(self, scenario_name: str = "dc_clos_service", **kwargs):
        super().__init__()
        self.net_env, self.runtime = init_problem(scenario_name, **kwargs)
        self.faulty_devices: list[str] = []

    def inject_fault(self, params: DNSLookupLatencyParams):
        host = params.host_name
        self.faulty_devices = [host]
        self.runtime.tc_set_netem(host, params.intf_name, delay_ms=params.delay_ms)

    def verify_fault(self, params: DNSLookupLatencyParams) -> dict:
        """Verify tc qdisc on DNS server interface has a delay configured."""
        host = params.host_name
        intf = params.intf_name
        tc_output = self.runtime.exec(host, f"tc qdisc show dev {intf}").strip()
        verified = "delay" in tc_output
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"host": host, "intf": intf, "tc_output": tc_output},
        )


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


class LoadBalancerOverloadParams(BaseModel):
    """Parameters for injecting a load balancer overload fault."""

    host_name: str = Field(description="Target load balancer host name.")
    duration: int = Field(default=300, description="Stress duration in seconds.")


class LoadBalancerOverloadBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.RESOURCE_CONTENTION
    root_cause_name: str = "load_balancer_overload"
    TAGS: str = ["load_balancer", "http"]

    Params = LoadBalancerOverloadParams

    def __init__(self, scenario_name: str = "load_balancer", **kwargs):
        super().__init__()
        self.net_env, self.runtime = init_problem(scenario_name, **kwargs)
        self.faulty_devices: list[str] = []

    def inject_fault(self, params: LoadBalancerOverloadParams):
        host = params.host_name
        self.faulty_devices = [host]
        self.runtime.exec(host, f"nohup stress-ng --cpu 0 --cpu-load 100 --iomix 0 --sock 0 --hdd 2 --vm 0 --vm-bytes 75% --timeout {params.duration} </dev/null >/dev/null 2>&1 &")

    def verify_fault(self, params: LoadBalancerOverloadParams) -> dict:
        """Verify stress-ng is running on the load balancer."""
        host = params.host_name
        pgrep_output = self.runtime.exec(host, "pgrep -a stress-ng 2>/dev/null || echo NONE").strip()
        verified = "stress-ng" in pgrep_output and pgrep_output != "NONE"
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"host": host, "pgrep_output": pgrep_output},
        )


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
