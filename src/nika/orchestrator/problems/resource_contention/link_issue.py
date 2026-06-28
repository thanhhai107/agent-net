import logging

from pydantic import BaseModel, Field

from nika.generator.fault.injector_tc import FaultInjectorTC
from nika.generator.traffic.od_flows import ODFLowGenerator
from nika.net_env.net_env_pool import get_net_env_instance
from nika.orchestrator.problems.problem_base import ProblemMeta, RootCauseCategory, TaskDescription, TaskLevel, build_verify_result
from nika.orchestrator.tasks.detection import DetectionTask
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask
from nika.service.kathara import KatharaAPIALL
from nika.utils.logger import system_logger

# ==================================================================
# Problem: High link packet corruption between devices causing performance degradation.
# ==================================================================


class LinkHighPacketCorruptionParams(BaseModel):
    """Parameters for injecting a high packet corruption fault."""

    host_name: str = Field(description="Target host name.")
    corruption_percentage: int = Field(default=60, description="Corruption percentage.")


class LinkHighPacketCorruptionBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.RESOURCE_CONTENTION
    root_cause_name: str = "link_high_packet_corruption"
    TAGS: str = ["link"]

    Params = LinkHighPacketCorruptionParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaAPIALL(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorTC(lab_name=self.net_env.lab.name)
        self.faulty_devices: list[str] = []

    def inject_fault(self, params: LinkHighPacketCorruptionParams):
        host = params.host_name
        self.faulty_devices = [host]
        intf_name = self.kathara_api.get_host_interfaces(host)[-1]
        self.injector.inject_packet_corruption(
            host_name=host,
            intf_name=intf_name,
            corruption_percentage=params.corruption_percentage,
        )

    def verify_fault(self, params: LinkHighPacketCorruptionParams) -> dict:
        """Verify tc qdisc on the host's last interface has corruption configured."""
        host = params.host_name
        intf = self.kathara_api.get_host_interfaces(host)[-1]
        tc_output = self.kathara_api.exec_cmd(host, f"tc qdisc show dev {intf}").strip()
        verified = "corrupt" in tc_output
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"host": host, "intf": intf, "tc_output": tc_output},
        )


class LinkHighPacketCorruptionDetection(LinkHighPacketCorruptionBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=LinkHighPacketCorruptionBase.root_cause_category,
        root_cause_name=LinkHighPacketCorruptionBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class LinkHighPacketCorruptionLocalization(LinkHighPacketCorruptionBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=LinkHighPacketCorruptionBase.root_cause_category,
        root_cause_name=LinkHighPacketCorruptionBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class LinkHighPacketCorruptionRCA(LinkHighPacketCorruptionBase, RCATask):
    META = ProblemMeta(
        root_cause_category=LinkHighPacketCorruptionBase.root_cause_category,
        root_cause_name=LinkHighPacketCorruptionBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==================================================================
# Problem: Bandwidth throttling on a link causing performance degradation.
# ==================================================================


class LinkBandwidthThrottlingParams(BaseModel):
    """Parameters for injecting a bandwidth throttling fault."""

    host_name: str = Field(description="Target host name.")
    rate: str = Field(default="30kbit", description="Bandwidth rate.")
    burst: str = Field(default="64kb", description="TBF burst.")
    limit: str = Field(default="500kb", description="TBF limit.")


class LinkBandwidthThrottlingBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.RESOURCE_CONTENTION
    root_cause_name: str = "link_bandwidth_throttling"
    TAGS: str = ["link"]

    Params = LinkBandwidthThrottlingParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaAPIALL(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorTC(lab_name=self.net_env.lab.name)
        self.faulty_devices: list[str] = []
        self.scenario_name = scenario_name

    def inject_fault(self, params: LinkBandwidthThrottlingParams):
        host = params.host_name
        self.faulty_devices = [host]
        intf_name = self.kathara_api.get_host_interfaces(host)[0]
        self.injector.inject_bandwidth_limit(
            host_name=host,
            intf_name=intf_name,
            rate=params.rate,
            burst=params.burst,
            limit=params.limit,
        )
        generator = ODFLowGenerator(lab_name=self.net_env.lab.name)
        od_dict = {}
        mbps = 20
        for h in self.net_env.hosts:
            if h != host:
                od_dict.setdefault(h, {})
                od_dict[h][host] = mbps
        res = generator.start_traffic_background(od_dicts=od_dict, interval=300, unit="M", udp=True)
        system_logger.info(f"Started background traffic generation {res} to amplify the bandwidth throttling effect.")

    def verify_fault(self, params: LinkBandwidthThrottlingParams) -> dict:
        """Verify tc qdisc on the host's first interface has TBF (token bucket filter) configured."""
        host = params.host_name
        intf = self.kathara_api.get_host_interfaces(host)[0]
        tc_output = self.kathara_api.exec_cmd(host, f"tc qdisc show dev {intf}").strip()
        verified = "tbf" in tc_output
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"host": host, "intf": intf, "tc_output": tc_output},
        )


class LinkBandwidthThrottlingDetection(LinkBandwidthThrottlingBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=LinkBandwidthThrottlingBase.root_cause_category,
        root_cause_name=LinkBandwidthThrottlingBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class LinkBandwidthThrottlingLocalization(LinkBandwidthThrottlingBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=LinkBandwidthThrottlingBase.root_cause_category,
        root_cause_name=LinkBandwidthThrottlingBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class LinkBandwidthThrottlingRCA(LinkBandwidthThrottlingBase, RCATask):
    META = ProblemMeta(
        root_cause_category=LinkBandwidthThrottlingBase.root_cause_category,
        root_cause_name=LinkBandwidthThrottlingBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==================================================================
# Problem: incast traffic causing performance degradation.
# ==================================================================


class IncastTrafficNetworkLimitationParams(BaseModel):
    """Parameters for injecting an incast traffic network limitation fault."""

    host_name: str = Field(description="Target web server host name.")
    rate: str = Field(default="1mbit", description="Bandwidth rate.")
    burst: str = Field(default="500kb", description="TBF burst.")
    limit: str = Field(default="500kb", description="TBF limit.")
    delay_ms: int = Field(default=20, description="Netem delay milliseconds.")


class IncastTrafficNetworkLimitationBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.RESOURCE_CONTENTION
    root_cause_name: str = "incast_traffic_network_limitation"
    TAGS: str = ["http"]

    Params = IncastTrafficNetworkLimitationParams

    def __init__(self, scenario_name: str = "dc_clos_service", **kwargs):
        super().__init__()
        self.scenario_name = scenario_name
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaAPIALL(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorTC(lab_name=self.net_env.lab.name)
        self.faulty_devices: list[str] = []

    def inject_fault(self, params: IncastTrafficNetworkLimitationParams):
        host = params.host_name
        self.faulty_devices = [host]
        self.kathara_api.tc_set_netem(host_name=host, intf_name="eth0", delay_ms=params.delay_ms, handle="1")
        self.kathara_api.tc_set_tbf(
            host_name=host,
            intf_name="eth0",
            rate=params.rate,
            burst=params.burst,
            limit=params.limit,
            handle="10",
            parent="1:1",
        )
        system_logger.info(f"Injected network limitation on host {host}")
        generator = ODFLowGenerator(lab_name=self.net_env.lab.name)
        od_dict = {}
        mbps = 20
        for h in self.net_env.hosts:
            if h != host:
                od_dict.setdefault(h, {})
                od_dict[h][host] = mbps
        res = generator.start_traffic_background(od_dicts=od_dict, interval=300, unit="M", udp=True)
        system_logger.info(f"Started background traffic generation {res} to amplify the network limitation effect.")

    def verify_fault(self, params: IncastTrafficNetworkLimitationParams) -> dict:
        """Verify tc qdisc on eth0 has netem or tbf (incast network limitation)."""
        host = params.host_name
        tc_output = self.kathara_api.exec_cmd(host, "tc qdisc show dev eth0").strip()
        verified = "netem" in tc_output or "tbf" in tc_output
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"host": host, "tc_output": tc_output},
        )


class IncastTrafficNetworkLimitationDetection(IncastTrafficNetworkLimitationBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=IncastTrafficNetworkLimitationBase.root_cause_category,
        root_cause_name=IncastTrafficNetworkLimitationBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class IncastTrafficNetworkLimitationLocalization(IncastTrafficNetworkLimitationBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=IncastTrafficNetworkLimitationBase.root_cause_category,
        root_cause_name=IncastTrafficNetworkLimitationBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class IncastTrafficNetworkLimitationRCA(IncastTrafficNetworkLimitationBase, RCATask):
    META = ProblemMeta(
        root_cause_category=IncastTrafficNetworkLimitationBase.root_cause_category,
        root_cause_name=IncastTrafficNetworkLimitationBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )
