import logging
import random

from nika.generator.fault.injector_tc import FaultInjectorTC
from nika.generator.traffic.od_flows import ODFLowGenerator
from nika.net_env.net_env_pool import get_net_env_instance
from nika.orchestrator.problems.problem_base import ProblemMeta, RootCauseCategory, TaskDescription, TaskLevel
from nika.orchestrator.tasks.detection import DetectionTask
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask
from nika.service.kathara import KatharaAPIALL
from nika.utils.failure_params import FailureParamField, FailureParamSchema
from nika.utils.logger import system_logger

# ==================================================================
# Problem: High link packet corruption between devices causing performance degradation.
# ==================================================================


class LinkHighPacketCorruptionBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.RESOURCE_CONTENTION
    root_cause_name: str = "link_high_packet_corruption"
    TAGS: str = ["link"]
    FAILURE_PARAM_SCHEMA = FailureParamSchema(
        problem_name="link_high_packet_corruption",
        summary="Inject high packet corruption on one host interface.",
        fields=(
            FailureParamField("host_name", "str", "Target host name."),
            FailureParamField("corruption_percentage", "int", "Corruption percentage.", default=60),
        ),
        example="nika failure inject link_high_packet_corruption --set host_name=h1 --set corruption_percentage=60",
    )

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaAPIALL(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorTC(lab_name=self.net_env.lab.name)
        self.faulty_devices = [random.choice(self.net_env.hosts)]
        self.corruption_percentage = 60

    def inject_fault(self):
        intf_name = self.kathara_api.get_host_interfaces(self.faulty_devices[0])[-1]
        self.injector.inject_packet_corruption(
            host_name=self.faulty_devices[0],
            intf_name=intf_name,
            corruption_percentage=self.corruption_percentage,
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


class LinkBandwidthThrottlingBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.RESOURCE_CONTENTION
    root_cause_name: str = "link_bandwidth_throttling"
    TAGS: str = ["link"]
    FAILURE_PARAM_SCHEMA = FailureParamSchema(
        problem_name="link_bandwidth_throttling",
        summary="Throttle host bandwidth and amplify with background traffic.",
        fields=(
            FailureParamField("host_name", "str", "Target host name."),
            FailureParamField("rate", "str", "Bandwidth rate.", default="30kbit"),
            FailureParamField("burst", "str", "TBF burst.", default="64kb"),
            FailureParamField("limit", "str", "TBF limit.", default="500kb"),
        ),
        example="nika failure inject link_bandwidth_throttling --set host_name=h1 --set rate=30kbit",
    )

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaAPIALL(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorTC(lab_name=self.net_env.lab.name)
        self.faulty_devices = [random.choice(self.net_env.hosts)]
        self.scenario_name = scenario_name
        self.rate = "30kbit"
        self.burst = "64kb"
        self.limit = "500kb"

    def inject_fault(self):
        intf_name = self.kathara_api.get_host_interfaces(self.faulty_devices[0])[0]
        self.injector.inject_bandwidth_limit(
            host_name=self.faulty_devices[0],
            intf_name=intf_name,
            rate=self.rate,
            burst=self.burst,
            limit=self.limit,
        )

        generator = ODFLowGenerator(lab_name=self.scenario_name)
        od_dict = {}
        mbps = 20
        for host in self.net_env.hosts:
            if host != self.faulty_devices[0]:
                od_dict.setdefault(host, {})
                od_dict[host][self.faulty_devices[0]] = mbps
        res = generator.start_traffic_background(od_dicts=od_dict, interval=300, unit="M", udp=True)
        system_logger.info(f"Started background traffic generation {res} to amplify the bandwidth throttling effect.")

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


class IncastTrafficNetworkLimitationBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.RESOURCE_CONTENTION
    root_cause_name: str = "incast_traffic_network_limitation"
    TAGS: str = ["http"]
    FAILURE_PARAM_SCHEMA = FailureParamSchema(
        problem_name="incast_traffic_network_limitation",
        summary="Limit server interface then create incast background traffic.",
        fields=(
            FailureParamField("host_name", "str", "Target web server host name."),
            FailureParamField("rate", "str", "Bandwidth rate.", default="1mbit"),
            FailureParamField("burst", "str", "TBF burst.", default="500kb"),
            FailureParamField("limit", "str", "TBF limit.", default="500kb"),
            FailureParamField("delay_ms", "int", "Netem delay milliseconds.", default=20),
        ),
        example="nika failure inject incast_traffic_network_limitation --set host_name=web0 --set rate=1mbit",
    )

    def __init__(self, scenario_name: str = "dc_clos_service", **kwargs):
        super().__init__()
        self.scenario_name = scenario_name
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaAPIALL(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorTC(lab_name=self.net_env.lab.name)
        self.faulty_devices = [random.choice(self.net_env.servers["web"])]
        self.delay_ms = 20
        self.rate = "1mbit"
        self.burst = "500kb"
        self.limit = "500kb"

    def inject_fault(self):
        self.kathara_api.tc_set_netem(
            host_name=self.faulty_devices[0],
            intf_name="eth0",
            delay_ms=self.delay_ms,
            handle="1",
        )
        self.kathara_api.tc_set_tbf(
            host_name=self.faulty_devices[0],
            intf_name="eth0",
            rate=self.rate,
            burst=self.burst,
            limit=self.limit,
            handle="10",
            parent="1:1",
        )
        system_logger.info(f"Injected network limitation on host {self.faulty_devices[0]}")

        generator = ODFLowGenerator(lab_name=self.scenario_name)
        od_dict = {}
        mbps = 20
        for host in self.net_env.hosts:
            if host != self.faulty_devices[0]:
                od_dict.setdefault(host, {})
                od_dict[host][self.faulty_devices[0]] = mbps
        res = generator.start_traffic_background(od_dicts=od_dict, interval=300, unit="M", udp=True)
        system_logger.info(f"Started background traffic generation {res} to amplify the network limitation effect.")

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    task = LinkBandwidthThrottlingBase(scenario_name="dc_clos_bgp", topo_size="m")
    task.inject_fault()
