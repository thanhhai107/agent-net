import logging
import random

from nika.generator.fault.injector_host import FaultInjectorHost
from nika.net_env.net_env_pool import get_net_env_instance
from nika.orchestrator.problems.problem_base import ProblemMeta, RootCauseCategory, TaskDescription, TaskLevel
from nika.orchestrator.tasks.detection import DetectionTask
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask
from nika.service.kathara import KatharaBaseAPI
from nika.utils.failure_params import FailureParamField, FailureParamSchema
from nika.utils.logger import system_logger

# ==========================================
# Problem: Host missing IP address
# ==========================================


class HostMissingIPBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.END_HOST_FAILURE
    root_cause_name: str = "host_missing_ip"
    TAGS: str = ["host"]
    FAILURE_PARAM_SCHEMA = FailureParamSchema(
        problem_name="host_missing_ip",
        summary="Remove IP from host interface.",
        fields=(
            FailureParamField("host_name", "str", "Target host name."),
            FailureParamField("intf_name", "str", "Target interface name.", default="eth0"),
        ),
        example="nika failure inject host_missing_ip --set host_name=h1 --set intf_name=eth0",
    )

    symptom_desc = "Some hosts are unable to communicate with other devices in the network."

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.logger = system_logger
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaBaseAPI(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorHost(lab_name=self.net_env.lab.name)
        self.faulty_devices = [random.choice(self.net_env.hosts)]
        self.incorrect_ip: str | None = None
        self.intf_name = "eth0"

    def inject_fault(self):
        real_ip = self.kathara_api.get_host_ip(self.faulty_devices[0], self.intf_name, with_prefix=True)
        real_gateway = self.kathara_api.get_default_gateway(self.faulty_devices[0])
        self.kathara_api.exec_cmd(
            host_name=self.faulty_devices[0],
            command=f"ip addr del {real_ip} dev {self.intf_name}",
        )
        # backup the removed IP to a file for recovery
        self.kathara_api.exec_cmd(
            host_name=self.faulty_devices[0],
            command=f"echo '{real_ip} {real_gateway}' > /tmp/removed_ip.txt",
        )
        self.logger.info(f"Injected missing IP on {self.faulty_devices[0]} from {real_ip} and gateway {real_gateway}.")

class HostMissingIPDetection(HostMissingIPBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=HostMissingIPBase.root_cause_category,
        root_cause_name=HostMissingIPBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class HostMissingIPLocalization(HostMissingIPBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=HostMissingIPBase.root_cause_category,
        root_cause_name=HostMissingIPBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class HostMissingIPRCA(HostMissingIPBase, RCATask):
    META = ProblemMeta(
        root_cause_category=HostMissingIPBase.root_cause_category,
        root_cause_name=HostMissingIPBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==========================================
""" Problem: Host IP conflict """


class HostIPConflictBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.END_HOST_FAILURE
    root_cause_name: str = "host_ip_conflict"
    TAGS: str = ["host"]
    FAILURE_PARAM_SCHEMA = FailureParamSchema(
        problem_name="host_ip_conflict",
        summary="Assign one host the same IP as another host.",
        fields=(
            FailureParamField("host_name", "str", "Source host whose IP is copied."),
            FailureParamField("host_name_2", "str", "Target host to misconfigure."),
        ),
        example="nika failure inject host_ip_conflict --set host_name=h1 --set host_name_2=h2",
    )

    symptom_desc = "Some hosts experience intermittent connectivity issues."

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaBaseAPI(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorHost(lab_name=self.net_env.lab.name)
        self.faulty_devices = random.sample(self.net_env.hosts, 2)

    def inject_fault(self):
        self.injector.inject_ip_change(
            host_name=self.faulty_devices[1],
            old_ip=self.kathara_api.get_host_ip(self.faulty_devices[1], "eth0", with_prefix=True),
            new_ip=self.kathara_api.get_host_ip(self.faulty_devices[0], "eth0", with_prefix=True),
            intf_name="eth0",
            new_gateway=self.kathara_api.get_default_gateway(self.faulty_devices[0]),
        )

class HostIPConflictDetection(HostIPConflictBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=HostIPConflictBase.root_cause_category,
        root_cause_name=HostIPConflictBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class HostIPConflictLocalization(HostIPConflictBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=HostIPConflictBase.root_cause_category,
        root_cause_name=HostIPConflictBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class HostIPConflictRCA(HostIPConflictBase, RCATask):
    META = ProblemMeta(
        root_cause_category=HostIPConflictBase.root_cause_category,
        root_cause_name=HostIPConflictBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==========================================
# Problem: Incorrect Host IP
# ==========================================


class HostIncorrectIPBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.END_HOST_FAILURE
    root_cause_name: str = "host_incorrect_ip"
    TAGS: str = ["host"]
    FAILURE_PARAM_SCHEMA = FailureParamSchema(
        problem_name="host_incorrect_ip",
        summary="Set incorrect IP on one host.",
        fields=(
            FailureParamField("host_name", "str", "Target host name."),
            FailureParamField("incorrect_ip", "str", "Incorrect CIDR IP (optional)."),
        ),
        example="nika failure inject host_incorrect_ip --set host_name=h1",
    )

    symptom_desc = "Some hosts seem to be unreachable in the network."

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaBaseAPI(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorHost(lab_name=self.net_env.lab.name)
        self.faulty_devices = [random.choice(self.net_env.hosts)]

    def inject_fault(self):
        incorrect_ip = self.incorrect_ip or f"10.2.1.{random.randint(2, 254)}/24"
        ip_gateway = "10.2.1.1"
        self.injector.inject_ip_change(
            host_name=self.faulty_devices[0],
            old_ip=self.kathara_api.get_host_ip(self.faulty_devices[0], "eth0", with_prefix=True),
            new_ip=incorrect_ip,
            intf_name="eth0",
            new_gateway=ip_gateway,
        )

class HostIncorrectIPDetection(HostIncorrectIPBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=HostIncorrectIPBase.root_cause_category,
        root_cause_name=HostIncorrectIPBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class HostIncorrectIPLocalization(HostIncorrectIPBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=HostIncorrectIPBase.root_cause_category,
        root_cause_name=HostIncorrectIPBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class HostIncorrectIPRCA(HostIncorrectIPBase, RCATask):
    META = ProblemMeta(
        root_cause_category=HostIncorrectIPBase.root_cause_category,
        root_cause_name=HostIncorrectIPBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==========================================
# Problem: Incorrect Host Gateway
# ==========================================


class HostIncorrectGatewayBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.END_HOST_FAILURE
    root_cause_name: str = "host_incorrect_gateway"
    TAGS: str = ["host", "frr"]
    FAILURE_PARAM_SCHEMA = FailureParamSchema(
        problem_name="host_incorrect_gateway",
        summary="Set incorrect default gateway on one host.",
        fields=(
            FailureParamField("host_name", "str", "Target host name."),
            FailureParamField("new_gateway", "str", "Incorrect gateway IP (optional)."),
        ),
        example="nika failure inject host_incorrect_gateway --set host_name=h1",
    )

    symptom_desc = "Some hosts seem to be unreachable in the network."

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaBaseAPI(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorHost(lab_name=self.net_env.lab.name)
        self.faulty_devices = [random.choice(self.net_env.hosts)]
        self.new_gateway: str | None = None

    def inject_fault(self):
        if self.new_gateway:
            new_gateway = self.new_gateway
        else:
            try:
                new_gateway_list = self.kathara_api.get_default_gateway(self.faulty_devices[0]).split(".")
                new_gateway_list[-1] = "254"
                new_gateway = ".".join(new_gateway_list)
            except Exception:
                new_gateway = "10.0.0.254"
        self.injector.inject_ip_change(
            host_name=self.faulty_devices[0],
            old_ip=self.kathara_api.get_host_ip(self.faulty_devices[0], "eth0", with_prefix=True),
            new_ip=self.kathara_api.get_host_ip(self.faulty_devices[0], "eth0", with_prefix=True),
            intf_name="eth0",
            new_gateway=new_gateway,
        )

class HostIncorrectGatewayDetection(HostIncorrectGatewayBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=HostIncorrectGatewayBase.root_cause_category,
        root_cause_name=HostIncorrectGatewayBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class HostIncorrectGatewayLocalization(HostIncorrectGatewayBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=HostIncorrectGatewayBase.root_cause_category,
        root_cause_name=HostIncorrectGatewayBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class HostIncorrectGatewayRCA(HostIncorrectGatewayBase, RCATask):
    META = ProblemMeta(
        root_cause_category=HostIncorrectGatewayBase.root_cause_category,
        root_cause_name=HostIncorrectGatewayBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==========================================
# Problem: Incorrect Host netmask
# ==========================================
class HostIncorrectNetmaskBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.END_HOST_FAILURE
    root_cause_name: str = "host_incorrect_netmask"
    TAGS: str = ["host", "frr"]
    FAILURE_PARAM_SCHEMA = FailureParamSchema(
        problem_name="host_incorrect_netmask",
        summary="Set incorrect netmask on one host.",
        fields=(
            FailureParamField("host_name", "str", "Target host name."),
            FailureParamField("netmask_prefix", "int", "Incorrect prefix length.", default=8),
        ),
        example="nika failure inject host_incorrect_netmask --set host_name=h1 --set netmask_prefix=8",
    )

    symptom_desc = "Some hosts seem to be unreachable in the network."

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaBaseAPI(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorHost(lab_name=self.net_env.lab.name)
        self.faulty_devices = [random.choice(self.net_env.hosts)]
        self.netmask_prefix = 8

    def inject_fault(self):
        new_ip = self.kathara_api.get_host_ip(self.faulty_devices[0], "eth0", with_prefix=True)
        new_ip = new_ip.split("/")
        new_ip[-1] = str(self.netmask_prefix)
        new_ip = "/".join(new_ip)

        self.injector.inject_ip_change(
            host_name=self.faulty_devices[0],
            old_ip=self.kathara_api.get_host_ip(self.faulty_devices[0], "eth0", with_prefix=True),
            new_ip=new_ip,
            intf_name="eth0",
            new_gateway=self.kathara_api.get_default_gateway(self.faulty_devices[0]),
        )

class HostIncorrectNetmaskDetection(HostIncorrectNetmaskBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=HostIncorrectNetmaskBase.root_cause_category,
        root_cause_name=HostIncorrectNetmaskBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class HostIncorrectNetmaskLocalization(HostIncorrectNetmaskBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=HostIncorrectNetmaskBase.root_cause_category,
        root_cause_name=HostIncorrectNetmaskBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class HostIncorrectNetmaskRCA(HostIncorrectNetmaskBase, RCATask):
    META = ProblemMeta(
        root_cause_category=HostIncorrectNetmaskBase.root_cause_category,
        root_cause_name=HostIncorrectNetmaskBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==========================================
# Problem: Incorrect Host DNS resolvers
# =========================================


class HostIncorrectDNSBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.END_HOST_FAILURE
    root_cause_name: str = "host_incorrect_dns"
    TAGS: str = ["dns"]
    FAILURE_PARAM_SCHEMA = FailureParamSchema(
        problem_name="host_incorrect_dns",
        summary="Set incorrect DNS resolver on one host.",
        fields=(
            FailureParamField("host_name", "str", "Target host name."),
            FailureParamField("fake_dns_ip", "str", "Incorrect DNS IP.", default="8.8.8.8"),
        ),
        example="nika failure inject host_incorrect_dns --set host_name=h1 --set fake_dns_ip=8.8.8.8",
    )

    symptom_desc = "Some hosts are unable to access web services."

    def __init__(self, scenario_name: str | None, **kwargs):
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaBaseAPI(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorHost(lab_name=self.net_env.lab.name)
        self.faulty_devices = [random.choice(self.net_env.hosts)]
        self.fake_dns_ip = "8.8.8.8"

    def inject_fault(self):
        self.injector.inject_dns_misconfiguration(
            host_name=self.faulty_devices[0],
            fake_dns_ip=self.fake_dns_ip,
        )

class HostIncorrectDNSDetection(HostIncorrectDNSBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=HostIncorrectDNSBase.root_cause_category,
        root_cause_name=HostIncorrectDNSBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class HostIncorrectDNSLocalization(HostIncorrectDNSBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=HostIncorrectDNSBase.root_cause_category,
        root_cause_name=HostIncorrectDNSBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class HostIncorrectDNSRCA(HostIncorrectDNSBase, RCATask):
    META = ProblemMeta(
        root_cause_category=HostIncorrectDNSBase.root_cause_category,
        root_cause_name=HostIncorrectDNSBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
