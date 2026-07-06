import logging
from typing import Optional

from nika.orchestrator.problems.context import init_problem
from pydantic import BaseModel, Field

from nika.orchestrator.problems.inject_resolve import derive_incorrect_ip, derive_wrong_gateway
from nika.orchestrator.problems.problem_base import ProblemMeta, RootCauseCategory, TaskDescription, TaskLevel, build_verify_result
from nika.orchestrator.tasks.detection import DetectionTask
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask
from nika.utils.logger import system_logger


def _inject_ip_change(
    runtime,
    *,
    host_name: str,
    old_ip: str,
    new_ip: str,
    intf_name: str,
    new_gateway: str | None = None,
) -> None:
    runtime.exec(host_name, f"ip addr del {old_ip} dev {intf_name}")
    runtime.exec(host_name, f"ip addr add {new_ip} dev {intf_name}")
    if new_gateway:
        runtime.exec(host_name, f"ip route add default via {new_gateway}")


# ==========================================
# Problem: Host missing IP address
# ==========================================


class HostMissingIPParams(BaseModel):
    """Parameters for injecting a host-missing-IP fault."""

    host_name: str = Field(description="Target host name.")
    intf_name: str = Field(default="eth0", description="Target interface name.")


class HostMissingIPBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.END_HOST_FAILURE
    root_cause_name: str = "host_missing_ip"
    TAGS: str = ["pc"]

    Params = HostMissingIPParams

    symptom_desc = "Some hosts are unable to communicate with other devices in the network."

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.logger = system_logger
        self.net_env, self.runtime = init_problem(scenario_name, **kwargs)
        self.faulty_devices: list[str] = []
        self.intf_name = "eth0"

    def inject_fault(self, params: HostMissingIPParams):
        host = params.host_name
        self.faulty_devices = [host]
        intf = params.intf_name
        real_ip = self.runtime.get_host_ip(host, intf, with_prefix=True)
        real_gateway = self.runtime.get_default_gateway(host)
        self.runtime.exec(host, f"ip addr del {real_ip} dev {intf}")
        self.runtime.exec(host, f"echo '{real_ip} {real_gateway}' > /tmp/removed_ip.txt")
        self.logger.info(f"Injected missing IP on {host} from {real_ip} and gateway {real_gateway}.")

    def verify_fault(self, params: HostMissingIPParams) -> dict:
        """Verify that the host has no global IPv4 address on the interface."""
        host = params.host_name
        intf = params.intf_name
        ip_line = self.runtime.exec(
            host, f"ip -4 -o addr show dev {intf} scope global"
        ).strip()
        verified = "inet " not in ip_line
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"host": host, "intf": intf, "ip_line": ip_line},
        )


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


class HostIPConflictParams(BaseModel):
    """Parameters for injecting a host IP conflict fault."""

    host_name: str = Field(description="Source host whose IP is copied.")
    host_name_2: str = Field(description="Target host to misconfigure.")


class HostIPConflictBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.END_HOST_FAILURE
    root_cause_name: str = "host_ip_conflict"
    TAGS: str = ["pc"]

    Params = HostIPConflictParams

    symptom_desc = "Some hosts experience intermittent connectivity issues."

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env, self.runtime = init_problem(scenario_name, **kwargs)
        self.faulty_devices: list[str] = []

    def inject_fault(self, params: HostIPConflictParams):
        src_host = params.host_name
        dst_host = params.host_name_2
        self.faulty_devices = [src_host, dst_host]
        _inject_ip_change(self.runtime, 
            host_name=dst_host,
            old_ip=self.runtime.get_host_ip(dst_host, "eth0", with_prefix=True),
            new_ip=self.runtime.get_host_ip(src_host, "eth0", with_prefix=True),
            intf_name="eth0",
            new_gateway=self.runtime.get_default_gateway(src_host),
        )

    def verify_fault(self, params: HostIPConflictParams) -> dict:
        """Verify both hosts share the same eth0 IP (conflict)."""
        host_a = params.host_name
        host_b = params.host_name_2
        cmd = "ip -4 -o addr show dev eth0 scope global | awk '/inet /{print $4}'"
        ip_a_raw = self.runtime.exec(host_a, cmd).strip()
        ip_b_raw = self.runtime.exec(host_b, cmd).strip()
        ip_a = ip_a_raw.split("/")[0] if ip_a_raw else ""
        ip_b = ip_b_raw.split("/")[0] if ip_b_raw else ""
        verified = bool(ip_a) and ip_a == ip_b
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"host_a": host_a, "host_b": host_b, "ip_a": ip_a, "ip_b": ip_b},
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


class HostIncorrectIPParams(BaseModel):
    """Parameters for injecting an incorrect host IP fault."""

    host_name: str = Field(description="Target host name.")
    incorrect_ip: Optional[str] = Field(default=None, description="Incorrect CIDR IP. Derived at inject time if omitted.")


class HostIncorrectIPBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.END_HOST_FAILURE
    root_cause_name: str = "host_incorrect_ip"
    TAGS: str = ["pc"]

    Params = HostIncorrectIPParams

    symptom_desc = "Some hosts seem to be unreachable in the network."

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env, self.runtime = init_problem(scenario_name, **kwargs)
        self.faulty_devices: list[str] = []
        self._original_ip: str | None = None

    def inject_fault(self, params: HostIncorrectIPParams):
        host = params.host_name
        self.faulty_devices = [host]
        old_ip = self.runtime.get_host_ip(host, "eth0", with_prefix=True)
        self._original_ip = old_ip
        incorrect_ip = params.incorrect_ip or derive_incorrect_ip(self.runtime, host)
        _inject_ip_change(self.runtime, 
            host_name=host,
            old_ip=old_ip,
            new_ip=incorrect_ip,
            intf_name="eth0",
            new_gateway=self.runtime.get_default_gateway(host),
        )

    def verify_fault(self, params: HostIncorrectIPParams) -> dict:
        """Verify that the host eth0 IP differs from the original address at inject time."""
        host = params.host_name
        ip_line = self.runtime.exec(
            host, "ip -4 -o addr show dev eth0 scope global"
        ).strip()
        current_ip = None
        if "inet " in ip_line:
            parts = ip_line.split()
            for i, p in enumerate(parts):
                if p == "inet" and i + 1 < len(parts):
                    current_ip = parts[i + 1]
                    break
        verified = bool(current_ip) and current_ip != self._original_ip
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"host": host, "ip_line": ip_line, "original_ip": self._original_ip, "current_ip": current_ip},
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


class HostIncorrectGatewayParams(BaseModel):
    """Parameters for injecting an incorrect host gateway fault."""

    host_name: str = Field(description="Target host name.")
    new_gateway: Optional[str] = Field(default=None, description="Incorrect gateway IP. Derived at inject time if omitted.")


class HostIncorrectGatewayBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.END_HOST_FAILURE
    root_cause_name: str = "host_incorrect_gateway"
    TAGS: str = ["pc", "frr"]

    Params = HostIncorrectGatewayParams

    symptom_desc = "Some hosts seem to be unreachable in the network."

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env, self.runtime = init_problem(scenario_name, **kwargs)
        self.faulty_devices: list[str] = []
        self._injected_gateway: str | None = None

    def inject_fault(self, params: HostIncorrectGatewayParams):
        host = params.host_name
        self.faulty_devices = [host]
        new_gateway = params.new_gateway or derive_wrong_gateway(self.runtime, host)
        self._injected_gateway = new_gateway
        _inject_ip_change(self.runtime, 
            host_name=host,
            old_ip=self.runtime.get_host_ip(host, "eth0", with_prefix=True),
            new_ip=self.runtime.get_host_ip(host, "eth0", with_prefix=True),
            intf_name="eth0",
            new_gateway=new_gateway,
        )

    def verify_fault(self, params: HostIncorrectGatewayParams) -> dict:
        """Verify that the default route uses the injected wrong gateway."""
        host = params.host_name
        route_line = self.runtime.exec(host, "ip route show default").strip()
        expected_gateway = params.new_gateway or self._injected_gateway
        verified = bool(expected_gateway) and expected_gateway in route_line
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"host": host, "route_line": route_line, "expected_gateway": expected_gateway},
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


class HostIncorrectNetmaskParams(BaseModel):
    """Parameters for injecting an incorrect host netmask fault."""

    host_name: str = Field(description="Target host name.")
    netmask_prefix: int = Field(default=8, description="Incorrect prefix length.")


class HostIncorrectNetmaskBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.END_HOST_FAILURE
    root_cause_name: str = "host_incorrect_netmask"
    TAGS: str = ["pc", "frr"]

    Params = HostIncorrectNetmaskParams

    symptom_desc = "Some hosts seem to be unreachable in the network."

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env, self.runtime = init_problem(scenario_name, **kwargs)
        self.faulty_devices: list[str] = []
        self.netmask_prefix = 8

    def inject_fault(self, params: HostIncorrectNetmaskParams):
        host = params.host_name
        self.faulty_devices = [host]
        old_ip = self.runtime.get_host_ip(host, "eth0", with_prefix=True)
        ip_part = old_ip.split("/")[0]
        new_ip = f"{ip_part}/{params.netmask_prefix}"
        _inject_ip_change(self.runtime, 
            host_name=host,
            old_ip=old_ip,
            new_ip=new_ip,
            intf_name="eth0",
            new_gateway=self.runtime.get_default_gateway(host),
        )

    def verify_fault(self, params: HostIncorrectNetmaskParams) -> dict:
        """Verify that eth0 has a non-/24 prefix (injected wrong netmask)."""
        host = params.host_name
        expected_prefix = params.netmask_prefix
        ip_line = self.runtime.exec(
            host, "ip -4 -o addr show dev eth0 scope global"
        ).strip()
        prefix = None
        if "inet " in ip_line:
            parts = ip_line.split()
            for i, p in enumerate(parts):
                if p == "inet" and i + 1 < len(parts):
                    cidr = parts[i + 1]
                    if "/" in cidr:
                        prefix = int(cidr.split("/")[1])
                    break
        verified = prefix is not None and prefix != 24
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"host": host, "ip_line": ip_line, "expected_prefix": expected_prefix, "actual_prefix": prefix},
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


class HostIncorrectDNSParams(BaseModel):
    """Parameters for injecting an incorrect DNS resolver fault."""

    host_name: str = Field(description="Target host name.")
    fake_dns_ip: str = Field(default="8.8.8.8", description="Incorrect DNS IP.")


class HostIncorrectDNSBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.END_HOST_FAILURE
    root_cause_name: str = "host_incorrect_dns"
    TAGS: str = ["dns"]

    Params = HostIncorrectDNSParams

    symptom_desc = "Some hosts are unable to access web services."

    def __init__(self, scenario_name: str | None, **kwargs):
        self.net_env, self.runtime = init_problem(scenario_name, **kwargs)
        self.faulty_devices: list[str] = []
        self.fake_dns_ip = "8.8.8.8"

    def inject_fault(self, params: HostIncorrectDNSParams):
        host = params.host_name
        self.faulty_devices = [host]
        self.runtime.exec(host, f"echo 'nameserver {params.fake_dns_ip}' > /etc/resolv.conf")

    def verify_fault(self, params: HostIncorrectDNSParams) -> dict:
        """Verify the incorrect-DNS fault by checking /etc/resolv.conf contains the fake DNS IP."""
        host = params.host_name
        fake_dns_ip = params.fake_dns_ip
        resolv = self.runtime.exec(host, "cat /etc/resolv.conf 2>/dev/null || echo ''")
        verified = fake_dns_ip in resolv
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"host": host, "fake_dns_ip": fake_dns_ip, "resolv_conf": resolv.strip()},
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
