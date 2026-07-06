from nika.orchestrator.problems.context import init_problem
from pydantic import BaseModel, Field

from nika.orchestrator.problems.problem_base import ProblemMeta, RootCauseCategory, TaskDescription, TaskLevel, build_verify_result
from nika.orchestrator.tasks.detection import DetectionTask
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask

# ==================================================================
# Problem: BGP Access Policy Misconfiguration - ACL blocking BGP traffic
# ==================================================================


class BGPAclBlockParams(BaseModel):
    """Parameters for injecting a BGP ACL block fault."""

    host_name: str = Field(description="Target router host name.")


class BGPAclBlockBase:
    root_cause_category = RootCauseCategory.MISCONFIGURATION
    root_cause_name = "bgp_acl_block"
    TAGS: str = ["bgp"]

    Params = BGPAclBlockParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env, self.runtime = init_problem(scenario_name, **kwargs)
        self.faulty_devices: list[str] = []

    def inject_fault(self, params: BGPAclBlockParams):
        host = params.host_name
        self.faulty_devices = [host]
        self.runtime.add_nft_drop_rule(host, "tcp dport 179 drop", family="inet")
        self.runtime.add_nft_drop_rule(host, "tcp sport 179 drop", family="inet")

    def verify_fault(self, params: BGPAclBlockParams) -> dict:
        """Verify nftables has a rule blocking TCP port 179 (BGP)."""
        host = params.host_name
        self.faulty_devices = [host]
        nft_output = self.runtime.exec(host, "nft list ruleset 2>/dev/null").strip()
        verified = "tcp dport 179" in nft_output and "drop" in nft_output
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"host": host, "nft_snippet": nft_output},
        )


class BGPAclBlockDetection(BGPAclBlockBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=BGPAclBlockBase.root_cause_category,
        root_cause_name=BGPAclBlockBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class BGPAclBlockLocalization(BGPAclBlockBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=BGPAclBlockBase.root_cause_category,
        root_cause_name=BGPAclBlockBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class BGPAclBlockRCA(BGPAclBlockBase, RCATask):
    META = ProblemMeta(
        root_cause_category=BGPAclBlockBase.root_cause_category,
        root_cause_name=BGPAclBlockBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==================================================================
# Problem: OSPF Access Policy Misconfiguration - ACL blocking OSPF traffic
# ==================================================================


class OSPFAclBlockParams(BaseModel):
    """Parameters for injecting an OSPF ACL block fault."""

    host_name: str = Field(description="Target router host name.")


class OSPFAclBlockBase:
    root_cause_category = RootCauseCategory.MISCONFIGURATION
    root_cause_name = "ospf_acl_block"
    TAGS: str = ["ospf"]

    Params = OSPFAclBlockParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env, self.runtime = init_problem(scenario_name, **kwargs)
        self.faulty_devices: list[str] = []

    def inject_fault(self, params: OSPFAclBlockParams):
        host = params.host_name
        self.faulty_devices = [host]
        self.runtime.add_nft_drop_rule(host, "ip protocol ospf drop", family="inet")
        self.runtime.add_nft_drop_rule(host, "ip protocol ospf drop", family="inet")

    def verify_fault(self, params: OSPFAclBlockParams) -> dict:
        """Verify nftables has a rule blocking OSPF protocol."""
        host = params.host_name
        self.faulty_devices = [host]
        nft_output = self.runtime.exec(host, "nft list ruleset 2>/dev/null").strip()
        verified = "ospf" in nft_output and "drop" in nft_output
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"host": host, "nft_snippet": nft_output},
        )


class OSPFAclBlockDetection(OSPFAclBlockBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=OSPFAclBlockBase.root_cause_category,
        root_cause_name=OSPFAclBlockBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class OSPFAclBlockLocalization(OSPFAclBlockBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=OSPFAclBlockBase.root_cause_category,
        root_cause_name=OSPFAclBlockBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class OSPFAclBlockRCA(OSPFAclBlockBase, RCATask):
    META = ProblemMeta(
        root_cause_category=OSPFAclBlockBase.root_cause_category,
        root_cause_name=OSPFAclBlockBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==================================================================
# Problem: ARP Access Policy Misconfiguration - ACL blocking ARP traffic
# ==================================================================


class ARPAclBlockParams(BaseModel):
    """Parameters for injecting an ARP ACL block fault."""

    host_name: str = Field(description="Target host name.")


class ARPAclBlockBase:
    root_cause_category = RootCauseCategory.MISCONFIGURATION
    root_cause_name = "arp_acl_block"
    TAGS: str = ["arp"]

    Params = ARPAclBlockParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env, self.runtime = init_problem(scenario_name, **kwargs)
        self.faulty_devices: list[str] = []

    def inject_fault(self, params: ARPAclBlockParams):
        host = params.host_name
        self.faulty_devices = [host]
        self.runtime.add_nft_drop_rule(host, "drop", family="arp")
        self.runtime.exec(host, "ip neigh flush all")

    def verify_fault(self, params: ARPAclBlockParams) -> dict:
        """Verify nftables has a rule blocking ARP traffic."""
        host = params.host_name
        self.faulty_devices = [host]
        nft_output = self.runtime.exec(host, "nft list ruleset 2>/dev/null").strip()
        verified = "arp" in nft_output and "drop" in nft_output
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"host": host, "nft_snippet": nft_output},
        )


class ARPAclBlockDetection(ARPAclBlockBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=ARPAclBlockBase.root_cause_category,
        root_cause_name=ARPAclBlockBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class ARPAclBlockLocalization(ARPAclBlockBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=ARPAclBlockBase.root_cause_category,
        root_cause_name=ARPAclBlockBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class ARPAclBlockRCA(ARPAclBlockBase, RCATask):
    META = ProblemMeta(
        root_cause_category=ARPAclBlockBase.root_cause_category,
        root_cause_name=ARPAclBlockBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==================================================================
# Problem: ACL blocking ICMP traffic
# ==================================================================


class IcmpAclBlockParams(BaseModel):
    """Parameters for injecting an ICMP ACL block fault."""

    host_name: str = Field(description="Target host name.")


class IcmpAclBlockBase:
    root_cause_category = RootCauseCategory.MISCONFIGURATION
    root_cause_name = "icmp_acl_block"
    TAGS: str = ["icmp"]

    Params = IcmpAclBlockParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env, self.runtime = init_problem(scenario_name, **kwargs)
        self.faulty_devices: list[str] = []

    def inject_fault(self, params: IcmpAclBlockParams):
        host = params.host_name
        self.faulty_devices = [host]
        self.runtime.add_nft_drop_rule(host, "ip protocol icmp drop", family="ip")

    def verify_fault(self, params: IcmpAclBlockParams) -> dict:
        """Verify nftables has a rule blocking ICMP traffic."""
        host = params.host_name
        self.faulty_devices = [host]
        nft_output = self.runtime.exec(host, "nft list ruleset 2>/dev/null").strip()
        verified = "icmp" in nft_output and "drop" in nft_output
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"host": host, "nft_snippet": nft_output},
        )


class IcmpAclBlockDetection(IcmpAclBlockBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=IcmpAclBlockBase.root_cause_category,
        root_cause_name=IcmpAclBlockBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class IcmpAclBlockLocalization(IcmpAclBlockBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=IcmpAclBlockBase.root_cause_category,
        root_cause_name=IcmpAclBlockBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class IcmpAclBlockRCA(IcmpAclBlockBase, RCATask):
    META = ProblemMeta(
        root_cause_category=IcmpAclBlockBase.root_cause_category,
        root_cause_name=IcmpAclBlockBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==================================================================
# Problem: ACL blocking HTTP traffic
# ==================================================================


class HttpAclBlockParams(BaseModel):
    """Parameters for injecting an HTTP ACL block fault."""

    host_name: str = Field(description="Target host name.")


class HttpAclBlockBase:
    root_cause_category = RootCauseCategory.MISCONFIGURATION
    root_cause_name = "http_acl_block"
    TAGS: str = ["http", "pc"]

    Params = HttpAclBlockParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env, self.runtime = init_problem(scenario_name, **kwargs)
        self.faulty_devices: list[str] = []

    def inject_fault(self, params: HttpAclBlockParams):
        host = params.host_name
        self.faulty_devices = [host]
        self.runtime.add_nft_drop_rule(host, "tcp dport 80 drop", family="inet")

    def verify_fault(self, params: HttpAclBlockParams) -> dict:
        """Verify nftables has a rule blocking HTTP (port 80) traffic."""
        host = params.host_name
        self.faulty_devices = [host]
        nft_output = self.runtime.exec(host, "nft list ruleset 2>/dev/null").strip()
        verified = "tcp dport 80" in nft_output and "drop" in nft_output
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"host": host, "nft_snippet": nft_output},
        )


class HttpAclBlockDetection(HttpAclBlockBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=HttpAclBlockBase.root_cause_category,
        root_cause_name=HttpAclBlockBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class HttpAclBlockLocalization(HttpAclBlockBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=HttpAclBlockBase.root_cause_category,
        root_cause_name=HttpAclBlockBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class HttpAclBlockRCA(HttpAclBlockBase, RCATask):
    META = ProblemMeta(
        root_cause_category=HttpAclBlockBase.root_cause_category,
        root_cause_name=HttpAclBlockBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==================================================================
# Problem: DNS listener port blocked
# ==================================================================


class DNSPortBlockedParams(BaseModel):
    """Parameters for injecting a DNS port blocked fault."""

    host_name: str = Field(description="Target DNS server host name.")


class DNSPortBlockedBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.MISCONFIGURATION
    root_cause_name: str = "dns_port_blocked"

    TAGS: str = ["dns", "http"]

    Params = DNSPortBlockedParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env, self.runtime = init_problem(scenario_name, **kwargs)
        self.faulty_devices: list[str] = []

    def inject_fault(self, params: DNSPortBlockedParams):
        host = params.host_name
        self.faulty_devices = [host]
        self.runtime.add_nft_drop_rule(host, "tcp dport 53 drop", family="inet")
        self.runtime.add_nft_drop_rule(host, "udp dport 53 drop", family="inet")

    def verify_fault(self, params: DNSPortBlockedParams) -> dict:
        """Verify nftables has rules blocking DNS port 53."""
        host = params.host_name
        self.faulty_devices = [host]
        nft_output = self.runtime.exec(host, "nft list ruleset 2>/dev/null").strip()
        verified = "dport 53" in nft_output and "drop" in nft_output
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={"host": host, "nft_snippet": nft_output},
        )


class DNSPortBlockedDetection(DNSPortBlockedBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=DNSPortBlockedBase.root_cause_category,
        root_cause_name=DNSPortBlockedBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class DNSPortBlockedLocalization(DNSPortBlockedBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=DNSPortBlockedBase.root_cause_category,
        root_cause_name=DNSPortBlockedBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class DNSPortBlockedRCA(DNSPortBlockedBase, RCATask):
    META = ProblemMeta(
        root_cause_category=DNSPortBlockedBase.root_cause_category,
        root_cause_name=DNSPortBlockedBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )
