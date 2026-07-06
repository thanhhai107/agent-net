from pydantic import BaseModel, Field

from nika.orchestrator.problems.context import init_problem
from nika.orchestrator.problems.problem_base import ProblemMeta, RootCauseCategory, TaskDescription, TaskLevel, build_verify_result
from nika.orchestrator.tasks.detection import DetectionTask
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask
from nika.utils.logger import system_logger

# ==========================================
# Problem: VPN membership missing on end host causing inability to access services over VPN.
# ==========================================


class VPNMembershipMissingParams(BaseModel):
    """Parameters for injecting a VPN membership missing fault."""

    host_name: str = Field(description="Target host to remove from VPN.")
    host_name_2: str = Field(description="VPN server host name.")


class VPNMembershipMissingBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.END_HOST_FAILURE
    root_cause_name: str = "host_vpn_membership_missing"
    TAGS: str = ["vpn"]

    Params = VPNMembershipMissingParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.logger = system_logger
        self.net_env, self.runtime = init_problem(scenario_name, **kwargs)
        self.faulty_devices: list[str] = []

    def inject_fault(self, params: VPNMembershipMissingParams):
        target_host = params.host_name
        vpn_server = params.host_name_2
        self.faulty_devices = [target_host, vpn_server]

        self.runtime.exec(
            vpn_server,
            "cp /etc/wireguard/wg0.conf /etc/wireguard/wg0.conf.bak",
        )
        self.runtime.exec(
            vpn_server,
            f"sed -i '/# {target_host}/{{n; s/^/# /; n; s/^/# /; n; s/^/# /;}}' /etc/wireguard/wg0.conf",
        )
        self.runtime.exec(
            vpn_server,
            "wg-quick down wg0 && wg-quick up wg0",
        )
        self.logger.info(f"Removed VPN membership of {target_host} on {vpn_server}.")

    def verify_fault(self, params: VPNMembershipMissingParams) -> dict:
        """Verify the VPN config for target_host has commented-out lines."""
        target_host = params.host_name
        vpn_server = params.host_name_2
        wg_conf_snippet = self.runtime.exec(
            vpn_server,
            f"grep -A4 '# {target_host}' /etc/wireguard/wg0.conf 2>/dev/null || echo absent",
        ).strip()
        lines = wg_conf_snippet.splitlines()
        commented_lines = [ln for ln in lines if ln.strip().startswith("#")]
        verified = len(commented_lines) >= 3
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={
                "vpn_server": vpn_server,
                "target_host": target_host,
                "wg_conf_snippet": wg_conf_snippet,
            },
        )


class HostIncorrectDNSDetection(VPNMembershipMissingBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=VPNMembershipMissingBase.root_cause_category,
        root_cause_name=VPNMembershipMissingBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class HostIncorrectDNSLocalization(VPNMembershipMissingBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=VPNMembershipMissingBase.root_cause_category,
        root_cause_name=VPNMembershipMissingBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class HostIncorrectDNSRCA(VPNMembershipMissingBase, RCATask):
    META = ProblemMeta(
        root_cause_category=VPNMembershipMissingBase.root_cause_category,
        root_cause_name=VPNMembershipMissingBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )
