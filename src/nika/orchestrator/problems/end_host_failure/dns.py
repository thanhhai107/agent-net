from typing import Optional

from pydantic import BaseModel, Field

from nika.generator.fault.injector_base import FaultInjectorBase
from nika.net_env.net_env_pool import get_net_env_instance
from nika.orchestrator.problems.problem_base import ProblemMeta, RootCauseCategory, TaskDescription, TaskLevel, build_verify_result
from nika.orchestrator.tasks.detection import DetectionTask
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask
from nika.service.kathara import KatharaBaseAPI
from nika.utils.logger import system_logger

logger = system_logger

# ==================================================================
# Problem: DNS record error. Apps resolve domain but connect to wrong host.
# ==================================================================


class DNSRecordErrorParams(BaseModel):
    """Parameters for injecting a DNS record error fault."""

    host_name: str = Field(description="Target DNS server host name.")
    target_website: str = Field(description="Record host label.")
    target_domain: str = Field(description="DNS zone/domain.")
    wrong_ip: Optional[str] = Field(default=None, description="Incorrect IP to set. Derived at inject time if omitted.")


class DNSRecordErrorBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.END_HOST_FAILURE
    root_cause_name: str = "dns_record_error"

    symptom_desc = "Some hosts cannot access external websites."
    TAGS: str = ["dns"]

    Params = DNSRecordErrorParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaBaseAPI(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorBase(lab_name=self.net_env.lab.name)
        self.faulty_devices: list[str] = []
        self._wrong_ip: str | None = None

    def inject_fault(self, params: DNSRecordErrorParams):
        host = params.host_name
        self.faulty_devices = [host]
        target_website = params.target_website
        target_domain = params.target_domain
        wrong_ip = params.wrong_ip or self.kathara_api.get_host_ip(self.net_env.hosts[0])
        self._wrong_ip = wrong_ip
        right_ip = self.kathara_api.get_host_ip(host)

        self.kathara_api.exec_cmd(
            host,
            f"cp /etc/bind/db.{target_domain} /etc/bind/db.{target_domain}.bak",
        )
        cmd = r"sed -i 's/^\({name}[[:space:]]\+IN[[:space:]]\+A[[:space:]]\+\)[0-9\.]\+/\1{new_ip}/' /etc/bind/db.{domain}"
        cmd = cmd.format(name=target_website, new_ip=wrong_ip, domain=target_domain)
        self.kathara_api.exec_cmd(host, cmd)
        self.kathara_api.exec_cmd(
            host,
            "rndc reload 2>/dev/null || service named restart 2>/dev/null || true",
        )
        logger.info(
            f"Injecting DNS record error on {host}: mapping {target_website}:{target_domain} "
            f"to wrong IP {wrong_ip} instead of {right_ip}"
        )

    def verify_fault(self, params: DNSRecordErrorParams) -> dict:
        """Verify the DNS zone file contains the wrong IP and the running daemon serves it."""
        host = params.host_name
        target_website = params.target_website
        target_domain = params.target_domain
        wrong_ip = params.wrong_ip or self._wrong_ip
        grep_result = self.kathara_api.exec_cmd(
            host,
            f"grep '{target_website}.*{wrong_ip}' /etc/bind/db.{target_domain} 2>/dev/null && echo found || echo absent",
        ).strip()
        file_has_wrong_ip = "found" in grep_result
        dig_result = self.kathara_api.exec_cmd(
            host,
            f"dig +short {target_website}.{target_domain} @127.0.0.1 2>/dev/null || echo absent",
        ).strip()
        dns_resolves_wrong = wrong_ip in dig_result
        verified = file_has_wrong_ip and dns_resolves_wrong
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={
                "host": host,
                "target_website": target_website,
                "wrong_ip": wrong_ip,
                "grep_result": grep_result,
                "file_has_wrong_ip": file_has_wrong_ip,
                "dig_result": dig_result,
                "dns_resolves_wrong": dns_resolves_wrong,
            },
        )


class DNSRecordErrorDetection(DNSRecordErrorBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=DNSRecordErrorBase.root_cause_category,
        root_cause_name=DNSRecordErrorBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class DNSRecordErrorLocalization(DNSRecordErrorBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=DNSRecordErrorBase.root_cause_category,
        root_cause_name=DNSRecordErrorBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class DNSRecordErrorRCA(DNSRecordErrorBase, RCATask):
    META = ProblemMeta(
        root_cause_category=DNSRecordErrorBase.root_cause_category,
        root_cause_name=DNSRecordErrorBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )
