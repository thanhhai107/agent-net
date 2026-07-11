import ipaddress

from pydantic import BaseModel, Field

from nika.problems.problem_base import (
    RootCauseCategory,
    build_verify_result,
    ProblemBase,
)
from nika.utils.logger import system_logger

# ==================================================================
# Problem: DHCP missing subnet
# ==================================================================


class DHCPMissingSubnetParams(BaseModel):
    """Parameters for injecting a DHCP missing subnet fault."""

    host_name: str = Field(description="DHCP server host name.")
    host_name_2: str = Field(description="Affected client host name.")


class DHCPMissingSubnet(ProblemBase):
    root_cause_category: RootCauseCategory = RootCauseCategory.MISCONFIGURATION
    root_cause_name: str = "dhcp_missing_subnet"

    TAGS: str = ["dhcp"]

    Params = DHCPMissingSubnetParams

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__(scenario_name, **kwargs)

    def inject_fault(self, params: DHCPMissingSubnetParams):
        dhcp_server = params.host_name
        client_host = params.host_name_2
        self.set_faulty_devices([dhcp_server, client_host])
        system_logger.info(
            f"Injecting DHCP missing subnet fault: DHCP server {dhcp_server}, affected host {client_host}"
        )
        subnet = str(
            ipaddress.ip_network(
                self.runtime.get_host_ip(client_host, with_prefix=True), strict=False
            ).network_address
        )
        self.runtime.dhcp_delete_subnet(dhcp_server, subnet)
        self.runtime.renew_dhcp_leases(self.runtime.list_dhcp_client_nodes())
        self._injected_subnet = subnet

    def verify_fault(self, params: DHCPMissingSubnetParams) -> dict:
        """Verify the deleted subnet is absent from dhcpd.conf."""
        dhcp_server = params.host_name
        client_host = params.host_name_2
        self.set_faulty_devices([dhcp_server, client_host])
        subnet = getattr(self, "_injected_subnet", None)
        if subnet is None:
            subnet = str(
                ipaddress.ip_network(
                    self.runtime.get_host_ip(client_host, with_prefix=True),
                    strict=False,
                ).network_address
            )
        grep_result = self.runtime.exec(
            dhcp_server,
            f"grep 'subnet {subnet} netmask' /etc/dhcp/dhcpd.conf && echo found || echo absent",
        ).strip()
        verified = "absent" in grep_result
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={
                "dhcp_server": dhcp_server,
                "subnet": subnet,
                "grep_result": grep_result,
            },
        )
