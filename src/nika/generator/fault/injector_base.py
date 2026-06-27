from nika.service.kathara import KatharaAPIALL
from nika.service.kathara.docker_utils import get_machine_container
from nika.utils.logger import system_logger

""" Fault injector for Kathara """


class FaultInjectorBase:
    def __init__(self, lab_name: str):
        self.kathara_api = KatharaAPIALL(lab_name)
        self.logger = system_logger

    def inject_intf_down(self, host_name: str, intf_name: str):
        """Bring down a specific interface of a host."""
        self.kathara_api.intf_on_off(host_name=host_name, interface=intf_name, state="down")
        self.logger.info(f"Injected interface down on {host_name}:{intf_name}")

    def recover_intf_down(self, host_name: str, intf_name: str):
        """Recover from an interface down by enabling the interface."""
        self.kathara_api.intf_on_off(host_name=host_name, interface=intf_name, state="up")
        self.logger.info(f"Recovered interface down on {host_name}:{intf_name}")

    def inject_host_down(self, host_name: str):
        """Inject a host crash fault by pausing the container."""
        container = get_machine_container(lab_name=self.kathara_api.lab.name, host_name=host_name)
        container.reload()
        if container.status != "paused":
            container.pause()
        self.logger.info(f"Injected host down fault on {host_name} (container paused).")

    def recover_host_down(self, host_name: str):
        """Recover from host crash fault by unpausing or starting the container."""
        container = get_machine_container(lab_name=self.kathara_api.lab.name, host_name=host_name)
        container.reload()
        if container.status == "paused":
            container.unpause()
        elif container.status in {"created", "exited"}:
            container.start()
        self.logger.info(f"Recovered host down fault on {host_name} (container restored).")

    def inject_acl_rule(self, host_name: str, rule: str, table_name: str = "filter", family: str = "inet"):
        """Inject an ACL rule into a specific host."""
        self.kathara_api.nft_add_table(host_name=host_name, table_name=table_name, family=family)
        for chain_name in ["input", "forward", "output"]:
            self.kathara_api.nft_add_chain(
                host_name=host_name,
                family=family,
                table=table_name,
                chain=chain_name,
                hook=chain_name,
                type="filter",
                policy="accept",
            )
            self.kathara_api.nft_add_rule(
                host_name=host_name,
                family=family,
                table=table_name,
                chain=chain_name,
                rule=rule,
            )
        self.logger.info(f"Injected ACL rule on {host_name}: {rule}")

    def recover_acl_rule(self, host_name: str, table_name: str = "filter", family: str = "inet"):
        """Recover from an ACL rule by deleting the filter table."""
        self.kathara_api.nft_delete_table(host_name=host_name, table_name=table_name, family=family)
        self.logger.info(f"Recovered ACL rules on {host_name} by deleting table {table_name}.")

    def inject_service_down(self, host_name: str, service_name: str):
        """Inject a fault by stopping a service on a host."""
        self.logger.info(f"Injected service down fault on {host_name} for service {service_name}.")
        self.kathara_api.systemctl_ops(host_name=host_name, service_name=service_name, operation="stop")

    def recover_service_down(self, host_name: str, service_name: str):
        """Recover from a fault by starting a service on a host."""
        self.logger.info(f"Recovered service down fault on {host_name} for service {service_name}.")
        self.kathara_api.systemctl_ops(host_name=host_name, service_name=service_name, operation="start")

    def inject_process_kill(self, host_name: str, process_name: str):
        """Kill a process by name using pkill -9. Works in Kathara (no systemd required)."""
        self.kathara_api.exec_cmd(host_name, f"pkill -9 {process_name} 2>/dev/null; true")
        self.logger.info(f"Injected process kill on {host_name} for process {process_name}.")

    def inject_bmv2_down(self, host_name: str):
        """Inject a fault by stopping the bmv2 service on a host."""
        self.kathara_api.exec_cmd(host_name, "pkill simple_switch")
        self.logger.info(f"Injected bmv2 down fault on {host_name}.")

    def recover_bmv2_down(self, host_name: str):
        """Recover from a fault by starting the bmv2 service on a host.
        Note: make sure bmv2 is started via the host's startup script.
        """
        cmd = f"./hostlab/{host_name}.startup"
        self.kathara_api.exec_cmd(host_name, cmd)
        self.logger.info(f"Recovered bmv2 down fault on {host_name}.")
