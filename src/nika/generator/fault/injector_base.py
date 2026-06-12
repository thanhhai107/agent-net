import docker

from nika.service.kathara import KatharaAPIALL
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
        docker_client = docker.from_env()
        candidates = docker_client.containers.list(all=True, filters={"name": host_name})
        if not candidates:
            raise ValueError(f"No container found for host {host_name}")
        container = next((item for item in candidates if item.name == host_name), candidates[0])
        container.reload()
        if container.status != "paused":
            container.pause()
        self.logger.info(f"Injected host down fault on {host_name} (container paused).")

    def recover_host_down(self, host_name: str):
        """Recover from host crash fault by unpausing or starting the container."""
        docker_client = docker.from_env()
        candidates = docker_client.containers.list(all=True, filters={"name": host_name})
        if not candidates:
            raise ValueError(f"No container found for host {host_name}")
        container = next((item for item in candidates if item.name == host_name), candidates[0])
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

    def inject_bgp_misconfig(self, host_name: str, correct_asn: int, wrong_asn: int):
        """Inject a BGP ASN misconfiguration by changing the ASN on a router, from real_asn to target_asn."""
        self.kathara_api.exec_cmd(
            host_name,
            f"vtysh -c 'show running-config' | sed 's/^router bgp {correct_asn}$/router bgp {wrong_asn}/' > /etc/frr/frr.conf && systemctl restart frr",
        )
        self.logger.info(f"Injected BGP ASN misconfiguration on {host_name} from ASN {correct_asn} to {wrong_asn}.")

    def recover_bgp_misconfig(self, host_name: str, correct_asn: int, wrong_asn: int):
        """Recover from a BGP ASN misconfiguration by resetting the ASN on a router."""
        self.kathara_api.exec_cmd(
            host_name,
            f"vtysh -c 'show running-config' | sed 's/^router bgp {wrong_asn}$/router bgp {correct_asn}/' > /etc/frr/frr.conf && systemctl restart frr",
        )
        self.logger.info(f"Recovered BGP ASN misconfiguration on {host_name} from ASN {wrong_asn} to {correct_asn}.")

    def inject_bgp_remove_advertisement(self, host_name: str):
        """Inject a BGP missing route by commenting out the network advertisement."""
        self.kathara_api.exec_cmd(
            host_name,
            "sed -i.bak -E 's/^([[:space:]]*)network /\1# network /' /etc/frr/frr.conf && systemctl restart frr",
        )
        self.logger.info(f"Injected BGP missing route on {host_name}.")

    def recover_bgp_remove_advertisement(self, host_name: str):
        """Recover from a BGP missing route by recovering the backed up frr.conf file."""
        self.kathara_api.exec_cmd(
            host_name,
            "mv /etc/frr/frr.conf.bak /etc/frr/frr.conf && systemctl restart frr",
        )
        self.logger.info(f"Recovered BGP missing route on {host_name}.")

    def inject_bgp_add_interface(self, host_name: str, intf_name: str, ip_address: str):
        """Inject a BGP add interface by adding a new interface with IP address and configuring BGP."""
        cmd = f"vtysh -c 'configure terminal' -c 'interface {intf_name}' -c 'ip address {ip_address}' "
        self.kathara_api.exec_cmd(
            host_name,
            cmd,
        )
        self.logger.info(f"Injected BGP add interface on {host_name}: {intf_name} with IP {ip_address}.")

    def recover_bgp_add_interface(self, host_name: str, intf_name: str, ip_address: str = None):
        """Recover from a BGP add interface by removing the interface configuration."""
        if intf_name == "lo":
            cmd = f"vtysh -c 'configure terminal' -c 'interface {intf_name}' -c 'no ip address {ip_address}' -c 'end' -c 'write memory' "
        else:
            cmd = f"vtysh -c 'configure terminal' -c 'no interface {intf_name}' -c 'end' -c 'write memory' "
        self.kathara_api.exec_cmd(
            host_name,
            cmd,
        )
        self.logger.info(f"Recovered BGP add interface on {host_name}: {intf_name}.")

    def inject_bgp_add_advertisement(self, host_name: str, network: str, AS: str):
        """Inject a BGP add route by adding a network advertisement."""
        cmd = f"vtysh -c 'configure terminal' -c 'router bgp {AS}' -c 'network {network}' -c 'end' -c 'write memory' "
        self.kathara_api.exec_cmd(
            host_name,
            cmd,
        )
        self.kathara_api.exec_cmd(
            host_name,
            "systemctl restart frr",
        )
        self.logger.info(f"Injected BGP add route on {host_name}: {network}.")

    def recover_bgp_add_advertisement(self, host_name: str, network: str, AS: str):
        """Recover from a BGP add route by removing the network advertisement."""
        cmd = (
            f"vtysh -c 'configure terminal' -c 'router bgp {AS}' -c 'no network {network}' -c 'end' -c 'write memory' "
        )
        self.kathara_api.exec_cmd(
            host_name,
            cmd,
        )
        self.logger.info(f"Recovered BGP add route on {host_name}: {network}.")

    def inject_add_route_blackhole_nexthop(self, host_name: str, network: str):
        """Inject a fault by adding a static blackhole route on a host."""
        self.kathara_api.exec_cmd(
            host_name,
            f"ip route replace blackhole {network}",
        )
        self.logger.info(f"Injected addition of blackhole route {network} on {host_name}.")

    def recover_add_route_blackhole_nexthop(self, host_name: str, network: str):
        """Recover from a fault by deleting a static blackhole route on a host."""
        self.kathara_api.exec_cmd(
            host_name,
            f"ip route del blackhole {network}",
        )
        self.logger.info(f"Recovered addition of blackhole route {network} on {host_name}.")

    def inject_add_route_blackhole_advertise(self, host_name: str, network: str, AS: str):
        cmd = (
            "vtysh -c 'configure terminal' "
            f"-c 'ip route {network} Null0' "
            f"-c 'router bgp {AS}' "
            f"-c 'network {network}' "
            "-c 'end' "
            "-c 'write memory' "
        )
        self.kathara_api.exec_cmd(
            host_name,
            cmd,
        )
        self.logger.info(f"Injected BGP advertise blackhole route on {host_name}: {network}.")

    def recover_add_route_blackhole_advertise(self, host_name: str, network: str, AS: str):
        cmd = (
            "vtysh -c 'configure terminal' "
            f"-c 'no ip route {network} Null0' "
            f"-c 'router bgp {AS}' "
            f"-c 'no network {network}' "
            "-c 'end' "
            "-c 'write memory' "
        )
        self.kathara_api.exec_cmd(
            host_name,
            cmd,
        )
        self.logger.info(f"Recovered BGP advertise blackhole route on {host_name}: {network}.")

    def inject_rip_missing_route(self, host_name: str, network: str):
        """Inject a RIP missing route by commenting out the network advertisement."""
        cmd = f"vtysh -c 'configure terminal' -c 'router rip' -c 'no network {network}' -c 'end' -c 'write memory' && systemctl restart frr"
        res = self.kathara_api.exec_cmd(
            host_name,
            cmd,
        )
        self.logger.info(f"Injected RIP missing route on {host_name}: {network}.")

    def recover_rip_missing_route(self, host_name: str, network: str):
        """Recover from a RIP missing route by recovering the backed up frr.conf file."""
        cmd = f"vtysh -c 'configure terminal' -c 'router rip' -c 'network {network}' -c 'end' -c 'write memory' && systemctl restart frr"
        self.kathara_api.exec_cmd(
            host_name,
            cmd,
        )
        self.logger.info(f"Recovered RIP missing route on {host_name}: {network}.")


if __name__ == "__main__":
    # Example usage
    injector = FaultInjectorBase("simple_bgp")
    injector.recover_intf_down("pc1", "eth0")
