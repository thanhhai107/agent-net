import shlex

from pydantic import BaseModel, Field

from nika.orchestrator.problems.problem_base import (
    RootCauseCategory,
    build_verify_result,
    ProblemBase,
)
from nika.utils.logger import system_logger


# ==================================================================
# Problem: Link failure by ip link down on host interface
# ==================================================================


class LinkFailureParams(BaseModel):
    """Parameters for injecting a link-down fault."""

    host_name: str = Field(description="Target host name.")
    intf_name: str = Field(default="eth0", description="Target interface name.")


class LinkFailure(ProblemBase):
    root_cause_category: RootCauseCategory = RootCauseCategory.LINK_FAILURE
    root_cause_name: str = "link_down"
    TAGS: str = ["link"]

    Params = LinkFailureParams

    symptom_desc = "Users report connectivity issues to other hosts."

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__(scenario_name, **kwargs)
        self.faulty_intf = "eth0"
        self.down_time = 1
        self.up_time = 1

    def inject_fault(self, params: LinkFailureParams):
        self.set_faulty_devices([params.host_name])
        self.faulty_intf = params.intf_name
        self.runtime.set_interface_state(params.host_name, params.intf_name, "down")

    def verify_fault(self, params: LinkFailureParams) -> dict:
        """Verify the link-down fault is active by reading the interface operstate from the container."""
        operstate = self.runtime.get_interface_operstate(
            params.host_name, params.intf_name
        )
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=operstate == "down",
            details={
                "host": params.host_name,
                "intf": params.intf_name,
                "operstate": operstate,
            },
        )


# ==========================================
# Problem: Link flapping by manual script
# ==========================================


class LinkFlapParams(BaseModel):
    """Parameters for injecting a link-flap fault."""

    host_name: str = Field(description="Target host name.")
    intf_name: str = Field(default="eth0", description="Target interface name.")
    down_time: int = Field(default=1, description="Down duration in seconds.")
    up_time: int = Field(default=1, description="Up duration in seconds.")


class LinkFlap(ProblemBase):
    root_cause_category: RootCauseCategory = RootCauseCategory.LINK_FAILURE
    root_cause_name: str = "link_flap"
    TAGS: str = ["link"]

    Params = LinkFlapParams

    symptom_desc = "Users report connectivity issues to other hosts."

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__(scenario_name, **kwargs)
        self.faulty_intf = "eth0"

    def inject_fault(self, params: LinkFlapParams):
        self.set_faulty_devices([params.host_name])
        self.faulty_intf = params.intf_name
        if params.down_time <= 0 or params.up_time <= 0:
            raise ValueError("down_time and up_time must be positive integers")

        script_path = f"/tmp/link_flap_{params.intf_name}.sh"
        pid_path = f"/tmp/link_flap_{params.intf_name}.pid"
        log_path = f"/tmp/link_flap_{params.intf_name}.log"

        script = f"""#!/bin/bash
IFACE={shlex.quote(params.intf_name)}
DOWN_TIME={int(params.down_time)}
UP_TIME={int(params.up_time)}
PID_FILE={shlex.quote(pid_path)}

cleanup() {{
    ip link set $IFACE up >/dev/null 2>&1 || true
    rm -f $PID_FILE
}}
trap cleanup EXIT INT TERM

echo $$ > $PID_FILE
while true; do
    ip link set $IFACE down
    sleep $DOWN_TIME
    ip link set $IFACE up
    sleep $UP_TIME
done
"""
        write_cmd = f"cat <<'EOF' > {shlex.quote(script_path)}\n{script}\nEOF\nchmod +x {shlex.quote(script_path)}"
        self.runtime.exec(params.host_name, write_cmd)

        stop_previous_cmd = (
            f"if [ -f {shlex.quote(pid_path)} ]; then "
            f"kill $(cat {shlex.quote(pid_path)}) 2>/dev/null || true; "
            f"rm -f {shlex.quote(pid_path)}; "
            "fi"
        )
        self.runtime.exec(params.host_name, stop_previous_cmd)

        start_cmd = f"nohup {shlex.quote(script_path)} > {shlex.quote(log_path)} 2>&1 < /dev/null &"
        self.runtime.exec(params.host_name, start_cmd)
        system_logger.info(
            f"Injected link flap on {params.host_name}:{params.intf_name} "
            f"(down_time={params.down_time}, up_time={params.up_time})"
        )

    def verify_fault(self, params: LinkFlapParams) -> dict:
        """Verify the link-flap script is running by checking the pid file and process liveness."""
        pid_path = f"/tmp/link_flap_{params.intf_name}.pid"
        running = self.runtime.pidfile_running(params.host_name, pid_path)
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=running,
            details={
                "host": params.host_name,
                "intf": params.intf_name,
                "flap_process": "running" if running else "not_running",
            },
        )


# ==========================================
# Problem: Link detached.
# ==========================================


class LinkDetachParams(BaseModel):
    """Parameters for injecting a link-detach fault."""

    host_name: str = Field(description="Target host name.")
    intf_name: str = Field(default="eth0", description="Target interface name.")


class LinkDetach(ProblemBase):
    root_cause_category: RootCauseCategory = RootCauseCategory.LINK_FAILURE
    root_cause_name: str = "link_detach"
    TAGS: str = ["link"]

    Params = LinkDetachParams

    symptom_desc = "Users report connectivity issues to other hosts."

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__(scenario_name, **kwargs)
        self.faulty_intf = "eth0"

    def inject_fault(self, params: LinkDetachParams):
        self.set_faulty_devices([params.host_name])
        self.faulty_intf = params.intf_name
        self.runtime.exec(params.host_name, f"ip link del {params.intf_name}")
        system_logger.info(
            f"Injected link detach on {params.host_name}:{params.intf_name}"
        )

    def verify_fault(self, params: LinkDetachParams) -> dict:
        """Verify the link-detach fault is active by confirming the interface no longer exists."""
        detached = not self.runtime.interface_exists(params.host_name, params.intf_name)
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=detached,
            details={
                "host": params.host_name,
                "intf": params.intf_name,
                "interface_exists": not detached,
            },
        )


# ==========================================
# Problem: Link fragmentation disabled, drop large packets
# ==========================================


class LinkFragParams(BaseModel):
    """Parameters for injecting a link-fragmentation-disabled fault."""

    host_name: str = Field(description="Target host name.")
    mtu: int = Field(default=10, description="Packet size threshold.")


class LinkFrag(ProblemBase):
    root_cause_category: RootCauseCategory = RootCauseCategory.LINK_FAILURE
    root_cause_name: str = "link_fragmentation_disabled"
    TAGS: str = ["link"]

    Params = LinkFragParams

    symptom_desc = (
        "Users report partial packet loss when communicating with other hosts."
    )

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__(scenario_name, **kwargs)
        self.mtu = 10

    @staticmethod
    def _frag_drop_rule_args(mtu: int) -> str:
        return f"-m length --length {int(mtu)}:65535 -j DROP"

    def inject_fault(self, params: LinkFragParams):
        self.set_faulty_devices([params.host_name])
        self.mtu = params.mtu
        self.runtime.exec(
            params.host_name,
            f"iptables -A OUTPUT {self._frag_drop_rule_args(params.mtu)}",
        )
        system_logger.info(
            f"Injected fragmentation disabled on {params.host_name} with MTU {params.mtu}"
        )

    def verify_fault(self, params: LinkFragParams) -> dict:
        """Verify the fragmentation-disabled fault is active via the exact length-based DROP rule."""
        rule_args = self._frag_drop_rule_args(params.mtu)
        verified = self.runtime.iptables_rule_present(
            params.host_name, "OUTPUT", rule_args
        )
        iptables_output = self.runtime.exec(
            params.host_name, "iptables -S OUTPUT"
        ).strip()
        expected_rule = f"-A OUTPUT {rule_args}"
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=verified,
            details={
                "host": params.host_name,
                "mtu": params.mtu,
                "rule": expected_rule,
                "iptables_output": iptables_output,
            },
        )
