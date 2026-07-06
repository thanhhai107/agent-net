import shlex

from pydantic import BaseModel, Field

from nika.orchestrator.problems.context import init_problem
from nika.orchestrator.problems.problem_base import ProblemMeta, RootCauseCategory, TaskDescription, TaskLevel, build_verify_result
from nika.orchestrator.tasks.detection import DetectionTask
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask
from nika.utils.logger import system_logger


# ==================================================================
# Problem: Link failure by ip link down on host interface
# ==================================================================


class LinkFailureParams(BaseModel):
    """Parameters for injecting a link-down fault."""

    host_name: str = Field(description="Target host name.")
    intf_name: str = Field(default="eth0", description="Target interface name.")


class LinkFailureBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.LINK_FAILURE
    root_cause_name: str = "link_down"
    TAGS: str = ["link"]

    Params = LinkFailureParams

    symptom_desc = "Users report connectivity issues to other hosts."

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env, self.runtime = init_problem(scenario_name, **kwargs)
        self.faulty_devices: list[str] = []
        self.faulty_intf = "eth0"
        self.down_time = 1
        self.up_time = 1

    def inject_fault(self, params: LinkFailureParams):
        host = params.host_name
        self.faulty_devices = [host]
        self.faulty_intf = params.intf_name
        self.runtime.set_interface_state(host, params.intf_name, "down")

    def verify_fault(self, params: LinkFailureParams) -> dict:
        """Verify the link-down fault is active by reading the interface operstate from the container."""
        host = params.host_name
        intf = params.intf_name
        operstate = self.runtime.get_interface_operstate(host, intf)
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=operstate == "down",
            details={"host": host, "intf": intf, "operstate": operstate},
        )


class LinkFailureDetection(LinkFailureBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=LinkFailureBase.root_cause_category,
        root_cause_name=LinkFailureBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class LinkFailureLocalization(LinkFailureBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=LinkFailureBase.root_cause_category,
        root_cause_name=LinkFailureBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class LinkFailureRCA(LinkFailureBase, RCATask):
    META = ProblemMeta(
        root_cause_category=LinkFailureBase.root_cause_category,
        root_cause_name=LinkFailureBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
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


class LinkFlapBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.LINK_FAILURE
    root_cause_name: str = "link_flap"
    TAGS: str = ["link"]

    Params = LinkFlapParams

    symptom_desc = "Users report connectivity issues to other hosts."

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env, self.runtime = init_problem(scenario_name, **kwargs)
        self.faulty_devices: list[str] = []
        self.faulty_intf = "eth0"

    def inject_fault(self, params: LinkFlapParams):
        host = params.host_name
        self.faulty_devices = [host]
        self.faulty_intf = params.intf_name
        intf_name = params.intf_name
        down_time = params.down_time
        up_time = params.up_time
        if down_time <= 0 or up_time <= 0:
            raise ValueError("down_time and up_time must be positive integers")

        script_path = f"/tmp/link_flap_{intf_name}.sh"
        pid_path = f"/tmp/link_flap_{intf_name}.pid"
        log_path = f"/tmp/link_flap_{intf_name}.log"

        script = f"""#!/bin/bash
IFACE={shlex.quote(intf_name)}
DOWN_TIME={int(down_time)}
UP_TIME={int(up_time)}
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
        self.runtime.exec(host, write_cmd)

        stop_previous_cmd = (
            f"if [ -f {shlex.quote(pid_path)} ]; then "
            f"kill $(cat {shlex.quote(pid_path)}) 2>/dev/null || true; "
            f"rm -f {shlex.quote(pid_path)}; "
            "fi"
        )
        self.runtime.exec(host, stop_previous_cmd)

        start_cmd = f"nohup {shlex.quote(script_path)} > {shlex.quote(log_path)} 2>&1 < /dev/null &"
        self.runtime.exec(host, start_cmd)
        system_logger.info(f"Injected link flap on {host}:{intf_name} (down_time={down_time}, up_time={up_time})")

    def verify_fault(self, params: LinkFlapParams) -> dict:
        """Verify the link-flap script is running by checking the pid file and process liveness."""
        host = params.host_name
        intf = params.intf_name
        pid_path = shlex.quote(f"/tmp/link_flap_{intf}.pid")
        check_cmd = (
            f"if [ -f {pid_path} ] && kill -0 $(cat {pid_path}) 2>/dev/null; "
            f"then echo running; else echo not_running; fi"
        )
        output = self.runtime.exec(host, check_cmd).strip()
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=output == "running",
            details={"host": host, "intf": intf, "flap_process": output},
        )


class LinkFlapDetection(LinkFlapBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=LinkFlapBase.root_cause_category,
        root_cause_name=LinkFlapBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class LinkFlapLocalization(LinkFlapBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=LinkFlapBase.root_cause_category,
        root_cause_name=LinkFlapBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class LinkFlapRCA(LinkFlapBase, RCATask):
    META = ProblemMeta(
        root_cause_category=LinkFlapBase.root_cause_category,
        root_cause_name=LinkFlapBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==========================================
# Problem: Link detached.
# ==========================================


class LinkDetachParams(BaseModel):
    """Parameters for injecting a link-detach fault."""

    host_name: str = Field(description="Target host name.")
    intf_name: str = Field(default="eth0", description="Target interface name.")


class LinkDetachBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.LINK_FAILURE
    root_cause_name: str = "link_detach"
    TAGS: str = ["link"]

    Params = LinkDetachParams

    symptom_desc = "Users report connectivity issues to other hosts."

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env, self.runtime = init_problem(scenario_name, **kwargs)
        self.faulty_devices: list[str] = []
        self.faulty_intf = "eth0"

    def inject_fault(self, params: LinkDetachParams):
        host = params.host_name
        self.faulty_devices = [host]
        self.faulty_intf = params.intf_name
        intf_name = params.intf_name
        self.runtime.exec(host, f"ip link del {intf_name}")
        system_logger.info(f"Injected link detach on {host}:{intf_name}")

    def verify_fault(self, params: LinkDetachParams) -> dict:
        """Verify the link-detach fault is active by confirming the interface no longer exists in the container."""
        host = params.host_name
        intf = params.intf_name
        output = self.runtime.exec(host, f"ip link show {shlex.quote(intf)} 2>&1")
        detached = "does not exist" in output.lower() or "no such device" in output.lower()
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=detached,
            details={"host": host, "intf": intf, "ip_link_output": output.strip()},
        )


class LinkDetachDetection(LinkDetachBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=LinkDetachBase.root_cause_category,
        root_cause_name=LinkDetachBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class LinkDetachLocalization(LinkDetachBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=LinkDetachBase.root_cause_category,
        root_cause_name=LinkDetachBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class LinkDetachRCA(LinkDetachBase, RCATask):
    META = ProblemMeta(
        root_cause_category=LinkDetachBase.root_cause_category,
        root_cause_name=LinkDetachBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )


# ==========================================
# Problem: Link fragmentation disabled, drop large packets
# ==========================================


class LinkFragParams(BaseModel):
    """Parameters for injecting a link-fragmentation-disabled fault."""

    host_name: str = Field(description="Target host name.")
    mtu: int = Field(default=10, description="Packet size threshold.")


class LinkFragBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.LINK_FAILURE
    root_cause_name: str = "link_fragmentation_disabled"
    TAGS: str = ["link"]

    Params = LinkFragParams

    symptom_desc = "Users report partial packet loss when communicating with other hosts."

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env, self.runtime = init_problem(scenario_name, **kwargs)
        self.faulty_devices: list[str] = []
        self.mtu = 10

    @staticmethod
    def _frag_drop_rule_args(mtu: int) -> str:
        return f"-m length --length {int(mtu)}:65535 -j DROP"

    def inject_fault(self, params: LinkFragParams):
        host = params.host_name
        self.faulty_devices = [host]
        mtu = params.mtu
        self.mtu = mtu
        self.runtime.exec(
            host,
            f"iptables -A OUTPUT {self._frag_drop_rule_args(mtu)}",
        )
        system_logger.info(f"Injected fragmentation disabled on {host} with MTU {mtu}")

    def verify_fault(self, params: LinkFragParams) -> dict:
        """Verify the fragmentation-disabled fault is active via the exact length-based DROP rule."""
        host = params.host_name
        mtu = params.mtu
        rule_args = self._frag_drop_rule_args(mtu)
        check_cmd = f"iptables -C OUTPUT {rule_args} >/dev/null 2>&1 && echo present || echo absent"
        check_output = self.runtime.exec(host, check_cmd).strip()
        iptables_output = self.runtime.exec(host, "iptables -S OUTPUT").strip()
        expected_rule = f"-A OUTPUT {rule_args}"
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=check_output == "present",
            details={"host": host, "mtu": mtu, "rule": expected_rule, "iptables_output": iptables_output},
        )


class LinkFragDetection(LinkFragBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=LinkFragBase.root_cause_category,
        root_cause_name=LinkFragBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class LinkFragLocalization(LinkFragBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=LinkFragBase.root_cause_category,
        root_cause_name=LinkFragBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class LinkFragRCA(LinkFragBase, RCATask):
    META = ProblemMeta(
        root_cause_category=LinkFragBase.root_cause_category,
        root_cause_name=LinkFragBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )
