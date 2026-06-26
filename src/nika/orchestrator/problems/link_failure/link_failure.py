import random
import shlex
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

# ==================================================================
# Problem: Link failure by ip link down on host interface
# ==================================================================


class LinkFailureParams(BaseModel):
    """Parameters for injecting a link-down fault."""

    host_name: Optional[str] = Field(
        default=None,
        description="Target host name. Defaults to a randomly selected host when not set.",
    )
    intf_name: str = Field(default="eth0", description="Target interface name.")


class LinkFailureBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.LINK_FAILURE
    root_cause_name: str = "link_down"
    TAGS: str = ["link"]

    Params = LinkFailureParams

    symptom_desc = "Users report connectivity issues to other hosts."

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaBaseAPI(lab_name=self.net_env.lab.name)
        self.injector = FaultInjectorBase(lab_name=self.net_env.lab.name)
        self.faulty_devices = [random.choice(self.net_env.hosts)]
        self.faulty_intf = "eth0"
        self.down_time = 1
        self.up_time = 1

    def inject_fault(self, params: LinkFailureParams | None = None):
        if params is None:
            params = LinkFailureParams()
        host = params.host_name if params.host_name is not None else self.faulty_devices[0]
        self.injector.inject_intf_down(
            host_name=host,
            intf_name=params.intf_name,
        )

    def verify_fault(self, params: LinkFailureParams | None = None) -> dict:
        """Verify the link-down fault is active by reading the interface operstate from the container."""
        if params is None:
            params = LinkFailureParams()
        host = params.host_name if params.host_name is not None else self.faulty_devices[0]
        intf = params.intf_name
        output = self.kathara_api.exec_cmd(host, f"cat /sys/class/net/{shlex.quote(intf)}/operstate")
        operstate = output.strip().lower()
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

    host_name: Optional[str] = Field(
        default=None,
        description="Target host name. Defaults to a randomly selected host when not set.",
    )
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
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaBaseAPI(lab_name=self.net_env.lab.name)
        self.faulty_devices = [random.choice(self.net_env.hosts)]
        self.faulty_intf = "eth0"

    def inject_fault(self, params: LinkFlapParams | None = None):
        if params is None:
            params = LinkFlapParams()
        host = params.host_name if params.host_name is not None else self.faulty_devices[0]
        intf_name = params.intf_name
        down_time = params.down_time
        up_time = params.up_time
        if down_time <= 0 or up_time <= 0:
            raise ValueError("down_time and up_time must be positive integers")

        script_path = f"/tmp/link_flap_{intf_name}.sh"
        pid_path = f"/tmp/link_flap_{intf_name}.pid"
        log_path = f"/tmp/link_flap_{intf_name}.log"

        # Avoid double quotes in the script body: exec_cmd escapes " when wrapping
        # commands for bash -c, which would corrupt a heredoc-written script.
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
        self.kathara_api.exec_cmd(host, write_cmd)

        stop_previous_cmd = (
            f"if [ -f {shlex.quote(pid_path)} ]; then "
            f"kill $(cat {shlex.quote(pid_path)}) 2>/dev/null || true; "
            f"rm -f {shlex.quote(pid_path)}; "
            "fi"
        )
        self.kathara_api.exec_cmd(host, stop_previous_cmd)

        start_cmd = f"nohup {shlex.quote(script_path)} > {shlex.quote(log_path)} 2>&1 < /dev/null &"
        self.kathara_api.exec_cmd(host, start_cmd)
        system_logger.info(f"Injected link flap on {host}:{intf_name} (down_time={down_time}, up_time={up_time})")

    def verify_fault(self, params: LinkFlapParams | None = None) -> dict:
        """Verify the link-flap script is running by checking the pid file and process liveness."""
        if params is None:
            params = LinkFlapParams()
        host = params.host_name if params.host_name is not None else self.faulty_devices[0]
        intf = params.intf_name
        pid_path = shlex.quote(f"/tmp/link_flap_{intf}.pid")
        check_cmd = (
            f"if [ -f {pid_path} ] && kill -0 $(cat {pid_path}) 2>/dev/null; "
            f"then echo running; else echo not_running; fi"
        )
        output = self.kathara_api.exec_cmd(host, check_cmd).strip()
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

    host_name: Optional[str] = Field(
        default=None,
        description="Target host name. Defaults to a randomly selected host when not set.",
    )
    intf_name: str = Field(default="eth0", description="Target interface name.")


class LinkDetachBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.LINK_FAILURE
    root_cause_name: str = "link_detach"
    TAGS: str = ["link"]

    Params = LinkDetachParams

    symptom_desc = "Users report connectivity issues to other hosts."

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaBaseAPI(lab_name=self.net_env.lab.name)
        self.faulty_devices = [random.choice(self.net_env.hosts)]
        self.faulty_intf = "eth0"

    def inject_fault(self, params: LinkDetachParams | None = None):
        if params is None:
            params = LinkDetachParams()
        host = params.host_name if params.host_name is not None else self.faulty_devices[0]
        intf_name = params.intf_name
        self.kathara_api.exec_cmd(host, f"ip link del {intf_name}")
        system_logger.info(f"Injected link detach on {host}:{intf_name}")

    def verify_fault(self, params: LinkDetachParams | None = None) -> dict:
        """Verify the link-detach fault is active by confirming the interface no longer exists in the container."""
        if params is None:
            params = LinkDetachParams()
        host = params.host_name if params.host_name is not None else self.faulty_devices[0]
        intf = params.intf_name
        output = self.kathara_api.exec_cmd(host, f"ip link show {shlex.quote(intf)} 2>&1")
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

    host_name: Optional[str] = Field(
        default=None,
        description="Target host name. Defaults to a randomly selected host when not set.",
    )
    mtu: int = Field(default=10, description="Packet size threshold.")


class LinkFragBase:
    root_cause_category: RootCauseCategory = RootCauseCategory.LINK_FAILURE
    root_cause_name: str = "link_fragmentation_disabled"
    TAGS: str = ["link"]

    Params = LinkFragParams

    symptom_desc = "Users report partial packet loss when communicating with other hosts."

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.kathara_api = KatharaBaseAPI(lab_name=self.net_env.lab.name)
        self.faulty_devices = [random.choice(self.net_env.hosts)]
        self.mtu = 10

    @staticmethod
    def _frag_drop_rule_args(mtu: int) -> str:
        return f"-m length --length {int(mtu)}:65535 -j DROP"

    def inject_fault(self, params: LinkFragParams | None = None):
        if params is None:
            params = LinkFragParams()
        host = params.host_name if params.host_name is not None else self.faulty_devices[0]
        mtu = params.mtu
        self.kathara_api.exec_cmd(
            host,
            f"iptables -A OUTPUT {self._frag_drop_rule_args(mtu)}",
        )
        system_logger.info(f"Injected fragmentation disabled on {host} with MTU {mtu}")

    def verify_fault(self, params: LinkFragParams | None = None) -> dict:
        """Verify the fragmentation-disabled fault is active via the exact length-based DROP rule."""
        if params is None:
            params = LinkFragParams()
        host = params.host_name if params.host_name is not None else self.faulty_devices[0]
        mtu = params.mtu
        rule_args = self._frag_drop_rule_args(mtu)
        check_cmd = f"iptables -C OUTPUT {rule_args} >/dev/null 2>&1 && echo present || echo absent"
        check_output = self.kathara_api.exec_cmd(host, check_cmd).strip()
        iptables_output = self.kathara_api.exec_cmd(host, "iptables -S OUTPUT").strip()
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
