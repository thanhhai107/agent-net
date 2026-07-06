"""Kathara-backed LabRuntime implementation."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import docker
from func_timeout import FunctionTimedOut, func_timeout

from Kathara.manager.Kathara import Kathara

from nika.runtime.base import LabRuntime
from nika.service.kathara.docker_utils import get_machine_container, list_lab_containers

if TYPE_CHECKING:
    from docker.models.containers import Container

    from nika.net_env.base import NetworkEnvBase


class KatharaRuntime(LabRuntime):
    """Wrap existing Kathara deploy/exec behavior without changing semantics."""

    def __init__(self, net_env: NetworkEnvBase) -> None:
        self._net_env = net_env
        self._instance = net_env.instance or Kathara.get_instance()
        self._resolved_shell_cache: dict[str, str] = {}

    @staticmethod
    def _escape_for_shell_c(command: str) -> str:
        return command.replace("'", "'\\''").replace('"', '\\"')

    def _wrap_shell_command(self, shell: str, command: str) -> str:
        escaped = self._escape_for_shell_c(command)
        return f"{shell} -c '{escaped}'"

    def _exec_raw(self, node: str, cmd: str, *, timeout: float = 10.0) -> str:
        def _run() -> str:
            output_generator = self._instance.exec(
                machine_name=node,
                lab_name=self.lab_name,
                command=cmd,
                stream=False,
            )
            chunks: list[str] = []
            for item in output_generator:
                if not item or item == b"" or isinstance(item, int) or item is None or item == "None":
                    continue
                if isinstance(item, bytes):
                    chunks.append(item.decode("utf-8", errors="ignore"))
                elif isinstance(item, str):
                    chunks.append(item)
                else:
                    chunks.append(str(item))
            return "".join(chunks).strip()

        try:
            return func_timeout(timeout, _run)
        except FunctionTimedOut:
            return f"[TIMEOUT] Command '{cmd}' on '{node}' exceeded {timeout}s."

    def _resolve_shell(self, node: str) -> str:
        cached = self._resolved_shell_cache.get(node)
        if cached is not None:
            return cached
        lab = self._net_env.lab
        if lab is not None:
            machine = lab.machines.get(node)
            if machine is not None and "shell" in machine.meta:
                shell = machine.get_shell()
                self._resolved_shell_cache[node] = shell
                return shell
        probe_cmd = (
            "/bin/sh -c 'if [ -x /bin/bash ]; then echo /bin/bash; "
            "elif [ -x /bin/sh ]; then echo /bin/sh; else echo /bin/sh; fi'"
        )
        probed = self._exec_raw(node, probe_cmd).strip()
        shell = probed if probed in ("/bin/bash", "/bin/sh") else "/bin/sh"
        self._resolved_shell_cache[node] = shell
        return shell

    @property
    def lab_name(self) -> str:
        return self._net_env.name or self._net_env.lab.name

    def deploy(self) -> None:
        if self.exists():
            print(f"Lab {self.lab_name} exists")
            return
        self._net_env._ensure_docker_images()
        Kathara.get_instance().deploy_lab(lab=self._net_env.lab)
        time.sleep(5)

    def destroy(self) -> None:
        try:
            self._instance.undeploy_lab(lab_name=self.lab_name)
        except Exception as exc:
            print(f"Error undeploying lab {self.lab_name}: {exc}")

    def exists(self) -> bool:
        tmp_lab = self._instance.get_lab_from_api(lab_name=self.lab_name)
        if tmp_lab is None:
            return False
        tmp_machines = tmp_lab.machines
        if tmp_machines is None or len(tmp_machines) == 0:
            return False
        return True

    def inspect(self) -> list[dict[str, Any]]:
        return list_lab_containers(lab_name=self.lab_name)

    def list_nodes(self) -> list[str]:
        if self._net_env.lab and self._net_env.lab.machines:
            return sorted(self._net_env.lab.machines.keys())
        tmp_lab = self._instance.get_lab_from_api(lab_name=self.lab_name)
        if tmp_lab is None:
            return []
        return sorted(tmp_lab.machines.keys())

    def exec(self, node: str, cmd: str, *, timeout: float = 10.0) -> str:
        shell = self._resolve_shell(node)
        wrapped = self._wrap_shell_command(shell, cmd)
        return self._exec_raw(node, wrapped, timeout=timeout)

    def get_container(self, node: str) -> Container:
        return get_machine_container(lab_name=self.lab_name, host_name=node)

    def pause(self, node: str) -> None:
        container = self.get_container(node)
        container.reload()
        if container.status != "paused":
            container.pause()

    def unpause(self, node: str) -> None:
        container = self.get_container(node)
        container.reload()
        if container.status == "paused":
            container.unpause()
        elif container.status in {"created", "exited"}:
            container.start()

    def get_connected_devices(self, node: str) -> list[str]:
        links = next(self._instance.get_links_stats(lab_name=self.lab_name))
        results: list[str] = []
        for link in links.values():
            if not link.name:
                continue
            left = link.containers[0].labels["name"]
            right = link.containers[1].labels["name"]
            if node == left:
                results.append(right)
            elif node == right:
                results.append(left)
        return results

    def list_dhcp_client_nodes(self) -> list[str]:
        """Return lab nodes that typically receive DHCP leases."""
        nodes: list[str] = []
        if not self._net_env.lab or not self._net_env.lab.machines:
            return self.list_nodes()
        for name, machine in self._net_env.lab.machines.items():
            image = machine.get_image()
            if "base" in image and any(key in name for key in ("pc", "client")):
                nodes.append(name)
        return nodes
