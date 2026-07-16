"""Kathara-backed LabRuntime implementation."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import nika.runtime.kathara.patch  # noqa: F401
from Kathara.manager.Kathara import Kathara

from nika.runtime.base import LabRuntime
from nika.runtime.kathara.cleanup import undeploy_kathara_lab
from nika.runtime.docker_ops import pause_container, unpause_container
from nika.runtime.exec_utils import exec_with_timeout
from nika.service.shell import ShellResolver
from nika.service.kathara.docker_utils import get_machine_container, list_lab_containers

if TYPE_CHECKING:
    from docker.models.containers import Container

    from nika.net_env.base import NetworkEnvBase


class KatharaRuntime(LabRuntime):
    """Wrap existing Kathara deploy/exec behavior without changing semantics."""

    def __init__(self, net_env: NetworkEnvBase) -> None:
        self._net_env = net_env
        self._instance = net_env.instance or Kathara.get_instance()
        self._shell = ShellResolver()

    @property
    def backend(self) -> str:
        return "kathara"

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
                if (
                    not item
                    or item == b""
                    or isinstance(item, int)
                    or item is None
                    or item == "None"
                ):
                    continue
                if isinstance(item, bytes):
                    chunks.append(item.decode("utf-8", errors="ignore"))
                elif isinstance(item, str):
                    chunks.append(item)
                else:
                    chunks.append(str(item))
            return "".join(chunks).strip()

        return exec_with_timeout(_run, timeout=timeout, node=node, cmd=cmd)

    def _preferred_shell(self, node: str) -> str | None:
        lab = self._net_env.lab
        if lab is None:
            return None
        machine = lab.machines.get(node)
        if machine is None or "shell" not in machine.meta:
            return None
        return machine.get_shell()

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
        undeploy_kathara_lab(self._instance, lab_name=self.lab_name)

    def exists(self) -> bool:
        machines = self._instance.get_machines_api_objects(lab_name=self.lab_name)
        links = self._instance.get_links_api_objects(lab_name=self.lab_name)
        return bool(machines or links)

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
        return self._shell.exec_via_shell(
            node,
            cmd,
            self._exec_raw,
            preferred_shell=self._preferred_shell(node),
            timeout=timeout,
        )

    def get_container(self, node: str) -> Container:
        return get_machine_container(lab_name=self.lab_name, host_name=node)

    def pause(self, node: str) -> None:
        pause_container(self.get_container(node))

    def unpause(self, node: str) -> None:
        unpause_container(self.get_container(node))

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
