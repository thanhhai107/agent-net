"""Shared helpers for deterministic failure injection parameter resolution."""

from __future__ import annotations

import ipaddress

from nika.runtime.base import LabRuntime


def derive_incorrect_ip(runtime: LabRuntime, host: str, intf: str = "eth0") -> str:
    """Return a wrong CIDR by incrementing the host part of the current address."""
    current = runtime.get_host_ip(host, intf, with_prefix=True)
    if not current:
        raise ValueError(f"Cannot derive incorrect IP: no address on {host}:{intf}")
    network = ipaddress.ip_interface(current)
    prefix = network.network.prefixlen
    host_int = int(network.ip)
    for offset in (1, 2, 3, -1, -2):
        candidate = ipaddress.ip_address(host_int + offset)
        if candidate.is_multicast or candidate.is_reserved or candidate.is_loopback:
            continue
        return f"{candidate}/{prefix}"
    raise ValueError(f"Cannot derive incorrect IP for {host}:{intf}")


def derive_wrong_gateway(runtime: LabRuntime, host: str) -> str:
    """Return a wrong gateway by changing the last octet to 254."""
    gateway = runtime.get_default_gateway(host)
    if not gateway:
        raise ValueError(f"Cannot derive wrong gateway: no default route on {host}")
    parts = gateway.split(".")
    parts[-1] = "254"
    return ".".join(parts)


def resolve_victim_host(runtime: LabRuntime, router: str) -> str:
    """Pick the first sorted non-router/switch host connected to a router."""
    connected = runtime.get_connected_devices(router)
    hosts = sorted(
        dev
        for dev in connected
        if "switch" not in dev and "router" not in dev
    )
    if not hosts:
        raise ValueError(f"No victim host found for router {router}")
    return hosts[0]


def resolve_intf(runtime: LabRuntime, host: str, intf_name: str = "eth0") -> str:
    """Validate that ``intf_name`` exists on ``host``."""
    interfaces = runtime.get_host_interfaces(host)
    if intf_name not in interfaces:
        raise ValueError(f"Interface {intf_name!r} not found on {host}; available: {interfaces}")
    return intf_name
