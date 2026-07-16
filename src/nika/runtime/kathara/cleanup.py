"""Verified cleanup through Kathara's public lifecycle API."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable

from docker.errors import APIError, NotFound
from Kathara import utils as kathara_utils
from Kathara.manager.Kathara import Kathara

from nika.runtime.base import LabCleanupError

_CLEANUP_ATTEMPTS = 6
_CLEANUP_DELAY_SECONDS = 1.0
_ACTIVE_ENDPOINT_RECOVERY_THRESHOLD = 2
_NETWORK_ID_PATTERN = re.compile(r"/networks/([0-9a-f]{12,64})", re.IGNORECASE)
_ENDPOINT_NAME_PATTERN = re.compile(
    r'active endpoints.*?\(name:"([^"]+)"',
    re.IGNORECASE | re.DOTALL,
)


def _active_endpoint_error(error: Exception) -> bool:
    return (
        isinstance(error, APIError)
        and error.status_code in {403, 409}
        and "active endpoints" in str(error).lower()
    )


def _network_labels(network: object) -> dict[str, str]:
    attrs = getattr(network, "attrs", None)
    if not isinstance(attrs, dict):
        return {}
    labels = attrs.get("Labels") or attrs.get("labels")
    return labels if isinstance(labels, dict) else {}


def _network_belongs_to_lab(network: object, *, lab_hash: str) -> bool:
    labels = _network_labels(network)
    return labels.get("app") == "kathara" and labels.get("lab_hash") == lab_hash


def _endpoint_names(network: object, *, lab_hash: str) -> dict[str, str]:
    attrs = getattr(network, "attrs", None)
    if not isinstance(attrs, dict):
        return {}
    endpoints = attrs.get("Containers")
    if not isinstance(endpoints, dict):
        return {}

    scoped: dict[str, str] = {}
    for container_id, endpoint in endpoints.items():
        if not isinstance(endpoint, dict):
            continue
        name = str(endpoint.get("Name") or "")
        if lab_hash in name:
            scoped[str(container_id)] = name
    return scoped


def _recover_lab_active_endpoints(
    instance: Kathara,
    *,
    lab_name: str,
    error: Exception,
) -> int:
    """Force-disconnect stale endpoints, scoped to one Kathara lab."""

    lab_hash = kathara_utils.generate_urlsafe_hash(lab_name)
    error_text = str(error)
    network_match = _NETWORK_ID_PATTERN.search(error_text)
    endpoint_match = _ENDPOINT_NAME_PATTERN.search(error_text)
    network_id = network_match.group(1) if network_match else None
    reported_endpoint = endpoint_match.group(1) if endpoint_match else None

    networks = list(instance.get_links_api_objects(lab_name=lab_name))
    manager = getattr(instance, "manager", None)
    client = getattr(manager, "client", None)
    if network_id and client is not None:
        try:
            reported_network = client.networks.get(network_id)
        except NotFound:
            reported_network = None
        if reported_network is not None and not any(
            getattr(network, "id", None) == getattr(reported_network, "id", None)
            for network in networks
        ):
            networks.append(reported_network)

    recovered = 0
    for network in networks:
        try:
            network.reload()
        except NotFound:
            continue
        if not _network_belongs_to_lab(network, lab_hash=lab_hash):
            continue

        endpoints = _endpoint_names(network, lab_hash=lab_hash)
        if (
            network_id
            and reported_endpoint
            and lab_hash in reported_endpoint
            and str(getattr(network, "id", "")).startswith(network_id)
        ):
            endpoints.setdefault(reported_endpoint, reported_endpoint)

        for endpoint_ref, endpoint_name in endpoints.items():
            try:
                network.disconnect(endpoint_ref, force=True)
            except NotFound:
                continue
            recovered += 1
            logging.warning(
                "Force-disconnected stale Kathara endpoint %s from network %s "
                "while cleaning lab %s",
                endpoint_name,
                getattr(network, "name", getattr(network, "id", "unknown")),
                lab_name,
            )
    return recovered


def _verified_cleanup(
    *,
    operation: Callable[[], None],
    remaining: Callable[[], tuple[int, int]],
    scope: str,
    recover: Callable[[Exception], int] | None = None,
) -> None:
    last_error: Exception | None = None
    last_recovery_error: Exception | None = None
    machine_count = link_count = 0
    active_endpoint_failures = 0

    for attempt in range(_CLEANUP_ATTEMPTS):
        try:
            operation()
        except Exception as exc:
            if not _active_endpoint_error(exc):
                raise LabCleanupError(
                    f"Kathara cleanup failed for {scope}: {exc}"
                ) from exc
            last_error = exc
            active_endpoint_failures += 1
            if (
                recover is not None
                and active_endpoint_failures >= _ACTIVE_ENDPOINT_RECOVERY_THRESHOLD
            ):
                try:
                    recovered = recover(exc)
                    if recovered:
                        logging.warning(
                            "Recovered %d stale Docker endpoint(s) during Kathara "
                            "cleanup for %s",
                            recovered,
                            scope,
                        )
                except Exception as recovery_error:
                    last_recovery_error = recovery_error

        try:
            machine_count, link_count = remaining()
        except Exception as exc:
            raise LabCleanupError(
                f"Kathara cleanup could not be verified for {scope}: {exc}"
            ) from exc

        if machine_count == 0 and link_count == 0:
            return
        if attempt < _CLEANUP_ATTEMPTS - 1:
            time.sleep(_CLEANUP_DELAY_SECONDS)

    detail = f"; last error: {last_error}" if last_error is not None else ""
    if last_recovery_error is not None:
        detail += f"; last endpoint recovery error: {last_recovery_error}"
    raise LabCleanupError(
        f"Kathara cleanup was not complete for {scope} after "
        f"{_CLEANUP_ATTEMPTS} attempts "
        f"(machines={machine_count}, links={link_count}){detail}"
    )


def undeploy_kathara_lab(instance: Kathara, *, lab_name: str) -> None:
    """Undeploy one lab and prove that both its devices and links are gone."""

    def remaining() -> tuple[int, int]:
        machines = instance.get_machines_api_objects(lab_name=lab_name)
        links = instance.get_links_api_objects(lab_name=lab_name)
        return len(machines), len(links)

    _verified_cleanup(
        operation=lambda: instance.undeploy_lab(lab_name=lab_name),
        remaining=remaining,
        scope=f"lab {lab_name!r}",
        recover=lambda error: _recover_lab_active_endpoints(
            instance,
            lab_name=lab_name,
            error=error,
        ),
    )


def wipe_kathara_user_labs(instance: Kathara) -> None:
    """Wipe current-user labs and prove no Kathara resources remain."""

    def remaining() -> tuple[int, int]:
        machines = instance.get_machines_api_objects(all_users=False)
        links = instance.get_links_api_objects(all_users=False)
        return len(machines), len(links)

    _verified_cleanup(
        operation=lambda: instance.wipe(all_users=False),
        remaining=remaining,
        scope="current user",
    )
