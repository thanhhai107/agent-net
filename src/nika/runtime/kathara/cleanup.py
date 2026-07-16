"""Verified cleanup through Kathara's public lifecycle API."""

from __future__ import annotations

import time
from collections.abc import Callable

from docker.errors import APIError
from Kathara.manager.Kathara import Kathara

from nika.runtime.base import LabCleanupError

_CLEANUP_ATTEMPTS = 6
_CLEANUP_DELAY_SECONDS = 1.0


def _active_endpoint_error(error: Exception) -> bool:
    return (
        isinstance(error, APIError)
        and error.status_code in {403, 409}
        and "active endpoints" in str(error).lower()
    )


def _verified_cleanup(
    *,
    operation: Callable[[], None],
    remaining: Callable[[], tuple[int, int]],
    scope: str,
) -> None:
    last_error: Exception | None = None
    machine_count = link_count = 0

    for attempt in range(_CLEANUP_ATTEMPTS):
        try:
            operation()
        except Exception as exc:
            if not _active_endpoint_error(exc):
                raise LabCleanupError(
                    f"Kathara cleanup failed for {scope}: {exc}"
                ) from exc
            last_error = exc

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
