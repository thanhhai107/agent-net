"""Shared helpers for post-deploy net_env verification."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nika.net_env.base import NetworkEnvBase

LAB_VERIFY_MAX_WAIT_SEC = 180
LAB_VERIFY_RETRY_DELAY_SEC = 5


def build_lab_verify_result(
    *,
    scenario_name: str,
    verified: bool,
    checks: dict[str, bool],
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "verified": verified,
        "scenario_name": scenario_name,
        "checks": dict(checks),
        "details": details or {},
    }


def verify_lab_with_retry(net_env: NetworkEnvBase) -> dict[str, Any] | None:
    """Poll ``net_env.verify_lab()`` until success or timeout.

    Returns ``None`` when the scenario defines no startup verification.
    """
    result = net_env.verify_lab()
    if result is None:
        return None

    deadline = time.time() + LAB_VERIFY_MAX_WAIT_SEC
    last_result = result
    while time.time() < deadline:
        last_result = net_env.verify_lab()
        if last_result.get("verified", False):
            return last_result
        time.sleep(LAB_VERIFY_RETRY_DELAY_SEC)

    failed_checks = {
        name: ok for name, ok in (last_result.get("checks") or {}).items() if not ok
    }
    raise RuntimeError(
        f"Lab verification failed for {net_env.name!r} "
        f"within {LAB_VERIFY_MAX_WAIT_SEC}s; failed checks: {failed_checks or last_result}"
    )
