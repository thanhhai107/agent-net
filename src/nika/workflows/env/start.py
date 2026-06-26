"""Start a Kathara lab for one scenario and persist a new session."""

from datetime import datetime
from typing import Literal
from uuid import uuid4

from nika.net_env.net_env_pool import get_net_env_instance, scenario_requires_topo_size
from nika.utils.logger import bind_session_dir, log_event, refresh_logger
from nika.utils.session import Session


def _normalize_topo_size(raw: str | None) -> Literal["s", "m", "l"] | None:
    """Return ``None`` for missing/blank input; otherwise validate ``s``/``m``/``l``."""
    if raw is None or raw == "":
        return None
    if raw not in ("s", "m", "l"):
        raise ValueError("Topology size must be one of: s, m, l.")
    return raw  # type: ignore[return-value]


def start_net_env(
    scenario: str,
    topo_size: str | None,
    *,
    redeploy: bool = True,
    instance_tag: str | None = None,
) -> str:
    """Deploy the lab for ``scenario`` and create a new runtime session."""
    size = _normalize_topo_size(topo_size)
    if scenario_requires_topo_size(scenario) and size is None:
        raise ValueError(f"Scenario '{scenario}' requires an explicit topology size (-s s|m|l).")
    if not scenario_requires_topo_size(scenario) and size is not None:
        raise ValueError(f"Scenario '{scenario}' does not use topology sizes; omit -s/--size.")

    refresh_logger()
    suffix = uuid4().hex[:6]
    tag = f"{instance_tag}-{suffix}" if instance_tag else f"{datetime.now().strftime('%m%d%H%M%S')}-{suffix}"
    lab_name = f"{scenario}__{tag}"
    net_env = get_net_env_instance(scenario, topo_size=size, lab_name=lab_name)
    if net_env.lab_exists() and redeploy:
        net_env.undeploy()
        net_env.deploy()
    elif not net_env.lab_exists():
        net_env.deploy()

    session_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{suffix}"

    session = Session()
    scenario_params: dict = {"lab_name": net_env.lab.name}
    if size is not None:
        scenario_params["topo_size"] = size
    session.init_session(
        session_id=session_id,
        scenario_name=scenario,
        lab_name=net_env.lab.name,
        scenario_topo_size=size,
        scenario_params=scenario_params,
    )
    bind_session_dir(session.session_dir)
    log_event(
        "env_start",
        f"Started network environment: {scenario} (size={size}) — session {session_id}, lab {net_env.lab.name}",
        scenario=scenario,
        topo_size=size,
        session_id=session_id,
        lab_name=net_env.lab.name,
    )
    return session_id
