"""Start a network lab for one scenario and persist a new session."""

from datetime import datetime
from typing import Literal
from uuid import uuid4

from nika.net_env.net_env_pool import (
    get_net_env_instance,
    scenario_backend,
    scenario_requires_topo_size,
)
from nika.runtime.base import LabCleanupError
from nika.net_env.verify import verify_lab_with_retry
from nika.utils.logger import (
    bind_session_dir,
    log_error_event,
    log_event,
    refresh_logger,
)
from nika.utils.session import Session
from nika.utils.session_id import make_session_id


def _normalize_topo_size(raw: str | None) -> Literal["s", "m", "l"] | None:
    """Return ``None`` for missing/blank input; otherwise validate ``s``/``m``/``l``."""
    if raw is None or raw == "":
        return None
    if raw not in ("s", "m", "l"):
        raise ValueError("Topology size must be one of: s, m, l.")
    return raw  # type: ignore[return-value]


def _record_failed_start(
    session: Session,
    net_env,
    *,
    error: BaseException,
) -> Exception | None:
    """Best-effort rollback for a session whose lab did not start cleanly."""
    try:
        net_env.undeploy()
    except Exception as cleanup_error:
        try:
            session.update_session("startup_error", f"{type(error).__name__}: {error}")
            session.update_session(
                "cleanup_error",
                f"{type(cleanup_error).__name__}: {cleanup_error}",
            )
        except Exception:
            pass
        log_error_event(
            "env_start_cleanup_failed",
            f"Failed to clean partially started lab {net_env.name}: {cleanup_error}",
            session_id=getattr(session, "session_id", None),
            lab_name=net_env.name,
            error=str(cleanup_error),
            error_type=type(cleanup_error).__name__,
        )
        return cleanup_error

    session.update_session("startup_error", f"{type(error).__name__}: {error}")
    session.clear_session(status="failed")
    log_event(
        "env_start_cleanup",
        f"Cleaned partially started lab {net_env.name}.",
        session_id=getattr(session, "session_id", None),
        lab_name=net_env.name,
    )
    return None


def start_net_env(
    scenario: str,
    topo_size: str | None,
    *,
    redeploy: bool = True,
    instance_tag: str | None = None,
    session_tag: str | None = None,
    result_dir: str | None = None,
) -> str:
    """Deploy the lab for ``scenario`` and create a new runtime session."""
    size = _normalize_topo_size(topo_size)
    if scenario_requires_topo_size(scenario) and size is None:
        raise ValueError(
            f"Scenario '{scenario}' requires an explicit topology size (-s s|m|l)."
        )
    if not scenario_requires_topo_size(scenario) and size is not None:
        raise ValueError(
            f"Scenario '{scenario}' does not use topology sizes; omit -s/--size."
        )

    backend = scenario_backend(scenario)

    refresh_logger()
    suffix = uuid4().hex[:6]
    tag = (
        f"{instance_tag}-{suffix}"
        if instance_tag
        else f"{datetime.now().strftime('%m%d%H%M%S')}-{suffix}"
    )
    lab_name = f"{scenario}__{tag}"
    session_id = make_session_id(session_tag=session_tag, suffix=suffix)
    net_env = get_net_env_instance(
        scenario, backend=backend, topo_size=size, lab_name=lab_name
    )
    if backend == "containerlab":
        net_env._ensure_runtime_files()

    session = Session()
    scenario_params: dict = {"lab_name": net_env.name, "backend": backend}
    if size is not None:
        scenario_params["topo_size"] = size
    topology_file = getattr(net_env, "topology_file", None)
    runtime_workdir = getattr(net_env, "runtime_workdir", None)
    session.init_session(
        session_id=session_id,
        scenario_name=scenario,
        lab_name=net_env.name,
        scenario_topo_size=size,
        scenario_params=scenario_params,
        result_dir=result_dir,
        backend=backend,
        topology_file=topology_file,
        runtime_workdir=runtime_workdir,
    )
    bind_session_dir(session.session_dir)

    phase = "deploy"
    try:
        lab_exists = net_env.lab_exists()
        if lab_exists and redeploy:
            net_env.undeploy()
            net_env.deploy()
        elif not lab_exists:
            net_env.deploy()

        phase = "verify"
        verify_result = verify_lab_with_retry(net_env)
        if verify_result is not None:
            log_event(
                "env_verify",
                f"Lab verification passed for {scenario} ({net_env.name})",
                scenario=scenario,
                lab_name=net_env.name,
                checks=verify_result.get("checks"),
            )
    except BaseException as exc:
        event_type = "env_verify_failed" if phase == "verify" else "env_start_failed"
        log_error_event(
            event_type,
            f"Failed to start network environment: {scenario} ({session_id}): {exc}",
            scenario=scenario,
            backend=backend,
            topo_size=size,
            session_id=session_id,
            lab_name=net_env.name,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        cleanup_error = _record_failed_start(session, net_env, error=exc)
        if cleanup_error is not None:
            raise LabCleanupError(
                f"Failed to clean partially started lab {net_env.name!r}"
            ) from exc
        raise

    log_event(
        "env_start",
        f"Started network environment: {scenario} (backend={backend}, size={size}) — session {session_id}, lab {net_env.name}",
        scenario=scenario,
        backend=backend,
        topo_size=size,
        session_id=session_id,
        lab_name=net_env.name,
    )
    return session_id
