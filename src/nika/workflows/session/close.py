"""Close running sessions: undeploy lab, end failures, clear runtime state."""

import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from Kathara.manager.Kathara import Kathara

from nika.config import (
    RESULTS_DIR,
    RUNTIME_DIR,
    SESSIONS_DB,
    SESSIONS_DIR,
    resolve_results_root,
)
from nika.net_env.net_env_pool import get_net_env_instance
from nika.runtime.base import LabCleanupError
from nika.runtime.factory import resolve_backend, runtime_for_session
from nika.runtime.meta import meta_get, meta_path
from nika.utils.logger import bind_session_dir, log_error_event, log_event
from nika.utils.session import Session
from nika.utils.session_resolve import resolve_running_session_id
from nika.utils.session_store import SessionStore


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _resolve_runtime_workdir(session_meta: dict) -> Path | None:
    """Return the lab runtime workdir path from session metadata."""
    workdir = meta_path(session_meta, "runtime_workdir", scenario_params=True)
    if workdir is not None:
        return workdir
    topology_file = meta_path(session_meta, "topology_file", scenario_params=True)
    if topology_file is not None:
        return topology_file.parent
    lab_name = meta_get(session_meta, "lab_name", scenario_params=True)
    if lab_name:
        return Path(RUNTIME_DIR) / "containerlab" / str(lab_name)
    return None


def _is_safe_runtime_removal(path: Path, session_meta: dict) -> bool:
    """Only delete paths under ``runtime/`` and never under ``results/``."""
    resolved = path.resolve()
    runtime_root = Path(RUNTIME_DIR).resolve()
    if not resolved.is_relative_to(runtime_root):
        return False
    session_dir = session_meta.get("session_dir")
    if session_dir and resolved == Path(str(session_dir)).resolve():
        return False
    for results_root in {Path(RESULTS_DIR).resolve(), resolve_results_root()}:
        if resolved.is_relative_to(results_root):
            return False
    return True


def remove_session_runtime_workdir(session_meta: dict) -> bool:
    """Delete the session's runtime working directory when present."""
    path = _resolve_runtime_workdir(session_meta)
    if path is None or not path.exists():
        return False
    if not _is_safe_runtime_removal(path, session_meta):
        log_error_event(
            "runtime_workdir_skip",
            f"Refusing to remove unsafe runtime workdir path: {path}",
            session_id=session_meta.get("session_id"),
            path=str(path),
        )
        return False
    _remove_path(path)
    return True


def wipe_runtime_artifacts(
    *,
    runtime_dir: str | Path | None = None,
    sessions_dir: str | Path | None = None,
    db_path: str | Path | None = None,
) -> int:
    """Remove all runtime working files except session documents and the index."""
    runtime_root = Path(runtime_dir or RUNTIME_DIR)
    sessions_root = Path(sessions_dir or SESSIONS_DIR)
    db_file = Path(db_path or SESSIONS_DB).resolve()
    removed = 0
    if not runtime_root.exists():
        return removed

    sessions_resolved = sessions_root.resolve()
    for path in runtime_root.iterdir():
        resolved = path.resolve()
        if resolved == sessions_resolved or resolved == db_file:
            continue
        if path.name.startswith(f"{db_file.name}-"):
            continue
        _remove_path(path)
        removed += 1
    return removed


def clean_emulation_environment() -> None:
    """Remove all Kathara and Containerlab labs owned by this benchmark host."""
    errors: list[str] = []
    kathara_error: Exception | None = None
    for attempt in range(3):
        try:
            Kathara.get_instance().wipe(all_users=False)
            kathara_error = None
            break
        except Exception as exc:
            kathara_error = exc
            if attempt < 2:
                time.sleep(1)
    if kathara_error is not None:
        errors.append(f"Kathara cleanup failed after 3 attempts: {kathara_error}")

    if shutil.which("clab") is not None:
        result = subprocess.run(
            [
                "clab",
                "destroy",
                "--all",
                "--cleanup",
                "--yes",
                "--log-level",
                "error",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            errors.append(f"Containerlab cleanup failed: {detail or result.returncode}")

    if errors:
        raise LabCleanupError("; ".join(errors))


def _stop_session_record(
    session_meta: dict,
    *,
    undeploy: bool = True,
    final_status: str = "finished",
) -> None:
    session = Session()
    for key, value in session_meta.items():
        setattr(session, key, value)
    scenario = session.scenario_name
    if not scenario:
        raise ValueError(
            "Session has no scenario_name; cannot determine which lab to stop."
        )

    backend = resolve_backend(session_meta)
    net_env_kwargs: dict = {"backend": backend}
    if getattr(session, "scenario_topo_size", None) is not None:
        net_env_kwargs["topo_size"] = session.scenario_topo_size
    if getattr(session, "lab_name", None):
        net_env_kwargs["lab_name"] = session.lab_name
    if backend == "containerlab":
        topology_file = meta_path(session_meta, "topology_file", scenario_params=True)
        runtime_workdir = meta_path(
            session_meta, "runtime_workdir", scenario_params=True
        )
        if topology_file is not None:
            net_env_kwargs["topology_file"] = topology_file
        if runtime_workdir is not None:
            net_env_kwargs["runtime_workdir"] = runtime_workdir
    net_env = get_net_env_instance(scenario, **net_env_kwargs)
    if (
        backend == "containerlab"
        and net_env.runtime is None
        and meta_path(session_meta, "topology_file", scenario_params=True)
    ):
        net_env.runtime = runtime_for_session(session_meta)

    session_dir = session_meta.get("session_dir")
    if not session_dir:
        raise ValueError(f"Session '{session.session_id}' has no session_dir.")
    bind_session_dir(session_dir)

    try:
        lab_exists = net_env.lab_exists() if undeploy else False
    except Exception as exc:
        log_error_event(
            "env_stop_failed",
            f"Failed to inspect network environment before cleanup: {scenario} "
            f"({session.session_id}): {exc}",
            scenario=scenario,
            session_id=session.session_id,
            backend=backend,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        raise LabCleanupError(
            f"Failed to inspect lab {session.lab_name!r} before cleanup"
        ) from exc

    should_undeploy = undeploy and (lab_exists or backend == "containerlab")
    if should_undeploy:
        try:
            net_env.undeploy()
        except Exception as exc:
            log_error_event(
                "env_stop_failed",
                f"Failed to stop network environment: {scenario} ({session.session_id}): {exc}",
                scenario=scenario,
                session_id=session.session_id,
                backend=backend,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            if isinstance(exc, LabCleanupError):
                raise
            raise LabCleanupError(f"Failed to clean lab {session.lab_name!r}") from exc
        log_event(
            "env_stop",
            f"Stopped network environment: {scenario} ({session.session_id})",
            scenario=scenario,
            session_id=session.session_id,
            backend=backend,
        )
    elif undeploy:
        log_event(
            "env_stop_skipped",
            f"Network environment {scenario} ({session.session_id}) is not deployed.",
            scenario=scenario,
            session_id=session.session_id,
            backend=backend,
        )

    ended_cnt = SessionStore().mark_session_failures_ended(
        session.session_id, end_time=datetime.now().timestamp()
    )
    if ended_cnt:
        log_event(
            "failures_ended",
            f"Marked {ended_cnt} failure record(s) as ended for session {session.session_id}",
            session_id=session.session_id,
            count=ended_cnt,
        )

    if remove_session_runtime_workdir(session_meta):
        log_event(
            "runtime_workdir_removed",
            f"Removed runtime workdir for session {session.session_id}",
            session_id=session.session_id,
            backend=backend,
        )

    session.clear_session(status=final_status)
    log_event(
        "session_cleared",
        f"Cleared session {session.session_id} for scenario {scenario}",
        session_id=session.session_id,
        scenario=scenario,
    )


def close_session(
    session_id: str | None = None,
    *,
    undeploy: bool = True,
    stop_all: bool = False,
    final_status: str = "finished",
) -> None:
    """Close one or all running sessions and clear runtime state."""
    store = SessionStore()
    running = store.list_running_sessions()

    if stop_all:
        try:
            for session_meta in running:
                full_meta = store.get_session(session_meta["session_id"])
                _stop_session_record(
                    full_meta,
                    undeploy=undeploy,
                    final_status=final_status,
                )
        finally:
            if undeploy:
                clean_emulation_environment()
                removed = wipe_runtime_artifacts()
                if removed:
                    log_event(
                        "runtime_artifacts_wiped",
                        f"Removed {removed} leftover runtime entr"
                        f"{'y' if removed == 1 else 'ies'}",
                        count=removed,
                    )
        return

    if not running:
        raise FileNotFoundError(
            "No running session found. Run `nika env run <scenario>` first."
        )

    resolved_id = resolve_running_session_id(session_id, store=store)
    _stop_session_record(
        store.get_session(resolved_id),
        undeploy=undeploy,
        final_status=final_status,
    )


def close_session_after_failure(
    session_id: str,
    error: BaseException,
) -> Exception | None:
    """Close a failed case without masking the exception that caused it."""
    try:
        close_session(
            session_id=session_id,
            undeploy=True,
            final_status="failed",
        )
    except FileNotFoundError:
        try:
            Session().load_closed_session(session_id=session_id).update_run_meta(
                "status", "failed"
            )
        except (FileNotFoundError, ValueError) as state_error:
            if hasattr(error, "add_note"):
                error.add_note(
                    "The lab was already closed, but its result status could not "
                    f"be changed to failed: {type(state_error).__name__}: "
                    f"{state_error}"
                )
            log_error_event(
                "case_status_update_failed",
                f"Failed to mark closed session {session_id} as failed: {state_error}",
                session_id=session_id,
                error=str(state_error),
                error_type=type(state_error).__name__,
            )
        return None
    except Exception as cleanup_error:
        if hasattr(error, "add_note"):
            error.add_note(
                "Cleanup after the case failure also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        log_error_event(
            "case_cleanup_failed",
            f"Failed to clean session {session_id}: {cleanup_error}",
            session_id=session_id,
            error=str(cleanup_error),
            error_type=type(cleanup_error).__name__,
        )
        return cleanup_error
    return None
