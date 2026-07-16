"""Resume support for benchmark batch runs by scanning session artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from nika.config import SESSIONS_DIR, resolve_results_root
from nika.runtime.factory import runtime_for_session
from nika.utils.session_artifacts import RUN_FILENAME, iter_session_dirs
from nika.utils.session_index import SessionIndex
from nika.utils.session_store import SessionStore
from nika.workflows.session.close import close_session


def benchmark_row_fingerprint(row: dict[str, Any]) -> str:
    payload = {
        "scenario": row["scenario"],
        "problem": row["problem"],
        "topo_size": row.get("topo_size") or "",
        "inject": {
            str(k): str(v) for k, v in sorted((row.get("inject") or {}).items())
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def benchmark_row_from_case(
    *,
    scenario: str,
    problem: str,
    topo_size: str,
    inject_params: dict[str, str],
) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "problem": problem,
        "topo_size": topo_size or "",
        "inject": inject_params,
    }


def _read_run_meta(session_dir: Path) -> dict[str, Any] | None:
    run_path = session_dir / RUN_FILENAME
    if not run_path.exists():
        return None
    try:
        return json.loads(run_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def session_matches_row(run_meta: dict[str, Any], row: dict[str, Any]) -> bool:
    stored_fp = run_meta.get("benchmark_fingerprint")
    if not stored_fp:
        return False
    return stored_fp == benchmark_row_fingerprint(row)


def is_benchmark_case_complete(session_dir: str | Path, row: dict[str, Any]) -> bool:
    """Return True when a benchmark row finished the full pipeline (session closed)."""
    run_meta = _read_run_meta(Path(session_dir))
    if run_meta is None or run_meta.get("status") != "finished":
        return False
    return session_matches_row(run_meta, row)


def cleanup_benchmark_session(
    session_id: str | None,
    session_dir: str | Path | None,
) -> None:
    """Remove a partial or failed benchmark session and any runtime state."""
    result_path = Path(session_dir) if session_dir else None
    persisted_meta = _read_run_meta(result_path) if result_path else None
    session_closed = False
    indexed_meta: dict[str, Any] | None = None
    if session_id:
        try:
            indexed_meta = SessionStore().get_session(session_id)
            if indexed_meta.get("status") == "running":
                close_session(session_id=session_id, undeploy=True)
                session_closed = True
        except FileNotFoundError:
            pass

        cleanup_meta = persisted_meta or indexed_meta
        if (
            not session_closed
            and cleanup_meta
            and cleanup_meta.get("lab_name")
            and cleanup_meta.get("scenario_name")
        ):
            runtime_for_session(cleanup_meta).destroy()

        SessionIndex().purge(session_id)
        runtime_path = Path(SESSIONS_DIR) / f"{session_id}.json"
        if runtime_path.exists():
            runtime_path.unlink()

    if result_path and result_path.exists():
        shutil.rmtree(result_path)


def _sessions_under_root(results_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    sessions: list[tuple[Path, dict[str, Any]]] = []
    for session_dir in iter_session_dirs(results_root):
        run_meta = _read_run_meta(session_dir)
        if run_meta is not None:
            sessions.append((session_dir, run_meta))
    return sessions


def scan_benchmark_cases(
    *,
    rows: list[dict[str, Any]],
    result_dir: str | Path | None,
    resume: bool,
) -> tuple[Path, list[int]]:
    """Scan session dirs under the results root and return row indices still to run."""
    results_root = resolve_results_root(result_dir)
    results_root.mkdir(parents=True, exist_ok=True)
    total = len(rows)

    if not resume:
        return results_root, list(range(total))

    pool = _sessions_under_root(results_root)
    claimed: set[str] = set()
    completed = 0
    pending: list[int] = []

    for index, row in enumerate(rows):
        label = f"[{index + 1}/{total}] {row['scenario']}/{row['problem']}"
        matched_dir: Path | None = None

        for session_dir, run_meta in pool:
            key = str(session_dir)
            if key in claimed:
                continue
            if is_benchmark_case_complete(session_dir, row):
                matched_dir = session_dir
                claimed.add(key)
                break

        if matched_dir is not None:
            completed += 1
            print(f"{label} skip (already complete: {matched_dir})")
            continue

        for session_dir, run_meta in pool:
            key = str(session_dir)
            if key in claimed or run_meta.get("status") == "finished":
                continue
            if session_matches_row(run_meta, row):
                print(f"{label} cleaning incomplete session")
                cleanup_benchmark_session(
                    str(run_meta.get("session_id") or session_dir.name), session_dir
                )

        pending.append(index)

    if completed and pending:
        print(
            f"Resuming benchmark: {completed}/{total} complete, "
            f"{len(pending)} remaining under {results_root}"
        )
    elif not pending:
        print(f"All {total} benchmark case(s) already complete under {results_root}")

    return results_root, pending
