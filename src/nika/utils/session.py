import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from nika.config import RESULTS_DIR
from nika.utils.session_index import extract_gt_fields, extract_index_fields
from nika.utils.session_resolve import resolve_running_session_id
from nika.utils.session_store import SessionStore


RUN_FILENAME = "run.json"


def _is_finished_session(run_meta: dict[str, Any]) -> bool:
    return run_meta.get("status") == "finished" or run_meta.get("end_time") is not None


def _iter_session_dirs() -> list[Path]:
    root = Path(RESULTS_DIR)
    if not root.exists():
        return []
    return [
        run_path.parent
        for run_path in sorted(root.rglob(RUN_FILENAME))
        if "0_summary" not in run_path.relative_to(root).parts
    ]


def find_session_dir(session_id: str, *, results_dir: str | Path | None = None) -> Path:
    root = Path(results_dir or RESULTS_DIR)
    direct = root / session_id
    if (direct / RUN_FILENAME).exists():
        return direct
    if not root.exists():
        raise FileNotFoundError(f"Session '{session_id}' not found under results/.")
    for run_path in sorted(root.rglob(RUN_FILENAME)):
        if "0_summary" in run_path.relative_to(root).parts:
            continue
        try:
            run_meta = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(run_meta.get("session_id") or run_path.parent.name) == session_id:
            return run_path.parent
    raise FileNotFoundError(f"Session '{session_id}' not found under results/.")


class Session:
    def __init__(self) -> None:
        self.store = SessionStore()

    def init_session(
        self,
        *,
        session_id: str,
        scenario_name: str,
        lab_name: str,
        scenario_topo_size: str | None,
        results_root: str | Path | None = None,
        scenario_params: dict | None = None,
        topology: list[tuple[str, str]] | None = None,
    ) -> None:
        self.session_id = session_id
        self.scenario_name = scenario_name
        self.lab_name = lab_name
        self.scenario_topo_size = scenario_topo_size
        self.scenario_params = scenario_params or {}
        self.topology = topology or []
        self.session_dir = os.path.join(str(results_root or RESULTS_DIR), session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        self.store.create_session(
            {
                "session_id": self.session_id,
                "lab_name": self.lab_name,
                "scenario_name": self.scenario_name,
                "scenario_topo_size": self.scenario_topo_size,
                "scenario_params": self.scenario_params,
                "topology": self.topology,
                "session_dir": self.session_dir,
                "status": "running",
            }
        )
        self._write_run_json({k: v for k, v in self.__dict__.items() if k != "store"})

    def load_running_session(self, session_id: str | None = None):
        resolved_id = resolve_running_session_id(session_id, store=self.store)
        session_meta = self.store.get_session(resolved_id)
        for key, value in session_meta.items():
            setattr(self, key, value)
        return self

    def load_closed_session(self, session_id: str | None = None):
        """Load a finished session from ``results/{session_id}/run.json`` for offline eval."""
        if session_id is not None:
            return self._load_closed_session_from_id(session_id)

        candidates: list[tuple[float, dict]] = []
        for session_dir in _iter_session_dirs():
            run_path = session_dir / RUN_FILENAME
            run_meta = json.loads(run_path.read_text(encoding="utf-8"))
            if not _is_finished_session(run_meta):
                continue
            sid = run_meta.get("session_id") or session_dir.name
            if self._session_is_still_running(sid):
                continue
            candidates.append((run_path.stat().st_mtime, run_meta))

        if not candidates:
            raise FileNotFoundError(
                "No closed session found under results/. Close a session with `nika session close` first."
            )
        if len(candidates) > 1:
            raise ValueError(
                "Multiple closed sessions found under results/. Please pass --session-id to select one."
            )
        return self._apply_closed_session_meta(candidates[0][1])

    def _session_is_still_running(self, session_id: str) -> bool:
        try:
            return self.store.get_session(session_id).get("status") == "running"
        except FileNotFoundError:
            return False

    def _load_closed_session_from_id(self, session_id: str):
        if self._session_is_still_running(session_id):
            raise ValueError(
                f"Session '{session_id}' is still running. Close it with `nika session close` before running eval."
            )

        session_dir = find_session_dir(session_id)
        run_path = session_dir / RUN_FILENAME

        run_meta = json.loads(run_path.read_text(encoding="utf-8"))
        if not _is_finished_session(run_meta):
            raise ValueError(
                f"Session '{session_id}' is not closed. Run `nika session close` before running eval."
            )
        return self._apply_closed_session_meta(run_meta, session_dir=session_dir)

    def _apply_closed_session_meta(self, run_meta: dict, *, session_dir: Path | None = None):
        for key, value in run_meta.items():
            setattr(self, key, value)
        resolved_dir = session_dir or find_session_dir(run_meta.get("session_id") or "")
        self.session_dir = str(resolved_dir)
        return self

    def _write_session(self) -> str:
        if not hasattr(self, "session_id"):
            raise ValueError("Session ID is not set.")
        payload = {k: v for k, v in self.__dict__.items() if k != "store"}
        self.store.update_session(self.session_id, payload)
        if getattr(self, "session_dir", None):
            self._write_run_json(payload)
        return self.session_id

    def _write_run_json(self, payload: dict) -> None:
        """Write/update run.json in the session results directory."""
        os.makedirs(self.session_dir, exist_ok=True)
        run_path = os.path.join(self.session_dir, "run.json")
        serializable = {
            k: v for k, v in payload.items() if k not in ("store", "failure_injections")
        }
        with open(run_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, default=str)

    def update_session(self, key: str, value: Any):
        setattr(self, key, value)
        if hasattr(self, "problem_names") and hasattr(self, "session_id"):
            problem_names = list(getattr(self, "problem_names", []) or [])
            if len(problem_names) > 1:
                self.root_cause_name = "multiple_faults"
            elif problem_names:
                self.root_cause_name = problem_names[0]
        self._write_session()

    def update_run_meta(self, key: str, value: Any):
        """Update ``run.json`` for a closed session (no runtime session document)."""
        setattr(self, key, value)
        if hasattr(self, "problem_names") and hasattr(self, "session_id"):
            problem_names = list(getattr(self, "problem_names", []) or [])
            if len(problem_names) > 1:
                self.root_cause_name = "multiple_faults"
            elif problem_names:
                self.root_cause_name = problem_names[0]
        payload = {k: v for k, v in self.__dict__.items() if k != "store"}
        self._write_run_json(payload)
        if hasattr(self, "session_id"):
            fields = extract_index_fields(payload)
            fields["session_id"] = self.session_id
            fields.setdefault("status", "finished")
            self.store.index.upsert(fields)

    def write_gt(self, gt: dict[str, Any]):
        os.makedirs(self.session_dir, exist_ok=True)
        with open(self.session_dir + "/ground_truth.json", "w") as f:
            f.write(json.dumps(gt, indent=4))
        if hasattr(self, "session_id"):
            self.store.index.upsert(
                {"session_id": self.session_id, **extract_gt_fields(gt)},
            )

    def clear_session(self):
        if not hasattr(self, "session_id"):
            raise ValueError("Session ID is not set.")
        payload = {k: v for k, v in self.__dict__.items() if k != "store"}
        payload["status"] = "finished"
        if getattr(self, "session_dir", None):
            self._write_run_json(payload)
        self.store.delete_session(self.session_id)

    def start_session(self):
        self.start_time = datetime.now().isoformat()
        self._write_session()

    def end_session(self):
        self.end_time = datetime.now().isoformat()
        self._write_session()

    def __str__(self) -> str:
        payload = {k: v for k, v in self.__dict__.items() if k != "store"}
        return str(payload)
