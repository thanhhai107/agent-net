"""Helpers for the Streamlit experiment runner."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.llm.model_factory import DEFAULT_LLM_BACKEND, DEFAULT_MODEL
from nika.config import RESULTS_DIR, RUNTIME_DIR, _REPO_ROOT
from nika.utils.agent_config import resolve_max_steps
from nika.utils.experiment_naming import next_experiment_id
from nika.utils.kathara_cleanup import ensure_kathara_clean
from nika.workflows.benchmark.run import default_benchmark_yaml_path

RUNS_DIR = RUNTIME_DIR / "streamlit_runs"
LOG_FILENAME = "run.log"
SPEC_FILENAME = "spec.json"
META_FILENAME = "meta.json"

MODULE_LABELS = {
    "tool_evolution": "Tool Evolution",
    "memory_evolution": "Memory Evolution",
}
AGENT_LABELS = {
    "react": "ReAct",
    "plan-execute": "Plan-Execute",
    "reflexion": "Reflexion",
    "mock": "Mock",
}


@dataclass(frozen=True)
class CommandPlan:
    name: str
    command: list[str]
    variant: str = "setup"


def _python_module_command(*args: str) -> list[str]:
    return [sys.executable, "-m", "nika.cli.main", *args]


def _str(value: Any, default: str) -> str:
    if value is None or value == "":
        return default
    return str(value)


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _common_agent_args(
    config: dict[str, Any],
    *,
    default_agent: str = "react",
) -> list[str]:
    return [
        "-a",
        _str(config.get("agent_type"), default_agent),
        "-b",
        _str(config.get("llm_backend"), DEFAULT_LLM_BACKEND),
        "-m",
        _str(config.get("model"), DEFAULT_MODEL),
        "-n",
        str(_int(config.get("max_steps"), resolve_max_steps(None))),
        "-r",
        str(_int(config.get("max_attempts"), 3)),
    ]


def _judge_args(config: dict[str, Any]) -> list[str]:
    if not config.get("run_judge"):
        return []
    return [
        "--judge",
        "--judge-backend",
        _str(
            config.get("judge_backend"),
            _str(config.get("llm_backend"), DEFAULT_LLM_BACKEND),
        ),
        "--judge-model",
        _str(
            config.get("judge_model"),
            _str(config.get("model"), DEFAULT_MODEL),
        ),
    ]


def _benchmark_command(config: dict[str, Any]) -> list[str]:
    command = _python_module_command(
        "benchmark",
        "run",
        "--file",
        _str(config.get("benchmark_file"), default_benchmark_yaml_path()),
        *_common_agent_args(config),
        *_judge_args(config),
    )
    if config.get("result_root"):
        command.extend(["--result-root", str(config["result_root"])])
    if config.get("resume"):
        command.append("--resume")
    return command


def selected_modules(config: dict[str, Any]) -> set[str]:
    modules = config.get("modules") or []
    return {str(item) for item in modules if item}


def agent_type(config: dict[str, Any]) -> str:
    return str(config.get("agent_type") or "react").lower()


def _command_experiment_id(config: dict[str, Any]) -> str:
    configured = str(config.get("experiment_id") or "").strip()
    if configured:
        return configured
    return next_experiment_id(
        _str(config.get("benchmark_file"), default_benchmark_yaml_path())
    )


def experiment_label(config: dict[str, Any]) -> str:
    modules = selected_modules(config)
    labels = [AGENT_LABELS.get(agent_type(config), agent_type(config))]
    ordered = ["tool_evolution", "memory_evolution"]
    labels.extend(MODULE_LABELS[item] for item in ordered if item in modules)
    label = " + ".join(labels)
    if config.get("resume"):
        label = f"{label} Resume"
    return label


def build_experiment_command(config: dict[str, Any]) -> list[str]:
    """Build the CLI command for one run with all selected modules enabled."""
    modules = selected_modules(config)
    tool_enabled = "tool_evolution" in modules
    memory_enabled = "memory_evolution" in modules
    default_library_id = _command_experiment_id(config)

    command = _benchmark_command(config)

    if tool_enabled:
        command.extend(
            [
                "--tools",
                _str(
                    config.get("tool_library_id"),
                    default_library_id,
                ),
                "--tool-doc-chars",
                str(_int(config.get("tool_doc_chars"), 500)),
                "--tool-prompt-doc-limit",
                str(_int(config.get("tool_prompt_doc_limit"), 6)),
                "--tool-scoped-prompt-doc-limit",
                str(_int(config.get("tool_scoped_prompt_doc_limit"), 4)),
                "--tool-planned-checks",
                str(_int(config.get("tool_planned_checks"), 4)),
                "--tool-next-checks",
                str(_int(config.get("tool_next_checks"), 2)),
                "--tool-convergence-threshold",
                _str(config.get("tool_convergence_threshold"), "0.75"),
            ]
        )

    if memory_enabled:
        command.extend(
            [
                "--memory",
                _str(
                    config.get("memory_bank"),
                    default_library_id,
                ),
                "--memory-k",
                str(_int(config.get("memory_k"), 5)),
                "--memory-tokens",
                str(_int(config.get("memory_tokens"), 1500)),
                "--memory-selector",
                _str(config.get("memory_selector"), "lcb"),
                "--memory-meta-controller",
                _str(config.get("memory_meta_controller"), "heuristic"),
                "--memory-max-skill-age",
                str(_int(config.get("memory_max_skill_age"), 4)),
                "--memory-selector-min-lcb",
                _str(config.get("memory_selector_min_lcb"), "-0.05"),
                "--memory-selector-nominee-k",
                str(_int(config.get("memory_selector_nominee_k"), 3)),
                "--memory-pool-size",
                str(_int(config.get("memory_pool_size"), 32)),
                "--memory-evolution-threshold",
                str(_int(config.get("memory_evolution_threshold"), 3)),
                "--memory-best-of-n",
                str(_int(config.get("memory_best_of_n"), 3)),
                "--memory-ppo-epsilon",
                _str(config.get("memory_ppo_epsilon"), "0.2"),
            ]
        )

    return command


def build_command_plan(config: dict[str, Any]) -> list[CommandPlan]:
    plan: list[CommandPlan] = []
    plan.append(
        CommandPlan(
            name=experiment_label(config),
            command=build_experiment_command(config),
            variant="benchmark",
        )
    )
    return plan


def prepare_experiment_config(config: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(config)
    benchmark_file = _str(prepared.get("benchmark_file"), default_benchmark_yaml_path())
    run_id = _str(prepared.get("experiment_id"), "")
    if not run_id:
        run_id = next_experiment_id(benchmark_file)
    prepared["experiment_id"] = run_id
    if not str(prepared.get("result_root") or "").strip():
        prepared["result_root"] = str(RESULTS_DIR / run_id)
    modules = selected_modules(prepared)
    if "tool_evolution" in modules and not str(prepared.get("tool_library_id") or "").strip():
        prepared["tool_library_id"] = run_id
    if "memory_evolution" in modules and not str(prepared.get("memory_bank") or "").strip():
        prepared["memory_bank"] = run_id
    return prepared


def prepare_resume_config(
    source_spec: dict[str, Any],
    *,
    resume_run_id: str | None = None,
) -> dict[str, Any]:
    config = dict(source_spec.get("config") or {})
    if not config:
        raise ValueError("Selected run does not have a resumable config.")
    source_run_id = str(source_spec.get("run_id") or config.get("experiment_id") or "").strip()
    if not source_run_id:
        raise ValueError("Selected run does not have a source run id.")

    if not str(config.get("result_root") or "").strip():
        config["result_root"] = str(RESULTS_DIR / source_run_id)
    modules = selected_modules(config)
    if "tool_evolution" in modules and not str(config.get("tool_library_id") or "").strip():
        config["tool_library_id"] = source_run_id
    if "memory_evolution" in modules and not str(config.get("memory_bank") or "").strip():
        config["memory_bank"] = source_run_id

    config["experiment_id"] = resume_run_id or source_run_id
    config["resume"] = True
    return prepare_experiment_config(config)


def create_run(config: dict[str, Any]) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    config = prepare_experiment_config(config)
    run_id = str(config["experiment_id"])
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    plan = build_command_plan(config)
    spec = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "commands": [asdict(item) for item in plan],
    }
    (run_dir / SPEC_FILENAME).write_text(json.dumps(spec, indent=2), encoding="utf-8")
    log_path = run_dir / LOG_FILENAME

    # Check if there is already a run running
    is_busy = False
    for r in list_runs():
        if run_status(r).get("status") == "running":
            is_busy = True
            break

    if is_busy:
        meta = {
            "run_id": run_id,
            "pid": 0,
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "log_path": str(log_path),
            "spec_path": str(run_dir / SPEC_FILENAME),
        }
        (run_dir / META_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    else:
        log_handle = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "nika.visualization.experiment_runner", str(run_dir / SPEC_FILENAME)],
            cwd=_REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        log_handle.close()
        meta = {
            "run_id": run_id,
            "pid": proc.pid,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "log_path": str(log_path),
            "spec_path": str(run_dir / SPEC_FILENAME),
        }
        (run_dir / META_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return run_dir


def resume_run(run_dir: Path) -> Path:
    """Resume a stopped/failed Studio run in-place using the same run directory."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = Path(run_dir)
    spec = read_run_spec(run_dir)
    source_run_id = str(spec.get("run_id") or run_dir.name)
    current_status = run_status(run_dir).get("status")
    if current_status in {"running", "queued"}:
        raise ValueError(f"Run {source_run_id} is already {current_status}.")

    config = prepare_resume_config(spec, resume_run_id=source_run_id)
    plan = build_command_plan(config)
    updated_spec = {
        **spec,
        "run_id": source_run_id,
        "created_at": spec.get("created_at") or datetime.now(timezone.utc).isoformat(),
        "resumed_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "commands": [asdict(item) for item in plan],
    }
    (run_dir / SPEC_FILENAME).write_text(json.dumps(updated_spec, indent=2), encoding="utf-8")
    log_path = run_dir / LOG_FILENAME
    resume_payload = {
        "run_id": source_run_id,
        "result_root": str(config.get("result_root") or ""),
    }
    _clean_resume_log(run_dir)

    is_busy = any(
        path != run_dir and run_status(path).get("status") == "running"
        for path in list_runs()
    )
    if is_busy:
        meta = {
            **_read_json(run_dir / META_FILENAME),
            "run_id": source_run_id,
            "pid": 0,
            "status": "queued",
            "resumed_at": datetime.now(timezone.utc).isoformat(),
            "log_path": str(log_path),
            "spec_path": str(run_dir / SPEC_FILENAME),
        }
        _write_run_meta(run_dir, meta)
        _append_run_log(
            run_dir,
            "ui_run_resumed "
            + json.dumps({**resume_payload, "queued": True}, ensure_ascii=False),
        )
        return run_dir

    _append_run_log(
        run_dir,
        "ui_run_resumed " + json.dumps(resume_payload, ensure_ascii=False),
    )
    log_handle = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "nika.visualization.experiment_runner", str(run_dir / SPEC_FILENAME)],
        cwd=_REPO_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    log_handle.close()
    meta = {
        **_read_json(run_dir / META_FILENAME),
        "run_id": source_run_id,
        "pid": proc.pid,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "resumed_at": datetime.now(timezone.utc).isoformat(),
        "log_path": str(log_path),
        "spec_path": str(run_dir / SPEC_FILENAME),
    }
    _write_run_meta(run_dir, meta)
    return run_dir


def list_runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted((path for path in RUNS_DIR.iterdir() if path.is_dir()), reverse=True)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def read_run_spec(run_dir: Path) -> dict[str, Any]:
    return _read_json(run_dir / SPEC_FILENAME)


def read_run_log(run_dir: Path) -> str:
    path = run_dir / LOG_FILENAME
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def tail_log(run_dir: Path, *, max_lines: int = 300) -> str:
    lines = read_run_log(run_dir).splitlines()
    return "\n".join(lines[-max_lines:])


def _pid_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        # Try to reap the child process if it has exited (this runs inside the Streamlit parent process)
        reaped_pid, status = os.waitpid(pid, os.WNOHANG)
        if reaped_pid == pid:
            return False
    except ChildProcessError:
        pass
    except OSError:
        pass

    try:
        os.kill(pid, 0)
    except OSError:
        return False

    # Check /proc/<pid>/status as a fallback for zombie state on Linux
    try:
        from pathlib import Path
        status_path = Path(f"/proc/{pid}/status")
        if status_path.exists():
            for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("State:"):
                    if "zombie" in line.lower() or "defunct" in line.lower():
                        return False
    except Exception:
        pass

    return True


def run_status(run_dir: Path) -> dict[str, Any]:
    meta = _read_json(run_dir / META_FILENAME)
    if meta.get("status") == "queued":
        return {**meta, "status": "queued", "exit_code": None}
    if meta.get("status") == "stopped":
        return {**meta, "status": "stopped", "exit_code": None}
    log_lines = read_run_log(run_dir).splitlines()
    last_resume = -1
    for index, line in enumerate(log_lines):
        if line.startswith("ui_run_resumed "):
            last_resume = index
    current_lines = log_lines[last_resume + 1 :]
    done = [line for line in current_lines if line.startswith("ui_run_done ")]
    if done:
        payload = _parse_json_suffix(done[-1], "ui_run_done ")
        code = _int(payload.get("exit_code"), 1)
        return {**meta, "status": "finished" if code == 0 else "failed", "exit_code": code}
    pid = _int(meta.get("pid"), 0)
    if _pid_running(pid):
        return {**meta, "status": "running", "exit_code": None}
    return {**meta, "status": "stopped", "exit_code": None}


def check_and_start_next_queued() -> None:
    if any(run_status(run).get("status") == "running" for run in list_runs()):
        return
    # list_runs() returns newest first. We reverse it to find the oldest queued run.
    runs = list(reversed(list_runs()))
    for r in runs:
        meta = _read_json(r / META_FILENAME)
        if meta.get("status") == "queued":
            spec_file = r / SPEC_FILENAME
            log_path = r / LOG_FILENAME
            log_handle = log_path.open("a", encoding="utf-8")
            try:
                proc = subprocess.Popen(
                    [sys.executable, "-u", "-m", "nika.visualization.experiment_runner", str(spec_file)],
                    cwd=_REPO_ROOT,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                log_handle.close()
                meta["pid"] = proc.pid
                meta["status"] = "running"
                meta["started_at"] = datetime.now(timezone.utc).isoformat()
                (r / META_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
                break
            except Exception as e:
                log_handle.close()
                print(f"Failed to auto-start queued run {r.name}: {e}")


def _write_run_meta(run_dir: Path, meta: dict[str, Any]) -> None:
    (run_dir / META_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _append_run_log(run_dir: Path, message: str) -> None:
    with (run_dir / LOG_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def _clean_resume_log(run_dir: Path) -> None:
    """Remove stale UI terminal markers before resuming a run in place."""
    log_path = run_dir / LOG_FILENAME
    if not log_path.exists():
        return
    terminal_prefixes = (
        "ui_step_done ",
        "ui_run_done ",
        "ui_run_stopped ",
    )
    lines = [
        line
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if not line.startswith(terminal_prefixes)
    ]
    log_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def stop_run(run_dir: Path) -> None:
    meta = _read_json(run_dir / META_FILENAME)
    pid = _int(meta.get("pid"), 0)
    if not pid:
        return
    meta["status"] = "stopped"
    meta["stopped_at"] = datetime.now(timezone.utc).isoformat()
    _write_run_meta(run_dir, meta)
    _append_run_log(
        run_dir,
        "ui_run_stopped "
        + json.dumps({"reason": "user_stop"}, ensure_ascii=False),
    )
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for _ in range(20):
        if not _pid_running(pid):
            break
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            break
        import time

        time.sleep(0.2)
    if _pid_running(pid):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    
    # Clean up lingering Kathara container environments.
    try:
        ensure_kathara_clean(context="studio stop")
    except Exception as exc:
        _append_run_log(
            run_dir,
            "ui_cleanup_failed " + json.dumps({"error": str(exc)}, ensure_ascii=False),
        )

    check_and_start_next_queued()


def _parse_json_suffix(line: str, prefix: str) -> dict[str, Any]:
    try:
        value = json.loads(line[len(prefix) :].strip())
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def parse_progress_events(log_text: str) -> list[dict[str, str]]:
    prefixes = (
        "ui_step_start ",
        "ui_step_done ",
        "ui_run_resumed ",
        "ui_run_stopped ",
        "ui_run_done ",
        "benchmark_start ",
        "benchmark_skip ",
        "benchmark_progress ",
        "benchmark_done ",
        "benchmark_failed ",
        "benchmark_summary ",
    )
    rows: list[dict[str, str]] = []
    for line in log_text.splitlines():
        prefix = next((item for item in prefixes if line.startswith(item)), None)
        if prefix is None:
            continue
        event = prefix.strip()
        rest = line[len(prefix) :]
        row: dict[str, str] = {"event": event, "raw": line}
        if prefix.startswith("ui_"):
            row.update({key: str(value) for key, value in _parse_json_suffix(line, prefix).items()})
        else:
            for part in rest.split():
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                row[key] = value
        rows.append(row)
    return rows


def run_spec_file(spec_path: str | Path) -> int:
    spec = _read_json(Path(spec_path))
    commands = spec.get("commands") or []
    exit_code = 0
    total = len(commands)
    for index, item in enumerate(commands, start=1):
        name = str(item.get("name") or f"step-{index}")
        command = [str(part) for part in item.get("command") or []]
        print(
            "ui_step_start "
            + json.dumps(
                {"index": index, "total": total, "name": name, "command": command},
                ensure_ascii=False,
            ),
            flush=True,
        )
        try:
            import os
            sub_env = os.environ.copy()
            sub_env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen(
                command,
                cwd=_REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=sub_env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
            return_code = proc.wait()
        except OSError as exc:
            return_code = 127
            print(f"Failed to start command: {exc}", flush=True)
        print(
            "ui_step_done "
            + json.dumps(
                {"index": index, "total": total, "name": name, "returncode": return_code},
                ensure_ascii=False,
            ),
            flush=True,
        )
        if return_code != 0:
            exit_code = return_code
            break
    print(
        "ui_run_done " + json.dumps({"exit_code": exit_code}, ensure_ascii=False),
        flush=True,
    )
    try:
        check_and_start_next_queued()
    except Exception as e:
        print(f"Error starting next queued run: {e}", flush=True)
    return exit_code


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m nika.visualization.experiment_runner SPEC")
    raise SystemExit(run_spec_file(sys.argv[1]))


if __name__ == "__main__":
    main()
