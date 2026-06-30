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

from agent.defaults import DEFAULT_MAX_STEPS
from agent.llm.model_factory import DEFAULT_LLM_BACKEND, DEFAULT_MODEL
from nika.config import RUNTIME_DIR, _REPO_ROOT
from nika.utils.kathara_cleanup import ensure_kathara_clean
from nika.workflows.benchmark.run import default_benchmark_csv_path

RUNS_DIR = RUNTIME_DIR / "streamlit_runs"
LOG_FILENAME = "run.log"
SPEC_FILENAME = "spec.json"
META_FILENAME = "meta.json"

MODULE_LABELS = {
    "tool_evolution": "Tool Evolution",
    "memory_evolution": "Memory Evolution",
    "harness_evolution": "Harness Evolution (SIA-H)",
}


@dataclass(frozen=True)
class CommandPlan:
    name: str
    command: list[str]
    variant: str = "setup"


def _now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")


def _python_module_command(*args: str) -> list[str]:
    return [sys.executable, "-m", "nika.codex_cli.main", *args]


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
        str(_int(config.get("max_steps"), DEFAULT_MAX_STEPS)),
        "-r",
        str(_int(config.get("max_attempts"), 3)),
    ]


def _harness_model_args(config: dict[str, Any]) -> list[str]:
    return [
        "-b",
        _str(config.get("llm_backend"), DEFAULT_LLM_BACKEND),
        "-m",
        _str(config.get("model"), DEFAULT_MODEL),
        "-n",
        str(_int(config.get("max_steps"), DEFAULT_MAX_STEPS)),
    ]


def _judge_args(config: dict[str, Any]) -> list[str]:
    if not config.get("run_judge"):
        return []
    return [
        "--judge",
        "--judge-backend",
        _str(config.get("judge_backend"), _str(config.get("llm_backend"), DEFAULT_LLM_BACKEND)),
        "--judge-model",
        _str(config.get("judge_model"), _str(config.get("model"), DEFAULT_MODEL)),
    ]


def _benchmark_command(config: dict[str, Any]) -> list[str]:
    command = _python_module_command(
        "benchmark",
        "run",
        "--file",
        _str(config.get("benchmark_file"), default_benchmark_csv_path()),
        *_common_agent_args(config),
        *_judge_args(config),
    )
    if config.get("oracle_routing"):
        command.append("--oracle-routing")
    return command


def selected_modules(config: dict[str, Any]) -> set[str]:
    modules = config.get("modules") or []
    return {str(item) for item in modules if item}


def experiment_label(config: dict[str, Any]) -> str:
    modules = selected_modules(config)
    if not modules:
        return "Baseline"
    ordered = ["harness_evolution", "tool_evolution", "memory_evolution"]
    return " + ".join(MODULE_LABELS[item] for item in ordered if item in modules)


def build_experiment_command(config: dict[str, Any]) -> list[str]:
    """Build the CLI command for one run with all selected modules enabled."""
    modules = selected_modules(config)
    tool_enabled = "tool_evolution" in modules
    memory_enabled = "memory_evolution" in modules
    harness_evolution_enabled = "harness_evolution" in modules

    if harness_evolution_enabled:
        command = _python_module_command(
            "evolve",
            "run",
            "--file",
            _str(config.get("benchmark_file"), default_benchmark_csv_path()),
            "--max-gen",
            str(_int(config.get("max_generations"), 3)),
            *_harness_model_args(config),
            "--feedback-mode",
            _str(config.get("feedback_mode"), "auto"),
            "--feedback-backend",
            _str(config.get("feedback_backend"), _str(config.get("llm_backend"), DEFAULT_LLM_BACKEND)),
            "--feedback-model",
            _str(config.get("feedback_model"), _str(config.get("model"), DEFAULT_MODEL)),
            *_judge_args(config),
        )
    else:
        command = _benchmark_command(config)

    if tool_enabled:
        command.extend(
            [
                "--tools",
                _str(config.get("tool_library_id"), "tools-streamlit"),
                "--tool-mode",
                _str(config.get("tool_mode"), "dual"),
            ]
        )

    if memory_enabled:
        command.extend(
            [
                "--memory",
                _str(config.get("memory_bank"), "memory-streamlit"),
                "--memory-k",
                str(_int(config.get("memory_k"), 5)),
                "--memory-tokens",
                str(_int(config.get("memory_tokens"), 1500)),
            ]
        )

    return command


def build_command_plan(config: dict[str, Any]) -> list[CommandPlan]:
    modules = selected_modules(config)
    plan: list[CommandPlan] = []
    if config.get("ensure_memory_services") and "memory_evolution" in modules:
        plan.append(
            CommandPlan(
                name="Memory services",
                command=["docker", "compose", "up", "-d", "postgres", "qdrant"],
            )
        )
    plan.append(
        CommandPlan(
            name=experiment_label(config),
            command=build_experiment_command(config),
            variant=(
                "harness_evolution"
                if "harness_evolution" in modules
                else "benchmark"
            ),
        )
    )
    return plan


def create_run(config: dict[str, Any]) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"ui-{_now_id()}"
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
    log_text = read_run_log(run_dir)
    done = [line for line in log_text.splitlines() if line.startswith("ui_run_done ")]
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
        "ui_run_stopped ",
        "ui_run_done ",
        "benchmark_start ",
        "benchmark_progress ",
        "benchmark_done ",
        "benchmark_failed ",
        "benchmark_summary ",
        "kathara_cleanup_start ",
        "kathara_cleanup_done ",
        "evolve_generation_start ",
        "evolve_generation_done ",
        "evolve_summary ",
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
    print("kathara_cleanup_start context=studio_run", flush=True)
    ensure_kathara_clean(context="studio run")
    print("kathara_cleanup_done context=studio_run", flush=True)

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
