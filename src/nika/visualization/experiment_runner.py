"""Helpers for the Streamlit experiment runner."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agent.composition import ProceduralMemoryConfig, ToolRefinementConfig
from agent.module_config import (
    ENV_MODULE_CONFIG_PATH,
    module_defaults,
)
from agent.extensions.config import (
    DEFAULT_LLM_PROVIDER as DEFAULT_LLM_BACKEND,
    DEFAULT_MODEL,
)
from nika.config import RESULTS_DIR, RUNTIME_DIR, _REPO_ROOT
from nika.config import BENCHMARK_DIR
from nika.utils.experiment_naming import next_experiment_id
from nika.workflows.session.close import clean_emulation_environment

RUNS_DIR = RUNTIME_DIR / "streamlit_runs"
LOG_FILENAME = "run.log"
SPEC_FILENAME = "spec.json"
META_FILENAME = "meta.json"
MODULE_CONFIG_SNAPSHOT_FILENAME = "modules.yaml"
RESOLVED_MODULE_DEFAULTS = module_defaults()
BASELINE_DEFAULTS = RESOLVED_MODULE_DEFAULTS.baseline
TOOL_MODULE_DEFAULTS = RESOLVED_MODULE_DEFAULTS.tool_refinement
DEFAULT_STUDIO_LEARNING_BENCHMARK = str(
    BENCHMARK_DIR / BASELINE_DEFAULTS.learning_benchmark
)
DEFAULT_STUDIO_EVALUATE_BENCHMARK = str(
    BENCHMARK_DIR / BASELINE_DEFAULTS.evaluate_benchmark
)
DEFAULT_STUDIO_MAX_STEPS = BASELINE_DEFAULTS.max_steps
TOOL_REFINEMENT_DEFAULTS = ToolRefinementConfig()
PROCEDURAL_MEMORY_DEFAULTS = ProceduralMemoryConfig()
_STOP_GRACE_SECONDS = 120.0
_STOP_KILL_GRACE_SECONDS = 10.0
_STOP_POLL_SECONDS = 0.2

MODULE_LABELS = {
    "tool_refinement": "Tool Refinement",
    "procedural_memory": "Procedural Memory",
}
AGENT_LABELS = {
    "react": "ReAct",
    "plan-execute": "Plan-and-Execute",
    "reflexion": "Reflexion",
}


@dataclass(frozen=True)
class CommandPlan:
    name: str
    command: list[str]
    variant: str = "setup"


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
    default_agent: str = BASELINE_DEFAULTS.agent_type,
) -> list[str]:
    return [
        "--agent",
        _str(config.get("agent_type"), default_agent),
        "--provider",
        _str(config.get("llm_backend"), DEFAULT_LLM_BACKEND),
        "--model",
        _str(config.get("model"), DEFAULT_MODEL),
        "--max-steps",
        str(_int(config.get("max_steps"), DEFAULT_STUDIO_MAX_STEPS)),
        "--max-attempts",
        str(_int(config.get("max_attempts"), BASELINE_DEFAULTS.max_attempts)),
    ]


def _judge_args(config: dict[str, Any]) -> list[str]:
    if not config.get("run_judge", BASELINE_DEFAULTS.judge_evaluation):
        return []
    return [
        "--judge",
        "--judge-provider",
        _str(
            config.get("judge_backend"),
            BASELINE_DEFAULTS.judge_provider,
        ),
        "--judge-model",
        _str(
            config.get("judge_model"),
            BASELINE_DEFAULTS.judge_model,
        ),
    ]


def _benchmark_command(config: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "nika.extensions.benchmark",
        "--evaluate-benchmark",
        _str(
            config.get("evaluate_benchmark_file"),
            DEFAULT_STUDIO_EVALUATE_BENCHMARK,
        ),
        *_common_agent_args(config),
        *_judge_args(config),
    ]
    if selected_modules(config):
        command.extend(
            [
                "--learning-benchmark",
                _str(
                    config.get("learning_benchmark_file"),
                    DEFAULT_STUDIO_LEARNING_BENCHMARK,
                ),
            ]
        )
    if config.get("result_root"):
        command.extend(["--result-dir", str(config["result_root"])])
    if config.get("resume"):
        command.append("--resume")
    else:
        command.append("--no-resume")
    return command


def selected_modules(config: dict[str, Any]) -> set[str]:
    modules = config.get("modules") or []
    return {str(item) for item in modules if item}


def agent_type(config: dict[str, Any]) -> str:
    configured = str(config.get("agent_type") or BASELINE_DEFAULTS.agent_type).lower()
    return "react" if configured == "byo.langgraph" else configured


def _command_experiment_id(config: dict[str, Any]) -> str:
    configured = str(config.get("experiment_id") or "").strip()
    if configured:
        return configured
    return next_experiment_id(
        _str(
            config.get("evaluate_benchmark_file"),
            DEFAULT_STUDIO_EVALUATE_BENCHMARK,
        )
    )


def experiment_label(config: dict[str, Any]) -> str:
    modules = selected_modules(config)
    labels = [AGENT_LABELS.get(agent_type(config), agent_type(config))]
    ordered = ["tool_refinement", "procedural_memory"]
    labels.extend(MODULE_LABELS[item] for item in ordered if item in modules)
    label = " + ".join(labels)
    if config.get("resume"):
        label = f"{label} Resume"
    return label


def build_experiment_command(config: dict[str, Any]) -> list[str]:
    """Build the CLI command for one run with all selected modules enabled."""
    modules = selected_modules(config)
    tool_enabled = "tool_refinement" in modules
    procedural_memory_enabled = "procedural_memory" in modules
    default_library_id = _command_experiment_id(config)

    command = _benchmark_command(config)

    if tool_enabled:
        command.extend(
            [
                "--tool-refinement",
                _str(
                    config.get("tool_library_id"),
                    default_library_id,
                ),
                "--tool-refinement-doc-chars",
                str(
                    _int(
                        config.get("tool_doc_chars"),
                        TOOL_REFINEMENT_DEFAULTS.tool_doc_chars,
                    )
                ),
                "--tool-refinement-convergence-threshold",
                _str(
                    config.get("tool_convergence_threshold"),
                    str(TOOL_REFINEMENT_DEFAULTS.convergence_threshold),
                ),
                "--tool-refinement-exploration-similarity-threshold",
                _str(
                    config.get("tool_exploration_similarity_threshold"),
                    str(TOOL_REFINEMENT_DEFAULTS.exploration_similarity_threshold),
                ),
                "--tool-refinement-explorer-reflection-limit",
                str(
                    _int(
                        config.get("tool_explorer_reflection_limit"),
                        TOOL_REFINEMENT_DEFAULTS.explorer_reflection_limit,
                    )
                ),
                "--tool-refinement-explorer-model",
                _str(
                    config.get("tool_explorer_model"),
                    TOOL_MODULE_DEFAULTS.llm_model,
                ),
                "--tool-refinement-analyzer-model",
                _str(
                    config.get("tool_analyzer_model"),
                    TOOL_MODULE_DEFAULTS.llm_model,
                ),
                "--tool-refinement-rewriter-model",
                _str(
                    config.get("tool_rewriter_model"),
                    TOOL_MODULE_DEFAULTS.llm_model,
                ),
                "--tool-refinement-update-interval",
                str(
                    _int(
                        config.get("tool_update_interval"),
                        TOOL_REFINEMENT_DEFAULTS.update_interval,
                    )
                ),
                "--tool-refinement-min-new-trials",
                str(
                    _int(
                        config.get("tool_min_new_trials"),
                        TOOL_REFINEMENT_DEFAULTS.min_new_trials,
                    )
                ),
                "--tool-refinement-max-tools-per-update",
                str(
                    _int(
                        config.get("tool_max_tools_per_update"),
                        TOOL_REFINEMENT_DEFAULTS.max_tools_per_update,
                    )
                ),
                "--tool-refinement-publish-min-utility",
                _str(
                    config.get("tool_publish_min_utility"),
                    str(TOOL_REFINEMENT_DEFAULTS.publish_min_utility),
                ),
            ]
        )
    if procedural_memory_enabled:
        command.extend(
            [
                "--procedural-memory",
                _str(
                    config.get("procedural_memory_bank"),
                    default_library_id,
                ),
                "--procedural-memory-token-budget",
                str(
                    _int(
                        config.get(
                            "procedural_memory_token_budget",
                            config.get("procedural_memory_tokens"),
                        ),
                        PROCEDURAL_MEMORY_DEFAULTS.token_budget,
                    )
                ),
                "--procedural-memory-max-skill-age",
                str(
                    _int(
                        config.get("procedural_memory_max_skill_age"),
                        PROCEDURAL_MEMORY_DEFAULTS.max_skill_age,
                    )
                ),
                "--procedural-memory-pool-size",
                str(
                    _int(
                        config.get("procedural_memory_pool_size"),
                        PROCEDURAL_MEMORY_DEFAULTS.pool_size,
                    )
                ),
                "--procedural-memory-update-threshold",
                str(
                    _int(
                        config.get("procedural_memory_update_threshold"),
                        PROCEDURAL_MEMORY_DEFAULTS.evolution_threshold,
                    )
                ),
                "--procedural-memory-best-of-n",
                str(
                    _int(
                        config.get("procedural_memory_best_of_n"),
                        PROCEDURAL_MEMORY_DEFAULTS.best_of_n,
                    )
                ),
                "--procedural-memory-ppo-epsilon",
                _str(
                    config.get("procedural_memory_ppo_epsilon"),
                    str(PROCEDURAL_MEMORY_DEFAULTS.ppo_epsilon),
                ),
                "--procedural-memory-selection-epsilon",
                _str(
                    config.get("procedural_memory_selection_epsilon"),
                    str(PROCEDURAL_MEMORY_DEFAULTS.selection_epsilon),
                ),
                "--procedural-memory-experience-pool-size",
                str(
                    _int(
                        config.get("procedural_memory_experience_pool_size"),
                        PROCEDURAL_MEMORY_DEFAULTS.experience_pool_size,
                    )
                ),
                "--procedural-memory-baseline-ema-alpha",
                _str(
                    config.get("procedural_memory_baseline_ema_alpha"),
                    str(PROCEDURAL_MEMORY_DEFAULTS.baseline_ema_alpha),
                ),
                "--procedural-memory-selection-epsilon-decay-cases",
                str(
                    _int(
                        config.get("procedural_memory_selection_epsilon_decay_cases"),
                        PROCEDURAL_MEMORY_DEFAULTS.selection_epsilon_decay_cases,
                    )
                ),
                "--procedural-memory-acceptance-margin",
                _str(
                    config.get("procedural_memory_acceptance_margin"),
                    str(PROCEDURAL_MEMORY_DEFAULTS.acceptance_margin),
                ),
                "--procedural-memory-evolver-model",
                _str(
                    config.get("procedural_memory_evolver_model"),
                    PROCEDURAL_MEMORY_DEFAULTS.evolver_model,
                ),
                "--procedural-memory-policy-scorer-model",
                _str(
                    config.get("procedural_memory_policy_scorer_model"),
                    PROCEDURAL_MEMORY_DEFAULTS.policy_scorer_model,
                ),
                "--procedural-memory-verifier",
                _str(
                    config.get("procedural_memory_verifier"),
                    PROCEDURAL_MEMORY_DEFAULTS.verifier,
                ),
                "--procedural-memory-holdout-size",
                str(
                    _int(
                        config.get("procedural_memory_holdout_size"),
                        PROCEDURAL_MEMORY_DEFAULTS.holdout_size,
                    )
                ),
                "--procedural-memory-min-positive-advantage",
                str(
                    _int(
                        config.get("procedural_memory_min_positive_advantage"),
                        PROCEDURAL_MEMORY_DEFAULTS.min_positive_advantage,
                    )
                ),
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
    prepared["agent_type"] = agent_type(prepared)
    learning_benchmark_file = _str(
        prepared.get("learning_benchmark_file"),
        DEFAULT_STUDIO_LEARNING_BENCHMARK,
    )
    evaluate_benchmark_file = _str(
        prepared.get("evaluate_benchmark_file"),
        DEFAULT_STUDIO_EVALUATE_BENCHMARK,
    )
    prepared["learning_benchmark_file"] = learning_benchmark_file
    prepared["evaluate_benchmark_file"] = evaluate_benchmark_file
    run_id = _str(prepared.get("experiment_id"), "")
    if not run_id:
        run_id = next_experiment_id(evaluate_benchmark_file)
    prepared["experiment_id"] = run_id
    if not str(prepared.get("result_root") or "").strip():
        prepared["result_root"] = str(RESULTS_DIR / run_id)
    modules = selected_modules(prepared)
    if (
        "tool_refinement" in modules
        and not str(prepared.get("tool_library_id") or "").strip()
    ):
        prepared["tool_library_id"] = run_id
    if (
        "procedural_memory" in modules
        and not str(prepared.get("procedural_memory_bank") or "").strip()
    ):
        prepared["procedural_memory_bank"] = run_id
    return prepared


def prepare_resume_config(
    source_spec: dict[str, Any],
    *,
    resume_run_id: str | None = None,
) -> dict[str, Any]:
    config = dict(source_spec.get("config") or {})
    if not config:
        raise ValueError("Selected run does not have a resumable config.")
    if not all(
        str(config.get(key) or "").strip()
        for key in ("learning_benchmark_file", "evaluate_benchmark_file")
    ):
        raise ValueError(
            "Selected run predates the two-benchmark Studio schema and cannot "
            "be resumed. Start a new run instead."
        )
    source_run_id = str(
        source_spec.get("run_id") or config.get("experiment_id") or ""
    ).strip()
    if not source_run_id:
        raise ValueError("Selected run does not have a source run id.")

    if not str(config.get("result_root") or "").strip():
        config["result_root"] = str(RESULTS_DIR / source_run_id)
    modules = selected_modules(config)
    if (
        "tool_refinement" in modules
        and not str(config.get("tool_library_id") or "").strip()
    ):
        config["tool_library_id"] = source_run_id
    if (
        "procedural_memory" in modules
        and not str(config.get("procedural_memory_bank") or "").strip()
    ):
        config["procedural_memory_bank"] = source_run_id

    config["experiment_id"] = resume_run_id or source_run_id
    config["resume"] = True
    return prepare_experiment_config(config)


def create_run(config: dict[str, Any]) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    config = prepare_experiment_config(config)
    run_id = str(config["experiment_id"])
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    module_snapshot = run_dir / MODULE_CONFIG_SNAPSHOT_FILENAME
    module_snapshot.write_text(
        yaml.safe_dump(
            asdict(RESOLVED_MODULE_DEFAULTS),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    config["module_config_snapshot"] = str(module_snapshot)
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
        (run_dir / META_FILENAME).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
    else:
        log_handle = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "nika.visualization.experiment_runner",
                str(run_dir / SPEC_FILENAME),
            ],
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
        (run_dir / META_FILENAME).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

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
    module_snapshot = Path(
        str(config.get("module_config_snapshot") or "").strip()
        or run_dir / MODULE_CONFIG_SNAPSHOT_FILENAME
    )
    if not module_snapshot.exists():
        module_snapshot.write_text(
            yaml.safe_dump(
                asdict(RESOLVED_MODULE_DEFAULTS),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
    config["module_config_snapshot"] = str(module_snapshot)
    plan = build_command_plan(config)
    updated_spec = {
        **spec,
        "run_id": source_run_id,
        "created_at": spec.get("created_at") or datetime.now(timezone.utc).isoformat(),
        "resumed_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "commands": [asdict(item) for item in plan],
    }
    (run_dir / SPEC_FILENAME).write_text(
        json.dumps(updated_spec, indent=2), encoding="utf-8"
    )
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
        [
            sys.executable,
            "-u",
            "-m",
            "nika.visualization.experiment_runner",
            str(run_dir / SPEC_FILENAME),
        ],
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
            for line in status_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                if line.startswith("State:"):
                    if "zombie" in line.lower() or "defunct" in line.lower():
                        return False
    except Exception:
        pass

    return True


def _process_group_running(pgid: int | None) -> bool:
    if not pgid:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run_status(run_dir: Path) -> dict[str, Any]:
    meta = _read_json(run_dir / META_FILENAME)
    if meta.get("status") == "queued":
        return {**meta, "status": "queued", "exit_code": None}
    if meta.get("stop_requested"):
        return {**meta, "status": "running", "exit_code": None}
    if meta.get("status") == "stopped":
        return {**meta, "status": "stopped", "exit_code": None}
    if meta.get("status") == "failed":
        return {**meta, "status": "failed", "exit_code": 1}
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
        return {
            **meta,
            "status": "finished" if code == 0 else "failed",
            "exit_code": code,
        }
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
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "nika.visualization.experiment_runner",
                        str(spec_file),
                    ],
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
                (r / META_FILENAME).write_text(
                    json.dumps(meta, indent=2), encoding="utf-8"
                )
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
    meta["stop_requested"] = True
    meta["stop_requested_at"] = datetime.now(timezone.utc).isoformat()
    _write_run_meta(run_dir, meta)
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    grace_polls = max(1, int(_STOP_GRACE_SECONDS / _STOP_POLL_SECONDS))
    for _ in range(grace_polls):
        _pid_running(pid)
        if not _process_group_running(pid):
            break
        time.sleep(_STOP_POLL_SECONDS)
    _pid_running(pid)
    forced = False
    if _process_group_running(pid):
        forced = True
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        kill_polls = max(1, int(_STOP_KILL_GRACE_SECONDS / _STOP_POLL_SECONDS))
        for _ in range(kill_polls):
            _pid_running(pid)
            if not _process_group_running(pid):
                break
            time.sleep(_STOP_POLL_SECONDS)

    if _process_group_running(pid):
        meta["status"] = "failed"
        meta["cleanup_error"] = "Process group did not exit after SIGKILL"
        _write_run_meta(run_dir, meta)
        _append_run_log(
            run_dir,
            "ui_cleanup_failed "
            + json.dumps(
                {
                    "error_type": "ProcessExitError",
                    "error": meta["cleanup_error"],
                },
                ensure_ascii=False,
            ),
        )
        return

    if forced:
        try:
            clean_emulation_environment()
        except Exception as exc:
            meta["status"] = "failed"
            meta["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            _write_run_meta(run_dir, meta)
            _append_run_log(
                run_dir,
                "ui_cleanup_failed "
                + json.dumps(
                    {"error_type": type(exc).__name__, "error": str(exc)},
                    ensure_ascii=False,
                ),
            )
            return

    meta["status"] = "stopped"
    meta["stopped_at"] = datetime.now(timezone.utc).isoformat()
    meta.pop("stop_requested", None)
    _write_run_meta(run_dir, meta)
    _append_run_log(
        run_dir,
        "ui_run_stopped "
        + json.dumps({"reason": "user_stop", "forced": forced}, ensure_ascii=False),
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
        "benchmark_aborted ",
        "benchmark_summary ",
        "benchmark_stage_start ",
        "benchmark_stage_done ",
        "learning_barrier_created ",
        "learning_barrier_reused ",
        "benchmark_pipeline_blocked ",
        "benchmark_pipeline_done ",
    )
    json_events = {
        "benchmark_stage_start",
        "benchmark_stage_done",
        "learning_barrier_created",
        "learning_barrier_reused",
        "benchmark_pipeline_blocked",
        "benchmark_pipeline_done",
    }
    rows: list[dict[str, str]] = []
    for line in log_text.splitlines():
        prefix = next((item for item in prefixes if line.startswith(item)), None)
        if prefix is None:
            continue
        event = prefix.strip()
        rest = line[len(prefix) :]
        row: dict[str, str] = {"event": event, "raw": line}
        if prefix.startswith("ui_") or event in json_events:
            row.update(
                {
                    key: str(value)
                    for key, value in _parse_json_suffix(line, prefix).items()
                }
            )
        else:
            for part in rest.split():
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                row[key] = value
            # The upstream single-case runner historically emitted a second,
            # context-free completion line inside Studio's enriched batch event.
            # It has no case index, topology size, or inject parameters and must
            # not be rendered as a separate case.
            if event == "benchmark_done" and "index" not in row:
                continue
        rows.append(row)
    return rows


def run_spec_file(spec_path: str | Path) -> int:
    spec = _read_json(Path(spec_path))
    commands = spec.get("commands") or []
    config = spec.get("config") or {}
    exit_code = 0
    total = len(commands)
    stop_requested = False

    def _request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_sigterm = signal.signal(signal.SIGTERM, _request_stop)
    try:
        for index, item in enumerate(commands, start=1):
            name = str(item.get("name") or f"step-{index}")
            command = [str(part) for part in item.get("command") or []]
            print(
                "ui_step_start "
                + json.dumps(
                    {
                        "index": index,
                        "total": total,
                        "name": name,
                        "command": command,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            try:
                sub_env = os.environ.copy()
                sub_env["PYTHONUNBUFFERED"] = "1"
                module_snapshot = str(
                    config.get("module_config_snapshot") or ""
                ).strip()
                if module_snapshot:
                    sub_env[ENV_MODULE_CONFIG_PATH] = module_snapshot
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
                    {
                        "index": index,
                        "total": total,
                        "name": name,
                        "returncode": return_code,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if return_code != 0:
                exit_code = return_code
                break
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    print(
        "ui_run_done " + json.dumps({"exit_code": exit_code}, ensure_ascii=False),
        flush=True,
    )
    if not stop_requested:
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
