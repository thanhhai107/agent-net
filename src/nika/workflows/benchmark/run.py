"""Batch or single-case benchmark runs (env → inject → agent → eval)."""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from agent.composition import (
    AgentRunConfig,
    MemoryConfig,
    ToolEvolutionConfig,
    validate_agent_extensions,
)
from nika.config import BENCHMARK_DIR, RESULTS_DIR
from nika.net_env.net_env_pool import scenario_requires_topo_tier
from nika.utils.kathara_cleanup import ensure_kathara_clean
from nika.workflows.agent.run import start_agent
from nika.workflows.benchmark.inject_defaults import resolve_inject_params
from nika.workflows.benchmark.load_config import load_benchmark_yaml
from nika.workflows.env.start import start_net_env
from nika.workflows.eval.session import eval_results
from nika.workflows.failure.inject import inject_failure
from nika.workflows.session.close import close_session

_BENCHMARK_DONE_PREFIX = "benchmark_done "
_BENCHMARK_FAILED_PREFIX = "benchmark_failed "
_BENCHMARK_PROGRESS_PREFIX = "benchmark_progress "
_BENCHMARK_START_PREFIX = "benchmark_start "
_BENCHMARK_SUMMARY_PREFIX = "benchmark_summary "


def default_benchmark_yaml_path() -> str:
    return str(BENCHMARK_DIR / "benchmark_test.yaml")


def _slugify_benchmark_name(raw: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip(".-")
    return slug or "benchmark"


def _new_benchmark_results_root(benchmark_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    root = Path(RESULTS_DIR) / f"{_slugify_benchmark_name(benchmark_name)}-{timestamp}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def _stable_fault_seed(benchmark_name: str, row: dict) -> str:
    parts = [
        benchmark_name,
        str(row.get("scenario", "")),
        str(row.get("problem", "")),
        str(row.get("topo_size") or ""),
        repr(sorted((row.get("inject") or {}).items())),
        str(row.get("benchmark_index") or ""),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _benchmark_row_cli_args(
    row: dict,
    *,
    agent_type: str,
    llm_backend: str,
    model: str,
    max_steps: int,
    max_attempts: int,
    memory: MemoryConfig | None = None,
    run_judge: bool = False,
    judge_llm_backend: str | None = None,
    judge_model: str | None = None,
    tool_evolution: ToolEvolutionConfig | None = None,
    result_root: str | Path | None = None,
    fault_seed: str | None = None,
) -> list[str]:
    memory = memory or MemoryConfig()
    tool_evolution = tool_evolution or ToolEvolutionConfig()
    args = [
        row["scenario"],
        "--problem",
        row["problem"],
        "-a",
        agent_type,
        "-b",
        llm_backend,
        "-m",
        model,
        "-n",
        str(max_steps),
        "-r",
        str(max_attempts),
    ]
    if memory.mode == "evolve":
        args += [
            "--memory",
            memory.bank,
            "--memory-k",
            str(memory.top_k),
            "--memory-tokens",
            str(memory.token_budget),
        ]
    elif memory.mode == "read":
        args += [
            "--memory-read",
            memory.bank,
            "--memory-k",
            str(memory.top_k),
            "--memory-tokens",
            str(memory.token_budget),
        ]
    if tool_evolution.enabled:
        args += [
            "--tools",
            tool_evolution.library_id,
        ]
    topo = row.get("topo_size") or ""
    if topo and topo != "-":
        args += ["-t", topo]
    inject = row.get("inject") or {}
    for key, value in inject.items():
        args += ["--set", f"{key}={value}"]
    if run_judge:
        args += [
            "--judge",
            "--judge-backend",
            judge_llm_backend,
            "--judge-model",
            judge_model,
        ]
    if result_root is not None:
        args += ["--result-root", str(result_root)]
    if fault_seed is not None:
        args += ["--fault-seed", fault_seed]
    if row.get("benchmark_index") is not None:
        args += ["--benchmark-index", str(row["benchmark_index"])]
    return args


def _run_benchmark_row_subprocess(
    row: dict,
    *,
    agent_type: str,
    llm_backend: str,
    model: str,
    max_steps: int,
    max_attempts: int,
    memory: MemoryConfig,
    run_judge: bool,
    judge_llm_backend: str | None,
    judge_model: str | None,
    tool_evolution: ToolEvolutionConfig,
    result_root: str | Path | None = None,
    fault_seed: str | None = None,
) -> str | None:
    """Run one YAML case via a subprocess for isolated Tool Evolution execution."""
    cli_args = _benchmark_row_cli_args(
        row,
        agent_type=agent_type,
        llm_backend=llm_backend,
        model=model,
        max_steps=max_steps,
        max_attempts=max_attempts,
        memory=memory,
        run_judge=run_judge,
        judge_llm_backend=judge_llm_backend,
        judge_model=judge_model,
        tool_evolution=tool_evolution,
        result_root=result_root,
        fault_seed=fault_seed,
    )
    proc = subprocess.run(
        [sys.executable, "-m", "nika.codex_cli.main", "benchmark", "run", *cli_args],
        capture_output=True,
        text=True,
    )
    output = proc.stdout
    if proc.stderr:
        output += proc.stderr
    if proc.returncode != 0:
        scenario = row.get("scenario", "?")
        problem = row.get("problem", "?")
        raise RuntimeError(
            f"[{scenario}/{problem}] `nika benchmark run {' '.join(cli_args)}` "
            f"exited {proc.returncode}:\n{output}"
        )
    done_lines = [
        line for line in output.splitlines() if line.startswith(_BENCHMARK_DONE_PREFIX)
    ]
    if done_lines:
        print(done_lines[-1], flush=True)
        for part in done_lines[-1].split():
            if part.startswith("session_id="):
                return part.split("=", 1)[1]
    return None


def _progress_row_label(row: dict) -> str:
    topo = row.get("topo_size") or "-"
    return (
        f"scenario={row.get('scenario', '?')} problem={row.get('problem', '?')} "
        f"topo={topo}"
    )


def run_single_benchmark(
    problem: str,
    scenario: str,
    topo_size: str,
    agent_type: str,
    llm_backend: str,
    model: str,
    max_steps: int,
    max_attempts: int = 3,
    *,
    memory: MemoryConfig | None = None,
    run_judge: bool = False,
    judge_llm_backend: str | None = None,
    judge_model: str | None = None,
    tool_evolution: ToolEvolutionConfig | None = None,
    result_root: str | Path | None = None,
    fault_seed: str | None = None,
    benchmark_index: int | None = None,
    inject_params: dict[str, str] | None = None,
) -> str:
    """
    Run a single benchmark case.

    Returns:
        The session id for the completed run.
    """
    ensure_kathara_clean(context="benchmark case")
    print(
        f"Running benchmark for Problem: {problem}, Scenario: {scenario}, Topo Size: {topo_size}"
    )
    memory = memory or MemoryConfig()
    tool_evolution = tool_evolution or ToolEvolutionConfig()
    validate_agent_extensions(
        AgentRunConfig(
            agent_type=agent_type,
            llm_backend=llm_backend,
            model=model,
            max_steps=max_steps,
            max_attempts=max_attempts,
            tool_evolution=tool_evolution,
            memory=memory,
        )
    )

    tier = topo_size if topo_size else None
    if scenario_requires_topo_tier(scenario) and not tier:
        raise ValueError(
            f"Scenario '{scenario}' requires a non-empty topology tier (-t s|m|l)."
        )
    if not scenario_requires_topo_tier(scenario):
        tier = None

    session_id = start_net_env(scenario, tier, redeploy=True, results_root=result_root)
    from nika.utils.session import Session

    session = Session().load_running_session(session_id=session_id)
    session_dir = Path(
        getattr(session, "session_dir", Path(result_root or RESULTS_DIR) / session_id)
    )
    if fault_seed is not None:
        session.update_session("fault_seed", fault_seed)
    if benchmark_index is not None:
        session.update_session("benchmark_index", benchmark_index)

    params = dict(inject_params or resolve_inject_params(problem, scenario, topo_size or ""))
    inject_failure(
        problem_names=[problem],
        session_id=session_id,
        param_overrides=params,
    )
    session = Session().load_running_session(session_id=session_id)
    session_dir = Path(
        getattr(session, "session_dir", Path(result_root or RESULTS_DIR) / session_id)
    )

    try:
        start_agent(
            AgentRunConfig(
                agent_type=agent_type,
                llm_backend=llm_backend,
                model=model,
                max_steps=max_steps,
                max_attempts=max_attempts,
                stream_output=False,
                tool_evolution=tool_evolution,
                memory=memory,
            ),
            session_id=session_id,
        )
        eval_results(
            session_id=session_id,
            run_judge=run_judge,
            judge_llm_backend=judge_llm_backend,
            judge_model=judge_model,
        )
    except Exception:
        try:
            close_session(session_id=session_id, undeploy=True)
        except (FileNotFoundError, ValueError):
            pass
        raise

    print(
        f"{_BENCHMARK_DONE_PREFIX}session_id={session_id} scenario={scenario} "
        f"problem={problem} session_dir={session_dir}"
    )
    return session_id


def run_benchmark_from_yaml(
    benchmark_file: str,
    agent_type: str,
    llm_backend: str,
    model: str,
    max_steps: int,
    max_attempts: int = 3,
    *,
    memory: MemoryConfig | None = None,
    run_judge: bool = False,
    judge_llm_backend: str | None = None,
    judge_model: str | None = None,
    tool_evolution: ToolEvolutionConfig | None = None,
    result_root: str | Path | None = None,
) -> None:
    """
    Run benchmark cases defined in a YAML file.

    The YAML file must contain a top-level ``cases`` list. Each case includes
    ``scenario``, ``problem``, optional ``topo_size``, and deterministic
    ``inject`` parameters.
    """
    memory_config = memory or MemoryConfig()
    tool_config = tool_evolution or ToolEvolutionConfig()
    validate_agent_extensions(
        AgentRunConfig(
            agent_type=agent_type,
            llm_backend=llm_backend,
            model=model,
            max_steps=max_steps,
            max_attempts=max_attempts,
            tool_evolution=tool_config,
            memory=memory_config,
        )
    )
    rows = load_benchmark_yaml(benchmark_file)

    if not rows:
        print(f"No benchmark cases found in {benchmark_file}")
        return
    benchmark_name = Path(benchmark_file).stem
    benchmark_root = (
        Path(result_root)
        if result_root is not None
        else _new_benchmark_results_root(benchmark_name)
    )
    benchmark_root.mkdir(parents=True, exist_ok=True)
    if tool_config.enabled:
        print(f"tool_evolution_library id={tool_config.library_id}")

    total = len(rows)
    prepared_rows: list[dict] = []
    for index, raw_row in enumerate(rows):
        row = dict(raw_row)
        row["benchmark_index"] = str(index)
        if not row.get("fault_seed"):
            row["fault_seed"] = _stable_fault_seed(benchmark_name, row)
        prepared_rows.append(row)
    completed = 0
    failed = 0
    batch_started_at = time.monotonic()
    print(
        f"{_BENCHMARK_PROGRESS_PREFIX}total={total} completed=0 failed=0 "
        f"yaml={benchmark_file} result_root={benchmark_root}",
        flush=True,
    )

    for index, row in enumerate(prepared_rows):
        case_started_at = time.monotonic()
        print(
            f"{_BENCHMARK_START_PREFIX}index={index + 1}/{total} "
            f"{_progress_row_label(row)}",
            flush=True,
        )
        try:
            if tool_config.enabled:
                session_id = _run_benchmark_row_subprocess(
                    row,
                    agent_type=agent_type,
                    llm_backend=llm_backend,
                    model=model,
                    max_steps=max_steps,
                    max_attempts=max_attempts,
                    memory=memory_config,
                    run_judge=run_judge,
                    judge_llm_backend=judge_llm_backend,
                    judge_model=judge_model,
                    tool_evolution=tool_config,
                    result_root=benchmark_root,
                    fault_seed=row.get("fault_seed"),
                )
            else:
                session_id = run_single_benchmark(
                    problem=row["problem"],
                    scenario=row["scenario"],
                    topo_size=(
                        ""
                        if (row.get("topo_size") or "") == "-"
                        else (row.get("topo_size") or "")
                    ),
                    agent_type=agent_type,
                    llm_backend=llm_backend,
                    model=model,
                    max_steps=max_steps,
                    max_attempts=max_attempts,
                    memory=memory_config,
                    run_judge=run_judge,
                    judge_llm_backend=judge_llm_backend,
                    judge_model=judge_model,
                    tool_evolution=tool_config,
                    result_root=benchmark_root,
                    fault_seed=row.get("fault_seed"),
                    inject_params=row.get("inject"),
                    benchmark_index=int(row["benchmark_index"]),
                )
        except Exception as exc:
            failed += 1
            elapsed = time.monotonic() - case_started_at
            error = str(exc).replace("\n", " ")[:500]
            print(
                f"{_BENCHMARK_FAILED_PREFIX}index={index + 1}/{total} "
                f"completed={completed} failed={failed} elapsed_sec={elapsed:.1f} "
                f"error_type={type(exc).__name__} error={error!r} "
                f"{_progress_row_label(row)}",
                flush=True,
            )
            raise
        completed += 1
        elapsed = time.monotonic() - case_started_at
        print(
            f"{_BENCHMARK_PROGRESS_PREFIX}index={index + 1}/{total} "
            f"completed={completed} failed={failed} elapsed_sec={elapsed:.1f} "
            f"session_id={session_id or '-'} {_progress_row_label(row)}",
            flush=True,
        )
    total_elapsed = time.monotonic() - batch_started_at
    print(
        f"{_BENCHMARK_SUMMARY_PREFIX}total={total} completed={completed} "
        f"failed={failed} elapsed_sec={total_elapsed:.1f}",
        flush=True,
    )
