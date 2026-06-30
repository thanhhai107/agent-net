"""Batch or single-case benchmark runs (env → inject → agent → eval)."""

from __future__ import annotations

import csv
import hashlib
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from agent.composition import (
    AgentRunConfig,
    HarnessConfig,
    MemoryConfig,
    ToolEvolutionConfig,
    validate_agent_extensions,
)
from agent.harness.dataset import build_public_case_dataset
from agent.harness.runner import HarnessExecutionConfig, run_harness_target
from nika.config import BENCHMARK_DIR, RESULTS_DIR
from nika.net_env.net_env_pool import scenario_requires_topo_tier
from nika.utils.kathara_cleanup import ensure_kathara_clean
from nika.utils.logger import bind_session_dir, log_event
from nika.workflows.agent.run import start_agent
from nika.workflows.env.start import start_net_env
from nika.workflows.eval.session import eval_results
from nika.workflows.failure.inject import inject_failure
from nika.workflows.session.close import close_session

_BENCHMARK_DONE_PREFIX = "benchmark_done "
_BENCHMARK_FAILED_PREFIX = "benchmark_failed "
_BENCHMARK_PROGRESS_PREFIX = "benchmark_progress "
_BENCHMARK_START_PREFIX = "benchmark_start "
_BENCHMARK_SUMMARY_PREFIX = "benchmark_summary "
_REQUIRED_CSV_COLUMNS = {"problem", "scenario", "topo_size"}
_REMOVED_CSV_METADATA_COLUMNS = {"stream_id", "stream", "split", "sequence_index"}


def default_benchmark_csv_path() -> str:
    return str(BENCHMARK_DIR / "benchmark_test.csv")


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
    oracle_routing: bool = False,
    tool_evolution: ToolEvolutionConfig | None = None,
    harness: HarnessConfig | None = None,
    harness_allow_failure: bool = False,
    result_root: str | Path | None = None,
    fault_seed: str | None = None,
) -> list[str]:
    memory = memory or MemoryConfig()
    tool_evolution = tool_evolution or ToolEvolutionConfig()
    harness = harness or HarnessConfig()
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
    if oracle_routing:
        args.append("--oracle-routing")
    if tool_evolution.enabled:
        args += [
            "--tools",
            tool_evolution.library_id,
            "--tool-mode",
            tool_evolution.mode,
        ]
    topo = row.get("topo_size") or ""
    if topo and topo != "-":
        args += ["-t", topo]
    if run_judge:
        args += [
            "--judge",
            "--judge-backend",
            judge_llm_backend,
            "--judge-model",
            judge_model,
        ]
    if harness.enabled:
        args += ["--harness", str(harness.target_agent_path)]
    if harness_allow_failure:
        args.append("--harness-allow-failure")
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
    oracle_routing: bool,
    tool_evolution: ToolEvolutionConfig,
    harness: HarnessConfig,
    harness_allow_failure: bool,
    result_root: str | Path | None = None,
    fault_seed: str | None = None,
) -> str | None:
    """Run one CSV row via a subprocess for isolated Tool Evolution execution."""
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
        oracle_routing=oracle_routing,
        tool_evolution=tool_evolution,
        harness=harness,
        harness_allow_failure=harness_allow_failure,
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
    oracle_routing: bool = False,
    tool_evolution: ToolEvolutionConfig | None = None,
    harness: HarnessConfig | None = None,
    harness_allow_failure: bool = False,
    result_root: str | Path | None = None,
    fault_seed: str | None = None,
    benchmark_index: int | None = None,
) -> str:
    """
    Run a single benchmark case.

    Returns:
        The session id for the completed run.
    """
    print(
        f"Running benchmark for Problem: {problem}, Scenario: {scenario}, Topo Size: {topo_size}"
    )
    memory = memory or MemoryConfig()
    tool_evolution = tool_evolution or ToolEvolutionConfig()
    harness = harness or HarnessConfig()
    if not harness.enabled and agent_type.lower() == "harness":
        raise ValueError(
            "agent_type 'harness' requires the internal --harness target_agent.py path."
        )
    if not harness.enabled:
        validate_agent_extensions(
            AgentRunConfig(
                agent_type=agent_type,
                llm_backend=llm_backend,
                model=model,
                max_steps=max_steps,
                max_attempts=max_attempts,
                oracle_routing=oracle_routing,
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

    inject_failure(problem_names=[problem], session_id=session_id)
    session = Session().load_running_session(session_id=session_id)
    session_dir = Path(
        getattr(session, "session_dir", Path(result_root or RESULTS_DIR) / session_id)
    )

    try:
        if harness.enabled:
            session.update_session("agent_type", "harness")
            session.update_session("harness_target_agent_path", harness.target_agent_path)
            session.update_session("llm_backend", llm_backend)
            session.update_session("model", model)
            session.update_session("max_steps", max_steps)
            session.update_session("memory_mode", memory.mode)
            if memory.enabled:
                session.update_session("memory_bank", memory.bank)
                session.update_session("memory_top_k", memory.top_k)
                session.update_session("memory_token_budget", memory.token_budget)
            session.update_session("oracle_routing", oracle_routing)
            session.update_session("tool_evolution_enabled", tool_evolution.enabled)
            if tool_evolution.enabled:
                session.update_session("tool_library_id", tool_evolution.library_id)
                session.update_session("tool_evolution_mode", tool_evolution.mode)
            session.start_session()
            bind_session_dir(session.session_dir)
            log_event(
                "harness_start",
                f"Starting harness target agent in session {session_id}",
                session_id=session_id,
                target_agent_path=harness.target_agent_path,
                model=model,
            )
            try:
                dataset_dir = build_public_case_dataset(
                    session=session,
                    output_dir=session_dir / "public_case",
                    memory=memory,
                    tool_evolution=tool_evolution,
                    llm_backend=llm_backend,
                    model=model,
                )
                result = run_harness_target(
                    HarnessExecutionConfig(
                        target_agent_path=str(harness.target_agent_path),
                        session_id=session_id,
                        dataset_dir=dataset_dir,
                        working_dir=session_dir,
                        llm_backend=llm_backend,
                        model=model,
                        max_steps=max_steps,
                        allow_failure=harness_allow_failure,
                    )
                )
                session.update_session("harness_returncode", result.returncode)
            finally:
                session.end_session()
                log_event(
                    "harness_end",
                    f"Harness target agent ended for session {session_id}",
                    session_id=session_id,
                )
        else:
            start_agent(
                AgentRunConfig(
                    agent_type=agent_type,
                    llm_backend=llm_backend,
                    model=model,
                    max_steps=max_steps,
                    max_attempts=max_attempts,
                    stream_output=False,
                    oracle_routing=oracle_routing,
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


def run_benchmark_from_csv(
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
    oracle_routing: bool = False,
    tool_evolution: ToolEvolutionConfig | None = None,
    harness: HarnessConfig | None = None,
    harness_allow_failure: bool = False,
    result_root: str | Path | None = None,
) -> None:
    """
    Run benchmark cases defined in a CSV file.

    The CSV file must contain the following columns:
    - problem
    - scenario
    - topo_size (same values as ``nika env run -t``: s, m, l, or empty)
    """
    ensure_kathara_clean(context="benchmark run")

    memory_config = memory or MemoryConfig()
    tool_config = tool_evolution or ToolEvolutionConfig()
    harness_config = harness or HarnessConfig()
    if not harness_config.enabled and agent_type.lower() == "harness":
        raise ValueError(
            "agent_type 'harness' requires the internal --harness target_agent.py path."
        )
    if not harness_config.enabled:
        validate_agent_extensions(
            AgentRunConfig(
                agent_type=agent_type,
                llm_backend=llm_backend,
                model=model,
                max_steps=max_steps,
                max_attempts=max_attempts,
                oracle_routing=oracle_routing,
                tool_evolution=tool_config,
                memory=memory_config,
            )
        )
    with open(benchmark_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = _REQUIRED_CSV_COLUMNS - fieldnames
        if missing:
            raise ValueError(
                "benchmark CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )
        removed = _REMOVED_CSV_METADATA_COLUMNS & fieldnames
        if removed:
            raise ValueError(
                "benchmark CSV metadata columns are no longer supported: "
                + ", ".join(sorted(removed))
                + ". Keep only problem, scenario, topo_size."
            )
        rows = list(reader)

    if not rows:
        print(f"No benchmark rows found in {benchmark_file}")
        return
    benchmark_name = Path(benchmark_file).stem
    benchmark_root = (
        Path(result_root)
        if result_root is not None
        else _new_benchmark_results_root(benchmark_name)
    )
    benchmark_root.mkdir(parents=True, exist_ok=True)
    if tool_config.enabled and tool_config.library_id == "default":
        tool_config = ToolEvolutionConfig(
            enabled=True,
            library_id=f"{Path(benchmark_file).stem}-{uuid.uuid4().hex[:8]}",
            mode=tool_config.mode,
        )
    if tool_config.enabled:
        print(
            f"tool_evolution_library id={tool_config.library_id} "
            f"mode={tool_config.mode}"
        )

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
        f"csv={benchmark_file} result_root={benchmark_root}",
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
                    oracle_routing=oracle_routing,
                    tool_evolution=tool_config,
                    harness=harness_config,
                    harness_allow_failure=harness_allow_failure,
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
                    oracle_routing=oracle_routing,
                    tool_evolution=tool_config,
                    harness=harness_config,
                    harness_allow_failure=harness_allow_failure,
                    result_root=benchmark_root,
                    fault_seed=row.get("fault_seed"),
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
