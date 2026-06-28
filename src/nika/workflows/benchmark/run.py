"""Batch or single-case benchmark runs (env → inject → agent → eval)."""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from nika.config import BENCHMARK_DIR, RESULTS_DIR
from nika.net_env.net_env_pool import scenario_requires_topo_size
from nika.workflows.agent.run import start_agent
from nika.workflows.benchmark.inject_defaults import resolve_inject_params
from nika.workflows.benchmark.load_config import load_benchmark_yaml
from nika.workflows.env.start import start_net_env
from nika.workflows.eval.session import eval_results
from nika.workflows.failure.inject import inject_failure

_BENCHMARK_DONE_PREFIX = "benchmark_done "


def default_benchmark_yaml_path() -> str:
    return str(BENCHMARK_DIR / "benchmark_selected.yaml")


def _benchmark_row_cli_args(
    row: dict,
    *,
    agent_type: str,
    llm_provider: str | None,
    model: str | None,
    max_steps: int,
    run_judge: bool,
    judge_llm_provider: str | None,
    judge_model: str | None,
) -> list[str]:
    args = [
        row["scenario"],
        "--problem",
        row["problem"],
        "-a",
        agent_type,
    ]
    if llm_provider:
        args += ["-p", llm_provider]
    if model:
        args += ["-m", model]
    args += [
        "-n",
        str(max_steps),
    ]
    topo = row.get("topo_size") or ""
    if topo:
        args += ["-s", topo]
    inject = row.get("inject") or {}
    for key, value in inject.items():
        args += ["--set", f"{key}={value}"]
    if run_judge:
        args += ["--judge", "--judge-provider", judge_llm_provider, "--judge-model", judge_model]
    return args


def _run_benchmark_row_subprocess(
    row: dict,
    *,
    agent_type: str,
    llm_provider: str | None,
    model: str | None,
    max_steps: int,
    run_judge: bool,
    judge_llm_provider: str | None,
    judge_model: str | None,
) -> None:
    """Run one YAML row via a subprocess for thread-safe parallel batch execution."""
    cli_args = _benchmark_row_cli_args(
        row,
        agent_type=agent_type,
        llm_provider=llm_provider,
        model=model,
        max_steps=max_steps,
        run_judge=run_judge,
        judge_llm_provider=judge_llm_provider,
        judge_model=judge_model,
    )
    proc = subprocess.run(
        [sys.executable, "-m", "nika.cli.main", "benchmark", "run", *cli_args],
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
    if output:
        print(output, end="" if output.endswith("\n") else "\n")


def _run_benchmark_batch_parallel(
    rows: list[dict],
    *,
    agent_type: str,
    llm_provider: str | None,
    model: str | None,
    max_steps: int,
    run_judge: bool,
    judge_llm_provider: str | None,
    judge_model: str | None,
) -> None:
    """Run all rows in *rows* simultaneously (one subprocess each), then return."""
    with ThreadPoolExecutor(max_workers=len(rows)) as pool:
        futures = [
            pool.submit(
                _run_benchmark_row_subprocess,
                row,
                agent_type=agent_type,
                llm_provider=llm_provider,
                model=model,
                max_steps=max_steps,
                run_judge=run_judge,
                judge_llm_provider=judge_llm_provider,
                judge_model=judge_model,
            )
            for row in rows
        ]
        for future in as_completed(futures):
            future.result()


def run_single_benchmark(
    problem: str,
    scenario: str,
    topo_size: str,
    agent_type: str,
    llm_provider: str | None,
    model: str | None,
    max_steps: int,
    *,
    inject_params: dict[str, str] | None = None,
    run_judge: bool = False,
    judge_llm_provider: str | None = None,
    judge_model: str | None = None,
) -> str:
    """Run a single benchmark case.

    Returns:
        The session id for the completed run.
    """
    print(f"Running benchmark for Problem: {problem}, Scenario: {scenario}, Topo Size: {topo_size}")

    size = topo_size if topo_size else None
    if scenario_requires_topo_size(scenario) and not size:
        raise ValueError(f"Scenario '{scenario}' requires a non-empty topology size (-s s|m|l).")
    if not scenario_requires_topo_size(scenario):
        size = None

    params = dict(inject_params or resolve_inject_params(problem, scenario, topo_size or ""))

    session_id = start_net_env(scenario, size, redeploy=True)
    session_dir = Path(RESULTS_DIR) / session_id

    inject_failure(problem_names=[problem], session_id=session_id, param_overrides=params)

    start_agent(
        agent_type=agent_type,
        llm_provider=llm_provider,
        model=model,
        max_steps=max_steps,
        session_id=session_id,
        stream_output=False,
    )

    eval_results(
        session_id=session_id,
        run_judge=run_judge,
        judge_llm_provider=judge_llm_provider,
        judge_model=judge_model,
    )

    print(
        f"{_BENCHMARK_DONE_PREFIX}session_id={session_id} scenario={scenario} "
        f"problem={problem} session_dir={session_dir}"
    )
    return session_id


def run_benchmark_from_yaml(
    benchmark_file: str,
    agent_type: str,
    llm_provider: str | None,
    model: str | None,
    max_steps: int,
    *,
    batch_size: int = 1,
    run_judge: bool = False,
    judge_llm_provider: str | None = None,
    judge_model: str | None = None,
) -> None:
    """
    Run benchmark cases defined in a YAML file.

    Each case must include scenario, problem, optional topo_size, and inject params.

    When ``batch_size == 1`` (default) rows are executed sequentially.
    When ``batch_size > 1`` rows are chunked into groups of that size; every row in a
    chunk runs in parallel (one subprocess each) and the next chunk starts only after
    all rows in the current chunk have finished.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    rows = load_benchmark_yaml(benchmark_file)

    if not rows:
        print(f"No benchmark rows found in {benchmark_file}")
        return

    _shared_kwargs = dict(
        agent_type=agent_type,
        llm_provider=llm_provider,
        model=model,
        max_steps=max_steps,
        run_judge=run_judge,
        judge_llm_provider=judge_llm_provider,
        judge_model=judge_model,
    )

    if batch_size == 1:
        for row in rows:
            run_single_benchmark(
                problem=row["problem"],
                scenario=row["scenario"],
                topo_size=row.get("topo_size") or "",
                inject_params=row.get("inject"),
                **_shared_kwargs,
            )
        return

    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        print(
            f"[batch {i // batch_size + 1}] running {len(chunk)} session(s) in parallel "
            f"(rows {i + 1}–{i + len(chunk)} of {len(rows)})"
        )
        _run_benchmark_batch_parallel(chunk, **_shared_kwargs)
