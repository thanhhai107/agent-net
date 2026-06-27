"""Batch or single-case benchmark runs (env → inject → agent → eval)."""

from __future__ import annotations

import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from nika.config import BENCHMARK_DIR, RESULTS_DIR
from nika.net_env.net_env_pool import scenario_requires_topo_size
from nika.workflows.agent.run import _resolve_agent_model, start_agent
from nika.workflows.env.start import start_net_env
from nika.workflows.eval.session import eval_results
from nika.workflows.failure.inject import inject_failure

_BENCHMARK_DONE_PREFIX = "benchmark_done "


def default_benchmark_csv_path() -> str:
    return str(BENCHMARK_DIR / "benchmark_selected.csv")


def _benchmark_row_cli_args(
    row: dict,
    *,
    agent_type: str,
    llm_provider: str,
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
        "-p",
        llm_provider,
    ]
    if model:
        args += ["-m", model]
    args += [
        "-n",
        str(max_steps),
    ]
    topo = row.get("topo_size") or ""
    if topo:
        args += ["-t", topo]
    if run_judge:
        args += ["--judge", "--judge-provider", judge_llm_provider, "--judge-model", judge_model]
    return args


def _run_benchmark_row_subprocess(
    row: dict,
    *,
    agent_type: str,
    llm_provider: str,
    model: str | None,
    max_steps: int,
    run_judge: bool,
    judge_llm_provider: str | None,
    judge_model: str | None,
) -> None:
    """Run one CSV row via a subprocess for thread-safe parallel batch execution."""
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
    if output:
        print(output, end="" if output.endswith("\n") else "\n")


def _run_benchmark_batch_parallel(
    rows: list[dict],
    *,
    agent_type: str,
    llm_provider: str,
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
    llm_provider: str,
    model: str | None,
    max_steps: int,
    *,
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

    session_id = start_net_env(scenario, size, redeploy=True)
    session_dir = Path(RESULTS_DIR) / session_id

    inject_failure(problem_names=[problem], session_id=session_id)

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


def run_benchmark_from_csv(
    benchmark_file: str,
    agent_type: str,
    llm_provider: str,
    model: str | None,
    max_steps: int,
    *,
    batch_size: int = 1,
    run_judge: bool = False,
    judge_llm_provider: str | None = None,
    judge_model: str | None = None,
) -> None:
    """
    Run benchmark cases defined in a CSV file.

    The CSV file must contain the following columns:
    - problem
    - scenario
    - topo_size (same values as ``nika env run -t``: s, m, l, or empty)

    When ``batch_size == 1`` (default) rows are executed sequentially.
    When ``batch_size > 1`` rows are chunked into groups of that size; every row in a
    chunk runs in parallel (one subprocess each) and the next chunk starts only after
    all rows in the current chunk have finished.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    with open(benchmark_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

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
