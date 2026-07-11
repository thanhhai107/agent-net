"""Custom benchmark controls around the unmodified NIKA pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import textwrap
from pathlib import Path
from typing import Any

import yaml

from agent.composition import AgentRunConfig, MemoryConfig, ToolEvolutionConfig
from agent.extensions.config import default_llm_provider, default_model
from agent.extensions.react_agent import configure_custom_provider_environment
from agent.extensions.run import start_agent
from agent.memory.workflow import evolve_session_memory
from agent.tool_evolution.curator import finalize_tool_evolution_session
from nika.evaluator.result_log import EVAL_METRICS_FILENAME
from nika.net_env.net_env_pool import (
    get_net_env_instance,
    scenario_backend,
    scenario_requires_topo_size,
)
from nika.utils.session import Session
from nika.utils.session_store import SessionStore
from nika.workflows.benchmark.resume import (
    benchmark_row_fingerprint,
    scan_benchmark_cases,
)
from nika.workflows.benchmark.run import run_single_case
from nika.workflows.env.start import start_net_env
from nika.workflows.eval.session import eval_results
from nika.workflows.failure.inject import inject_failure
from nika.workflows.session.close import close_session

NO_FAULT_NAMES = frozenset({"no_fault", "clean", "normal", "healthy", "none"})


def is_no_fault(problem: str) -> bool:
    return problem.strip().lower() in NO_FAULT_NAMES


def load_custom_benchmark(path: str | Path) -> list[dict[str, Any]]:
    """Load NIKA rows while allowing an empty inject map for clean controls."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cases = data.get("cases") if isinstance(data, dict) else None
    if not isinstance(cases, list):
        raise ValueError(f"Invalid benchmark YAML (missing list 'cases'): {path}")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(cases):
        if not isinstance(raw, dict) or not raw.get("scenario") or not raw.get("problem"):
            raise ValueError(f"Benchmark case {index} requires scenario and problem")
        inject = raw.get("inject") or {}
        if not isinstance(inject, dict):
            raise ValueError(f"Benchmark case {index} inject must be a mapping")
        problem = str(raw["problem"])
        if not inject and not is_no_fault(problem):
            raise ValueError(f"Benchmark case {index} requires non-empty inject params")
        rows.append(
            {
                "scenario": str(raw["scenario"]),
                "problem": problem,
                "topo_size": str(raw.get("topo_size") or ""),
                "inject": {str(key): str(value) for key, value in inject.items()},
            }
        )
    return rows


def _clean_task_description(session: Session) -> str:
    net_env = get_net_env_instance(
        session.scenario_name,
        backend=scenario_backend(session.scenario_name),
        topo_size=session.scenario_topo_size,
        lab_name=session.lab_name,
    )
    return textwrap.dedent(
        f"""\
        You are provided with the following network description and its current state:
        {net_env.get_info()}

        Your goal is to analyze the network condition and, if needed, use the available tools.
        You need to generate a troubleshooting diagnosis report.
        The report should reflect your assessment of the network's health,
        indicate any abnormal behavior you identify, and describe relevant
        findings based on your analysis.

        Focus on producing an informative and coherent diagnostic report
        derived from the network state.
        Do not need to propose any solutions or remediation steps at this stage.
        """
    ).strip()


def _prepare_clean_control(session: Session) -> None:
    session.update_session("problem_names", ["no_fault"])
    session.update_session("root_cause_category", "none")
    session.update_session("task_description", _clean_task_description(session))
    session.write_gt(
        {"is_anomaly": False, "faulty_devices": [], "root_cause_name": []}
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _update_learning(session_id: str, config: AgentRunConfig) -> None:
    session = Session().load_closed_session(session_id=session_id)
    session_dir = Path(session.session_dir)
    metrics = _read_json(session_dir / EVAL_METRICS_FILENAME)
    run_meta = _read_json(session_dir / "run.json")
    if config.tool_evolution.enabled:
        finalize_tool_evolution_session(session_id=session_id, metrics=metrics)
    if config.memory.mode == "evolve":
        asyncio.run(
            evolve_session_memory(
                run_meta=run_meta,
                metrics=metrics,
                session_dir=session_dir,
            )
        )


def run_extended_case(
    row: dict[str, Any],
    *,
    config: AgentRunConfig,
    result_dir: str | None,
    run_judge: bool = False,
    judge_provider: str | None = None,
    judge_model: str | None = None,
) -> tuple[str, Path]:
    """Run a clean control or a learning-enabled row on upstream primitives."""
    scenario = row["scenario"]
    topo_size = row.get("topo_size") or ""
    size = topo_size or None
    if scenario_requires_topo_size(scenario) and size is None:
        raise ValueError(f"Scenario {scenario!r} requires topology size")
    if not scenario_requires_topo_size(scenario):
        size = None

    session_id = start_net_env(
        scenario,
        size,
        redeploy=True,
        result_dir=result_dir,
    )
    session = Session().load_running_session(session_id=session_id)
    session_dir = Path(SessionStore().get_session(session_id)["session_dir"])
    try:
        if is_no_fault(row["problem"]):
            _prepare_clean_control(session)
        else:
            inject_failure(
                problem_names=[row["problem"]],
                session_id=session_id,
                param_overrides=row["inject"],
            )
            session = Session().load_running_session(session_id=session_id)
        session.update_session(
            "benchmark_fingerprint", benchmark_row_fingerprint(row)
        )
        start_agent(config, session_id=session_id)
        eval_results(
            session_id=session_id,
            run_judge=run_judge,
            judge_llm_provider=judge_provider,
            judge_model=judge_model,
        )
        _update_learning(session_id, config)
    except Exception:
        try:
            close_session(session_id=session_id, undeploy=True)
        except (FileNotFoundError, ValueError):
            pass
        raise
    return session_id, session_dir


def run_batch(args: argparse.Namespace) -> int:
    rows = load_custom_benchmark(args.config)
    memory_mode = "evolve" if args.memory else "read" if args.memory_read else "off"
    config = AgentRunConfig(
        agent_type="byo.langgraph",
        llm_provider=args.provider,
        model=args.model,
        max_steps=args.max_steps,
        memory=MemoryConfig(
            mode=memory_mode,
            bank=args.memory or args.memory_read or "default",
            top_k=args.memory_k,
            token_budget=args.memory_tokens,
            max_skill_age=args.memory_max_skill_age,
            pool_size=args.memory_pool_size,
            evolution_threshold=args.memory_evolution_threshold,
            best_of_n=args.memory_best_of_n,
            ppo_epsilon=args.memory_ppo_epsilon,
        ),
        tool_evolution=ToolEvolutionConfig(
            enabled=bool(args.tools),
            library_id=args.tools or "default",
            tool_doc_chars=args.tool_doc_chars,
            convergence_threshold=args.tool_convergence_threshold,
        ),
    )
    _root, pending = scan_benchmark_cases(
        rows=rows,
        result_dir=args.result_dir,
        resume=args.resume,
    )
    failed = 0
    completed = len(rows) - len(pending)
    for index in pending:
        row = rows[index]
        label = (
            f"index={index + 1}/{len(rows)} scenario={row['scenario']} "
            f"topo_size={row.get('topo_size') or '-'} problem={row['problem']}"
        )
        inject_label = " ".join(
            f"inject_{key}={value}" for key, value in sorted(row["inject"].items())
        )
        print(f"benchmark_start {label} {inject_label}".rstrip(), flush=True)
        try:
            if config.extensions_enabled or is_no_fault(row["problem"]):
                session_id, session_dir = run_extended_case(
                    row,
                    config=config,
                    result_dir=args.result_dir,
                    run_judge=args.judge,
                    judge_provider=args.judge_provider,
                    judge_model=args.judge_model,
                )
            else:
                session_id, session_dir = run_single_case(
                    problem=row["problem"],
                    scenario=row["scenario"],
                    topo_size=row.get("topo_size") or "",
                    agent_type="byo.langgraph",
                    llm_provider=args.provider,
                    model=args.model,
                    max_steps=args.max_steps,
                    inject_params=row["inject"],
                    run_judge=args.judge,
                    judge_llm_provider=args.judge_provider,
                    judge_model=args.judge_model,
                    result_dir=args.result_dir,
                )
            completed += 1
            print(
                f"benchmark_done {label} session_id={session_id} session_dir={session_dir}",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            print(
                f"benchmark_failed {label} error_type={type(exc).__name__} error={exc}",
                flush=True,
            )
    print(
        f"benchmark_summary total={len(rows)} completed={completed} failed={failed}",
        flush=True,
    )
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--provider", default=default_llm_provider())
    parser.add_argument("--model", default=default_model())
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--result-dir")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-provider")
    parser.add_argument("--judge-model")
    parser.add_argument("--tools")
    parser.add_argument("--tool-doc-chars", type=int, default=500)
    parser.add_argument("--tool-convergence-threshold", type=float, default=0.75)
    parser.add_argument("--memory")
    parser.add_argument("--memory-read")
    parser.add_argument("--memory-k", type=int, default=5)
    parser.add_argument("--memory-tokens", type=int, default=1500)
    parser.add_argument("--memory-max-skill-age", type=int, default=4)
    parser.add_argument("--memory-pool-size", type=int, default=32)
    parser.add_argument("--memory-evolution-threshold", type=int, default=3)
    parser.add_argument("--memory-best-of-n", type=int, default=3)
    parser.add_argument("--memory-ppo-epsilon", type=float, default=0.2)
    return parser


def main() -> None:
    configure_custom_provider_environment()
    args = build_parser().parse_args()
    if args.memory and args.memory_read:
        raise SystemExit("Use either --memory or --memory-read, not both")
    if args.judge and (not args.judge_provider or not args.judge_model):
        raise SystemExit("--judge-provider and --judge-model are required with --judge")
    raise SystemExit(run_batch(args))


if __name__ == "__main__":
    main()
