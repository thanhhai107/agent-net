"""Custom benchmark controls around the unmodified NIKA pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from agent.composition import (
    AgentRunConfig,
    ProceduralMemoryConfig,
    ToolRefinementConfig,
    validate_agent_extensions,
)
from agent.extensions.config import default_llm_provider, default_model
from agent.module_config import module_defaults
from agent.extensions.react_agent import configure_custom_provider_environment
from agent.extensions.run import start_agent
from agent.procedural_memory.service import ProceduralMemoryModule
from agent.procedural_memory.workflow import update_procedural_memory_from_session
from agent.tool_refinement.curator import finalize_tool_refinement_session
from nika.evaluator.result_log import EVAL_METRICS_FILENAME
from nika.runtime.base import LabCleanupError
from nika.utils.logger import log_event
from nika.net_env.net_env_pool import (
    scenario_requires_topo_size,
)
from nika.utils.session import Session
from nika.utils.session_store import SessionStore
from nika.workflows.benchmark.resume import (
    benchmark_row_fingerprint,
    scan_benchmark_cases,
)
from nika.workflows.benchmark.load_config import (
    is_no_fault_problem,
    load_benchmark_evolve_first_cases,
    load_benchmark_yaml,
)
from nika.workflows.benchmark.run import (
    normalize_no_fault_metrics,
    prepare_no_fault_case,
    run_single_case,
)
from nika.workflows.env.start import start_net_env
from nika.workflows.eval.session import eval_results
from nika.workflows.failure.inject import inject_failure
from nika.workflows.session.close import close_session_after_failure


def is_no_fault(problem: str) -> bool:
    return is_no_fault_problem(problem)


def load_custom_benchmark(path: str | Path) -> list[dict[str, Any]]:
    """Load extension benchmark rows with the shared NIKA validator."""

    return load_benchmark_yaml(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _update_learning(session_id: str, config: AgentRunConfig) -> dict[str, float]:
    session = Session().load_closed_session(session_id=session_id)
    session_dir = Path(session.session_dir)
    metrics = _read_json(session_dir / EVAL_METRICS_FILENAME)
    run_meta = _read_json(session_dir / "run.json")
    errors: list[dict[str, str]] = []
    timings: dict[str, float] = {}
    error_path = session_dir / "learning_errors.json"
    if (
        config.tool_refinement.enabled
        and config.tool_refinement.learning_mode == "evolve"
    ):
        started = time.perf_counter()
        try:
            finalize_tool_refinement_session(
                session_id=session_id,
                metrics=metrics,
                rewrite=config.tool_refinement.update_due,
                min_new_trials=config.tool_refinement.min_new_trials,
                max_tools_per_update=config.tool_refinement.max_tools_per_update,
                publish_min_utility=config.tool_refinement.publish_min_utility,
            )
        except Exception as exc:
            errors.append(
                {
                    "module": "tool_refinement",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            timings["tool_refinement_duration"] = round(
                time.perf_counter() - started,
                6,
            )
    if config.procedural_memory.mode == "evolve":
        started = time.perf_counter()
        try:
            memory_report = asyncio.run(
                update_procedural_memory_from_session(
                    run_meta=run_meta,
                    metrics=metrics,
                    session_dir=session_dir,
                )
            )
            metrics["procedural_memory"] = memory_report
            metrics_path = session_dir / EVAL_METRICS_FILENAME
            metrics_path.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            try:
                session.update_run_meta("eval_metrics", metrics)
            except (AttributeError, FileNotFoundError, ValueError):
                pass
        except Exception as exc:
            errors.append(
                {
                    "module": "procedural_memory",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            timings["procedural_memory_duration"] = round(
                time.perf_counter() - started,
                6,
            )
    if errors:
        error_path.write_text(
            json.dumps(errors, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        for error in errors:
            log_event(
                "learning_update_failed",
                f"{error['module']} update failed: {error['error']}",
                session_id=session_id,
                module=error["module"],
                error=error["error"],
            )
    else:
        error_path.unlink(missing_ok=True)
    timings["learning_duration"] = round(sum(timings.values()), 6)
    (session_dir / "learning_timing.json").write_text(
        json.dumps(timings, indent=2),
        encoding="utf-8",
    )
    return timings


def run_extended_case(
    row: dict[str, Any],
    *,
    config: AgentRunConfig,
    result_dir: str | None,
    run_judge: bool = False,
    judge_provider: str | None = None,
    judge_model: str | None = None,
    benchmark_index: int | None = None,
    benchmark_phase: str | None = None,
) -> tuple[str, Path]:
    """Run a clean control or a learning-enabled row on upstream primitives."""
    scenario = row["scenario"]
    topo_size = row.get("topo_size") or ""
    size = topo_size or None
    if scenario_requires_topo_size(scenario) and size is None:
        raise ValueError(f"Scenario {scenario!r} requires topology size")
    if not scenario_requires_topo_size(scenario):
        size = None

    session_id: str | None = None
    try:
        session_id = start_net_env(
            scenario,
            size,
            redeploy=True,
            result_dir=result_dir,
        )
        session = Session().load_running_session(session_id=session_id)
        session_dir = Path(SessionStore().get_session(session_id)["session_dir"])
        if benchmark_index is not None:
            session.update_session("benchmark_index", benchmark_index)
        if benchmark_phase:
            session.update_session("benchmark_phase", benchmark_phase)
        if is_no_fault(row["problem"]):
            prepare_no_fault_case(session)
        else:
            inject_failure(
                problem_names=[row["problem"]],
                session_id=session_id,
                param_overrides=row["inject"],
            )
            session = Session().load_running_session(session_id=session_id)
        session.update_session("benchmark_fingerprint", benchmark_row_fingerprint(row))
        agent_started = time.perf_counter()
        start_agent(config, session_id=session_id)
        agent_duration = time.perf_counter() - agent_started
        evaluation_started = time.perf_counter()
        eval_results(
            session_id=session_id,
            run_judge=run_judge,
            judge_llm_provider=judge_provider,
            judge_model=judge_model,
        )
        evaluation_duration = time.perf_counter() - evaluation_started
        if is_no_fault(row["problem"]):
            normalize_no_fault_metrics(session_id, session_dir)
        learning_timings = _update_learning(session_id, config) or {}
        tool_runtime = _read_json(session_dir / "tool_refinement_session.json")
        tool_exploration_duration = float(tool_runtime.get("explorer_duration") or 0.0)
        metrics_path = session_dir / EVAL_METRICS_FILENAME
        final_metrics = _read_json(metrics_path)
        final_metrics.update(
            {
                "agent_duration": round(agent_duration, 6),
                "evaluation_duration": round(evaluation_duration, 6),
                "tool_exploration_duration": round(
                    tool_exploration_duration,
                    6,
                ),
                "learning_overhead_duration": round(
                    float(learning_timings.get("learning_duration") or 0.0)
                    + tool_exploration_duration,
                    6,
                ),
                **learning_timings,
            }
        )
        metrics_path.write_text(
            json.dumps(final_metrics, indent=2),
            encoding="utf-8",
        )
        try:
            Session().load_closed_session(session_id=session_id).update_run_meta(
                "eval_metrics",
                final_metrics,
            )
        except (AttributeError, FileNotFoundError, ValueError):
            pass
    except BaseException as exc:
        if session_id is not None:
            cleanup_error = close_session_after_failure(session_id, exc)
            if cleanup_error is not None:
                raise cleanup_error from exc
        raise
    return session_id, session_dir


def run_batch(args: argparse.Namespace) -> int:
    baseline_defaults = module_defaults().baseline
    tool_defaults = ToolRefinementConfig()
    memory_defaults = ProceduralMemoryConfig()
    rows = load_custom_benchmark(args.config)
    evolve_until = getattr(args, "evolve_until", None)
    if evolve_until is None:
        evolve_until = load_benchmark_evolve_first_cases(args.config)
    if evolve_until is not None and not 0 <= evolve_until <= len(rows):
        raise ValueError(
            f"--evolve-until must be between 0 and the benchmark size ({len(rows)})"
        )
    procedural_memory_mode = "evolve" if args.procedural_memory else "off"
    config = AgentRunConfig(
        agent_type=getattr(args, "agent", baseline_defaults.agent_type),
        llm_provider=args.provider,
        model=args.model,
        max_steps=args.max_steps,
        max_attempts=getattr(args, "max_attempts", baseline_defaults.max_attempts),
        procedural_memory=ProceduralMemoryConfig(
            mode=procedural_memory_mode,
            bank=args.procedural_memory or "default",
            token_budget=args.procedural_memory_tokens,
            max_skill_age=args.procedural_memory_max_skill_age,
            pool_size=args.procedural_memory_pool_size,
            evolution_threshold=args.procedural_memory_update_threshold,
            best_of_n=args.procedural_memory_best_of_n,
            ppo_epsilon=args.procedural_memory_ppo_epsilon,
            selection_epsilon=getattr(
                args,
                "procedural_memory_selection_epsilon",
                memory_defaults.selection_epsilon,
            ),
            experience_pool_size=getattr(
                args,
                "procedural_memory_experience_pool_size",
                memory_defaults.experience_pool_size,
            ),
            baseline_ema_alpha=getattr(
                args,
                "procedural_memory_baseline_ema_alpha",
                memory_defaults.baseline_ema_alpha,
            ),
            selection_epsilon_decay_cases=getattr(
                args,
                "procedural_memory_selection_epsilon_decay_cases",
                memory_defaults.selection_epsilon_decay_cases,
            ),
            acceptance_margin=getattr(
                args,
                "procedural_memory_acceptance_margin",
                memory_defaults.acceptance_margin,
            ),
            verifier=getattr(
                args, "procedural_memory_verifier", memory_defaults.verifier
            ),
            holdout_size=getattr(
                args, "procedural_memory_holdout_size", memory_defaults.holdout_size
            ),
            min_positive_advantage=getattr(
                args,
                "procedural_memory_min_positive_advantage",
                memory_defaults.min_positive_advantage,
            ),
            evolver_model=getattr(
                args,
                "procedural_memory_evolver_model",
                memory_defaults.evolver_model,
            ),
            policy_scorer_model=getattr(
                args,
                "procedural_memory_policy_scorer_model",
                memory_defaults.policy_scorer_model,
            ),
        ),
        tool_refinement=ToolRefinementConfig(
            enabled=bool(args.tool_refinement),
            library_id=args.tool_refinement or "default",
            tool_doc_chars=args.tool_refinement_doc_chars,
            convergence_threshold=args.tool_refinement_convergence_threshold,
            exploration_similarity_threshold=getattr(
                args,
                "tool_refinement_exploration_similarity_threshold",
                tool_defaults.exploration_similarity_threshold,
            ),
            explorer_reflection_limit=getattr(
                args,
                "tool_refinement_explorer_reflection_limit",
                tool_defaults.explorer_reflection_limit,
            ),
            update_interval=getattr(
                args,
                "tool_refinement_update_interval",
                tool_defaults.update_interval,
            ),
            min_new_trials=getattr(
                args,
                "tool_refinement_min_new_trials",
                tool_defaults.min_new_trials,
            ),
            max_tools_per_update=getattr(
                args,
                "tool_refinement_max_tools_per_update",
                tool_defaults.max_tools_per_update,
            ),
            publish_min_utility=getattr(
                args,
                "tool_refinement_publish_min_utility",
                tool_defaults.publish_min_utility,
            ),
            explorer_model=getattr(args, "tool_refinement_explorer_model", ""),
            analyzer_model=getattr(args, "tool_refinement_analyzer_model", ""),
            rewriter_model=getattr(args, "tool_refinement_rewriter_model", ""),
        ),
    )
    validate_agent_extensions(config)
    results_root, pending = scan_benchmark_cases(
        rows=rows,
        result_dir=args.result_dir,
        resume=args.resume,
    )
    frozen_memory_hash = ""
    memory_freezer: ProceduralMemoryModule | None = None
    freeze_manifest_path = results_root / "procedural_memory_freeze.json"
    if config.procedural_memory.mode == "evolve" and evolve_until is not None:
        memory_freezer = ProceduralMemoryModule(bank_id=config.procedural_memory.bank)
        training_pending = any(index < evolve_until for index in pending)
        read_pending = any(index >= evolve_until for index in pending)
        if read_pending and not training_pending and freeze_manifest_path.exists():
            previous_manifest = _read_json(freeze_manifest_path)
            expected_hash = str(previous_manifest.get("state_hash") or "")
            current_hash = memory_freezer.bank_state_hash()
            if expected_hash and current_hash != expected_hash:
                raise ValueError(
                    "Procedural Memory bank differs from the frozen resume snapshot: "
                    f"expected {expected_hash}, got {current_hash}"
                )
            frozen_memory_hash = expected_hash

    def freeze_memory_bank() -> str:
        if memory_freezer is None:
            return ""
        manifest = memory_freezer.freeze_for_evaluation(
            output_path=results_root / "procedural_memory_frozen_bank.jsonl"
        )
        freeze_manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            "procedural_memory_frozen "
            f"bank={manifest['bank_id']} iteration={manifest['iteration']} "
            f"state_hash={manifest['state_hash']}",
            flush=True,
        )
        return str(manifest["state_hash"])

    failed = 0
    completed = len(rows) - len(pending)
    for index in pending:
        row = rows[index]
        case_config = config
        learning_phase_end = evolve_until if evolve_until is not None else len(rows)
        benchmark_phase = None
        if evolve_until is not None:
            benchmark_phase = "evolve" if index < evolve_until else "read"
        if (
            memory_freezer is not None
            and index >= learning_phase_end
            and not frozen_memory_hash
        ):
            frozen_memory_hash = freeze_memory_bank()
        tool_learning_enabled = (
            config.tool_refinement.enabled and index < learning_phase_end
        )
        tool_update_due = tool_learning_enabled and (
            (index + 1) % config.tool_refinement.update_interval == 0
            or index + 1 == learning_phase_end
        )
        if config.tool_refinement.enabled:
            case_config = replace(
                case_config,
                tool_refinement=replace(
                    config.tool_refinement,
                    learning_mode="evolve" if tool_learning_enabled else "read",
                    update_due=tool_update_due,
                ),
            )
        if config.procedural_memory.mode == "evolve" and evolve_until is not None:
            case_mode = "evolve" if index < evolve_until else "read"
            case_config = replace(
                case_config,
                procedural_memory=replace(
                    config.procedural_memory,
                    mode=case_mode,
                ),
            )
        label = (
            f"index={index + 1}/{len(rows)} scenario={row['scenario']} "
            f"topo_size={row.get('topo_size') or '-'} problem={row['problem']}"
        )
        if benchmark_phase:
            label += f" benchmark_phase={benchmark_phase}"
        if case_config.procedural_memory.enabled:
            label += f" procedural_memory_mode={case_config.procedural_memory.mode}"
        if case_config.tool_refinement.enabled:
            label += (
                f" tool_refinement_mode={case_config.tool_refinement.learning_mode}"
                f" tool_refinement_update_due={case_config.tool_refinement.update_due}"
            )
        inject_label = " ".join(
            f"inject_{key}={value}" for key, value in sorted(row["inject"].items())
        )
        print(f"benchmark_start {label} {inject_label}".rstrip(), flush=True)
        try:
            if (
                case_config.extensions_enabled
                or case_config.normalized_agent_type != "react"
                or is_no_fault(row["problem"])
            ):
                session_id, session_dir = run_extended_case(
                    row,
                    config=case_config,
                    result_dir=args.result_dir,
                    run_judge=args.judge,
                    judge_provider=args.judge_provider,
                    judge_model=args.judge_model,
                    benchmark_index=index + 1,
                    benchmark_phase=benchmark_phase,
                )
            else:
                session_id, session_dir = run_single_case(
                    problem=row["problem"],
                    scenario=row["scenario"],
                    topo_size=row.get("topo_size") or "",
                    agent_type="react",
                    llm_provider=args.provider,
                    model=args.model,
                    max_steps=args.max_steps,
                    inject_params=row["inject"],
                    run_judge=args.judge,
                    judge_llm_provider=args.judge_provider,
                    judge_model=args.judge_model,
                    result_dir=args.result_dir,
                    emit_completion_event=False,
                    benchmark_index=index + 1,
                    benchmark_phase=benchmark_phase,
                )
            if case_config.procedural_memory.mode == "read" and frozen_memory_hash:
                current_hash = (
                    memory_freezer.bank_state_hash() if memory_freezer else ""
                )
                if current_hash != frozen_memory_hash:
                    raise RuntimeError(
                        "Procedural Memory bank changed during read-only evaluation: "
                        f"expected {frozen_memory_hash}, got {current_hash}"
                    )
                try:
                    Session().load_closed_session(
                        session_id=session_id
                    ).update_run_meta(
                        "procedural_memory_frozen_state_hash",
                        frozen_memory_hash,
                    )
                except (AttributeError, FileNotFoundError, ValueError):
                    pass
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
            if isinstance(exc, LabCleanupError):
                print(
                    f"benchmark_aborted {label} reason=lab_cleanup_failed",
                    flush=True,
                )
                break
    print(
        f"benchmark_summary total={len(rows)} completed={completed} failed={failed}",
        flush=True,
    )
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    baseline_defaults = module_defaults().baseline
    tool_defaults = ToolRefinementConfig()
    memory_defaults = ProceduralMemoryConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--provider", default=default_llm_provider())
    parser.add_argument("--model", default=default_model())
    parser.add_argument(
        "--agent",
        choices=("react", "plan-execute", "reflexion"),
        default=baseline_defaults.agent_type,
    )
    parser.add_argument("--max-steps", type=int, default=baseline_defaults.max_steps)
    parser.add_argument(
        "--max-attempts", type=int, default=baseline_defaults.max_attempts
    )
    parser.add_argument("--result-dir")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--judge",
        action=argparse.BooleanOptionalAction,
        default=baseline_defaults.judge_evaluation,
    )
    parser.add_argument("--judge-provider", default=baseline_defaults.judge_provider)
    parser.add_argument("--judge-model", default=baseline_defaults.judge_model)
    parser.add_argument("--tool-refinement", metavar="LIBRARY_ID")
    parser.add_argument(
        "--tool-refinement-doc-chars",
        dest="tool_refinement_doc_chars",
        type=int,
        default=tool_defaults.tool_doc_chars,
        metavar="CHARS",
    )
    parser.add_argument(
        "--tool-refinement-convergence-threshold",
        dest="tool_refinement_convergence_threshold",
        type=float,
        default=tool_defaults.convergence_threshold,
        metavar="THRESHOLD",
    )
    parser.add_argument(
        "--tool-refinement-exploration-similarity-threshold",
        dest="tool_refinement_exploration_similarity_threshold",
        type=float,
        default=tool_defaults.exploration_similarity_threshold,
    )
    parser.add_argument(
        "--tool-refinement-explorer-reflection-limit",
        dest="tool_refinement_explorer_reflection_limit",
        type=int,
        default=tool_defaults.explorer_reflection_limit,
    )
    parser.add_argument(
        "--tool-refinement-update-interval",
        type=int,
        default=tool_defaults.update_interval,
    )
    parser.add_argument(
        "--tool-refinement-min-new-trials",
        type=int,
        default=tool_defaults.min_new_trials,
    )
    parser.add_argument(
        "--tool-refinement-max-tools-per-update",
        type=int,
        default=tool_defaults.max_tools_per_update,
    )
    parser.add_argument(
        "--tool-refinement-publish-min-utility",
        type=float,
        default=tool_defaults.publish_min_utility,
    )
    parser.add_argument("--tool-refinement-explorer-model", default="")
    parser.add_argument("--tool-refinement-analyzer-model", default="")
    parser.add_argument("--tool-refinement-rewriter-model", default="")
    parser.add_argument("--procedural-memory", metavar="BANK_ID")
    parser.add_argument(
        "--evolve-until",
        dest="evolve_until",
        type=int,
        metavar="CASE_COUNT",
        help=(
            "Tag the first CASE_COUNT benchmark cases as evolve and later cases "
            "as read/evaluation. Enabled learning modules evolve before the "
            "boundary and run read-only after it. Zero reads for all cases."
        ),
    )
    parser.add_argument(
        "--procedural-memory-tokens",
        "--procedural-memory-token-budget",
        dest="procedural_memory_tokens",
        type=int,
        default=memory_defaults.token_budget,
        metavar="TOKEN_BUDGET",
    )
    parser.add_argument(
        "--procedural-memory-max-skill-age",
        dest="procedural_memory_max_skill_age",
        type=int,
        default=memory_defaults.max_skill_age,
        metavar="MAX_SKILL_AGE",
    )
    parser.add_argument(
        "--procedural-memory-pool-size",
        dest="procedural_memory_pool_size",
        type=int,
        default=memory_defaults.pool_size,
        metavar="POOL_SIZE",
    )
    parser.add_argument(
        "--procedural-memory-update-threshold",
        dest="procedural_memory_update_threshold",
        type=int,
        default=memory_defaults.evolution_threshold,
        metavar="UPDATE_THRESHOLD",
    )
    parser.add_argument(
        "--procedural-memory-best-of-n",
        dest="procedural_memory_best_of_n",
        type=int,
        default=memory_defaults.best_of_n,
        metavar="BEST_OF_N",
    )
    parser.add_argument(
        "--procedural-memory-ppo-epsilon",
        dest="procedural_memory_ppo_epsilon",
        type=float,
        default=memory_defaults.ppo_epsilon,
        metavar="PPO_EPSILON",
    )
    parser.add_argument(
        "--procedural-memory-selection-epsilon",
        dest="procedural_memory_selection_epsilon",
        type=float,
        default=memory_defaults.selection_epsilon,
        metavar="SELECTION_EPSILON",
    )
    parser.add_argument(
        "--procedural-memory-experience-pool-size",
        dest="procedural_memory_experience_pool_size",
        type=int,
        default=memory_defaults.experience_pool_size,
    )
    parser.add_argument(
        "--procedural-memory-baseline-ema-alpha",
        dest="procedural_memory_baseline_ema_alpha",
        type=float,
        default=memory_defaults.baseline_ema_alpha,
    )
    parser.add_argument(
        "--procedural-memory-selection-epsilon-decay-cases",
        dest="procedural_memory_selection_epsilon_decay_cases",
        type=int,
        default=memory_defaults.selection_epsilon_decay_cases,
    )
    parser.add_argument(
        "--procedural-memory-acceptance-margin",
        dest="procedural_memory_acceptance_margin",
        type=float,
        default=memory_defaults.acceptance_margin,
    )
    parser.add_argument(
        "--procedural-memory-verifier",
        choices=("behavioral_replay", "structured_replay", "policy_logprob"),
        default=memory_defaults.verifier,
    )
    parser.add_argument(
        "--procedural-memory-holdout-size",
        type=int,
        default=memory_defaults.holdout_size,
    )
    parser.add_argument(
        "--procedural-memory-min-positive-advantage",
        type=int,
        default=memory_defaults.min_positive_advantage,
    )
    parser.add_argument(
        "--procedural-memory-evolver-model",
        default=memory_defaults.evolver_model,
    )
    parser.add_argument(
        "--procedural-memory-policy-scorer-model",
        default=memory_defaults.policy_scorer_model,
    )
    return parser


def main() -> None:
    configure_custom_provider_environment()
    args = build_parser().parse_args()
    if args.judge and (not args.judge_provider or not args.judge_model):
        raise SystemExit("--judge-provider and --judge-model are required with --judge")

    def _request_graceful_stop(_signum, _frame) -> None:
        raise KeyboardInterrupt("Benchmark stop requested")

    previous_sigterm = signal.signal(signal.SIGTERM, _request_graceful_stop)
    try:
        exit_code = run_batch(args)
    except KeyboardInterrupt:
        print("benchmark_stopped reason=signal", flush=True)
        exit_code = 130
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
