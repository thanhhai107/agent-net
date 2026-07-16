"""Training/evaluation benchmark pipeline for optional NIKA modules."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import signal
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


from agent.composition import (
    AgentRunConfig,
    ProceduralMemoryConfig,
    ToolRefinementConfig,
    validate_agent_extensions,
)
from agent.extensions.config import default_llm_provider, default_model
from agent.extensions.react_agent import configure_custom_provider_environment
from agent.extensions.run import start_agent
from agent.module_config import module_defaults
from agent.procedural_memory.service import ProceduralMemoryModule
from agent.procedural_memory.workflow import update_procedural_memory_from_session
from agent.tool_refinement.curator import finalize_tool_refinement_session
from agent.tool_refinement.store import ToolRefinementStore
from nika.config import resolve_results_root
from nika.evaluator.result_log import EVAL_METRICS_FILENAME
from nika.net_env.net_env_pool import scenario_requires_topo_size
from nika.runtime.base import LabCleanupError
from nika.utils.logger import log_event
from nika.utils.session import Session
from nika.utils.session_store import SessionStore
from nika.workflows.benchmark.load_config import (
    BenchmarkManifest,
    is_no_fault_problem,
    load_benchmark_manifest,
    load_benchmark_yaml,
)
from nika.workflows.benchmark.resume import (
    benchmark_row_fingerprint,
    scan_benchmark_cases,
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
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _update_training(session_id: str, config: AgentRunConfig) -> dict[str, float]:
    """Update enabled modules after one completed training case."""

    if not config.allow_training_updates:
        return {}
    session = Session().load_closed_session(session_id=session_id)
    session_dir = Path(session.session_dir)
    metrics = _read_json(session_dir / EVAL_METRICS_FILENAME)
    run_meta = _read_json(session_dir / "run.json")
    errors: list[dict[str, str]] = []
    timings: dict[str, float] = {}
    error_path = session_dir / "training_errors.json"

    if config.tool_refinement.enabled:
        started = time.perf_counter()
        try:
            finalize_tool_refinement_session(
                session_id=session_id,
                metrics=metrics,
                allow_training_updates=config.allow_training_updates,
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

    if config.procedural_memory.enabled:
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
                "training_update_failed",
                f"{error['module']} update failed: {error['error']}",
                session_id=session_id,
                module=error["module"],
                error=error["error"],
            )
    else:
        error_path.unlink(missing_ok=True)
    timings["training_duration"] = round(sum(timings.values()), 6)
    timings["training_error_count"] = float(len(errors))
    (session_dir / "training_timing.json").write_text(
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
    benchmark_role: str | None = None,
    benchmark_total: int | None = None,
    benchmark_file: str | None = None,
    benchmark_manifest_hash: str | None = None,
) -> tuple[str, Path]:
    """Run a clean control or a module-enabled benchmark case."""

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
        metadata = {
            "benchmark_index": benchmark_index,
            "benchmark_role": benchmark_role,
            "benchmark_total": benchmark_total,
            "benchmark_file": benchmark_file,
            "benchmark_manifest_hash": benchmark_manifest_hash,
        }
        for key, value in metadata.items():
            if value not in (None, ""):
                session.update_session(key, value)
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
        training_timings = _update_training(session_id, config)
        tool_runtime = _read_json(session_dir / "tool_refinement_session.json")
        tool_exploration_duration = float(tool_runtime.get("explorer_duration") or 0.0)
        metrics_path = session_dir / EVAL_METRICS_FILENAME
        final_metrics = _read_json(metrics_path)
        final_metrics.update(
            {
                "agent_duration": round(agent_duration, 6),
                "evaluation_duration": round(evaluation_duration, 6),
                "tool_exploration_duration": round(tool_exploration_duration, 6),
                "training_overhead_duration": round(
                    float(training_timings.get("training_duration") or 0.0)
                    + tool_exploration_duration,
                    6,
                ),
                **training_timings,
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


@dataclass(frozen=True)
class StageResult:
    role: str
    total: int
    completed: int
    failed: int
    training_update_failures: int = 0

    @property
    def successful(self) -> bool:
        return (
            self.completed == self.total
            and self.failed == 0
            and self.training_update_failures == 0
        )


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _tool_library_state_hash(library_id: str) -> str:
    return _sha256_path(ToolRefinementStore(library_id).state_path)


def _pipeline_config_fingerprint(
    args: argparse.Namespace,
    *,
    training_manifest: BenchmarkManifest | None,
    evaluation_manifest: BenchmarkManifest,
) -> str:
    ignored = {"resume", "result_dir", "training_benchmark", "evaluate_benchmark"}
    execution = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in sorted(vars(args).items())
        if key not in ignored
    }
    payload = {
        "version": 1,
        "training_manifest_hash": (
            training_manifest.fingerprint if training_manifest else ""
        ),
        "evaluation_manifest_hash": evaluation_manifest.fingerprint,
        "execution": execution,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_agent_config(
    args: argparse.Namespace,
    *,
    allow_training_updates: bool,
    memory_store_path: Path | None = None,
    tool_state_path: Path | None = None,
) -> AgentRunConfig:
    baseline_defaults = module_defaults().baseline
    tool_defaults = ToolRefinementConfig()
    memory_defaults = ProceduralMemoryConfig()
    config = AgentRunConfig(
        agent_type=getattr(args, "agent", baseline_defaults.agent_type),
        llm_provider=args.provider,
        model=args.model,
        max_steps=args.max_steps,
        max_attempts=getattr(args, "max_attempts", baseline_defaults.max_attempts),
        allow_training_updates=allow_training_updates,
        procedural_memory=ProceduralMemoryConfig(
            enabled=bool(args.procedural_memory),
            bank=args.procedural_memory or "default",
            store_path=memory_store_path,
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
                args,
                "procedural_memory_verifier",
                memory_defaults.verifier,
            ),
            holdout_size=getattr(
                args,
                "procedural_memory_holdout_size",
                memory_defaults.holdout_size,
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
            state_path=tool_state_path,
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
    return config


def _freeze_training_modules(
    *,
    config: AgentRunConfig,
    results_root: Path,
    training_manifest: BenchmarkManifest,
    evaluation_manifest: BenchmarkManifest,
    config_fingerprint: str,
) -> dict[str, Any]:
    memory_payload: dict[str, Any] = {"enabled": False, "state_hash": ""}
    if config.procedural_memory.enabled:
        module = ProceduralMemoryModule(bank_id=config.procedural_memory.bank)
        # ``freeze_for_evaluation`` also retires unresolved probationary skills.
        # Its historical JSONL export is useful for inspection, but the runtime
        # store consumes one canonical JSON state document.  Keep the barrier
        # snapshot in that loadable format so evaluation can point at it directly.
        freeze_report = module.freeze_for_evaluation(
            output_path=results_root / "procedural_memory_frozen_bank.jsonl"
        )
        snapshot_path = results_root / "procedural_memory_frozen_state.json"
        module.store.snapshot(snapshot_path)
        memory_payload = {
            "enabled": True,
            **freeze_report,
            "snapshot_path": str(snapshot_path),
            "snapshot_hash": _sha256_path(snapshot_path),
        }

    tool_payload: dict[str, Any] = {"enabled": False, "state_hash": ""}
    if config.tool_refinement.enabled:
        store = ToolRefinementStore(config.tool_refinement.library_id)
        snapshot_path = results_root / "tool_refinement_frozen_state.json"
        store.snapshot(snapshot_path)
        tool_payload = {
            "enabled": True,
            "library_id": config.tool_refinement.library_id,
            "state_hash": store.state_hash(),
            "snapshot_path": str(snapshot_path),
            "snapshot_hash": _sha256_path(snapshot_path),
        }

    barrier = {
        "version": 1,
        "status": "ready",
        "training_manifest_hash": training_manifest.fingerprint,
        "evaluation_manifest_hash": evaluation_manifest.fingerprint,
        "config_fingerprint": config_fingerprint,
        "training_cases": len(training_manifest.cases),
        "evaluation_cases": len(evaluation_manifest.cases),
        "procedural_memory": memory_payload,
        "tool_refinement": tool_payload,
    }
    barrier_path = results_root / "training_barrier.json"
    barrier_path.write_text(
        json.dumps(barrier, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "training_barrier_created "
        + json.dumps(
            {
                "path": str(barrier_path),
                "training_manifest_hash": training_manifest.fingerprint,
                "procedural_memory_hash": memory_payload.get("state_hash", ""),
                "tool_refinement_hash": tool_payload.get("state_hash", ""),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return barrier


def _verify_frozen_modules(
    *,
    barrier: dict[str, Any],
    config: AgentRunConfig,
) -> None:
    def verify_snapshot(module_name: str, payload: dict[str, Any]) -> None:
        raw_path = str(payload.get("snapshot_path") or "").strip()
        expected_snapshot_hash = str(payload.get("snapshot_hash") or "").strip()
        if not raw_path or not expected_snapshot_hash:
            raise RuntimeError(f"{module_name} barrier is missing its frozen snapshot")
        snapshot_path = Path(raw_path)
        if not snapshot_path.is_file():
            raise RuntimeError(
                f"{module_name} frozen snapshot does not exist: {snapshot_path}"
            )
        current_snapshot_hash = _sha256_path(snapshot_path)
        if current_snapshot_hash != expected_snapshot_hash:
            raise RuntimeError(
                f"{module_name} frozen snapshot changed: expected "
                f"{expected_snapshot_hash}, got {current_snapshot_hash}"
            )

    memory_payload = barrier.get("procedural_memory") or {}
    if memory_payload.get("enabled"):
        verify_snapshot("Procedural Memory", memory_payload)
        expected = str(memory_payload.get("state_hash") or "")
        current = ProceduralMemoryModule(
            bank_id=config.procedural_memory.bank,
            read_only=True,
        ).bank_state_hash()
        if current != expected:
            raise RuntimeError(
                "Procedural Memory changed after the training barrier: "
                f"expected {expected}, got {current}"
            )

    tool_payload = barrier.get("tool_refinement") or {}
    if tool_payload.get("enabled"):
        verify_snapshot("Tool Refinement", tool_payload)
        expected = str(tool_payload.get("state_hash") or "")
        current = _tool_library_state_hash(config.tool_refinement.library_id)
        if current != expected:
            raise RuntimeError(
                "Tool Refinement changed after the training barrier: "
                f"expected {expected}, got {current}"
            )


def _validate_training_barrier(
    barrier: dict[str, Any],
    *,
    training_manifest: BenchmarkManifest,
    evaluation_manifest: BenchmarkManifest,
    config_fingerprint: str,
    config: AgentRunConfig,
) -> None:
    if barrier.get("version") != 1 or barrier.get("status") != "ready":
        raise ValueError(
            "Training barrier is incomplete or uses an unsupported version"
        )
    expected = {
        "training_manifest_hash": training_manifest.fingerprint,
        "evaluation_manifest_hash": evaluation_manifest.fingerprint,
        "config_fingerprint": config_fingerprint,
        "training_cases": len(training_manifest.cases),
        "evaluation_cases": len(evaluation_manifest.cases),
    }
    for key, value in expected.items():
        actual = barrier.get(key)
        if actual != value:
            raise ValueError(
                f"Training barrier mismatch for {key}: expected {value}, "
                f"found {actual!r}"
            )
    _verify_frozen_modules(barrier=barrier, config=config)


def _run_stage(
    *,
    args: argparse.Namespace,
    manifest: BenchmarkManifest,
    role: str,
    results_root: Path,
    barrier: dict[str, Any] | None = None,
) -> StageResult:
    allow_updates = role == "training"
    memory_store_path: Path | None = None
    tool_state_path: Path | None = None
    if not allow_updates and barrier is not None:
        memory_info = barrier.get("procedural_memory") or {}
        tool_info = barrier.get("tool_refinement") or {}
        if memory_info.get("enabled") and memory_info.get("snapshot_path"):
            memory_store_path = Path(str(memory_info["snapshot_path"]))
        if tool_info.get("enabled") and tool_info.get("snapshot_path"):
            tool_state_path = Path(str(tool_info["snapshot_path"]))
    config = _build_agent_config(
        args,
        allow_training_updates=allow_updates,
        memory_store_path=memory_store_path,
        tool_state_path=tool_state_path,
    )
    stage_root = results_root / role
    stage_root.mkdir(parents=True, exist_ok=True)
    rows = manifest.cases
    _, pending = scan_benchmark_cases(
        rows=rows,
        result_dir=stage_root,
        resume=args.resume,
    )
    completed = len(rows) - len(pending)
    failed = 0
    print(
        "benchmark_stage_start "
        + json.dumps(
            {
                "role": role,
                "benchmark": str(manifest.path),
                "manifest_hash": manifest.fingerprint,
                "total": len(rows),
                "pending": len(pending),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if barrier is not None:
        _verify_frozen_modules(barrier=barrier, config=config)

    for index in pending:
        row = rows[index]
        case_config = config
        if config.tool_refinement.enabled:
            update_due = allow_updates and (
                (index + 1) % config.tool_refinement.update_interval == 0
                or index + 1 == len(rows)
                or index == pending[-1]
            )
            case_config = replace(
                config,
                tool_refinement=replace(
                    config.tool_refinement,
                    update_due=update_due,
                ),
            )
        label = (
            f"role={role} index={index + 1}/{len(rows)} "
            f"scenario={row['scenario']} topo_size={row.get('topo_size') or '-'} "
            f"problem={row['problem']}"
        )
        inject_label = " ".join(
            f"inject_{key}={value}" for key, value in sorted(row["inject"].items())
        )
        print(f"benchmark_start {label} {inject_label}".rstrip(), flush=True)
        try:
            common_meta = {
                "benchmark_index": index + 1,
                "benchmark_role": role,
                "benchmark_total": len(rows),
                "benchmark_file": str(manifest.path),
                "benchmark_manifest_hash": manifest.fingerprint,
            }
            if (
                case_config.extensions_enabled
                or case_config.normalized_agent_type != "react"
                or is_no_fault(row["problem"])
            ):
                session_id, session_dir = run_extended_case(
                    row,
                    config=case_config,
                    result_dir=str(stage_root),
                    run_judge=args.judge,
                    judge_provider=args.judge_provider,
                    judge_model=args.judge_model,
                    **common_meta,
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
                    result_dir=str(stage_root),
                    emit_completion_event=False,
                    **common_meta,
                )
            if barrier is not None:
                _verify_frozen_modules(barrier=barrier, config=case_config)
                try:
                    barrier_hash = hashlib.sha256(
                        json.dumps(
                            barrier,
                            sort_keys=True,
                            default=str,
                        ).encode("utf-8")
                    ).hexdigest()
                    Session().load_closed_session(
                        session_id=session_id
                    ).update_run_meta("training_barrier_hash", barrier_hash)
                except (AttributeError, FileNotFoundError, ValueError):
                    pass
            completed += 1
            print(
                f"benchmark_done {label} session_id={session_id} "
                f"session_dir={session_dir}",
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

    update_failures = (
        sum(1 for _ in stage_root.rglob("training_errors.json")) if allow_updates else 0
    )
    result = StageResult(
        role=role,
        total=len(rows),
        completed=completed,
        failed=failed,
        training_update_failures=update_failures,
    )
    print(
        "benchmark_stage_done "
        + json.dumps(
            {
                "role": role,
                "total": result.total,
                "completed": result.completed,
                "failed": result.failed,
                "training_update_failures": result.training_update_failures,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return result


def run_batch(args: argparse.Namespace) -> int:
    """Run Training Benchmark, freeze modules, then run Evaluate Benchmark."""

    evaluation_manifest = load_benchmark_manifest(
        args.evaluate_benchmark,
        expected_role="evaluation",
    )
    modules_enabled = bool(args.procedural_memory or args.tool_refinement)
    training_manifest: BenchmarkManifest | None = None
    if modules_enabled:
        if not args.training_benchmark:
            raise ValueError(
                "--training-benchmark is required when a training module is enabled"
            )
        training_manifest = load_benchmark_manifest(
            args.training_benchmark,
            expected_role="training",
        )

    results_root = resolve_results_root(args.result_dir)
    results_root.mkdir(parents=True, exist_ok=True)
    config_fingerprint = _pipeline_config_fingerprint(
        args,
        training_manifest=training_manifest,
        evaluation_manifest=evaluation_manifest,
    )
    barrier: dict[str, Any] | None = None
    barrier_path = results_root / "training_barrier.json"

    if training_manifest is not None:
        training_config = _build_agent_config(args, allow_training_updates=True)
        if args.resume and barrier_path.exists():
            barrier = _read_json(barrier_path)
            _validate_training_barrier(
                barrier,
                training_manifest=training_manifest,
                evaluation_manifest=evaluation_manifest,
                config_fingerprint=config_fingerprint,
                config=training_config,
            )
            print(
                "training_barrier_reused "
                + json.dumps({"path": str(barrier_path)}, ensure_ascii=False),
                flush=True,
            )
        else:
            training_result = _run_stage(
                args=args,
                manifest=training_manifest,
                role="training",
                results_root=results_root,
            )
            if not training_result.successful:
                print(
                    "benchmark_pipeline_blocked "
                    + json.dumps(
                        {
                            "reason": "training_incomplete",
                            "completed": training_result.completed,
                            "total": training_result.total,
                            "failed": training_result.failed,
                            "training_update_failures": (
                                training_result.training_update_failures
                            ),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                return 1
            barrier = _freeze_training_modules(
                config=training_config,
                results_root=results_root,
                training_manifest=training_manifest,
                evaluation_manifest=evaluation_manifest,
                config_fingerprint=config_fingerprint,
            )

    evaluation_result = _run_stage(
        args=args,
        manifest=evaluation_manifest,
        role="evaluation",
        results_root=results_root,
        barrier=barrier,
    )
    exit_code = 0 if evaluation_result.successful else 1
    print(
        "benchmark_pipeline_done "
        + json.dumps(
            {
                "exit_code": exit_code,
                "training_cases": (
                    training_manifest.counts["total"] if training_manifest else 0
                ),
                "evaluation_cases": evaluation_manifest.counts["total"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    baseline_defaults = module_defaults().baseline
    tool_defaults = ToolRefinementConfig()
    memory_defaults = ProceduralMemoryConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-benchmark",
        help="Training Benchmark YAML. Required when a training module is enabled.",
    )
    parser.add_argument(
        "--evaluate-benchmark",
        required=True,
        help="Evaluate Benchmark YAML.",
    )
    parser.add_argument("--provider", default=default_llm_provider())
    parser.add_argument("--model", default=default_model())
    parser.add_argument(
        "--agent",
        choices=("react", "plan-execute", "reflexion"),
        default=baseline_defaults.agent_type,
    )
    parser.add_argument("--max-steps", type=int, default=baseline_defaults.max_steps)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=baseline_defaults.max_attempts,
    )
    parser.add_argument("--result-dir")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
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
