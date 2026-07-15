"""Batch or single-case benchmark runs (env → inject → agent → eval)."""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from nika.config import BENCHMARK_DIR
from nika.evaluator.result_log import EVAL_METRICS_FILENAME
from nika.evaluator.scoring import score_detection
from nika.net_env.net_env_pool import (
    get_net_env_instance,
    scenario_backend,
    scenario_requires_topo_size,
)
from nika.utils.agent_config import (
    resolve_agent_model,
    resolve_agent_type,
    resolve_llm_provider,
    resolve_max_steps,
)
from nika.utils.logger import log_event
from nika.utils.session import Session
from nika.utils.session_store import SessionStore
from nika.problems.prob_pool import get_problem_instance
from nika.workflows.agent.run import start_agent
from nika.workflows.benchmark.load_config import (
    is_no_fault_problem,
    load_benchmark_yaml,
)
from nika.workflows.benchmark.resume import (
    benchmark_row_fingerprint,
    benchmark_row_from_case,
    scan_benchmark_cases,
)
from nika.workflows.env.start import start_net_env
from nika.workflows.eval.session import eval_results
from nika.workflows.failure.inject import inject_failure
from nika.workflows.session.close import (
    clean_emulation_environment,
    close_session_after_failure,
)

_BENCHMARK_DONE_PREFIX = "benchmark_done "
_BENCHMARK_DONE_RE = re.compile(
    r"benchmark_done session_id=(\S+) scenario=(\S+) problem=(\S+) session_dir=(\S+)"
)


def _run_benchmark_agent(
    *,
    agent_type: str | None,
    llm_provider: str | None,
    model: str | None,
    max_steps: int | None,
    session_id: str,
) -> None:
    """Dispatch the three public workflows through their appropriate runners."""
    resolved_agent_type = resolve_agent_type(agent_type)
    if resolved_agent_type in {"plan-execute", "reflexion"}:
        from agent.composition import AgentRunConfig
        from agent.extensions.run import start_agent as start_extension_agent

        start_extension_agent(
            AgentRunConfig(
                agent_type=resolved_agent_type,
                llm_provider=resolve_llm_provider(
                    llm_provider, agent_type=resolved_agent_type
                )
                or "",
                model=resolve_agent_model(resolved_agent_type, model),
                max_steps=resolve_max_steps(max_steps),
            ),
            session_id=session_id,
        )
        return

    start_agent(
        agent_type=resolved_agent_type,
        llm_provider=llm_provider,
        model=model,
        max_steps=max_steps,
        session_id=session_id,
    )


def default_benchmark_yaml_path() -> str:
    return str(BENCHMARK_DIR / "benchmark_selected.yaml")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _no_fault_task_description(session: Session) -> str:
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


def prepare_no_fault_case(session: Session) -> None:
    """Create clean-control metadata without injecting a problem."""

    session.update_session("problem_names", ["no_fault"])
    session.update_session("root_cause_category", "none")
    session.update_session("task_description", _no_fault_task_description(session))
    session.write_gt(
        {
            "is_anomaly": False,
            "faulty_devices": [],
            "root_cause_category": "none",
            "root_cause_name": [],
            "detailed_cause": "",
        }
    )


def _empty_list_score(value: Any) -> float:
    return 1.0 if isinstance(value, list) and not value else 0.0


def no_fault_scores(submission: dict[str, Any]) -> dict[str, float]:
    """Score an explicit clean-control submission with empty-set semantics."""

    localization = _empty_list_score(submission.get("faulty_devices"))
    rca = _empty_list_score(submission.get("root_cause_name"))
    return {
        "detection_score": score_detection(submission, {"is_anomaly": False}),
        "localization_accuracy": localization,
        "localization_precision": localization,
        "localization_recall": localization,
        "localization_f1": localization,
        "rca_accuracy": rca,
        "rca_precision": rca,
        "rca_recall": rca,
        "rca_f1": rca,
    }


def normalize_no_fault_metrics(session_id: str, session_dir: Path) -> None:
    """Apply empty-set semantics after the upstream evaluator completes."""

    submission_path = session_dir / "submission.json"
    metrics_path = session_dir / EVAL_METRICS_FILENAME
    if not submission_path.exists() or not metrics_path.exists():
        return
    submission = _read_json(submission_path)
    metrics = _read_json(metrics_path)
    metrics.update(no_fault_scores(submission))
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    Session().load_closed_session(session_id=session_id).update_run_meta(
        "eval_metrics", metrics
    )
    log_event(
        "clean_control_metrics_saved",
        f"Applied no-fault empty-set scoring for session {session_id}.",
        session_id=session_id,
    )


def validate_inject_params(
    problem: str,
    scenario: str,
    topo_size: str,
    params: dict[str, str],
) -> None:
    """Raise ValueError if inject params do not satisfy the problem schema."""
    if is_no_fault_problem(problem):
        if params:
            raise ValueError("The no_fault control must use an empty inject map.")
        return
    if not params:
        raise ValueError(
            f"Missing inject parameters for {problem!r}. "
            f"Use --config with a YAML case or pass complete --set key=value flags. "
            f"Run `nika failure describe {problem}` for required fields."
        )

    kwargs: dict = {}
    if topo_size:
        kwargs["topo_size"] = topo_size
    problem_inst = get_problem_instance(
        problem_names=[problem],
        scenario_name=scenario,
        **kwargs,
    )
    params_class = getattr(type(problem_inst), "Params", None)
    if params_class is None:
        if params:
            raise ValueError(f"Problem {problem!r} does not accept inject parameters.")
        return
    try:
        params_class(**params)
    except ValidationError as exc:
        raise ValueError(
            f"Invalid or incomplete inject parameters for {problem!r}: {exc}. "
            f"Run `nika failure describe {problem}` for required fields."
        ) from exc


def run_single_case(
    problem: str,
    scenario: str,
    topo_size: str,
    agent_type: str,
    llm_provider: str | None,
    model: str | None,
    max_steps: int | None,
    *,
    inject_params: dict[str, str],
    run_judge: bool = False,
    judge_llm_provider: str | None = None,
    judge_model: str | None = None,
    result_dir: str | None = None,
    session_tag: str | None = None,
    emit_completion_event: bool = True,
) -> tuple[str, Path]:
    """Run one benchmark case (env → inject → agent → eval).

    Returns:
        The session id and session directory for the completed run.
    """
    print(
        f"Running benchmark for Problem: {problem}, Scenario: {scenario}, Topo Size: {topo_size}"
    )

    size = topo_size if topo_size else None
    if scenario_requires_topo_size(scenario) and not size:
        raise ValueError(
            f"Scenario '{scenario}' requires a non-empty topology size (-s s|m|l)."
        )
    if not scenario_requires_topo_size(scenario):
        size = None

    no_fault = is_no_fault_problem(problem)
    validate_inject_params(problem, scenario, topo_size or "", inject_params)
    params = dict(inject_params)

    clean_emulation_environment()
    session_id: str | None = None
    try:
        session_id = start_net_env(
            scenario,
            size,
            redeploy=True,
            result_dir=result_dir,
            session_tag=session_tag,
        )
        session_dir = Path(SessionStore().get_session(session_id)["session_dir"])
        if no_fault:
            prepare_no_fault_case(Session().load_running_session(session_id=session_id))
        else:
            inject_failure(
                problem_names=[problem], session_id=session_id, param_overrides=params
            )

        row = benchmark_row_from_case(
            scenario=scenario,
            problem=problem,
            topo_size=topo_size,
            inject_params=params,
        )
        Session().load_running_session(session_id=session_id).update_session(
            "benchmark_fingerprint",
            benchmark_row_fingerprint(row),
        )

        _run_benchmark_agent(
            agent_type=agent_type,
            llm_provider=llm_provider,
            model=model,
            max_steps=max_steps,
            session_id=session_id,
        )

        eval_results(
            session_id=session_id,
            run_judge=run_judge,
            judge_llm_provider=judge_llm_provider,
            judge_model=judge_model,
        )
        if no_fault:
            normalize_no_fault_metrics(session_id, session_dir)
    except BaseException as exc:
        if session_id is not None:
            cleanup_error = close_session_after_failure(session_id, exc)
            if cleanup_error is not None:
                raise cleanup_error from exc
        raise
    finally:
        clean_emulation_environment()

    if emit_completion_event:
        print(
            f"{_BENCHMARK_DONE_PREFIX}session_id={session_id} scenario={scenario} "
            f"problem={problem} session_dir={session_dir}"
        )
    return session_id, session_dir


def run_benchmark_from_yaml(
    benchmark_file: str,
    agent_type: str,
    llm_provider: str | None,
    model: str | None,
    max_steps: int | None,
    *,
    batch_size: int = 1,
    run_judge: bool = False,
    judge_llm_provider: str | None = None,
    judge_model: str | None = None,
    result_dir: str | None = None,
    resume: bool = True,
    session_tag: str | None = None,
) -> None:
    """
    Run benchmark cases defined in a YAML file.

    Each case must include scenario, problem, optional topo_size, and inject params.

    All rows are scanned first against existing session dirs under ``--result_dir``:
    completed cases are skipped and incomplete ones are cleaned. Remaining cases run
    sequentially. Re-run the same command to resume after an interruption.
    """
    if batch_size != 1:
        raise ValueError(
            "batch_size must be 1 because each case exclusively owns and cleans "
            "the emulation environment"
        )

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
        result_dir=result_dir,
        session_tag=session_tag,
    )

    _results_root, pending = scan_benchmark_cases(
        rows=rows,
        result_dir=result_dir,
        resume=resume,
    )
    if not pending:
        return

    for index in pending:
        row = rows[index]
        label = f"[{index + 1}/{len(rows)}] {row['scenario']}/{row['problem']}"
        print(f"{label} running")
        run_single_case(
            problem=row["problem"],
            scenario=row["scenario"],
            topo_size=row.get("topo_size") or "",
            inject_params=row["inject"],
            **_shared_kwargs,
        )
