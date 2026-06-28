import csv
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from nika.config import RESULTS_DIR
from nika.evaluator.llm_judge import JudgeResponse
from nika.evaluator.trace_parser import AgentTraceParser
from nika.orchestrator.problems.prob_pool import get_problem_instance

load_dotenv()

EVAL_METRICS_FILENAME = "eval_metrics.json"
RUN_FILENAME = "run.json"
GROUND_TRUTH_FILENAME = "ground_truth.json"
SUBMISSION_FILENAME = "submission.json"
LLM_JUDGE_FILENAME = "llm_judge.json"
MESSAGES_FILENAME = "messages.jsonl"

SUMMARY_REQUIRED_ARTIFACTS = (
    RUN_FILENAME,
    GROUND_TRUTH_FILENAME,
    EVAL_METRICS_FILENAME,
)


@dataclass
class EvalResult:
    agent_type: str = None
    model: str = None
    root_cause_category: str = None
    root_cause_name: str = None
    net_env: str = None
    scenario_topo_size: str = None
    session_id: str = None
    in_tokens: int = None
    out_tokens: int = None
    steps: int = None
    tool_calls: int = None
    tool_errors: int = None
    primitive_calls: int = None
    composite_calls: int = None
    generated_tool_calls: int = None
    evolved_tools_created: int = None
    mastery_updates: int = None
    tool_evolution_enabled: bool | None = None
    tool_library_id: str = None
    tool_evolution_mode: str = None
    evolution_stream: str = None
    evolution_split: str = None
    evolution_sequence_index: int = None
    tool_selection_recall: float = None
    argument_validity: float = None
    error_recovery_count: int = None
    tool_reuse_count: int = None
    tool_promotion_count: int = None
    tool_regression_count: int = None
    library_candidates: int = None
    library_promoted: int = None
    library_generated_tools: int = None
    library_generated_candidates: int = None
    library_generated_promoted: int = None
    library_mastered_primitives: int = None
    tool_card_revisions: int = None
    capability_gaps: int = None
    verified_composites: int = None
    verified_generated_tools: int = None
    unverified_ephemeral_tools: int = None
    cross_model_reused_tools: int = None
    generated_tool_reuse_count: int = None
    retrieved_tool_available_count: int = None
    retrieved_tool_started_count: int = None
    retrieved_tool_started_unique_count: int = None
    retrieved_tool_called_count: int = None
    retrieved_tool_called_unique_count: int = None
    incident_success: bool = None
    efficiency_evolution_rate: float = None
    evolutionary_gain: float = None
    time_taken: float = None
    llm_judge_relevance_score: int = None
    llm_judge_correctness_score: int = None
    llm_judge_efficiency_score: int = None
    llm_judge_clarity_score: int = None
    llm_judge_final_outcome_score: int = None
    llm_judge_overall_score: int = None
    detection_score: float = None
    localization_accuracy: float = None
    localization_precision: float = None
    localization_recall: float = None
    localization_f1: float = None
    rca_accuracy: float = None
    rca_precision: float = None
    rca_recall: float = None
    rca_f1: float = None


def _session_duration_seconds(start_time, end_time) -> float | None:
    if start_time is None or end_time is None:
        return None
    try:
        return round(float(end_time) - float(start_time), 2)
    except (TypeError, ValueError):
        start_dt = datetime.fromisoformat(str(start_time))
        end_dt = datetime.fromisoformat(str(end_time))
        return round((end_dt - start_dt).total_seconds(), 2)


def default_summary_csv_path() -> str:
    return os.path.join(RESULTS_DIR, "0_summary", "evaluation_summary.csv")


def missing_summary_artifacts(session_dir: Path) -> list[str]:
    return [
        name for name in SUMMARY_REQUIRED_ARTIFACTS if not (session_dir / name).exists()
    ]


def is_finished_session(run_meta: dict) -> bool:
    if run_meta.get("status") == "finished":
        return True
    return run_meta.get("end_time") is not None


def iter_session_dirs(results_dir: str | Path | None = None) -> list[Path]:
    root = Path(results_dir or RESULTS_DIR)
    if not root.exists():
        return []
    session_dirs: list[Path] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name == "0_summary":
            continue
        if (entry / RUN_FILENAME).exists():
            session_dirs.append(entry)
    return session_dirs


def resolve_root_cause_category(run_meta: dict) -> str | None:
    category = run_meta.get("root_cause_category")
    if category:
        return str(category)
    problem_names = run_meta.get("problem_names") or []
    if not problem_names:
        return None
    try:
        problem = get_problem_instance(
            problem_names=problem_names,
            task_level="detection",
            scenario_name=run_meta.get("scenario_name", ""),
            **(run_meta.get("scenario_params") or {}),
        )
        return str(problem.root_cause_category)
    except Exception:
        return None


def build_eval_result_from_session_dir(session_dir: Path) -> EvalResult:
    missing = missing_summary_artifacts(session_dir)
    if missing:
        raise FileNotFoundError(
            f"Session directory {session_dir} is missing artifacts required for summary CSV: {', '.join(missing)}"
        )

    run_meta = json.loads((session_dir / RUN_FILENAME).read_text(encoding="utf-8"))
    if not is_finished_session(run_meta):
        raise ValueError(
            f"Session {run_meta.get('session_id', session_dir.name)} is not finished."
        )

    metrics_blob = json.loads(
        (session_dir / EVAL_METRICS_FILENAME).read_text(encoding="utf-8")
    )

    judge_response: JudgeResponse | None = None
    judge_path = session_dir / LLM_JUDGE_FILENAME
    if judge_path.exists():
        judge_response = JudgeResponse.model_validate_json(
            judge_path.read_text(encoding="utf-8")
        )

    trace_metrics = {
        "in_tokens": metrics_blob.get("in_tokens"),
        "out_tokens": metrics_blob.get("out_tokens"),
        "steps": metrics_blob.get("steps"),
        "tool_calls": metrics_blob.get("tool_calls"),
        "tool_errors": metrics_blob.get("tool_errors"),
        "primitive_calls": metrics_blob.get("primitive_calls"),
        "composite_calls": metrics_blob.get("composite_calls"),
        "evolved_tools_created": metrics_blob.get("evolved_tools_created"),
        "mastery_updates": metrics_blob.get("mastery_updates"),
    }
    if not any(v is not None for v in trace_metrics.values()):
        trace_path = session_dir / MESSAGES_FILENAME
        if trace_path.exists():
            trace_metrics = AgentTraceParser(trace_path=str(trace_path)).parse_trace()

    if judge_response:
        relevance_score = judge_response.scores.relevance.score
        correctness_score = judge_response.scores.correctness.score
        efficiency_score = judge_response.scores.efficiency.score
        clarity_score = judge_response.scores.clarity.score
        final_outcome_score = judge_response.scores.final_outcome.score
        overall_score = judge_response.scores.overall_score.score
    else:
        relevance_score = correctness_score = efficiency_score = clarity_score = None
        final_outcome_score = overall_score = None

    return EvalResult(
        agent_type=run_meta.get("agent_type"),
        model=run_meta.get("model"),
        root_cause_category=resolve_root_cause_category(run_meta),
        root_cause_name=run_meta.get("root_cause_name"),
        net_env=run_meta.get("scenario_name"),
        scenario_topo_size=run_meta.get("scenario_topo_size"),
        session_id=run_meta.get("session_id") or session_dir.name,
        in_tokens=trace_metrics.get("in_tokens"),
        out_tokens=trace_metrics.get("out_tokens"),
        steps=trace_metrics.get("steps"),
        tool_calls=trace_metrics.get("tool_calls"),
        tool_errors=trace_metrics.get("tool_errors"),
        primitive_calls=trace_metrics.get("primitive_calls"),
        composite_calls=trace_metrics.get("composite_calls"),
        generated_tool_calls=trace_metrics.get("generated_tool_calls"),
        evolved_tools_created=trace_metrics.get("evolved_tools_created"),
        mastery_updates=trace_metrics.get("mastery_updates"),
        tool_evolution_enabled=bool(run_meta.get("tool_evolution_enabled", False)),
        tool_library_id=metrics_blob.get("tool_library_id")
        or run_meta.get("tool_library_id"),
        tool_evolution_mode=metrics_blob.get("tool_evolution_mode")
        or run_meta.get("tool_evolution_mode"),
        evolution_stream=run_meta.get("evolution_stream"),
        evolution_split=run_meta.get("evolution_split"),
        evolution_sequence_index=run_meta.get("evolution_sequence_index"),
        tool_selection_recall=metrics_blob.get("tool_selection_recall"),
        argument_validity=metrics_blob.get("argument_validity"),
        error_recovery_count=metrics_blob.get("error_recovery_count"),
        tool_reuse_count=metrics_blob.get("tool_reuse_count"),
        tool_promotion_count=metrics_blob.get("tool_promotion_count"),
        tool_regression_count=metrics_blob.get("tool_regression_count"),
        library_candidates=metrics_blob.get("library_candidates"),
        library_promoted=metrics_blob.get("library_promoted"),
        library_generated_tools=metrics_blob.get("library_generated_tools"),
        library_generated_candidates=metrics_blob.get("library_generated_candidates"),
        library_generated_promoted=metrics_blob.get("library_generated_promoted"),
        library_mastered_primitives=metrics_blob.get("library_mastered_primitives"),
        tool_card_revisions=metrics_blob.get("tool_card_revisions"),
        capability_gaps=metrics_blob.get("capability_gaps"),
        verified_composites=metrics_blob.get("verified_composites"),
        verified_generated_tools=metrics_blob.get("verified_generated_tools"),
        unverified_ephemeral_tools=metrics_blob.get("unverified_ephemeral_tools"),
        cross_model_reused_tools=metrics_blob.get("cross_model_reused_tools"),
        generated_tool_reuse_count=metrics_blob.get("generated_tool_reuse_count"),
        retrieved_tool_available_count=metrics_blob.get(
            "retrieved_tool_available_count"
        ),
        retrieved_tool_started_count=metrics_blob.get("retrieved_tool_started_count"),
        retrieved_tool_started_unique_count=metrics_blob.get(
            "retrieved_tool_started_unique_count"
        ),
        retrieved_tool_called_count=metrics_blob.get("retrieved_tool_called_count"),
        retrieved_tool_called_unique_count=metrics_blob.get(
            "retrieved_tool_called_unique_count"
        ),
        incident_success=all(
            metrics_blob.get(key) == 1.0
            for key in (
                "detection_score",
                "localization_accuracy",
                "rca_accuracy",
            )
        ),
        time_taken=_session_duration_seconds(
            run_meta.get("start_time"), run_meta.get("end_time")
        ),
        llm_judge_relevance_score=relevance_score,
        llm_judge_correctness_score=correctness_score,
        llm_judge_efficiency_score=efficiency_score,
        llm_judge_clarity_score=clarity_score,
        llm_judge_final_outcome_score=final_outcome_score,
        llm_judge_overall_score=overall_score,
        detection_score=metrics_blob.get("detection_score", -1.0),
        localization_accuracy=metrics_blob.get("localization_accuracy", -1.0),
        localization_precision=metrics_blob.get("localization_precision", -1.0),
        localization_recall=metrics_blob.get("localization_recall", -1.0),
        localization_f1=metrics_blob.get("localization_f1", -1.0),
        rca_accuracy=metrics_blob.get("rca_accuracy", -1.0),
        rca_precision=metrics_blob.get("rca_precision", -1.0),
        rca_recall=metrics_blob.get("rca_recall", -1.0),
        rca_f1=metrics_blob.get("rca_f1", -1.0),
    )


def write_eval_summary_csv(
    eval_results: list[EvalResult], output_path: str | Path
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(EvalResult()).keys())
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in eval_results:
            writer.writerow(asdict(result))
    return out
