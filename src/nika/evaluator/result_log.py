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
    tool_evolution_enabled: bool | None = None
    tool_library_id: str = None
    memory_mode: str = None
    memory_bank: str = None
    memory_skill_selector_mode: str = None
    memory_meta_controller_mode: str = None
    benchmark_index: int = None
    draft_trials: int = None
    draft_trials_added: int = None
    draft_document_revisions: int = None
    draft_comprehension_gaps: int = None
    draft_frozen_documents: int = None
    draft_documented_tools: int = None
    draft_unique_trial_tools: int = None
    draft_explorations: int = None
    draft_planned_explorations: int = None
    draft_consumed_explorations: int = None
    draft_analyzer_suggestions: int = None
    draft_mastered_tools: int = None
    draft_documented_path_rate: float = None
    draft_success_path_rate: float = None
    draft_converged_documents: int = None
    draft_llm_attempts: int = None
    draft_llm_failures: int = None
    draft_llm_revisions: int = None
    memory_update_status: str = None
    memory_skill_id: str = None
    memory_runtime_skill_ids: list[str] = None
    memory_episode_reward: float = None
    memory_episode_baseline: float = None
    memory_episode_advantage: float = None
    memory_episode_success: bool | None = None
    memory_total_added_tokens: int = None
    memory_delta_prompt_tokens_per_step: float = None
    memory_prompt_added_tokens: int = None
    memory_tool_description_added_tokens: int = None
    memory_followup_added_tokens: int = None
    memory_ppo_j_score: float = None
    memory_candidate_alignment: float = None
    memory_baseline_alignment: float = None
    memory_semantic_gradient_source: str = None
    memory_semantic_gradient_llm_failed: bool | None = None
    memory_skills: int = None
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
    detection_valid: bool | None = None
    detection_tp: int = None
    detection_tn: int = None
    detection_fp: int = None
    detection_fn: int = None
    detection_precision: float = None
    detection_recall: float = None
    detection_f1: float = None
    detection_false_positive_rate: float = None
    detection_true_positive_rate: float = None
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
    for run_path in sorted(root.rglob(RUN_FILENAME)):
        if "0_summary" in run_path.relative_to(root).parts:
            continue
        session_dirs.append(run_path.parent)
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
    }
    if not any(v is not None for v in trace_metrics.values()):
        trace_path = session_dir / MESSAGES_FILENAME
        if trace_path.exists():
            trace_metrics = AgentTraceParser(trace_path=str(trace_path)).parse_trace()
    memory_update = metrics_blob.get("memory_update") or {}
    memory_decision = (
        memory_update.get("decision")
        if isinstance(memory_update.get("decision"), dict)
        else {}
    )

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
        tool_evolution_enabled=bool(run_meta.get("tool_evolution_enabled", False)),
        tool_library_id=metrics_blob.get("tool_library_id")
        or run_meta.get("tool_library_id"),
        memory_mode=run_meta.get("memory_mode"),
        memory_bank=run_meta.get("memory_bank"),
        memory_skill_selector_mode=run_meta.get("memory_skill_selector_mode"),
        memory_meta_controller_mode=run_meta.get("memory_meta_controller_mode"),
        benchmark_index=run_meta.get("benchmark_index"),
        draft_trials=metrics_blob.get("draft_trials"),
        draft_trials_added=metrics_blob.get("draft_trials_added"),
        draft_document_revisions=metrics_blob.get("draft_document_revisions"),
        draft_comprehension_gaps=metrics_blob.get("draft_comprehension_gaps"),
        draft_frozen_documents=metrics_blob.get("draft_frozen_documents"),
        draft_documented_tools=metrics_blob.get("draft_documented_tools"),
        draft_unique_trial_tools=metrics_blob.get("draft_unique_trial_tools"),
        draft_explorations=metrics_blob.get("draft_explorations"),
        draft_planned_explorations=metrics_blob.get(
            "draft_planned_explorations"
        ),
        draft_consumed_explorations=metrics_blob.get(
            "draft_consumed_explorations"
        ),
        draft_analyzer_suggestions=metrics_blob.get("draft_analyzer_suggestions"),
        draft_mastered_tools=metrics_blob.get("draft_mastered_tools"),
        draft_documented_path_rate=metrics_blob.get("draft_documented_path_rate"),
        draft_success_path_rate=metrics_blob.get("draft_success_path_rate"),
        draft_converged_documents=metrics_blob.get("draft_converged_documents"),
        draft_llm_attempts=metrics_blob.get("draft_llm_attempts"),
        draft_llm_failures=metrics_blob.get("draft_llm_failures"),
        draft_llm_revisions=metrics_blob.get("draft_llm_revisions"),
        memory_update_status=memory_update.get("status"),
        memory_skill_id=memory_update.get("skill_id"),
        memory_runtime_skill_ids=memory_update.get("runtime_skill_ids"),
        memory_episode_reward=memory_update.get("episode_reward"),
        memory_episode_baseline=memory_update.get("episode_baseline"),
        memory_episode_advantage=memory_update.get("episode_advantage"),
        memory_episode_success=memory_update.get("episode_success"),
        memory_total_added_tokens=memory_update.get("total_added_tokens"),
        memory_delta_prompt_tokens_per_step=memory_update.get(
            "delta_prompt_tokens_per_step"
        ),
        memory_prompt_added_tokens=memory_update.get("prompt_added_tokens"),
        memory_tool_description_added_tokens=memory_update.get(
            "tool_description_added_tokens"
        ),
        memory_followup_added_tokens=memory_update.get("followup_added_tokens"),
        memory_ppo_j_score=memory_decision.get("j_score"),
        memory_candidate_alignment=memory_decision.get("candidate_alignment"),
        memory_baseline_alignment=memory_decision.get("baseline_alignment"),
        memory_semantic_gradient_source=memory_update.get("semantic_gradient_source"),
        memory_semantic_gradient_llm_failed=memory_update.get(
            "semantic_gradient_llm_failed"
        ),
        memory_skills=memory_update.get("skills"),
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
        detection_valid=metrics_blob.get("detection_valid"),
        detection_tp=metrics_blob.get("detection_tp"),
        detection_tn=metrics_blob.get("detection_tn"),
        detection_fp=metrics_blob.get("detection_fp"),
        detection_fn=metrics_blob.get("detection_fn"),
        detection_precision=metrics_blob.get("detection_precision"),
        detection_recall=metrics_blob.get("detection_recall"),
        detection_f1=metrics_blob.get("detection_f1"),
        detection_false_positive_rate=metrics_blob.get(
            "detection_false_positive_rate"
        ),
        detection_true_positive_rate=metrics_blob.get("detection_true_positive_rate"),
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
