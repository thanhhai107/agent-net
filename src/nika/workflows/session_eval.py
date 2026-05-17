"""Session evaluation: numeric metrics, LLM judge, and publish to CSV + teardown."""

import json
import os
import textwrap
from pathlib import Path

from nika.evaluator.llm_judge import JudgeResponse, LLMJudge
from nika.evaluator.result_log import EvalResult, record_eval_result
from nika.evaluator.trace_parser import AgentTraceParser
from nika.net_env.net_env_pool import get_net_env_instance
from nika.orchestrator.problems.prob_pool import get_problem_instance
from nika.orchestrator.tasks.detection import DetectionSubmission
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask
from nika.utils.logger import system_logger
from nika.utils.session import Session
from nika.utils.session_store import SessionStore

logger = system_logger

EVAL_METRICS_FILENAME = "eval_metrics.json"


def generic_eval(gt, submission):
    """Score detection, localization, and RCA from structured ``gt`` and ``submission``."""
    try:
        parsed_detect_sub = DetectionSubmission.model_validate({"is_anomaly": submission.get("is_anomaly", False)})
        if gt["is_anomaly"] == parsed_detect_sub.is_anomaly:
            detection_score = 1.0
        else:
            detection_score = 0.0
    except Exception:
        detection_score = -1.0

    try:
        loc_acc, loc_prec, loc_rec, loc_f1 = LocalizationTask().eval(
            submission={"faulty_devices": submission.get("faulty_devices", [])},
            gt={"faulty_devices": gt.get("faulty_devices", [])},
        )
    except Exception:
        loc_acc, loc_prec, loc_rec, loc_f1 = -1.0, -1.0, -1.0, -1.0

    try:
        rca_acc, rca_prec, rca_rec, rca_f1 = RCATask().eval(
            submission={"root_cause_name": submission.get("root_cause_name", [])},
            gt={"root_cause_name": gt.get("root_cause_name", [])},
        )
    except Exception:
        rca_acc, rca_prec, rca_rec, rca_f1 = -1.0, -1.0, -1.0, -1.0

    return (
        detection_score,
        loc_acc,
        loc_prec,
        loc_rec,
        loc_f1,
        rca_acc,
        rca_prec,
        rca_rec,
        rca_f1,
    )


def run_eval_metrics(*, session_id: str | None = None) -> None:
    """Compute rule-based scores and trace stats; write ``eval_metrics.json`` under the session dir."""
    session = Session()
    session.load_running_session(session_id=session_id)

    gt_path = Path(session.session_dir) / "ground_truth.json"
    gt = json.loads(gt_path.read_text())

    submission_path = Path(session.session_dir) / "submission.json"
    if submission_path.exists():
        submission = json.loads(submission_path.read_text())
        (
            detection_score,
            loc_acc,
            loc_prec,
            loc_rec,
            loc_f1,
            rca_acc,
            rca_prec,
            rca_rec,
            rca_f1,
        ) = generic_eval(gt, submission)
    else:
        logger.error(f"Submission file not found: {submission_path}")
        detection_score = -1.0
        loc_acc = loc_prec = loc_rec = loc_f1 = -1.0
        rca_acc = rca_prec = rca_rec = rca_f1 = -1.0

    trace_path = os.path.join(session.session_dir, "conversation_diagnosis_agent.log")
    trace_metrics = AgentTraceParser(trace_path=trace_path).parse_trace()

    payload = {
        "detection_score": detection_score,
        "localization_accuracy": loc_acc,
        "localization_precision": loc_prec,
        "localization_recall": loc_rec,
        "localization_f1": loc_f1,
        "rca_accuracy": rca_acc,
        "rca_precision": rca_prec,
        "rca_recall": rca_rec,
        "rca_f1": rca_f1,
        "in_tokens": trace_metrics.get("in_tokens"),
        "out_tokens": trace_metrics.get("out_tokens"),
        "steps": trace_metrics.get("steps"),
        "tool_calls": trace_metrics.get("tool_calls"),
        "tool_errors": trace_metrics.get("tool_errors"),
    }
    out_path = Path(session.session_dir) / EVAL_METRICS_FILENAME
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    session.update_session("eval_metrics_json", payload)
    logger.info(f"Wrote numeric eval metrics to {out_path}")


def run_llm_judge(judge_llm_backend: str, judge_model: str, *, session_id: str | None = None) -> None:
    """Run LLM-as-judge only; writes ``llm_judge.json`` under the session dir."""
    session = Session()
    session.load_running_session(session_id=session_id)

    gt_path = Path(session.session_dir) / "ground_truth.json"
    gt = json.loads(gt_path.read_text())

    trace_path = os.path.join(session.session_dir, "conversation_diagnosis_agent.log")
    logger.info(f"Evaluating session {session.session_id} using LLM-as-Judge.")

    llm_judge = LLMJudge(judge_llm_backend=judge_llm_backend, judge_model=judge_model)
    llm_judge.evaluate_agent(
        ground_truth=textwrap.dedent(
            f"""\
                The root cause is {gt["root_cause_name"]}.
                The faulty devices are: {", ".join(gt["faulty_devices"])}.
            """
        ),
        trace_path=trace_path,
        save_path=f"{session.session_dir}/llm_judge.json",
    )
    judge_path = Path(session.session_dir) / "llm_judge.json"
    if judge_path.exists():
        session.update_session("llm_judge_json", json.loads(judge_path.read_text(encoding="utf-8")))


def publish_session_eval(*, destroy_env: bool = True, session_id: str | None = None) -> None:
    """Merge artifacts, append one CSV row via ``record_eval_result``, then undeploy and clear session."""
    session = Session()
    session.load_running_session(session_id=session_id)

    metrics_path = Path(session.session_dir) / EVAL_METRICS_FILENAME
    metrics_blob: dict = {}
    if metrics_path.exists():
        metrics_blob = json.loads(metrics_path.read_text(encoding="utf-8"))

    judge_path = Path(session.session_dir) / "llm_judge.json"
    judge_response: JudgeResponse | None = None
    if judge_path.exists():
        judge_response = JudgeResponse.model_validate_json(judge_path.read_text(encoding="utf-8"))

    trace_path = os.path.join(session.session_dir, "conversation_diagnosis_agent.log")
    trace_metrics = {
        "in_tokens": metrics_blob.get("in_tokens"),
        "out_tokens": metrics_blob.get("out_tokens"),
        "steps": metrics_blob.get("steps"),
        "tool_calls": metrics_blob.get("tool_calls"),
        "tool_errors": metrics_blob.get("tool_errors"),
    }
    if not any(v is not None for v in trace_metrics.values()) and os.path.exists(trace_path):
        trace_metrics = AgentTraceParser(trace_path=trace_path).parse_trace()

    detection_score = metrics_blob.get("detection_score", -1.0)
    loc_acc = metrics_blob.get("localization_accuracy", -1.0)
    loc_prec = metrics_blob.get("localization_precision", -1.0)
    loc_rec = metrics_blob.get("localization_recall", -1.0)
    loc_f1 = metrics_blob.get("localization_f1", -1.0)
    rca_acc = metrics_blob.get("rca_accuracy", -1.0)
    rca_prec = metrics_blob.get("rca_precision", -1.0)
    rca_rec = metrics_blob.get("rca_recall", -1.0)
    rca_f1 = metrics_blob.get("rca_f1", -1.0)

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

    problem = get_problem_instance(
        problem_names=session.problem_names,
        task_level="detection",
        scenario_name=session.scenario_name,
        **(session.scenario_params if hasattr(session, "scenario_params") else {}),
    )

    if session.end_time is None:
        session.end_session()

    logger.info(
        f"Publish eval for session {session.session_id}: "
        f"overall_judge={overall_score}, detection={detection_score}, loc_acc={loc_acc}, rca_acc={rca_acc}."
    )

    eval_result = EvalResult(
        agent_type=session.agent_type,
        model=session.model,
        root_cause_category=problem.root_cause_category,
        root_cause_name=problem.root_cause_name,
        net_env=session.scenario_name,
        scenario_topo_size=session.scenario_topo_size,
        session_id=session.session_id,
        in_tokens=trace_metrics.get("in_tokens", None),
        out_tokens=trace_metrics.get("out_tokens", None),
        steps=trace_metrics.get("steps", None),
        tool_calls=trace_metrics.get("tool_calls", None),
        tool_errors=trace_metrics.get("tool_errors", None),
        time_taken=round(float(session.end_time) - float(session.start_time), 2),
        llm_judge_relevance_score=relevance_score,
        llm_judge_correctness_score=correctness_score,
        llm_judge_efficiency_score=efficiency_score,
        llm_judge_clarity_score=clarity_score,
        llm_judge_final_outcome_score=final_outcome_score,
        llm_judge_overall_score=overall_score,
        detection_score=detection_score,
        localization_accuracy=loc_acc,
        localization_precision=loc_prec,
        localization_recall=loc_rec,
        localization_f1=loc_f1,
        rca_accuracy=rca_acc,
        rca_precision=rca_prec,
        rca_recall=rca_rec,
        rca_f1=rca_f1,
    )

    record_eval_result(eval_result)
    session.update_session(
        "eval_summary_json",
        {
            "detection_score": detection_score,
            "localization_accuracy": loc_acc,
            "rca_accuracy": rca_acc,
            "overall_judge_score": overall_score,
        },
    )

    net_env_kwargs = {}
    if session.scenario_topo_size is not None:
        net_env_kwargs["topo_size"] = session.scenario_topo_size
    net_env = get_net_env_instance(session.scenario_name, **net_env_kwargs)
    if destroy_env and net_env.lab_exists():
        net_env.undeploy()
    logger.info(f"Destroyed network environment: {session.scenario_name} with session ID: {session.session_id}")
    ended_cnt = SessionStore().mark_session_failures_ended(session.session_id)
    if ended_cnt:
        logger.info(f"Marked {ended_cnt} failure record(s) as ended for session {session.session_id}")
    session.clear_session()


def eval_results(
    judge_llm_backend: str,
    judge_model: str,
    *,
    destroy_env: bool = True,
    session_id: str | None = None,
) -> None:
    """Run metrics, LLM judge, and publish in one call (benchmark / legacy pipeline)."""
    run_eval_metrics(session_id=session_id)
    run_llm_judge(judge_llm_backend, judge_model, session_id=session_id)
    publish_session_eval(destroy_env=destroy_env, session_id=session_id)
