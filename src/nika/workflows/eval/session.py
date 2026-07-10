"""Session evaluation: numeric metrics, LLM judge, and publish on closed sessions."""

import asyncio
import json
import os
import textwrap
from pathlib import Path

from nika.evaluator.llm_judge import LLMJudge
from nika.evaluator.result_log import EVAL_METRICS_FILENAME, MESSAGES_FILENAME
from nika.evaluator.trace_parser import AgentTraceParser
from nika.orchestrator.tasks.detection import DetectionSubmission
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask
from nika.utils.logger import bind_session_dir, log_event, system_logger
from nika.utils.session import Session
from nika.workflows.session.close import close_session

logger = system_logger


def _load_closed_session(
    session: Session,
    *,
    session_id: str | None = None,
    results_dir: str | Path | None = None,
) -> None:
    if results_dir is None:
        session.load_closed_session(session_id=session_id)
    else:
        session.load_closed_session(session_id=session_id, results_dir=results_dir)


def generic_eval(gt, submission):
    """Score detection, localization, and RCA from structured ``gt`` and ``submission``."""
    try:
        parsed_detect_sub = DetectionSubmission.model_validate(
            {"is_anomaly": submission.get("is_anomaly", False)}
        )
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


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def invalid_detection_confusion_metrics() -> dict[str, float | int | bool | None]:
    return {
        "detection_valid": False,
        "detection_tp": 0,
        "detection_tn": 0,
        "detection_fp": 0,
        "detection_fn": 0,
        "detection_precision": None,
        "detection_recall": None,
        "detection_f1": None,
        "detection_false_positive_rate": None,
        "detection_true_positive_rate": None,
    }


def detection_confusion_metrics(gt, submission) -> dict[str, float | int | bool | None]:
    """Return binary detection confusion counts and derived metrics for one session."""
    try:
        gt_is_anomaly = bool(gt["is_anomaly"])
        parsed_detect_sub = DetectionSubmission.model_validate(
            {"is_anomaly": submission["is_anomaly"]}
        )
        submitted_is_anomaly = bool(parsed_detect_sub.is_anomaly)
    except Exception:
        return invalid_detection_confusion_metrics()

    tp = int(gt_is_anomaly and submitted_is_anomaly)
    tn = int((not gt_is_anomaly) and (not submitted_is_anomaly))
    fp = int((not gt_is_anomaly) and submitted_is_anomaly)
    fn = int(gt_is_anomaly and (not submitted_is_anomaly))
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = (
        None
        if precision is None or recall is None or precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    false_positive_rate = _safe_ratio(fp, fp + tn)
    true_positive_rate = recall

    return {
        "detection_valid": True,
        "detection_tp": tp,
        "detection_tn": tn,
        "detection_fp": fp,
        "detection_fn": fn,
        "detection_precision": precision,
        "detection_recall": recall,
        "detection_f1": f1,
        "detection_false_positive_rate": false_positive_rate,
        "detection_true_positive_rate": true_positive_rate,
    }


def run_eval_metrics(
    *,
    session_id: str | None = None,
    results_dir: str | Path | None = None,
) -> None:
    """Compute rule-based scores and trace stats; write ``eval_metrics.json`` under the session dir."""
    session = Session()
    _load_closed_session(session, session_id=session_id, results_dir=results_dir)
    bind_session_dir(session.session_dir)

    gt_path = Path(session.session_dir) / "ground_truth.json"
    gt = json.loads(gt_path.read_text())

    submission_path = Path(session.session_dir) / "submission.json"
    if submission_path.exists():
        submission = json.loads(submission_path.read_text())
        detection_breakdown = detection_confusion_metrics(gt, submission)
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
        detection_breakdown = invalid_detection_confusion_metrics()
        loc_acc = loc_prec = loc_rec = loc_f1 = -1.0
        rca_acc = rca_prec = rca_rec = rca_f1 = -1.0

    trace_path = os.path.join(session.session_dir, MESSAGES_FILENAME)
    trace_metrics = AgentTraceParser(trace_path=trace_path).parse_trace()

    payload = {
        "detection_score": detection_score,
        **detection_breakdown,
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
    if bool(getattr(session, "tool_evolution_enabled", False)):
        from agent.tool_evolution.curator import finalize_tool_evolution_session

        evolution = finalize_tool_evolution_session(
            session_id=session.session_id,
            metrics=payload,
        )
        payload.update(
            {
                "tool_library_id": evolution.get("library_id"),
                "draft_trials": evolution.get("draft_trials"),
                "draft_trials_added": evolution.get("draft_trials_added"),
                "draft_document_revisions": evolution.get(
                    "draft_document_revisions"
                ),
                "draft_comprehension_gaps": evolution.get(
                    "draft_comprehension_gaps"
                ),
                "draft_frozen_documents": evolution.get("draft_frozen_documents"),
                "draft_documented_tools": evolution.get("draft_documented_tools"),
                "draft_unique_trial_tools": evolution.get(
                    "draft_unique_trial_tools"
                ),
                "draft_explorations": evolution.get("draft_explorations"),
                "draft_planned_explorations": evolution.get(
                    "draft_planned_explorations"
                ),
                "draft_consumed_explorations": evolution.get(
                    "draft_consumed_explorations"
                ),
                "draft_analyzer_suggestions": evolution.get(
                    "draft_analyzer_suggestions"
                ),
                "draft_mastered_tools": evolution.get("draft_mastered_tools"),
                "draft_documented_path_rate": evolution.get(
                    "draft_documented_path_rate"
                ),
                "draft_success_path_rate": evolution.get(
                    "draft_success_path_rate"
                ),
                "draft_converged_documents": evolution.get(
                    "draft_converged_documents"
                ),
                "draft_llm_attempts": evolution.get("draft_llm_attempts"),
                "draft_llm_failures": evolution.get("draft_llm_failures"),
                "draft_llm_revisions": evolution.get("draft_llm_revisions"),
                "draft_llm_analyzer_revisions": evolution.get(
                    "draft_llm_analyzer_revisions"
                ),
                "draft_llm_analyzer_failures": evolution.get(
                    "draft_llm_analyzer_failures"
                ),
                "draft_llm_errors": evolution.get("draft_llm_errors"),
            }
        )
    out_path = Path(session.session_dir) / EVAL_METRICS_FILENAME
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    session.update_run_meta("eval_metrics", payload)
    log_event(
        "eval_metrics_saved",
        f"Wrote numeric eval metrics to {out_path}",
        session_id=session.session_id,
    )

    if getattr(session, "memory_mode", "off") == "evolve":
        try:
            from agent.memory.workflow import evolve_session_memory

            run_meta = {
                key: value for key, value in session.__dict__.items() if key != "store"
            }
            memory_report = asyncio.run(
                evolve_session_memory(
                    run_meta=run_meta,
                    metrics=payload,
                    session_dir=session.session_dir,
                )
            )
            payload["memory_update"] = memory_report
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            session.update_run_meta("eval_metrics", payload)
            session.update_run_meta("memory_update", memory_report)
            log_event(
                "memory_evolution_completed",
                (
                    f"Memory evolution {memory_report.get('status')} for "
                    f"session {session.session_id}"
                ),
                session_id=session.session_id,
                memory_report=memory_report,
            )
        except Exception as exc:
            logger.exception(
                "Memory evolution failed for session %s; evaluation remains valid.",
                session.session_id,
            )
            memory_report = {"status": "failed", "error": str(exc)}
            payload["memory_update"] = memory_report
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            session.update_run_meta("eval_metrics", payload)
            session.update_run_meta("memory_update", memory_report)
            log_event(
                "memory_evolution_failed",
                f"Memory evolution failed: {exc}",
                session_id=session.session_id,
            )


def run_llm_judge(
    judge_llm_backend: str,
    judge_model: str,
    *,
    session_id: str | None = None,
    results_dir: str | Path | None = None,
) -> None:
    """Run LLM-as-judge only; writes ``llm_judge.json`` under the session dir."""
    session = Session()
    _load_closed_session(session, session_id=session_id, results_dir=results_dir)
    bind_session_dir(session.session_dir)

    gt_path = Path(session.session_dir) / "ground_truth.json"
    gt = json.loads(gt_path.read_text())

    trace_path = os.path.join(session.session_dir, MESSAGES_FILENAME)
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
        session.update_run_meta(
            "llm_judge", json.loads(judge_path.read_text(encoding="utf-8"))
        )


def publish_session_eval(
    *,
    session_id: str | None = None,
    results_dir: str | Path | None = None,
) -> None:
    """Validate eval artifacts on a closed session and record publish completion."""
    session = Session()
    _load_closed_session(session, session_id=session_id, results_dir=results_dir)
    bind_session_dir(session.session_dir)

    metrics_path = Path(session.session_dir) / EVAL_METRICS_FILENAME
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"eval_metrics.json not found under {session.session_dir}. Run `nika eval metrics` first."
        )

    log_event(
        "eval_publish",
        f"Published evaluation for session {session.session_id} (scenario {session.scenario_name}).",
        session_id=session.session_id,
        scenario=session.scenario_name,
    )


def eval_results(
    *,
    destroy_env: bool = True,
    session_id: str | None = None,
    run_judge: bool = False,
    judge_llm_backend: str | None = None,
    judge_model: str | None = None,
) -> None:
    """Close the session, then run metrics and publish; LLM judge runs only when ``run_judge`` is set."""
    if run_judge and (not judge_llm_backend or not judge_model):
        raise ValueError(
            "--judge-backend and --judge-model are required when run_judge is enabled."
        )

    session = Session()
    session.load_running_session(session_id=session_id)
    resolved_session_id = session.session_id
    results_dir = Path(session.session_dir).parent
    close_session(session_id=resolved_session_id, undeploy=destroy_env)
    run_eval_metrics(session_id=resolved_session_id, results_dir=results_dir)
    if run_judge:
        run_llm_judge(
            judge_llm_backend,
            judge_model,
            session_id=resolved_session_id,
            results_dir=results_dir,
        )
    publish_session_eval(session_id=resolved_session_id, results_dir=results_dir)
