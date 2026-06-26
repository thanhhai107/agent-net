"""Session evaluation: numeric metrics, LLM judge, and publish on closed sessions."""

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
    session.load_closed_session(session_id=session_id)
    bind_session_dir(session.session_dir)

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

    trace_path = os.path.join(session.session_dir, MESSAGES_FILENAME)
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
        "primitive_calls": trace_metrics.get("primitive_calls"),
        "composite_calls": trace_metrics.get("composite_calls"),
        "evolved_tools_created": trace_metrics.get("evolved_tools_created"),
        "mastery_updates": trace_metrics.get("mastery_updates"),
    }
    if bool(getattr(session, "tool_evolution_enabled", False)):
        from agent.tool_evolution.curator import finalize_tool_evolution_session

        evolution = finalize_tool_evolution_session(
            session_id=session.session_id,
            metrics=payload,
        )
        payload.update(
            {
                "primitive_calls": (
                    (evolution.get("primitive_calls") or 0)
                    + (trace_metrics.get("primitive_calls") or 0)
                ),
                "composite_calls": evolution.get("composite_calls", 0),
                "evolved_tools_created": len(evolution.get("created_tools", [])),
                "mastery_updates": evolution.get("mastery_updates", 0),
                "tool_library_id": evolution.get("library_id"),
                "tool_evolution_mode": evolution.get("mode"),
                "tool_selection_recall": evolution.get("tool_selection_recall"),
                "argument_validity": evolution.get("argument_validity"),
                "error_recovery_count": evolution.get("error_recovery_count"),
                "tool_reuse_count": evolution.get("tool_reuse_count"),
                "tool_promotion_count": len(evolution.get("promoted_tools", [])),
                "tool_regression_count": len(evolution.get("regressed_tools", [])),
                "library_candidates": evolution.get("library_candidates"),
                "library_promoted": evolution.get("library_promoted"),
                "library_mastered_primitives": evolution.get(
                    "library_mastered_primitives"
                ),
                "tool_card_revisions": evolution.get("tool_card_revisions"),
                "capability_gaps": evolution.get("capability_gaps"),
                "verified_composites": evolution.get("verified_composites"),
                "unverified_ephemeral_tools": evolution.get(
                    "unverified_ephemeral_tools"
                ),
                "cross_model_reused_tools": evolution.get(
                    "cross_model_reused_tools"
                ),
            }
        )
    out_path = Path(session.session_dir) / EVAL_METRICS_FILENAME
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    session.update_run_meta("eval_metrics", payload)
    log_event("eval_metrics_saved", f"Wrote numeric eval metrics to {out_path}", session_id=session.session_id)


def run_llm_judge(judge_llm_backend: str, judge_model: str, *, session_id: str | None = None) -> None:
    """Run LLM-as-judge only; writes ``llm_judge.json`` under the session dir."""
    session = Session()
    session.load_closed_session(session_id=session_id)
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
        session.update_run_meta("llm_judge", json.loads(judge_path.read_text(encoding="utf-8")))


def publish_session_eval(*, session_id: str | None = None) -> None:
    """Validate eval artifacts on a closed session and record publish completion."""
    session = Session()
    session.load_closed_session(session_id=session_id)
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
        raise ValueError("--judge-backend and --judge-model are required when run_judge is enabled.")

    session = Session()
    session.load_running_session(session_id=session_id)
    resolved_session_id = session.session_id
    close_session(session_id=resolved_session_id, undeploy=destroy_env)
    run_eval_metrics(session_id=resolved_session_id)
    if run_judge:
        run_llm_judge(judge_llm_backend, judge_model, session_id=resolved_session_id)
    publish_session_eval(session_id=resolved_session_id)
