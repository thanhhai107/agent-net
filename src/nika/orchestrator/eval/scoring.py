"""Rule-based scoring for detection, localization, and RCA submissions."""

from pydantic import ValidationError

from nika.orchestrator.eval.submissions import (
    DetectionSubmission,
    LocalizationSubmission,
    RCASubmission,
)


def score_detection(submission: dict, gt: dict) -> float:
    """Score binary anomaly detection."""
    try:
        is_anomaly = submission.get("is_anomaly", -1.0)
        if is_anomaly in ("True", "true", "1", 1, True, "yes", "Yes"):
            is_anomaly = True
        elif is_anomaly in ("False", "false", "0", 0, False, "no", "No"):
            is_anomaly = False
        else:
            return 0.0
        parsed = DetectionSubmission(is_anomaly=is_anomaly)
        return 1.0 if gt["is_anomaly"] == parsed.is_anomaly else 0.0
    except Exception:
        return -1.0


def score_localization(submission: dict, gt: dict) -> tuple[float, float, float, float]:
    """Score localization via set precision/recall/F1 on faulty devices."""
    try:
        parsed_submission = LocalizationSubmission.model_validate(
            {"faulty_devices": submission.get("faulty_devices", [])}
        )
    except ValidationError:
        return -1.0, -1.0, -1.0, -1.0

    parsed_gt = LocalizationSubmission.model_validate(
        {"faulty_devices": gt.get("faulty_devices", [])}
    )
    correct_components = set(parsed_gt.faulty_devices)
    submitted_components = set(parsed_submission.faulty_devices)

    tp = len(correct_components & submitted_components)
    fp = len(submitted_components - correct_components)
    fn = len(correct_components - submitted_components)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = tp / len(correct_components) if len(correct_components) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return (
        round(float(accuracy), 4),
        round(float(precision), 4),
        round(float(recall), 4),
        round(float(f1), 4),
    )


def score_rca(submission: dict, gt: dict) -> tuple[float, float, float, float]:
    """Score RCA via set precision/recall/F1 on root cause names."""
    sub_rc_names = submission.get("root_cause_name", None)
    if sub_rc_names is None:
        return -1.0, -1.0, -1.0, -1.0

    try:
        parsed_gt = RCASubmission.model_validate(
            {"root_cause_name": gt.get("root_cause_name", [])}
        )
    except ValidationError:
        return -1.0, -1.0, -1.0, -1.0

    correct_rc_names = set(parsed_gt.root_cause_name)
    submitted_rc_names = set(sub_rc_names)

    tp = len(correct_rc_names & submitted_rc_names)
    fp = len(submitted_rc_names - correct_rc_names)
    fn = len(correct_rc_names - submitted_rc_names)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = tp / len(correct_rc_names) if len(correct_rc_names) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return (
        round(float(accuracy), 4),
        round(float(precision), 4),
        round(float(recall), 4),
        round(float(f1), 4),
    )
