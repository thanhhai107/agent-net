from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask


def test_localization_eval_normalizes_device_case_and_whitespace() -> None:
    task = LocalizationTask()

    accuracy, precision, recall, f1 = task.eval(
        {"faulty_devices": [" PC2 "]},
        {"faulty_devices": ["pc2"]},
    )

    assert accuracy == 1.0
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0


def test_localization_accuracy_requires_exact_set() -> None:
    task = LocalizationTask()

    accuracy, precision, recall, f1 = task.eval(
        {"faulty_devices": ["pc2", "pc3"]},
        {"faulty_devices": ["pc2"]},
    )

    assert accuracy == 0.0
    assert precision == 0.5
    assert recall == 1.0
    assert f1 == 0.6667


def test_rca_accuracy_requires_exact_set() -> None:
    task = RCATask()

    accuracy, precision, recall, f1 = task.eval(
        {"root_cause_name": ["link_down", "host_incorrect_gateway"]},
        {"root_cause_name": ["link_down"]},
    )

    assert accuracy == 0.0
    assert precision == 0.5
    assert recall == 1.0
    assert f1 == 0.6667
