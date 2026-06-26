from nika.orchestrator.tasks.localization import LocalizationTask


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
