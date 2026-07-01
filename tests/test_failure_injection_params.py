import pytest
from pydantic import BaseModel, ValidationError

from nika.orchestrator.problems.link_failure.link_failure import LinkFailureParams
from nika.workflows.failure.inject import _sync_problem_runtime_from_params


class _DummyProblem:
    def __init__(self) -> None:
        self.faulty_devices = ["pc1"]
        self.faulty_intf = "eth0"


class _TwoHostParams(BaseModel):
    host_name: str | None = None
    host_name_2: str | None = None


class _InterfaceOnlyParams(BaseModel):
    intf_name: str | None = None


def test_typed_host_override_syncs_faulty_devices_and_interface() -> None:
    problem = _DummyProblem()

    _sync_problem_runtime_from_params(
        problem,
        LinkFailureParams(host_name="pc2", intf_name="eth7"),
    )

    assert problem.faulty_devices == ["pc2"]
    assert problem.faulty_intf == "eth7"


def test_interface_only_override_preserves_existing_faulty_devices() -> None:
    problem = _DummyProblem()

    _sync_problem_runtime_from_params(problem, _InterfaceOnlyParams(intf_name="eth1"))

    assert problem.faulty_devices == ["pc1"]
    assert problem.faulty_intf == "eth1"


def test_link_failure_params_require_host_name() -> None:
    with pytest.raises(ValidationError):
        LinkFailureParams(intf_name="eth1")


def test_two_host_override_updates_stable_slots() -> None:
    problem = _DummyProblem()
    problem.faulty_devices = ["router1", "pc1"]

    _sync_problem_runtime_from_params(
        problem,
        _TwoHostParams(host_name="router2", host_name_2="pc2"),
    )

    assert problem.faulty_devices == ["router2", "pc2"]
