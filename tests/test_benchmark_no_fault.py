from __future__ import annotations

from pathlib import Path

from langgraph.errors import GraphRecursionError

from nika.workflows.benchmark import run as benchmark_run


def test_no_fault_benchmark_skips_injection_and_writes_clean_gt(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, object]] = []

    class FakeSession:
        sessions: dict[str, "FakeSession"] = {}

        def load_running_session(self, session_id: str | None = None):
            stored = self.sessions[str(session_id)]
            self.__dict__.update(stored.__dict__)
            return self

        def update_session(self, key: str, value: object) -> None:
            setattr(self, key, value)
            self.sessions[self.session_id] = self
            calls.append((key, value))

        def write_gt(self, gt: dict[str, object]) -> None:
            self.gt = gt
            self.sessions[self.session_id] = self
            calls.append(("ground_truth", gt))

    class FakeNetEnv:
        def get_info(self) -> str:
            return "Network Description: clean test lab"

    def fake_start_net_env(*_: object, **__: object) -> str:
        session = FakeSession()
        session.session_id = "session-clean"
        session.session_dir = str(tmp_path / "session-clean")
        session.lab_name = "clean-lab"
        FakeSession.sessions[session.session_id] = session
        return session.session_id

    def fail_inject(*_: object, **__: object) -> None:
        raise AssertionError("no_fault benchmark must not inject failures")

    monkeypatch.setattr(benchmark_run, "ensure_kathara_clean", lambda **_: None)
    monkeypatch.setattr(benchmark_run, "validate_agent_extensions", lambda *_: None)
    monkeypatch.setattr(benchmark_run, "scenario_requires_topo_tier", lambda *_: False)
    monkeypatch.setattr(benchmark_run, "start_net_env", fake_start_net_env)
    monkeypatch.setattr(benchmark_run, "get_net_env_instance", lambda *_1, **_2: FakeNetEnv())
    monkeypatch.setattr(benchmark_run, "inject_failure", fail_inject)
    monkeypatch.setattr(benchmark_run, "start_agent", lambda *_1, **_2: None)
    monkeypatch.setattr(benchmark_run, "eval_results", lambda **_: None)
    monkeypatch.setattr(benchmark_run, "log_event", lambda *_1, **_2: None)

    import nika.utils.session as session_module

    monkeypatch.setattr(session_module, "Session", FakeSession)

    session_id = benchmark_run.run_single_benchmark(
        problem="no_fault",
        scenario="simple_bgp",
        topo_size="",
        agent_type="mock",
        llm_backend="mock",
        model="mock",
        max_steps=1,
        result_root=tmp_path,
        inject_params={},
    )

    assert session_id == "session-clean"
    assert ("problem_names", []) in calls
    assert ("root_cause_name", "no_fault") in calls
    assert (
        "ground_truth",
        {"is_anomaly": False, "faulty_devices": [], "root_cause_name": []},
    ) in calls


def test_single_benchmark_evaluates_agent_recursion_as_missing_submission(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    class FakeSession:
        sessions: dict[str, "FakeSession"] = {}

        def load_running_session(self, session_id: str | None = None):
            stored = self.sessions[str(session_id)]
            self.__dict__.update(stored.__dict__)
            return self

        def update_session(self, key: str, value: object) -> None:
            setattr(self, key, value)
            self.sessions[self.session_id] = self

        def write_gt(self, gt: dict[str, object]) -> None:
            self.gt = gt
            self.sessions[self.session_id] = self

    class FakeNetEnv:
        def get_info(self) -> str:
            return "Network Description: clean test lab"

    def fake_start_net_env(*_: object, **__: object) -> str:
        session = FakeSession()
        session.session_id = "session-recursion"
        session.session_dir = str(tmp_path / "session-recursion")
        session.lab_name = "clean-lab"
        FakeSession.sessions[session.session_id] = session
        return session.session_id

    def fake_start_agent(*_: object, **__: object) -> None:
        calls.append("start_agent")
        raise GraphRecursionError("recursion limit")

    def fake_eval_results(**_: object) -> None:
        calls.append("eval_results")

    monkeypatch.setattr(benchmark_run, "ensure_kathara_clean", lambda **_: None)
    monkeypatch.setattr(benchmark_run, "validate_agent_extensions", lambda *_: None)
    monkeypatch.setattr(benchmark_run, "scenario_requires_topo_tier", lambda *_: False)
    monkeypatch.setattr(benchmark_run, "start_net_env", fake_start_net_env)
    monkeypatch.setattr(benchmark_run, "get_net_env_instance", lambda *_1, **_2: FakeNetEnv())
    monkeypatch.setattr(benchmark_run, "start_agent", fake_start_agent)
    monkeypatch.setattr(benchmark_run, "eval_results", fake_eval_results)
    monkeypatch.setattr(benchmark_run, "log_event", lambda *_1, **_2: None)

    import nika.utils.session as session_module

    monkeypatch.setattr(session_module, "Session", FakeSession)

    session_id = benchmark_run.run_single_benchmark(
        problem="no_fault",
        scenario="simple_bgp",
        topo_size="",
        agent_type="mock",
        llm_backend="mock",
        model="mock",
        max_steps=1,
        result_root=tmp_path,
        inject_params={},
    )

    assert session_id == "session-recursion"
    assert calls == ["start_agent", "eval_results"]
