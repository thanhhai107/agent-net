import json
from pathlib import Path

from nika.visualization.data import (
    discover_sessions,
    faulty_devices,
    load_session_bundle,
    parse_topology,
    replay_steps,
    timeline_rows,
)
from nika.visualization.topology import render_topology_svg


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_discover_and_load_finished_session(tmp_path: Path) -> None:
    results = tmp_path / "results"
    sessions = tmp_path / "sessions"
    session_dir = results / "session-1"
    _write_json(
        session_dir / "run.json",
        {
            "session_id": "session-1",
            "status": "finished",
            "scenario_name": "simple_bgp",
            "created_at": "2026-01-01T00:00:00Z",
            "task_description": (
                "Topology: (router1:eth0, router2:eth0), "
                "(router1:eth1, pc1:eth0)"
            ),
        },
    )
    _write_json(
        session_dir / "ground_truth.json",
        {"is_anomaly": True, "faulty_devices": ["pc1"]},
    )
    (session_dir / "events.jsonl").write_text(
        json.dumps({"timestamp": "2026-01-01", "event": "env_start", "message": "started"})
        + "\n",
        encoding="utf-8",
    )

    discovered = discover_sessions(results_dir=results, sessions_dir=sessions)
    assert [item["session_id"] for item in discovered] == ["session-1"]

    bundle = load_session_bundle(
        "session-1", results_dir=results, sessions_dir=sessions
    )
    assert parse_topology(bundle.meta) == [
        ("router1:eth0", "router2:eth0"),
        ("router1:eth1", "pc1:eth0"),
    ]
    assert faulty_devices(bundle.ground_truth) == {"pc1"}
    assert timeline_rows(bundle)[0]["event"] == "env_start"


def test_discover_and_load_nested_benchmark_session(tmp_path: Path) -> None:
    results = tmp_path / "results"
    sessions = tmp_path / "sessions"
    session_dir = results / "test-20260629-010203-000001" / "session-1"
    _write_json(
        session_dir / "run.json",
        {
            "session_id": "session-1",
            "session_dir": str(session_dir),
            "status": "finished",
            "scenario_name": "simple_bgp",
        },
    )
    _write_json(session_dir / "ground_truth.json", {"faulty_devices": ["router1"]})

    discovered = discover_sessions(results_dir=results, sessions_dir=sessions)
    assert [item["session_id"] for item in discovered] == ["session-1"]
    assert discovered[0]["session_dir"] == str(session_dir)

    bundle = load_session_bundle(
        "session-1", results_dir=results, sessions_dir=sessions
    )
    assert bundle.session_dir == session_dir
    assert faulty_devices(bundle.ground_truth) == {"router1"}


def test_runtime_session_overrides_result_metadata(tmp_path: Path) -> None:
    results = tmp_path / "results"
    sessions = tmp_path / "sessions"
    _write_json(
        results / "session-2" / "run.json",
        {"session_id": "session-2", "status": "finished"},
    )
    _write_json(
        sessions / "session-2.json",
        {
            "session_id": "session-2",
            "status": "running",
            "topology": [["pc1:eth0", "router1:eth0"]],
        },
    )

    discovered = discover_sessions(results_dir=results, sessions_dir=sessions)
    assert discovered[0]["status"] == "running"
    assert parse_topology(discovered[0]) == [("pc1:eth0", "router1:eth0")]


def test_topology_svg_marks_actual_and_predicted_faults() -> None:
    svg = render_topology_svg(
        [("pc1:eth0", "router1:eth0")],
        actual_faulty={"pc1"},
        predicted_faulty={"router1"},
        fault_interfaces={("pc1", "eth0")},
    )
    assert "pc1" in svg
    assert "router1" in svg
    assert "#70263a" in svg
    assert "#704817" in svg
    assert 'stroke-dasharray="10 7"' in svg


def test_replay_steps_pair_tool_results_and_detect_devices(tmp_path: Path) -> None:
    results = tmp_path / "results"
    sessions = tmp_path / "sessions"
    session_dir = results / "session-replay"
    _write_json(
        session_dir / "run.json",
        {
            "session_id": "session-replay",
            "status": "finished",
            "topology": [["pc1:eth0", "router1:eth0"]],
        },
    )
    records = [
        {
            "timestamp": "2026-01-01T00:00:00",
            "agent": "diagnosis_agent",
            "event": "tool_start",
            "tool": {"name": "ping_pair"},
            "input": '{"host_a":"pc1","host_b":"router1"}',
        },
        {
            "timestamp": "2026-01-01T00:00:01",
            "agent": "diagnosis_agent",
            "event": "tool_end",
            "output": "pc1 cannot reach router1",
        },
        {
            "timestamp": "2026-01-01T00:00:02",
            "agent": "diagnosis_agent",
            "event": "llm_end",
            "text": "Suspect pc1.",
        },
    ]
    (session_dir / "messages.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    bundle = load_session_bundle(
        "session-replay", results_dir=results, sessions_dir=sessions
    )

    steps = replay_steps(bundle)
    assert len(steps) == 2
    assert steps[0].title == "ping_pair"
    assert steps[0].output == "pc1 cannot reach router1"
    assert steps[0].devices == ("pc1", "router1")
    assert steps[1].kind == "response"
