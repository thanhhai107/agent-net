"""Session cleanup invariants across normal, partial, and stop-all paths."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from nika.runtime.base import LabCleanupError
from nika.workflows.session import close as session_close


def _session_meta(session_id: str = "session-1") -> dict:
    return {
        "session_id": session_id,
        "session_dir": f"/tmp/{session_id}",
        "scenario_name": "simple_bgp",
        "lab_name": f"simple_bgp__{session_id}",
        "backend": "kathara",
        "scenario_params": {
            "lab_name": f"simple_bgp__{session_id}",
            "backend": "kathara",
        },
    }


def test_stop_record_undeploys_without_machine_only_exists_gate(monkeypatch) -> None:
    session = Mock()
    net_env = Mock()
    store = Mock()
    store.mark_session_failures_ended.return_value = 0

    monkeypatch.setattr(session_close, "Session", Mock(return_value=session))
    monkeypatch.setattr(
        session_close, "get_net_env_instance", Mock(return_value=net_env)
    )
    monkeypatch.setattr(session_close, "SessionStore", Mock(return_value=store))
    monkeypatch.setattr(session_close, "bind_session_dir", Mock())
    monkeypatch.setattr(
        session_close, "remove_session_runtime_workdir", Mock(return_value=False)
    )

    session_close._stop_session_record(_session_meta(), undeploy=True)

    net_env.undeploy.assert_called_once_with()
    net_env.lab_exists.assert_not_called()
    session.clear_session.assert_called_once_with(status="finished")


def test_stop_record_keeps_session_when_cleanup_is_unproven(monkeypatch) -> None:
    session = Mock()
    net_env = Mock()
    net_env.undeploy.side_effect = LabCleanupError("one link remains")
    store = Mock()

    monkeypatch.setattr(session_close, "Session", Mock(return_value=session))
    monkeypatch.setattr(
        session_close, "get_net_env_instance", Mock(return_value=net_env)
    )
    monkeypatch.setattr(session_close, "SessionStore", Mock(return_value=store))
    monkeypatch.setattr(session_close, "bind_session_dir", Mock())

    with pytest.raises(LabCleanupError, match="one link remains"):
        session_close._stop_session_record(_session_meta(), undeploy=True)

    store.mark_session_failures_ended.assert_not_called()
    session.clear_session.assert_not_called()


def test_stop_all_continues_then_reconciles_failed_records(monkeypatch) -> None:
    store = Mock()
    store.list_running_sessions.return_value = [
        {"session_id": "session-1"},
        {"session_id": "session-2"},
    ]
    store.get_session.side_effect = [
        _session_meta("session-1"),
        _session_meta("session-2"),
    ]
    calls: list[str] = []
    first_attempt = True

    def stop_record(meta, **_kwargs):
        nonlocal first_attempt
        calls.append(meta["session_id"])
        if meta["session_id"] == "session-1" and first_attempt:
            first_attempt = False
            raise LabCleanupError("transient teardown race")

    global_cleanup = Mock()
    monkeypatch.setattr(session_close, "SessionStore", Mock(return_value=store))
    monkeypatch.setattr(session_close, "_stop_session_record", stop_record)
    monkeypatch.setattr(session_close, "clean_emulation_environment", global_cleanup)
    monkeypatch.setattr(session_close, "wipe_runtime_artifacts", Mock(return_value=0))

    session_close.close_session(stop_all=True)

    assert calls == ["session-1", "session-2", "session-1"]
    global_cleanup.assert_called_once_with()
