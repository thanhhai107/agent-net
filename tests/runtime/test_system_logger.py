import json
from pathlib import Path

from nika.utils.logger import (
    bind_session_dir,
    log_error_event,
    log_event,
    refresh_logger,
)


def test_log_event_and_error_event_write_jsonl(tmp_path: Path) -> None:
    refresh_logger()
    bind_session_dir(str(tmp_path))

    log_event("env_start", "started", session_id="s1")
    log_error_event(
        "env_start_failed", "deploy failed", session_id="s1", error="timeout"
    )

    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    info_entry = json.loads(lines[0])
    assert info_entry["level"] == "INFO"
    assert info_entry["event"] == "env_start"
    assert info_entry["data"]["session_id"] == "s1"

    error_entry = json.loads(lines[1])
    assert error_entry["level"] == "ERROR"
    assert error_entry["event"] == "env_start_failed"
    assert error_entry["data"]["error"] == "timeout"
