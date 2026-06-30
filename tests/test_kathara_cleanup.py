from __future__ import annotations

import subprocess

import pytest

from nika.utils.kathara_cleanup import KatharaCleanupError, ensure_kathara_clean


def test_ensure_kathara_clean_wipes_prunes_and_verifies(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    ensure_kathara_clean(context="test")

    assert calls == [
        ["kathara", "wipe", "-f"],
        ["docker", "network", "prune", "-f", "--filter", "label=app=kathara"],
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            "label=app=kathara",
            "--format",
            "{{.ID}} {{.Names}} {{.Status}}",
        ],
        [
            "docker",
            "network",
            "ls",
            "--filter",
            "label=app=kathara",
            "--format",
            "{{.ID}} {{.Name}}",
        ],
    ]


def test_ensure_kathara_clean_fails_on_wipe_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="active endpoints",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(KatharaCleanupError, match="kathara wipe"):
        ensure_kathara_clean(context="test")


def test_ensure_kathara_clean_fails_when_kathara_resources_remain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        stdout = ""
        if command[:3] == ["docker", "ps", "-a"]:
            stdout = "abc123 kathara_pc Up 1 minute\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(KatharaCleanupError, match="cleanup incomplete"):
        ensure_kathara_clean(context="test")
