from __future__ import annotations

import subprocess

import pytest

from nika.workflows.benchmark import run as benchmark_run
from nika.workflows.session import close as session_close
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


def test_single_benchmark_cleans_before_deploy(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_clean(*, context: str) -> None:
        calls.append(f"clean:{context}")

    def fake_start_net_env(*_: object, **__: object) -> str:
        calls.append("start_net_env")
        raise RuntimeError("stop after deploy boundary")

    monkeypatch.setattr(benchmark_run, "ensure_kathara_clean", fake_clean)
    monkeypatch.setattr(benchmark_run, "validate_agent_extensions", lambda *_: None)
    monkeypatch.setattr(benchmark_run, "scenario_requires_topo_tier", lambda *_: False)
    monkeypatch.setattr(benchmark_run, "start_net_env", fake_start_net_env)

    with pytest.raises(RuntimeError, match="deploy boundary"):
        benchmark_run.run_single_benchmark(
            problem="packet_loss",
            scenario="simple_bgp",
            topo_size="",
            agent_type="mock",
            llm_backend="mock",
            model="mock",
            max_steps=1,
        )

    assert calls == ["clean:benchmark case", "start_net_env"]


def test_session_wipe_uses_verified_kathara_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_clean(*, context: str) -> None:
        calls.append(context)

    monkeypatch.setattr(session_close, "ensure_kathara_clean", fake_clean)

    session_close.wipe_kathara_labs()

    assert calls == ["session wipe"]
