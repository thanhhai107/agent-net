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

    assert calls[0] == ["kathara", "wipe", "-f"]
    assert [
        "docker",
        "ps",
        "-aq",
        "--filter",
        "label=app=kathara",
    ] in calls
    assert ["docker", "ps", "-aq", "--filter", "name=kathara_"] in calls
    assert ["docker", "network", "prune", "-f", "--filter", "label=app=kathara"] in calls
    assert [
        "docker",
        "network",
        "ls",
        "--filter",
        "label=app=kathara",
        "--format",
        "{{.ID}} {{.Name}}",
    ] in calls
    assert [
        "docker",
        "network",
        "ls",
        "--filter",
        "name=kathara_",
        "--format",
        "{{.ID}} {{.Name}}",
    ] in calls


def test_ensure_kathara_clean_fails_on_wipe_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["docker", "ps", "-aq"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="active endpoints",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(KatharaCleanupError, match="kathara wipe"):
        ensure_kathara_clean(context="test")


def test_ensure_kathara_clean_removes_stale_kathara_containers_before_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    wipe_attempts = 0

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        nonlocal wipe_attempts
        calls.append(command)
        if command == ["kathara", "wipe", "-f"]:
            wipe_attempts += 1
            if wipe_attempts == 1:
                return subprocess.CompletedProcess(command, 1, stdout="active endpoints", stderr="")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["docker", "ps", "-aq"]:
            stdout = "abc123\n" if command[-1] == "label=app=kathara" else ""
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("nika.utils.kathara_cleanup.time.sleep", lambda *_: None)

    ensure_kathara_clean(context="test")

    assert ["docker", "rm", "-f", "abc123"] in calls
    assert wipe_attempts == 2


def test_ensure_kathara_clean_removes_containers_attached_to_kathara_networks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    wipe_attempts = 0
    network_removed = False

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        nonlocal network_removed, wipe_attempts
        calls.append(command)
        if command == ["kathara", "wipe", "-f"]:
            wipe_attempts += 1
            if wipe_attempts == 1:
                return subprocess.CompletedProcess(
                    command,
                    1,
                    stdout='network kathara_stale has active endpoints (name:"orphan" id:"dead")',
                    stderr="",
                )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["docker", "ps", "-aq"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["docker", "network", "ls"] and command[4] == "name=kathara_":
            if network_removed:
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            return subprocess.CompletedProcess(command, 0, stdout="net123 kathara_stale\n", stderr="")
        if command[:3] == ["docker", "network", "ls"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["docker", "network", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout='{"orphan123": {"Name": "orphan"}}\n', stderr="")
        if command[:3] == ["docker", "network", "rm"]:
            network_removed = True
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("nika.utils.kathara_cleanup.time.sleep", lambda *_: None)

    ensure_kathara_clean(context="test")

    assert ["docker", "rm", "-f", "orphan123"] in calls
    assert wipe_attempts == 2


def test_ensure_kathara_clean_tolerates_empty_stale_network_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command == ["kathara", "wipe", "-f"]:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout='network kathara_stale has active endpoints (name:"gone" id:"dead")',
                stderr="",
            )
        if command[:3] == ["docker", "ps", "-aq"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["docker", "network", "ls"]:
            return subprocess.CompletedProcess(command, 0, stdout="net123 kathara_stale\n", stderr="")
        if command[:3] == ["docker", "network", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout="{}\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    ensure_kathara_clean(context="test")

    assert ["docker", "network", "inspect", "net123", "--format", "{{json .Containers}}"] in calls
    assert ["docker", "network", "prune", "-f", "--filter", "label=app=kathara"] in calls


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
