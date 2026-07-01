"""Run executable SIA-H target agents."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from nika.config import _REPO_ROOT


FORBIDDEN_SOURCE_PATTERNS = (
    "ground_truth.json",
    "failure_injections",
    "problem_names",
    "benchmark_selected",
    "benchmark_full",
    "benchmark_test.csv",
    "run_benchmark_from_csv",
    "inject_failure",
    "tool_evolution_mcp_server",
    "fastmcp",
    "mcp.server.fastmcp",
)


@dataclass(frozen=True)
class HarnessExecutionConfig:
    target_agent_path: str | Path
    session_id: str
    dataset_dir: str | Path
    working_dir: str | Path
    llm_backend: str
    model: str
    max_steps: int
    timeout_seconds: int | None = None
    allow_failure: bool = False


@dataclass(frozen=True)
class HarnessExecutionResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.returncode == 0


def validate_target_agent_source(path: str | Path) -> None:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"target agent not found: {target}")
    source = target.read_text(encoding="utf-8")
    compile(source, str(target), "exec")
    lowered = source.lower()
    found = [pattern for pattern in FORBIDDEN_SOURCE_PATTERNS if pattern in lowered]
    if found:
        raise ValueError(
            "target_agent.py contains forbidden benchmark/private references: "
            + ", ".join(sorted(found))
        )
    if "def main(" not in source or "--session-id" not in source:
        raise ValueError("target_agent.py must expose the NIKA SIA-H CLI contract")


def _subprocess_env(session_id: str) -> dict[str, str]:
    env = os.environ.copy()
    src = str(Path(_REPO_ROOT) / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    env["NIKA_SESSION_ID"] = session_id
    return env


def _write_error(working_dir: Path, payload: dict) -> None:
    working_dir.mkdir(parents=True, exist_ok=True)
    (working_dir / "harness_error.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def run_harness_target(config: HarnessExecutionConfig) -> HarnessExecutionResult:
    target = Path(config.target_agent_path)
    working_dir = Path(config.working_dir)
    stdout_path = working_dir / "target_agent_stdout.log"
    try:
        validate_target_agent_source(target)
    except Exception as exc:
        _write_error(
            working_dir,
            {
                "stage": "preflight",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        if not config.allow_failure:
            raise
        return HarnessExecutionResult(returncode=2, stdout="", stderr=str(exc))

    timeout = config.timeout_seconds or max(600, int(config.max_steps) * 90)
    cmd = [
        sys.executable,
        "-u",
        str(target),
        "--session-id",
        config.session_id,
        "--dataset-dir",
        str(config.dataset_dir),
        "--working-dir",
        str(working_dir),
        "--backend",
        config.llm_backend,
        "--model",
        config.model,
        "--max-steps",
        str(config.max_steps),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=_REPO_ROOT,
            env=_subprocess_env(config.session_id),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        stdout_path.write_text(stdout + stderr, encoding="utf-8")
        _write_error(
            working_dir,
            {
                "stage": "subprocess",
                "error_type": "TimeoutExpired",
                "error": f"target_agent.py timed out after {timeout}s",
            },
        )
        if not config.allow_failure:
            raise RuntimeError(f"target_agent.py timed out after {timeout}s") from exc
        return HarnessExecutionResult(returncode=124, stdout=stdout, stderr=stderr)

    combined = proc.stdout + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    stdout_path.write_text(combined, encoding="utf-8")
    if proc.returncode != 0:
        _write_error(
            working_dir,
            {
                "stage": "subprocess",
                "error_type": "NonZeroExit",
                "returncode": proc.returncode,
                "stderr": proc.stderr[-4000:],
            },
        )
        if not config.allow_failure:
            raise RuntimeError(
                f"target_agent.py exited {proc.returncode}: {proc.stderr[-1000:]}"
            )
    return HarnessExecutionResult(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )
