"""Shared helpers for ordered agent pipeline integration tests."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from nika.utils.session_index import SessionIndex
from nika.utils.session_store import SessionStore

SCENARIO = "simple_bgp"
PROBLEM = "link_down"


def load_test_env() -> None:
    """Load ``.env`` from the repository root (idempotent)."""
    from dotenv import load_dotenv

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")


def openai_api_key_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def codex_cli_available() -> bool:
    if shutil.which("codex") is None:
        return False
    return bool(os.environ.get("OPENAI_API_KEY")) or (Path.home() / ".codex" / "auth.json").is_file()


def claude_cli_available() -> bool:
    from agent.local_cli.claude_cli.config import claude_credentials_available

    return claude_credentials_available()


def deepseek_api_key_available() -> bool:
    return bool(os.environ.get("DEEPSEEK_API_KEY"))


def tool_text_list(result: object) -> list[str]:
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            return [result]
    if not isinstance(result, list):
        return [str(result)]
    return [str(item["text"]) if isinstance(item, dict) and "text" in item else str(item) for item in result]


class CommonPipelineSteps:
    """Mixin with shared step helpers for ordered pipeline test cases."""

    def _step_start_env(self) -> None:
        out = self._invoke_ok(["env", "run", SCENARIO])  # type: ignore[attr-defined]
        match = re.search(r"session_id=(\S+)", out.strip())
        self.assertIsNotNone(match, f"session_id missing from env run output:\n{out}")  # type: ignore[attr-defined]
        type(self).session_id = match.group(1)
        self._assert_session_ready(self.session_id, SCENARIO)  # type: ignore[attr-defined]

    def _step_inject_failure(self) -> None:
        self.assertIsNotNone(self.session_id)  # type: ignore[attr-defined]
        self._invoke_ok(  # type: ignore[attr-defined]
            [
                "failure",
                "inject",
                PROBLEM,
                "--session_id",
                self.session_id,
                "--set",
                "host_name=pc1",
                "--set",
                "intf_name=eth0",
            ]
        )
        row = SessionStore().get_session(self.session_id)
        self.assertIn(PROBLEM, row.get("problem_names", []))  # type: ignore[attr-defined]
        self.assertIn("task_description", row)  # type: ignore[attr-defined]
        type(self).session_dir = Path(row["session_dir"])
        gt = json.loads((type(self).session_dir / "ground_truth.json").read_text())
        self.assertTrue(gt["is_anomaly"])  # type: ignore[attr-defined]
        self.assertIn(PROBLEM, gt["root_cause_name"])  # type: ignore[attr-defined]

    def _step_close_and_verify(self, expected_agent_type: str) -> None:
        self.assertIsNotNone(self.session_id)  # type: ignore[attr-defined]
        self._invoke_ok(["session", "close", "--session_id", self.session_id, "-y"])  # type: ignore[attr-defined]
        type(self).env_destroyed = True
        run = self._load_json("run.json")  # type: ignore[attr-defined]
        self.assertEqual(run["status"], "finished")  # type: ignore[attr-defined]
        self.assertEqual(run["agent_type"], expected_agent_type)  # type: ignore[attr-defined]

    def _step_eval_metrics(self, min_tool_calls: int = 1) -> None:
        self.assertIsNotNone(self.session_id)  # type: ignore[attr-defined]
        self._invoke_ok(["eval", "metrics", "--session_id", self.session_id])  # type: ignore[attr-defined]
        metrics = self._load_json("eval_metrics.json")  # type: ignore[attr-defined]
        for field in ("detection_score", "localization_accuracy", "rca_accuracy", "tool_calls"):
            self.assertIn(field, metrics)  # type: ignore[attr-defined]
        self.assertGreaterEqual(metrics["detection_score"], 0.0)  # type: ignore[attr-defined]
        self.assertGreaterEqual(metrics["tool_calls"], min_tool_calls)  # type: ignore[attr-defined]

        run = self._load_json("run.json")  # type: ignore[attr-defined]
        self.assertIn("eval_metrics", run)  # type: ignore[attr-defined]

        index_row = SessionIndex().get_row(self.session_id)  # type: ignore[attr-defined]
        self.assertIsNotNone(index_row)  # type: ignore[attr-defined]
        self.assertIsNotNone(index_row.get("detection_score"))  # type: ignore[attr-defined]
