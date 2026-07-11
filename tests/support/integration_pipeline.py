"""Shared helpers for ordered agent pipeline integration tests."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from nika.utils.session_index import SessionIndex
from nika.utils.session_store import SessionStore
from nika.workflows.eval.session import run_eval_metrics
from nika.workflows.session.close import close_session
from tests.support.prerequisites import containerlab_prerequisites

SCENARIO = "simple_bgp"
PROBLEM = "link_down"
LINK_INJECT_PARAMS = {"host_name": "pc1", "intf_name": "eth0"}

CLAB_SCENARIO = "min3clos"
CLAB_LINK_INJECT_PARAMS = {"host_name": "leaf1", "intf_name": "e1-1"}

# Backward-compatible alias for agent pipeline tests.
_min3clos_prerequisites = containerlab_prerequisites


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
    return (
        bool(os.environ.get("OPENAI_API_KEY"))
        or (Path.home() / ".codex" / "auth.json").is_file()
    )


def claude_cli_available() -> bool:
    from agent.local_cli.claude_cli.config import claude_credentials_available

    return claude_credentials_available()


def deepseek_api_key_available() -> bool:
    return bool(os.environ.get("DEEPSEEK_API_KEY"))


def sade_available() -> bool:
    try:
        import claude_agent_sdk  # noqa: F401
    except ImportError:
        return False
    from agent.community.sade.config import sade_credentials_available

    return sade_credentials_available()


def claude_sdk_available() -> bool:
    try:
        import claude_agent_sdk  # noqa: F401
    except ImportError:
        return False
    from agent.sdk.claude_sdk.config import claude_sdk_credentials_available

    return claude_sdk_credentials_available()


def codex_sdk_available() -> bool:
    try:
        import openai_codex  # noqa: F401
    except ImportError:
        return False
    from agent.sdk.codex_sdk.config import codex_sdk_local_auth_available

    return codex_sdk_local_auth_available()


def tool_text_list(result: object) -> list[str]:
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            return [result]
    if not isinstance(result, list):
        return [str(result)]
    return [
        str(item["text"]) if isinstance(item, dict) and "text" in item else str(item)
        for item in result
    ]


class CommonPipelineSteps:
    """Mixin with shared step helpers for ordered pipeline test cases."""

    def _step_start_env(self) -> None:
        type(self).session_id = self._start_env(SCENARIO)  # type: ignore[attr-defined]
        self._assert_session_ready(self.session_id, SCENARIO)  # type: ignore[attr-defined]

    def _step_inject_failure(self) -> None:
        self.assertIsNotNone(self.session_id)  # type: ignore[attr-defined]
        self._inject_failure(PROBLEM, LINK_INJECT_PARAMS)  # type: ignore[attr-defined]
        row = SessionStore().get_session(self.session_id)
        self.assertIn(PROBLEM, row.get("problem_names", []))  # type: ignore[attr-defined]
        self.assertIn("task_description", row)  # type: ignore[attr-defined]
        type(self).session_dir = Path(row["session_dir"])
        gt = json.loads((type(self).session_dir / "ground_truth.json").read_text())
        self.assertTrue(gt["is_anomaly"])  # type: ignore[attr-defined]
        self.assertIn(PROBLEM, gt["root_cause_name"])  # type: ignore[attr-defined]

    def _step_close_and_verify(self, expected_agent_type: str) -> None:
        self.assertIsNotNone(self.session_id)  # type: ignore[attr-defined]
        close_session(session_id=self.session_id)  # type: ignore[attr-defined]
        type(self).env_destroyed = True
        run = self._load_json("run.json")  # type: ignore[attr-defined]
        self.assertEqual(run["status"], "finished")  # type: ignore[attr-defined]
        self.assertEqual(run["agent_type"], expected_agent_type)  # type: ignore[attr-defined]

    def _step_eval_metrics(self, min_tool_calls: int = 1) -> None:
        self.assertIsNotNone(self.session_id)  # type: ignore[attr-defined]
        run_eval_metrics(session_id=self.session_id)  # type: ignore[attr-defined]
        metrics = self._load_json("eval_metrics.json")  # type: ignore[attr-defined]
        for field in (
            "detection_score",
            "localization_accuracy",
            "rca_accuracy",
            "tool_calls",
        ):
            self.assertIn(field, metrics)  # type: ignore[attr-defined]
        self.assertGreaterEqual(metrics["detection_score"], 0.0)  # type: ignore[attr-defined]
        self.assertGreaterEqual(metrics["tool_calls"], min_tool_calls)  # type: ignore[attr-defined]

        run = self._load_json("run.json")  # type: ignore[attr-defined]
        self.assertIn("eval_metrics", run)  # type: ignore[attr-defined]

        index_row = SessionIndex().get_row(self.session_id)  # type: ignore[attr-defined]
        self.assertIsNotNone(index_row)  # type: ignore[attr-defined]
        self.assertIsNotNone(index_row.get("detection_score"))  # type: ignore[attr-defined]


class ClabCommonPipelineSteps:
    """Mixin with shared step helpers for containerlab min3clos pipeline tests."""

    def _step_start_env(self) -> None:
        type(self).session_id = self._start_env(  # type: ignore[attr-defined]
            CLAB_SCENARIO
        )
        self._assert_session_ready(self.session_id, CLAB_SCENARIO)  # type: ignore[attr-defined]

    def _step_inject_failure(self) -> None:
        self.assertIsNotNone(self.session_id)  # type: ignore[attr-defined]
        self._inject_failure(PROBLEM, CLAB_LINK_INJECT_PARAMS)  # type: ignore[attr-defined]
        row = SessionStore().get_session(self.session_id)
        self.assertIn(PROBLEM, row.get("problem_names", []))  # type: ignore[attr-defined]
        self.assertIn("task_description", row)  # type: ignore[attr-defined]
        type(self).session_dir = Path(row["session_dir"])
        gt = json.loads((type(self).session_dir / "ground_truth.json").read_text())
        self.assertTrue(gt["is_anomaly"])  # type: ignore[attr-defined]
        self.assertIn(PROBLEM, gt["root_cause_name"])  # type: ignore[attr-defined]

    def _step_close_and_verify(self, expected_agent_type: str) -> None:
        self.assertIsNotNone(self.session_id)  # type: ignore[attr-defined]
        close_session(session_id=self.session_id)  # type: ignore[attr-defined]
        type(self).env_destroyed = True
        run = self._load_json("run.json")  # type: ignore[attr-defined]
        self.assertEqual(run["status"], "finished")  # type: ignore[attr-defined]
        self.assertEqual(run["agent_type"], expected_agent_type)  # type: ignore[attr-defined]

    def _step_eval_metrics(self, min_tool_calls: int = 1) -> None:
        self.assertIsNotNone(self.session_id)  # type: ignore[attr-defined]
        run_eval_metrics(session_id=self.session_id)  # type: ignore[attr-defined]
        metrics = self._load_json("eval_metrics.json")  # type: ignore[attr-defined]
        for field in (
            "detection_score",
            "localization_accuracy",
            "rca_accuracy",
            "tool_calls",
        ):
            self.assertIn(field, metrics)  # type: ignore[attr-defined]
        self.assertGreaterEqual(metrics["detection_score"], 0.0)  # type: ignore[attr-defined]
        self.assertGreaterEqual(metrics["tool_calls"], min_tool_calls)  # type: ignore[attr-defined]

        run = self._load_json("run.json")  # type: ignore[attr-defined]
        self.assertIn("eval_metrics", run)  # type: ignore[attr-defined]

        index_row = SessionIndex().get_row(self.session_id)  # type: ignore[attr-defined]
        self.assertIsNotNone(index_row)  # type: ignore[attr-defined]
        self.assertIsNotNone(index_row.get("detection_score"))  # type: ignore[attr-defined]
