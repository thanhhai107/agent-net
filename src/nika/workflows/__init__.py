"""High-level pipeline steps grouped by CLI domain."""

from nika.workflows.agent import start_agent
from nika.workflows.benchmark import (
    default_benchmark_yaml_path,
    run_benchmark_from_yaml,
    run_single_case,
)
from nika.workflows.env import start_net_env
from nika.workflows.session import close_session, inspect_session, list_sessions, wipe_kathara_labs
from nika.workflows.eval import (
    EvalCleanReport,
    eval_results,
    run_eval_clean,
    run_eval_metrics,
    run_eval_summary,
    run_llm_judge,
)
from nika.workflows.exec import exec_command_in_host
from nika.workflows.failure import inject_failure

__all__ = [
    "EvalCleanReport",
    "close_session",
    "default_benchmark_yaml_path",
    "eval_results",
    "exec_command_in_host",
    "inspect_session",
    "inject_failure",
    "list_sessions",
    "run_benchmark_from_yaml",
    "run_eval_clean",
    "run_eval_metrics",
    "run_eval_summary",
    "run_llm_judge",
    "run_single_case",
    "start_agent",
    "start_net_env",
    "wipe_kathara_labs",
]
