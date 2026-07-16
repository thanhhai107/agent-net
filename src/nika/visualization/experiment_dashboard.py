"""Streamlit experiment runner for NIKA agent baselines and modules."""

from __future__ import annotations

import json
import html
import re
import shlex
from pathlib import Path
from typing import Any

import streamlit as st
from yaml import YAMLError

from agent.composition import ProceduralMemoryConfig, ToolRefinementConfig
from agent.module_config import module_defaults
from agent.extensions.config import (
    DEFAULT_LLM_PROVIDER as DEFAULT_LLM_BACKEND,
    DEFAULT_MODEL,
)
from nika.config import BENCHMARK_DIR, RESULTS_DIR
from nika.visualization.experiment_runner import (
    build_command_plan,
    create_run,
    list_runs,
    parse_progress_events,
    prepare_experiment_config,
    read_run_log,
    read_run_spec,
    resume_run,
    run_status,
    stop_run,
)
from nika.workflows.benchmark.load_config import (
    benchmark_case_identity,
    load_benchmark_manifest,
)


TOOL_REFINEMENT_DEFAULTS = ToolRefinementConfig()
PROCEDURAL_MEMORY_DEFAULTS = ProceduralMemoryConfig()
MODULE_DEFAULTS = module_defaults()
BASELINE_DEFAULTS = MODULE_DEFAULTS.baseline
DEFAULT_STUDIO_TRAINING_BENCHMARK = str(
    BENCHMARK_DIR / BASELINE_DEFAULTS.training_benchmark
)
DEFAULT_STUDIO_EVALUATE_BENCHMARK = str(
    BENCHMARK_DIR / BASELINE_DEFAULTS.evaluate_benchmark
)
DEFAULT_STUDIO_MAX_STEPS = BASELINE_DEFAULTS.max_steps


st.set_page_config(
    page_title="NIKA · Experiment Studio",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      :root {
        --nika-text: var(--text-color, #0f172a);
        --nika-bg: var(--background-color, #f8fafc);
        --nika-secondary: var(--secondary-background-color, #f1f5f9);
        --nika-accent: #2563eb;
        --nika-muted: color-mix(in srgb, var(--nika-text) 62%, transparent);
        --nika-panel: color-mix(in srgb, var(--nika-secondary) 88%, transparent);
        --nika-elevated: color-mix(in srgb, var(--nika-bg) 82%, var(--nika-secondary));
        --nika-line: color-mix(in srgb, var(--nika-text) 14%, transparent);
        --nika-shadow: color-mix(in srgb, var(--nika-text) 10%, transparent);
        --nika-accent-soft: color-mix(in srgb, var(--nika-accent) 14%, transparent);
        --nika-accent-text: color-mix(in srgb, var(--nika-accent) 72%, var(--nika-text));
        --nika-success: #22c55e;
        --nika-danger: #ef4444;
        --nika-warning: #f59e0b;
      }

      .stApp {
        background: var(--nika-bg);
        color: var(--nika-text);
      }
      header[data-testid="stHeader"] {display: none !important;}
      .block-container {max-width: 1480px; padding: 2.8rem 2rem 4rem !important;}
      section[data-testid="stSidebar"] {
        background: var(--nika-secondary);
        border-right: 1px solid var(--nika-line);
      }
      section[data-testid="stSidebar"] .block-container {padding: 1.35rem 1.15rem;}
      h1, h2, h3 {letter-spacing: -.025em;}
      h1 {font-size: clamp(2rem, 3vw, 3.15rem) !important; line-height: 1.05 !important;}
      h2 {font-size: 1.28rem !important;}
      h3 {font-size: 1.02rem !important; color: var(--nika-text) !important;}

      .section-card {
        border:1px solid var(--nika-line); border-radius:12px; padding:1.2rem;
        background:var(--nika-panel); margin:.55rem 0 1rem;
        box-shadow:0 14px 35px var(--nika-shadow);
      }
      .section-title {
        color: var(--nika-text); font-weight: 750; font-size: 1.1rem; margin: 1.2rem 0 .9rem;
      }
      .mini-grid {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.55rem;}
      .mini-item {
        border:1px solid var(--nika-line); border-radius:12px; background:var(--nika-elevated);
        padding:.55rem .65rem; min-height:58px;
      }
      .mini-label {font-size:.7rem; color:var(--nika-muted); font-weight:700;}
      .mini-value {font-size:.82rem; color:var(--nika-text); font-weight:750; overflow-wrap:anywhere;}

      .status-pill {
        display:inline-flex; align-items:center; gap:.45rem; padding:.34rem .65rem;
        border:1px solid var(--nika-line); border-radius:999px; color:var(--nika-text);
        background:var(--nika-elevated); font-size:.78rem; font-weight:700;
      }
      .status-dot {width:7px; height:7px; border-radius:50%; background:#8a98aa;}
      .status-running .status-dot {background:var(--nika-success); box-shadow:0 0 12px color-mix(in srgb, var(--nika-success) 45%, transparent);}
      .status-finished .status-dot {background:#3b82f6;}
      .status-failed .status-dot {background:var(--nika-danger); box-shadow:0 0 12px color-mix(in srgb, var(--nika-danger) 55%, transparent);}
      .status-queued .status-dot {background:var(--nika-warning); box-shadow:0 0 12px color-mix(in srgb, var(--nika-warning) 55%, transparent);}

      [data-testid="stMetric"] {
        min-height: 112px;
        border: 1px solid var(--nika-line);
        border-radius: 16px;
        padding: 17px 18px;
        background: var(--nika-elevated);
        box-shadow: 0 12px 35px var(--nika-shadow);
      }
      [data-testid="stMetricLabel"] {color: var(--nika-muted); font-size: .78rem;}
      [data-testid="stMetricValue"] {color: var(--nika-text); font-weight: 700;}

      [data-testid="stTabs"] {
        border: none !important;
        background: transparent !important;
        padding: 0px !important;
        box-shadow: none !important;
      }
      [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 1.5rem; background: transparent; border: none;
        border-bottom: 0 !important;
        border-radius: 0px; padding: 0px; margin: 0 0 1rem 0 !important;
      }
      [data-testid="stTabs"] [data-baseweb="tab-panel"] {
        padding-top: 0px !important;
        padding-bottom: 0px !important;
      }
      [data-testid="stTabs"] [data-baseweb="tab"] {
        height: auto; border-radius: 0px; padding: 0.5rem 0;
        background: transparent !important;
        color: var(--nika-muted);
        border-bottom: 2px solid transparent !important;
      }
      [data-testid="stTabs"] [aria-selected="true"] {
        background: transparent !important; color: var(--nika-accent-text);
        border-bottom: 2px solid transparent !important;
      }
      div[data-baseweb="tab-border"] {
        display: none !important;
      }
      [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background: var(--nika-accent) !important;
        height: 2px !important;
      }
      [data-testid="stDataFrame"] {
        border: 1px solid var(--nika-line); border-radius: 14px; overflow: hidden;
      }
      .nika-results-table-wrap {
        overflow-x: auto; border: 1px solid var(--nika-line); border-radius: 8px;
        background: var(--nika-panel);
      }
      .nika-results-table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
      .nika-results-table th, .nika-results-table td {
        padding: 0.6rem 0.75rem; border-bottom: 1px solid var(--nika-line);
        text-align: left; white-space: nowrap;
      }
      .nika-results-table th { background: var(--nika-secondary); font-weight: 700; }
      .nika-results-table tbody tr:last-child td { border-bottom: 0; }
      .nika-results-empty { color: var(--nika-muted); margin: 0.6rem 0; }
      [data-testid="stExpander"] {
        border: 1px solid var(--nika-line); border-radius: 13px;
        background: var(--nika-panel);
      }
      .module-expander-gap {height: 0.75rem;}

      div[data-testid="stCheckbox"] label {font-weight:700;}

      .stButton > button, .stDownloadButton > button {
        border-radius: 11px;
        font-weight: 800;
        transition: all 0.2s ease;
      }
      /* Stable action colors. Target explicit widget keys, not column order. */
      div.st-key-studio_run_button button,
      div.stButton button[data-testid="baseButton-primary"] {
        border: 1px solid #2563eb !important;
        background: #2563eb !important;
        color: #ffffff !important;
        border-radius: 11px !important;
        font-weight: 800 !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
      }
      div.st-key-studio_run_button button:hover,
      div.stButton button[data-testid="baseButton-primary"]:hover {
        border-color: #1d4ed8 !important;
        background: #1d4ed8 !important;
        color: #ffffff !important;
        transform: translateY(-1.5px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.35) !important;
      }
      div.st-key-studio_run_button button:active,
      div.stButton button[data-testid="baseButton-primary"]:active {
        transform: translateY(0.5px) !important;
      }

      /* Secondary button styling */
      button[data-testid="baseButton-secondary"], .stDownloadButton > button {
        border: 1px solid color-mix(in srgb, var(--nika-accent) 38%, transparent) !important;
        background: var(--nika-accent-soft) !important;
        color: var(--nika-accent-text) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
      }
      button[data-testid="baseButton-secondary"]:hover, .stDownloadButton > button:hover {
        border-color: var(--nika-accent) !important;
        color: var(--nika-accent-text) !important;
        background: color-mix(in srgb, var(--nika-accent) 22%, transparent) !important;
        transform: translateY(-1.5px);
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.15) !important;
      }
      button[data-testid="baseButton-secondary"]:active, .stDownloadButton > button:active {
        transform: translateY(0.5px);
      }
      div.st-key-studio_queue_button button {
        border: 1px solid #2563eb !important;
        background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
        color: #ffffff !important;
      }
      div.st-key-studio_queue_button button:hover {
        border-color: #1d4ed8 !important;
        background: linear-gradient(135deg, #60a5fa, #3b82f6) !important;
        color: #ffffff !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.24) !important;
      }
      div.st-key-studio_resume_button button {
        border: 1px solid color-mix(in srgb, var(--nika-warning) 48%, transparent) !important;
        background: color-mix(in srgb, var(--nika-warning) 16%, transparent) !important;
        color: color-mix(in srgb, var(--nika-warning) 72%, var(--nika-text)) !important;
      }
      div.st-key-studio_resume_button button:hover {
        border-color: var(--nika-warning) !important;
        background: color-mix(in srgb, var(--nika-warning) 25%, transparent) !important;
        color: color-mix(in srgb, var(--nika-warning) 72%, var(--nika-text)) !important;
        box-shadow: 0 6px 20px rgba(217, 119, 6, 0.18) !important;
      }
      div.st-key-studio_stop_button button {
        border: 1px solid color-mix(in srgb, var(--nika-danger) 48%, transparent) !important;
        background: color-mix(in srgb, var(--nika-danger) 14%, transparent) !important;
        color: color-mix(in srgb, var(--nika-danger) 76%, var(--nika-text)) !important;
      }
      div.st-key-studio_stop_button button:hover {
        border-color: var(--nika-danger) !important;
        background: color-mix(in srgb, var(--nika-danger) 24%, transparent) !important;
        color: color-mix(in srgb, var(--nika-danger) 76%, var(--nika-text)) !important;
        box-shadow: 0 6px 20px rgba(239, 68, 68, 0.18) !important;
      }

      div[data-testid="stCodeBlock"] {
        border: 2px solid var(--nika-accent) !important;
        border-radius: 12px !important;
        background: var(--nika-secondary) !important;
        box-shadow: 0 4px 18px rgba(14, 165, 233, 0.14) !important;
      }
      textarea {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
        font-size:.78rem !important;
        background-color: var(--nika-secondary) !important;
        color: var(--nika-text) !important;
        border: 1px solid var(--nika-line) !important;
      }

      .nika-brand {display:flex; align-items:center; gap:.75rem; margin:.2rem 0 1.6rem;}
      .nika-mark {
        width:38px; height:38px; display:grid; place-items:center; border-radius:12px;
        color:#ffffff; background:linear-gradient(135deg,#38bdf8,#0284c7);
        font-size:1.15rem; font-weight:900; box-shadow:0 0 24px rgba(14, 165, 233, .2);
      }
      .nika-brand-title {font-size:1.05rem; font-weight:800; letter-spacing:.08em; color: var(--nika-text);}
      .nika-brand-sub {font-size:.72rem; color:var(--nika-muted); letter-spacing:.04em;}

      .eyebrow {
        color:var(--nika-accent-text); font-size:.74rem; font-weight:800; letter-spacing:.14em;
        text-transform:uppercase; margin-bottom:.55rem;
      }

      /* Custom styling for multiselect tags */
      div[data-baseweb="tag"] {
        background-color: var(--nika-accent-soft) !important;
        border: 1px solid color-mix(in srgb, var(--nika-accent) 28%, transparent) !important;
        border-radius: 6px !important;
        color: var(--nika-accent-text) !important;
        padding: 2px 8px !important;
      }
      div[data-baseweb="tag"] span {
        color: var(--nika-accent-text) !important;
        font-weight: 600 !important;
      }
      div[data-baseweb="tag"] svg {
        fill: var(--nika-accent-text) !important;
      }
      @media (max-width: 900px) {.mini-grid {grid-template-columns:1fr 1fr;}}
    </style>
    """,
    unsafe_allow_html=True,
)


def _benchmark_path_from_name(value: str, *, default: str) -> Path:
    raw = value.strip() or Path(default).stem
    path = Path(raw).expanduser()
    if path.is_absolute() or path.parent != Path("."):
        return path if path.suffix in {".yaml", ".yml"} else path.with_suffix(".yaml")
    name = re.sub(r"\.ya?ml$", "", raw)
    return BENCHMARK_DIR / f"{name}.yaml"


def _count_rows(path: Path, *, expected_role: str | None = None) -> int:
    return len(load_benchmark_manifest(path, expected_role=expected_role).cases)


def _selected_run_dir() -> Path | None:
    active = st.session_state.get("active_run_dir")
    if active:
        path = Path(active)
        if path.exists():
            return path
    runs = list_runs()
    if not runs:
        return None
    for r in reversed(runs):
        if run_status(r).get("status") == "running":
            return r
    return runs[0]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg(values: list[float]) -> str:
    return "-" if not values else f"{sum(values) / len(values):.2f}"


def _sum(values: list[float]) -> str:
    return "-" if not values else f"{sum(values):.0f}"


def _ratio(numerator: float, denominator: float) -> str:
    return "-" if denominator <= 0 else f"{numerator / denominator:.2f}"


def _fmt_delta(value: float | None) -> str:
    return "-" if value is None else f"{value:+.2f}"


RESULT_SUMMARY_COLUMNS = [
    "result_root",
    "modules",
    "incident_success",
    "detection_score",
    "localization_f1",
    "rca_f1",
    "tool_calls",
    "tool_errors",
    "duration",
    "agent",
    "model",
]

RESULT_DETAIL_COLUMNS = [
    "result_root",
    "progress",
    "evaluation_cases",
    "training_cases",
    "detection_score",
    "incident_success",
    "localization_f1",
    "rca_f1",
    "localization_precision",
    "rca_precision",
    "tool_calls",
    "tool_errors",
    "detection_precision",
    "detection_recall",
    "detection_f1",
    "detection_fpr",
    "detection_fp",
    "detection_fn",
    "paired_baseline",
    "paired_cases",
    "paired_delta_detection",
    "paired_delta_localization_f1",
    "paired_delta_rca_f1",
    "paired_wins",
    "paired_losses",
    "paired_ties",
    "token_in",
    "token_out",
    "procedural_memory_reward",
    "procedural_memory_advantage",
    "procedural_memory_success",
    "procedural_memory_added_tokens",
    "procedural_memory_delta_tokens_step",
    "duration",
    "modules",
    "agent",
    "model",
]

RESULT_NUMERIC_COLUMNS = {
    "detection_score",
    "incident_success",
    "localization_f1",
    "rca_f1",
    "tool_calls",
    "detection_precision",
    "detection_recall",
    "detection_f1",
    "detection_fpr",
    "paired_delta_detection",
    "paired_delta_localization_f1",
    "paired_delta_rca_f1",
    "procedural_memory_reward",
    "procedural_memory_advantage",
    "procedural_memory_success",
    "procedural_memory_delta_tokens_step",
}

RESULT_INTEGER_COLUMNS = {
    "cases",
    "evaluation_cases",
    "training_cases",
    "finished",
    "failed",
    "submitted",
    "tool_errors",
    "detection_fp",
    "detection_fn",
    "paired_cases",
    "paired_wins",
    "paired_losses",
    "paired_ties",
    "token_in",
    "token_out",
    "total_tokens",
    "procedural_memory_added_tokens",
}


def _result_display_value(column: str, value: object) -> object:
    if value is None or value == "" or value == "-":
        return None
    number = _float(value)
    if column in RESULT_INTEGER_COLUMNS:
        return None if number is None else int(number)
    if column in RESULT_NUMERIC_COLUMNS:
        return number
    return value


def _result_completion(row: dict[str, object]) -> str:
    finished = row.get("finished")
    cases = row.get("cases")
    return f"{finished}/{cases}" if cases not in {None, "", "-"} else "-"


def _result_record(row: dict[str, object]) -> str:
    wins = row.get("paired_wins")
    losses = row.get("paired_losses")
    ties = row.get("paired_ties")
    if wins is None and losses is None and ties is None:
        return "-"
    return f"{wins or 0}/{losses or 0}/{ties or 0}"


def _summary_result_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for row in rows:
        display_row = dict(row)
        display_row["progress"] = _result_completion(row)
        display_row["paired_record"] = _result_record(row)
        token_in = _float(row.get("token_in"))
        token_out = _float(row.get("token_out"))
        if token_in is None and token_out is None:
            display_row["total_tokens"] = None
        else:
            display_row["total_tokens"] = int((token_in or 0.0) + (token_out or 0.0))
        summary.append(display_row)
    return summary


def _result_column_label(column: str) -> str:
    labels = {
        "result_root": "Result",
        "incident_success": "Incident Success",
        "detection_score": "Detection",
        "localization_f1": "Loc F1",
        "rca_f1": "RCA F1",
        "localization_precision": "Loc Precision",
        "rca_precision": "RCA Precision",
        "tool_calls": "Tool Calls",
        "tool_errors": "Tool Errors",
        "token_in": "Token In",
        "token_out": "Token Out",
        "procedural_memory_reward": "Memory Reward",
        "procedural_memory_advantage": "Memory Advantage",
        "procedural_memory_success": "Memory Success",
        "evaluation_cases": "Evaluate cases",
        "training_cases": "Training cases",
    }
    return labels.get(column, column.replace("_", " ").title())


def _result_cell_text(column: str, value: object) -> str:
    display_value = _result_display_value(column, value)
    if display_value is None:
        return "-"
    if column == "stage":
        return {"training": "Train", "evaluation": "Eval", "all": "Baseline"}.get(
            str(display_value),
            str(display_value),
        )
    if column in RESULT_INTEGER_COLUMNS:
        return str(int(display_value))
    if column in RESULT_NUMERIC_COLUMNS:
        return f"{float(display_value):.2f}"
    return str(display_value)


def _results_table_html(rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return "<p class='nika-results-empty'>No results yet.</p>"
    header = "".join(
        f"<th>{html.escape(_result_column_label(column))}</th>" for column in columns
    )
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(_result_cell_text(column, row.get(column)))}</td>"
            for column in columns
        )
        + "</tr>"
        for row in rows
    )
    return (
        "<div class='nika-results-table-wrap'><table class='nika-results-table'>"
        f"<thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _parse_duration(meta: dict) -> float | None:
    st = meta.get("start_time")
    et = meta.get("end_time")
    if not st or not et:
        return None
    try:
        from datetime import datetime

        t1 = datetime.fromisoformat(str(st).replace("Z", "+00:00"))
        t2 = datetime.fromisoformat(str(et).replace("Z", "+00:00"))
        return (t2 - t1).total_seconds()
    except Exception:
        return None


def _top_result_root(run_path: Path) -> Path:
    rel = run_path.parent.relative_to(RESULTS_DIR)
    if len(rel.parts) <= 1:
        return run_path.parent
    return RESULTS_DIR / rel.parts[0]


def _case_key(meta: dict) -> object:
    fingerprint = str(meta.get("benchmark_fingerprint") or "").strip()
    if fingerprint:
        return fingerprint
    if meta.get("benchmark_index") is not None:
        return meta.get("benchmark_index")
    problem_names = meta.get("problem_names") or []
    return (
        meta.get("root_cause_name"),
        tuple(problem_names),
        meta.get("scenario_name"),
        meta.get("fault_seed"),
    )


def _is_baseline_meta(meta: dict) -> bool:
    return not bool(meta.get("tool_refinement_enabled")) and not bool(
        meta.get("procedural_memory_enabled")
    )


def _benchmark_role(meta: dict) -> str:
    role = str(meta.get("benchmark_role") or "").strip().lower()
    return role if role in {"training", "evaluation"} else ""


def _metric_total(metrics: dict, *, is_anomaly: bool) -> float | None:
    del is_anomaly
    values = [
        _float(metrics.get("detection_score")),
        _float(metrics.get("localization_f1")),
        _float(metrics.get("rca_f1")),
    ]
    if any(value is None for value in values):
        return None
    return sum(value or 0.0 for value in values) / len(values)


def _root_case_map(run_paths: list[Path]) -> dict[object, dict[str, object]]:
    roles = [_benchmark_role(_read_json(path)) for path in run_paths]
    has_evaluation_role = "evaluation" in roles
    cases: dict[object, dict[str, object]] = {}
    for run_path in run_paths:
        meta = _read_json(run_path)
        if has_evaluation_role and _benchmark_role(meta) != "evaluation":
            continue
        metrics = _read_json(run_path.parent / "eval_metrics.json")
        ground_truth = _read_json(run_path.parent / "ground_truth.json")
        key = _case_key(meta)
        if key is None:
            continue
        cases[key] = {
            "meta": meta,
            "metrics": metrics,
            "is_anomaly": bool(ground_truth.get("is_anomaly", True)),
        }
    return cases


def _primary_result_paths(run_paths: list[Path]) -> list[Path]:
    """Use evaluation cases as the experiment's primary endpoint."""

    roles = [_benchmark_role(_read_json(path)) for path in run_paths]
    if any(roles):
        return [path for path, role in zip(run_paths, roles) if role == "evaluation"]
    return run_paths


def _paired_stats(
    *,
    root: Path,
    root_cases: dict[Path, dict[object, dict[str, object]]],
    baseline_roots: list[Path],
) -> dict[str, object]:
    target_cases = root_cases.get(root) or {}
    if not target_cases:
        return {}
    candidate_rows: list[tuple[int, float, Path, set[object]]] = []
    target_keys = set(target_cases)
    for baseline_root in baseline_roots:
        if baseline_root == root:
            continue
        baseline_cases = root_cases.get(baseline_root) or {}
        overlap = target_keys & set(baseline_cases)
        if not overlap:
            continue
        mtime = baseline_root.stat().st_mtime if baseline_root.exists() else 0.0
        candidate_rows.append((len(overlap), mtime, baseline_root, overlap))
    if not candidate_rows:
        return {}
    _, _, baseline_root, overlap = max(candidate_rows)
    deltas: dict[str, list[float]] = {
        "detection_score": [],
        "localization_f1": [],
        "rca_f1": [],
    }
    wins = losses = ties = 0
    for key in overlap:
        target_case = target_cases[key]
        baseline_case = root_cases[baseline_root][key]
        target_metrics = target_case["metrics"]
        baseline_metrics = baseline_case["metrics"]
        is_anomaly = bool(target_case.get("is_anomaly", True))
        for metric, values in deltas.items():
            if metric != "detection_score" and not is_anomaly:
                continue
            target_value = _float(target_metrics.get(metric))
            baseline_value = _float(baseline_metrics.get(metric))
            if target_value is not None and baseline_value is not None:
                values.append(target_value - baseline_value)
        target_total = _metric_total(target_metrics, is_anomaly=is_anomaly)
        baseline_total = _metric_total(baseline_metrics, is_anomaly=is_anomaly)
        if target_total is None or baseline_total is None:
            continue
        delta = target_total - baseline_total
        if delta > 0.01:
            wins += 1
        elif delta < -0.01:
            losses += 1
        else:
            ties += 1
    return {
        "paired_baseline": baseline_root.name,
        "paired_cases": len(overlap),
        "paired_delta_detection": _fmt_delta(_avg_float(deltas["detection_score"])),
        "paired_delta_localization_f1": _fmt_delta(
            _avg_float(deltas["localization_f1"])
        ),
        "paired_delta_rca_f1": _fmt_delta(_avg_float(deltas["rca_f1"])),
        "paired_wins": wins,
        "paired_losses": losses,
        "paired_ties": ties,
    }


def _avg_float(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _result_rows(
    *,
    benchmark_name: str | None = None,
    stage: str | None = None,
) -> list[dict[str, object]]:
    if not RESULTS_DIR.exists():
        return []
    grouped: dict[Path, list[Path]] = {}
    for run_path in RESULTS_DIR.rglob("run.json"):
        if "0_summary" in run_path.relative_to(RESULTS_DIR).parts:
            continue
        root = _top_result_root(run_path)
        if benchmark_name and not root.name.startswith(benchmark_name):
            continue
        group_key = root
        grouped.setdefault(group_key, []).append(run_path)

    primary_paths = {
        root: _primary_result_paths(run_paths) for root, run_paths in grouped.items()
    }
    root_cases = {
        root: _root_case_map(run_paths) for root, run_paths in primary_paths.items()
    }
    baseline_roots = [
        root
        for root, cases in root_cases.items()
        if cases
        and all(
            _is_baseline_meta(case["meta"])
            for case in cases.values()
            if isinstance(case.get("meta"), dict)
        )
    ]

    rows: list[dict[str, object]] = []
    for gkey, run_paths in sorted(
        grouped.items(),
        key=lambda item: item[0].stat().st_mtime if item[0].exists() else 0,
        reverse=True,
    ):
        if stage in {"training", "evaluation"}:
            run_paths = [
                path
                for path in run_paths
                if _benchmark_role(_read_json(path)) == stage
            ]
            if not run_paths:
                continue
        detections: list[float] = []
        incident_successes: list[float] = []
        localization_f1s: list[float] = []
        rca_f1s: list[float] = []
        localization_precisions: list[float] = []
        rca_precisions: list[float] = []
        detection_tp = 0.0
        detection_tn = 0.0
        detection_fp = 0.0
        detection_fn = 0.0
        tool_calls: list[float] = []
        in_tokens: list[float] = []
        out_tokens: list[float] = []
        tool_errors: list[float] = []
        durations: list[float] = []
        procedural_memory_rewards: list[float] = []
        procedural_memory_advantages: list[float] = []
        procedural_memory_successes: list[float] = []
        procedural_memory_added_tokens: list[float] = []
        procedural_memory_delta_tokens: list[float] = []
        submitted = 0
        finished = 0
        failed = 0
        training_cases = 0
        evaluation_cases = 0
        result_modules: set[str] = set()
        agents: set[str] = set()
        models: set[str] = set()

        metric_paths = set(primary_paths[gkey]).intersection(run_paths)
        for run_path in run_paths:
            session_dir = run_path.parent
            meta = _read_json(run_path)
            role = _benchmark_role(meta)
            if role == "training":
                training_cases += 1
            elif role == "evaluation":
                evaluation_cases += 1
            metrics = _read_json(session_dir / "eval_metrics.json")
            ground_truth = _read_json(session_dir / "ground_truth.json")
            is_anomaly = bool(ground_truth.get("is_anomaly", True))
            include_primary_metrics = run_path in metric_paths
            if meta.get("status") == "finished":
                finished += 1
            has_eval = (session_dir / "eval_metrics.json").exists()
            if (session_dir / "submission.json").exists():
                submitted += 1
            elif meta.get("status") != "running" and not has_eval:
                failed += 1

            metric_targets = [
                ("detection_score", detections),
                ("tool_calls", tool_calls),
                ("in_tokens", in_tokens),
                ("out_tokens", out_tokens),
                ("tool_errors", tool_errors),
            ]
            if is_anomaly:
                metric_targets.extend(
                    [
                        ("localization_f1", localization_f1s),
                        ("rca_f1", rca_f1s),
                        ("localization_precision", localization_precisions),
                        ("rca_precision", rca_precisions),
                    ]
                )
            for key, target in metric_targets:
                if not include_primary_metrics:
                    continue
                value = _float(metrics.get(key))
                if value is not None:
                    target.append(value)

            incident_score = (
                _metric_total(metrics, is_anomaly=is_anomaly)
                if include_primary_metrics
                else None
            )
            if incident_score is not None:
                incident_successes.append(1.0 if incident_score >= 1.0 else 0.0)

            if include_primary_metrics:
                detection_tp += _float(metrics.get("detection_tp")) or 0.0
                detection_tn += _float(metrics.get("detection_tn")) or 0.0
                detection_fp += _float(metrics.get("detection_fp")) or 0.0
                detection_fn += _float(metrics.get("detection_fn")) or 0.0

            procedural_memory_update = metrics.get("procedural_memory") or {}
            if isinstance(procedural_memory_update, dict):
                for key, target in (
                    ("episode_reward", procedural_memory_rewards),
                    ("episode_advantage", procedural_memory_advantages),
                    ("total_added_tokens", procedural_memory_added_tokens),
                    ("delta_prompt_tokens_per_step", procedural_memory_delta_tokens),
                ):
                    value = _float(procedural_memory_update.get(key))
                    if value is not None:
                        target.append(value)
                if procedural_memory_update.get("episode_success") is not None:
                    procedural_memory_successes.append(
                        1.0 if procedural_memory_update.get("episode_success") else 0.0
                    )
            dur = _parse_duration(meta)
            if include_primary_metrics and dur is not None:
                durations.append(dur)

            if meta.get("tool_refinement_enabled"):
                result_modules.add("Tool Refinement")
            if meta.get("procedural_memory_enabled"):
                result_modules.add("Procedural Memory")
            if meta.get("agent_type"):
                agent_name = str(meta["agent_type"])
                agents.add(agent_name)
            if meta.get("model"):
                models.add(str(meta["model"]))

        display_name = gkey.name
        if stage == "training":
            display_name = f"{display_name}-train"
        elif stage == "evaluation":
            display_name = f"{display_name}-eval"

        row = {
            "result_root": display_name,
            "stage": stage or "all",
            "cases": len(run_paths),
            "evaluation_cases": (
                evaluation_cases
                if training_cases or evaluation_cases
                else len(metric_paths)
            ),
            "training_cases": training_cases,
            "finished": finished,
            "failed": failed,
            "submitted": submitted,
            "detection_score": _avg(detections),
            "incident_success": _avg(incident_successes),
            "detection_precision": _ratio(detection_tp, detection_tp + detection_fp),
            "detection_recall": _ratio(detection_tp, detection_tp + detection_fn),
            "detection_f1": (
                "-"
                if detection_tp + detection_fp <= 0 or detection_tp + detection_fn <= 0
                else _ratio(
                    2 * detection_tp,
                    2 * detection_tp + detection_fp + detection_fn,
                )
            ),
            "detection_fpr": _ratio(detection_fp, detection_fp + detection_tn),
            "detection_fp": f"{detection_fp:.0f}",
            "detection_fn": f"{detection_fn:.0f}",
            "localization_f1": _avg(localization_f1s),
            "rca_f1": _avg(rca_f1s),
            "localization_precision": _avg(localization_precisions),
            "rca_precision": _avg(rca_precisions),
            "tool_calls": _avg(tool_calls),
            "tool_errors": _sum(tool_errors),
            "token_in": _sum(in_tokens),
            "token_out": _sum(out_tokens),
            "procedural_memory_reward": _avg(procedural_memory_rewards),
            "procedural_memory_advantage": _avg(procedural_memory_advantages),
            "procedural_memory_success": _avg(procedural_memory_successes),
            "procedural_memory_added_tokens": _sum(procedural_memory_added_tokens),
            "procedural_memory_delta_tokens_step": _avg(procedural_memory_delta_tokens),
            "duration": f"{int(sum(durations))}s" if durations else "-",
            "modules": ", ".join(sorted(result_modules)) or "-",
            "agent": ", ".join(sorted(agents)) or "-",
            "model": ", ".join(sorted(models)) or "-",
        }
        row.update(
            _paired_stats(
                root=gkey,
                root_cases=root_cases,
                baseline_roots=baseline_roots,
            )
        )
        rows.append(row)
    return rows


def _event_case_summary(ev: dict) -> str:
    scenario = ev.get("scenario") or "?"
    topo_size = ev.get("topo_size") or ev.get("topo") or "-"
    problem = ev.get("problem") or "?"
    return (
        f"scenario={html.escape(str(scenario))} "
        f"topo_size={html.escape(str(topo_size))} "
        f"problem={html.escape(str(problem))}"
    )


def _event_inject_summary(ev: dict) -> str:
    items = []
    for key, value in sorted(ev.items()):
        if key.startswith("inject_"):
            items.append(
                f"{html.escape(key.removeprefix('inject_'))}={html.escape(str(value))}"
            )
    return ", ".join(items) if items else "none"


def _case_event_html(
    ev: dict,
    *,
    style: str,
    verb: str,
    color: str,
) -> str:
    index = html.escape(str(ev.get("index") or "?"))
    role = str(ev.get("role") or "").strip().lower()
    case_label = f"{role.title()} case" if role else "Case"
    case_summary = _event_case_summary(ev)
    inject_summary = _event_inject_summary(ev)
    return (
        f"<div style='{style}; color: {color};'>"
        f"<b>[{html.escape(case_label)} {index}]</b> {verb}: "
        f"<code style='font-size: 0.75rem; background: var(--nika-secondary); "
        f"padding: 2px 4px; border-radius: 4px;'>{case_summary}</code>"
        f"<span style='margin-left: 0.5rem; color: var(--nika-muted);'>&nbsp;inject: "
        f"<code style='font-size: 0.75rem;'>{inject_summary}</code></span>"
        f"</div>"
    )


st.markdown('<div class="section-title">Studio</div>', unsafe_allow_html=True)

st.session_state.setdefault(
    "studio_training_benchmark",
    Path(DEFAULT_STUDIO_TRAINING_BENCHMARK).stem,
)
st.session_state.setdefault(
    "studio_evaluate_benchmark",
    Path(DEFAULT_STUDIO_EVALUATE_BENCHMARK).stem,
)

baseline_settings = st.expander("Experimental Setting", expanded=False)
baseline_settings.__enter__()
if baseline_settings is not None:
    b_col1, b_col2, b_col3, b_col4, b_col5, b_col6 = st.columns(
        [1.05, 1.05, 1.8, 0.75, 1.45, 1.0],
        gap="small",
    )
    with b_col1:
        agent_options = ["react", "plan-execute", "reflexion"]
        agent_type = st.selectbox(
            "Workflow",
            agent_options,
            index=agent_options.index(BASELINE_DEFAULTS.agent_type),
        )
    with b_col2:
        backend_options = ["custom", "openai", "deepseek", "ollama"]
        llm_backend = st.selectbox(
            "Backend",
            backend_options,
            index=backend_options.index(DEFAULT_LLM_BACKEND)
            if DEFAULT_LLM_BACKEND in backend_options
            else 0,
        )
    with b_col3:
        model = st.text_input("Model", value=DEFAULT_MODEL)
    with b_col4:
        default_max_steps = DEFAULT_STUDIO_MAX_STEPS
        max_steps_str = st.text_input("Steps", value=str(default_max_steps))
        max_steps = int(max_steps_str) if max_steps_str.isdigit() else default_max_steps
    with b_col5:
        training_benchmark_name = st.text_input(
            "Training Benchmark",
            key="studio_training_benchmark",
            help="Cases used to update enabled training modules.",
        )
    with b_col6:
        evaluate_benchmark_name = st.text_input(
            "Evaluate Benchmark",
            key="studio_evaluate_benchmark",
            help="Cases used to score the frozen module snapshot.",
        )
    training_benchmark_path = _benchmark_path_from_name(
        training_benchmark_name,
        default=DEFAULT_STUDIO_TRAINING_BENCHMARK,
    )
    evaluate_benchmark_path = _benchmark_path_from_name(
        evaluate_benchmark_name,
        default=DEFAULT_STUDIO_EVALUATE_BENCHMARK,
    )
    benchmark_errors: list[str] = []
    benchmark_warnings: list[str] = []
    benchmark_manifests: dict[str, Any] = {}
    for role, path in (
        ("training", training_benchmark_path),
        ("evaluation", evaluate_benchmark_path),
    ):
        try:
            benchmark_manifests[role] = load_benchmark_manifest(
                path,
                expected_role=role,
            )
        except FileNotFoundError:
            benchmark_errors.append(f"{role.title()} benchmark not found: {path}")
        except OSError as exc:
            benchmark_errors.append(f"Cannot read {role} benchmark: {exc}")
        except (ValueError, YAMLError) as exc:
            benchmark_errors.append(f"Invalid {role} benchmark: {exc}")
    training_manifest = benchmark_manifests.get("training")
    evaluation_manifest = benchmark_manifests.get("evaluation")
    if training_manifest is not None and evaluation_manifest is not None:
        training_ids = {benchmark_case_identity(row) for row in training_manifest.cases}
        evaluation_ids = {
            benchmark_case_identity(row) for row in evaluation_manifest.cases
        }
        overlap = training_ids & evaluation_ids
        if overlap:
            benchmark_warnings.append(
                f"Training and evaluation benchmarks overlap on {len(overlap)} "
                "case identities; evaluation is not fully held out."
            )
    if agent_type == "reflexion":
        attempts_col, _ = st.columns([1, 5], gap="small")
        with attempts_col:
            max_attempts = st.number_input(
                "Attempts",
                min_value=1,
                max_value=10,
                value=BASELINE_DEFAULTS.max_attempts,
            )
    else:
        max_attempts = BASELINE_DEFAULTS.max_attempts
    run_judge = BASELINE_DEFAULTS.judge_evaluation
    judge_backend = BASELINE_DEFAULTS.judge_provider
    judge_model = BASELINE_DEFAULTS.judge_model
    if run_judge:
        judge_col1, judge_col2 = st.columns(2, gap="small")
        with judge_col1:
            judge_backend = st.text_input("Judge backend", value=judge_backend)
        with judge_col2:
            judge_model = st.text_input("Judge model", value=judge_model)

st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)

# Keep config values stable while module-specific controls are conditionally hidden.
tool_library_id = ""
tool_doc_chars = TOOL_REFINEMENT_DEFAULTS.tool_doc_chars
tool_convergence_threshold = TOOL_REFINEMENT_DEFAULTS.convergence_threshold
tool_exploration_similarity_threshold = (
    TOOL_REFINEMENT_DEFAULTS.exploration_similarity_threshold
)
tool_explorer_reflection_limit = TOOL_REFINEMENT_DEFAULTS.explorer_reflection_limit
tool_explorer_model = MODULE_DEFAULTS.tool_refinement.llm_model
tool_analyzer_model = MODULE_DEFAULTS.tool_refinement.llm_model
tool_rewriter_model = MODULE_DEFAULTS.tool_refinement.llm_model
tool_update_interval = TOOL_REFINEMENT_DEFAULTS.update_interval
tool_min_new_trials = TOOL_REFINEMENT_DEFAULTS.min_new_trials
tool_max_tools_per_update = TOOL_REFINEMENT_DEFAULTS.max_tools_per_update
tool_publish_min_utility = TOOL_REFINEMENT_DEFAULTS.publish_min_utility
procedural_memory_bank = ""
procedural_memory_token_budget = PROCEDURAL_MEMORY_DEFAULTS.token_budget
procedural_memory_max_skill_age = PROCEDURAL_MEMORY_DEFAULTS.max_skill_age
procedural_memory_pool_size = PROCEDURAL_MEMORY_DEFAULTS.pool_size
procedural_memory_update_threshold = PROCEDURAL_MEMORY_DEFAULTS.evolution_threshold
procedural_memory_best_of_n = PROCEDURAL_MEMORY_DEFAULTS.best_of_n
procedural_memory_ppo_epsilon = PROCEDURAL_MEMORY_DEFAULTS.ppo_epsilon
procedural_memory_selection_epsilon = PROCEDURAL_MEMORY_DEFAULTS.selection_epsilon
procedural_memory_experience_pool_size = PROCEDURAL_MEMORY_DEFAULTS.experience_pool_size
procedural_memory_epsilon_decay_cases = (
    PROCEDURAL_MEMORY_DEFAULTS.selection_epsilon_decay_cases
)
procedural_memory_baseline_ema_alpha = PROCEDURAL_MEMORY_DEFAULTS.baseline_ema_alpha
procedural_memory_acceptance_margin = PROCEDURAL_MEMORY_DEFAULTS.acceptance_margin
procedural_memory_verifier = PROCEDURAL_MEMORY_DEFAULTS.verifier
procedural_memory_evolver_model = PROCEDURAL_MEMORY_DEFAULTS.evolver_model
procedural_memory_policy_scorer_model = PROCEDURAL_MEMORY_DEFAULTS.policy_scorer_model
procedural_memory_holdout_size = PROCEDURAL_MEMORY_DEFAULTS.holdout_size
procedural_memory_min_positive_advantage = (
    PROCEDURAL_MEMORY_DEFAULTS.min_positive_advantage
)

with st.container():
    with st.container(
        horizontal=True,
        horizontal_alignment="distribute",
        vertical_alignment="center",
        gap="small",
    ):
        st.markdown("Enable Procedural Memory")
        procedural_memory_selected = st.toggle(
            "Enabled",
            value=False,
            key="procedural_memory_enabled",
            label_visibility="collapsed",
            help="Enable Procedural Memory",
        )
    if procedural_memory_selected:
        procedural_memory_bank = st.text_input(
            "Procedural Memory bank",
            value="",
            placeholder="auto",
            disabled=not procedural_memory_selected,
            help=(
                "Studio updates this bank on the Training Benchmark, then "
                "evaluates an immutable snapshot on the Evaluate Benchmark."
            ),
        )
        m_col1, m_col2, m_col5 = st.columns(3, gap="small")
        with m_col1:
            procedural_memory_token_budget = st.number_input(
                "Skill context budget",
                min_value=100,
                max_value=8000,
                value=PROCEDURAL_MEMORY_DEFAULTS.token_budget,
                step=100,
                disabled=not procedural_memory_selected,
                help="Upper budget used to derive the bounded active-skill policy context.",
            )

        with m_col2:
            procedural_memory_max_skill_age = st.number_input(
                "Max actions / activation",
                min_value=1,
                max_value=100,
                value=PROCEDURAL_MEMORY_DEFAULTS.max_skill_age,
                disabled=not procedural_memory_selected,
            )

        with m_col5:
            procedural_memory_pool_size = st.number_input(
                "Skill pool capacity",
                min_value=1,
                max_value=512,
                value=PROCEDURAL_MEMORY_DEFAULTS.pool_size,
                disabled=not procedural_memory_selected,
            )
        m_col10, m_col11, m_col12, m_col13 = st.columns(4, gap="small")
        with m_col10:
            procedural_memory_update_threshold = st.number_input(
                "Evolution threshold (trajectories)",
                min_value=2,
                max_value=100,
                value=PROCEDURAL_MEMORY_DEFAULTS.evolution_threshold,
                disabled=not procedural_memory_selected,
            )
        with m_col11:
            procedural_memory_best_of_n = st.number_input(
                "Candidate skills / update",
                min_value=1,
                max_value=20,
                value=PROCEDURAL_MEMORY_DEFAULTS.best_of_n,
                disabled=not procedural_memory_selected,
            )
        with m_col12:
            procedural_memory_ppo_epsilon = st.number_input(
                "Trust-region clip epsilon",
                min_value=0.0,
                max_value=1.0,
                value=PROCEDURAL_MEMORY_DEFAULTS.ppo_epsilon,
                step=0.05,
                format="%.2f",
                disabled=not procedural_memory_selected,
            )
        with m_col13:
            procedural_memory_selection_epsilon = st.number_input(
                "Exploration epsilon",
                min_value=0.0,
                max_value=1.0,
                value=PROCEDURAL_MEMORY_DEFAULTS.selection_epsilon,
                step=0.05,
                format="%.2f",
                disabled=not procedural_memory_selected,
            )

        m_col14, m_col15, m_col17, m_col18 = st.columns(4, gap="small")
        with m_col14:
            procedural_memory_experience_pool_size = st.number_input(
                "Experience buffer capacity",
                min_value=1,
                max_value=10000,
                value=PROCEDURAL_MEMORY_DEFAULTS.experience_pool_size,
                step=100,
                disabled=not procedural_memory_selected,
            )
        with m_col15:
            procedural_memory_epsilon_decay_cases = st.number_input(
                "Exploration decay horizon (cases)",
                min_value=1,
                max_value=10000,
                value=PROCEDURAL_MEMORY_DEFAULTS.selection_epsilon_decay_cases,
                step=50,
                disabled=not procedural_memory_selected,
            )

        with m_col17:
            procedural_memory_baseline_ema_alpha = st.number_input(
                "Baseline EMA alpha",
                min_value=0.01,
                max_value=1.0,
                value=PROCEDURAL_MEMORY_DEFAULTS.baseline_ema_alpha,
                step=0.01,
                format="%.2f",
                disabled=not procedural_memory_selected,
            )
        with m_col18:
            procedural_memory_acceptance_margin = st.number_input(
                "Minimum surrogate improvement",
                min_value=0.0,
                max_value=1.0,
                value=PROCEDURAL_MEMORY_DEFAULTS.acceptance_margin,
                step=0.001,
                format="%.3f",
                disabled=not procedural_memory_selected,
            )
        verifier_options = (
            "behavioral_replay",
            "structured_replay",
            "policy_logprob",
        )
        verifier_col, m_col21, m_col22 = st.columns(3, gap="small")
        with verifier_col:
            procedural_memory_verifier = st.selectbox(
                "Candidate verifier",
                options=verifier_options,
                index=verifier_options.index(PROCEDURAL_MEMORY_DEFAULTS.verifier),
                format_func=lambda value: {
                    "behavioral_replay": "Behavioral replay (LLM)",
                    "structured_replay": "Structured replay",
                    "policy_logprob": "Policy log-prob (completion replay)",
                }[value],
                help=(
                    "Policy log-prob uses a completion endpoint to replay the exact "
                    "Skill system prompt and serialized historical action. Chat "
                    "messages and tool schemas remain provider-side context and "
                    "cannot be replayed by this endpoint."
                ),
                disabled=not procedural_memory_selected,
            )

        m_col19, m_col20 = st.columns(2, gap="small")
        with m_col19:
            procedural_memory_evolver_model = st.text_input(
                "Semantic-gradient / Evolver model",
                value=PROCEDURAL_MEMORY_DEFAULTS.evolver_model,
                disabled=not procedural_memory_selected,
            )
        with m_col20:
            procedural_memory_policy_scorer_model = st.text_input(
                "Verifier model",
                value=PROCEDURAL_MEMORY_DEFAULTS.policy_scorer_model,
                disabled=(
                    not procedural_memory_selected
                    or procedural_memory_verifier == "structured_replay"
                ),
            )

        maximum_holdout = max(1, int(procedural_memory_update_threshold) - 1)
        holdout_key = "procedural_memory_holdout_size_input"
        holdout_default = min(
            PROCEDURAL_MEMORY_DEFAULTS.holdout_size,
            maximum_holdout,
        )
        if int(st.session_state.get(holdout_key, 1)) > maximum_holdout:
            st.session_state[holdout_key] = maximum_holdout
        holdout_value = (
            {} if holdout_key in st.session_state else {"value": holdout_default}
        )
        with m_col21:
            procedural_memory_holdout_size = st.number_input(
                "Verification holdout (trajectories)",
                min_value=1,
                max_value=maximum_holdout,
                key=holdout_key,
                disabled=not procedural_memory_selected,
                **holdout_value,
            )
        maximum_positive_holdouts = int(procedural_memory_holdout_size)
        positive_holdout_key = "procedural_memory_positive_holdouts_input"
        if (
            int(st.session_state.get(positive_holdout_key, 0))
            > maximum_positive_holdouts
        ):
            st.session_state[positive_holdout_key] = maximum_positive_holdouts
        positive_holdout_default = min(
            PROCEDURAL_MEMORY_DEFAULTS.min_positive_advantage,
            maximum_positive_holdouts,
        )
        positive_holdout_value = (
            {}
            if positive_holdout_key in st.session_state
            else {"value": positive_holdout_default}
        )
        with m_col22:
            procedural_memory_min_positive_advantage = st.number_input(
                "Min positive-advantage holdouts",
                min_value=0,
                max_value=maximum_positive_holdouts,
                key=positive_holdout_key,
                disabled=not procedural_memory_selected,
                **positive_holdout_value,
            )
st.markdown('<div class="module-expander-gap"></div>', unsafe_allow_html=True)

with st.container():
    with st.container(
        horizontal=True,
        horizontal_alignment="distribute",
        vertical_alignment="center",
        gap="small",
    ):
        st.markdown("Enable Tool Refinement")
        tool_selected = st.toggle(
            "Enabled",
            value=False,
            key="tool_refinement_enabled",
            label_visibility="collapsed",
            help="Enable Tool Refinement",
        )
    if tool_selected:
        t_col1, t_col2 = st.columns([1.5, 1], gap="small")
        with t_col1:
            tool_library_id = st.text_input(
                "Tool library ID",
                value="",
                placeholder="auto",
                disabled=not tool_selected,
            )
        with t_col2:
            tool_doc_chars = st.number_input(
                "Refined documentation limit (chars)",
                min_value=100,
                max_value=4000,
                value=TOOL_REFINEMENT_DEFAULTS.tool_doc_chars,
                step=50,
                disabled=not tool_selected,
            )

        t_threshold_col, t_col3, t_col4 = st.columns(3, gap="small")
        with t_threshold_col:
            tool_convergence_threshold = st.number_input(
                "Convergence threshold",
                min_value=0.0,
                max_value=1.0,
                value=TOOL_REFINEMENT_DEFAULTS.convergence_threshold,
                step=0.05,
                format="%.2f",
                disabled=not tool_selected,
            )
        with t_col3:
            tool_exploration_similarity_threshold = st.number_input(
                "Exploration similarity threshold",
                min_value=0.0,
                max_value=1.0,
                value=(TOOL_REFINEMENT_DEFAULTS.exploration_similarity_threshold),
                step=0.05,
                format="%.2f",
                disabled=not tool_selected,
            )
        with t_col4:
            tool_explorer_reflection_limit = st.number_input(
                "Explorer reflection limit",
                min_value=0,
                max_value=20,
                value=TOOL_REFINEMENT_DEFAULTS.explorer_reflection_limit,
                disabled=not tool_selected,
            )
        t_col5, t_col6, t_col7 = st.columns(3, gap="small")
        with t_col5:
            tool_explorer_model = st.text_input(
                "Explorer model",
                value=MODULE_DEFAULTS.tool_refinement.llm_model,
                disabled=not tool_selected,
            )
        with t_col6:
            tool_analyzer_model = st.text_input(
                "Analyzer model",
                value=MODULE_DEFAULTS.tool_refinement.llm_model,
                disabled=not tool_selected,
            )
        with t_col7:
            tool_rewriter_model = st.text_input(
                "Rewriter model",
                value=MODULE_DEFAULTS.tool_refinement.llm_model,
                disabled=not tool_selected,
            )
        t_col8, t_col9, t_col10, t_col11 = st.columns(4, gap="small")
        with t_col8:
            tool_update_interval = st.number_input(
                "Rewrite interval (cases)",
                min_value=1,
                max_value=100,
                value=TOOL_REFINEMENT_DEFAULTS.update_interval,
                disabled=not tool_selected,
            )
        with t_col9:
            tool_min_new_trials = st.number_input(
                "Minimum new trials",
                min_value=1,
                max_value=100,
                value=TOOL_REFINEMENT_DEFAULTS.min_new_trials,
                disabled=not tool_selected,
            )
        with t_col10:
            tool_max_tools_per_update = st.number_input(
                "Tools / update",
                min_value=1,
                max_value=50,
                value=TOOL_REFINEMENT_DEFAULTS.max_tools_per_update,
                disabled=not tool_selected,
            )
        with t_col11:
            tool_publish_min_utility = st.number_input(
                "Minimum publication utility",
                min_value=0.0,
                max_value=1.0,
                value=TOOL_REFINEMENT_DEFAULTS.publish_min_utility,
                step=0.05,
                format="%.2f",
                disabled=not tool_selected,
            )

st.markdown('<div class="module-expander-gap"></div>', unsafe_allow_html=True)
with st.container():
    with st.container(
        horizontal=True,
        horizontal_alignment="distribute",
        vertical_alignment="center",
        gap="small",
    ):
        st.markdown("Enable LLM Judge")
        run_judge = st.toggle(
            "Enabled",
            value=BASELINE_DEFAULTS.judge_evaluation,
            key="llm_judge_enabled",
            label_visibility="collapsed",
            help="Enable LLM Judge and run the evaluator after each benchmark case.",
        )
    if run_judge:
        judge_col1, judge_col2 = st.columns(2, gap="small")
        with judge_col1:
            judge_backend = st.text_input("Judge backend", value=judge_backend)
        with judge_col2:
            judge_model = st.text_input("Judge model", value=judge_model)

modules = []
if tool_selected:
    modules.append("tool_refinement")
if procedural_memory_selected:
    modules.append("procedural_memory")

config = {
    "training_benchmark_file": str(training_benchmark_path),
    "evaluate_benchmark_file": str(evaluate_benchmark_path),
    "modules": modules,
    "agent_type": agent_type,
    "llm_backend": llm_backend,
    "model": model,
    "max_steps": int(max_steps),
    "max_attempts": int(max_attempts),
    "parallel": 1,
    "tool_library_id": tool_library_id,
    "tool_doc_chars": int(tool_doc_chars),
    "tool_convergence_threshold": float(tool_convergence_threshold),
    "tool_exploration_similarity_threshold": float(
        tool_exploration_similarity_threshold
    ),
    "tool_explorer_reflection_limit": int(tool_explorer_reflection_limit),
    "tool_explorer_model": tool_explorer_model,
    "tool_analyzer_model": tool_analyzer_model,
    "tool_rewriter_model": tool_rewriter_model,
    "tool_update_interval": int(tool_update_interval),
    "tool_min_new_trials": int(tool_min_new_trials),
    "tool_max_tools_per_update": int(tool_max_tools_per_update),
    "tool_publish_min_utility": float(tool_publish_min_utility),
    "procedural_memory_bank": procedural_memory_bank,
    "procedural_memory_token_budget": int(procedural_memory_token_budget),
    "procedural_memory_max_skill_age": int(procedural_memory_max_skill_age),
    "procedural_memory_pool_size": int(procedural_memory_pool_size),
    "procedural_memory_update_threshold": int(procedural_memory_update_threshold),
    "procedural_memory_best_of_n": int(procedural_memory_best_of_n),
    "procedural_memory_ppo_epsilon": float(procedural_memory_ppo_epsilon),
    "procedural_memory_selection_epsilon": float(procedural_memory_selection_epsilon),
    "procedural_memory_experience_pool_size": int(
        procedural_memory_experience_pool_size
    ),
    "procedural_memory_baseline_ema_alpha": float(procedural_memory_baseline_ema_alpha),
    "procedural_memory_selection_epsilon_decay_cases": int(
        procedural_memory_epsilon_decay_cases
    ),
    "procedural_memory_acceptance_margin": float(procedural_memory_acceptance_margin),
    "procedural_memory_evolver_model": procedural_memory_evolver_model,
    "procedural_memory_policy_scorer_model": (procedural_memory_policy_scorer_model),
    "procedural_memory_verifier": procedural_memory_verifier,
    "procedural_memory_holdout_size": int(procedural_memory_holdout_size),
    "procedural_memory_min_positive_advantage": int(
        procedural_memory_min_positive_advantage
    ),
    "run_judge": bool(run_judge),
    "judge_backend": judge_backend,
    "judge_model": judge_model,
}
baseline_settings.__exit__(None, None, None)
prepared_config = prepare_experiment_config(config)
plan = build_command_plan(prepared_config)

st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
with st.expander("Command Preview", expanded=False):
    for item in plan:

        def format_command_multiline(cmd_parts: list[str]) -> str:
            if not cmd_parts:
                return ""
            lines = []
            current_line = []
            for i, part in enumerate(cmd_parts):
                quoted = shlex.quote(part)
                if part.startswith("-") and i >= 3:
                    if current_line:
                        lines.append(" ".join(current_line))
                    current_line = [quoted]
                else:
                    current_line.append(quoted)
            if current_line:
                lines.append(" ".join(current_line))
            return " \\\n  ".join(lines)

        st.code(format_command_multiline(item.command), language="bash")


st.markdown('<div style="margin-top: 0.75rem;"></div>', unsafe_allow_html=True)

all_runs = list_runs()
run_statuses = {path: run_status(path) for path in all_runs}
has_running_run = any(
    status.get("status") == "running" for status in run_statuses.values()
)
has_active_queue = any(
    status.get("status") in {"running", "queued"} for status in run_statuses.values()
)

if has_active_queue:
    if st.button(
        "Add Queue",
        key="studio_queue_button",
        type="secondary",
        disabled=bool(benchmark_errors),
        width="stretch",
    ):
        run_dir = create_run(prepared_config)
        st.session_state["active_run_dir"] = str(run_dir)
        st.rerun()
else:
    if st.button(
        "Run",
        key="studio_run_button",
        type="primary",
        disabled=bool(benchmark_errors),
        width="stretch",
    ):
        run_dir = create_run(prepared_config)
        st.session_state["active_run_dir"] = str(run_dir)
        st.rerun()
for benchmark_error in benchmark_errors:
    st.error(benchmark_error)
for benchmark_warning in benchmark_warnings:
    st.warning(benchmark_warning)

runs = list_runs()
run_statuses = {path: run_status(path) for path in runs}
has_running_run = any(
    status.get("status") == "running" for status in run_statuses.values()
)
has_active_queue = any(
    status.get("status") in {"running", "queued"} for status in run_statuses.values()
)
selected = _selected_run_dir()
if runs:
    run_labels = []
    run_map = {}
    for path in runs:
        status_val = run_statuses.get(path, {}).get("status") or "unknown"
        label = f"{path.name} ({status_val})"
        run_labels.append(label)
        run_map[label] = path

    selected_label = None
    if selected is not None:
        selected_status = (
            run_statuses.get(selected, run_status(selected)).get("status") or "unknown"
        )
        selected_label = f"{selected.name} ({selected_status})"

    history_col, resume_col, stop_col = st.columns(
        [3, 1, 1],
        gap="medium",
        vertical_alignment="bottom",
    )
    with history_col:
        selected_label = st.selectbox(
            "Run history",
            options=run_labels,
            index=run_labels.index(selected_label)
            if selected_label in run_labels
            else 0,
        )
    selected = run_map[selected_label]
    st.session_state["active_run_dir"] = str(selected)
else:
    selected = None

if selected is not None:
    status = run_status(selected)
    log_text = read_run_log(selected)
    spec = read_run_spec(selected)
    events = parse_progress_events(log_text)
    log_key = f"log-{selected.name}"
else:
    status = {}
    log_text = ""
    events = []
    log_key = "log-none"

if selected is not None:
    selected_status = str(status.get("status") or "unknown")
    can_resume = (
        selected_status not in {"running", "queued"}
        and not has_active_queue
        and bool(spec.get("config"))
    )
    can_stop = selected_status == "running"
    with resume_col:
        if st.button(
            "Resume Selected",
            key="studio_resume_button",
            type="secondary",
            disabled=not can_resume,
            width="stretch",
        ):
            try:
                run_dir = resume_run(selected)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state["active_run_dir"] = str(run_dir)
                st.rerun()
    with stop_col:
        if st.button(
            "Stop Selected",
            key="studio_stop_button",
            type="secondary",
            disabled=not can_stop,
            width="stretch",
        ):
            stop_run(selected)
            st.rerun()


def format_event_message_html(ev: dict) -> str | None:
    event = ev.get("event")
    style = "font-size: 0.8rem; padding: 6px 0; border-bottom: 1px solid var(--nika-line); color: var(--nika-text); line-height: 1.4;"

    if event == "ui_step_start":
        cmd = ev.get("command") or ""
        if isinstance(cmd, str) and cmd.startswith("[") and cmd.endswith("]"):
            try:
                import ast

                cmd = ast.literal_eval(cmd)
            except Exception:
                pass
        if isinstance(cmd, list):
            import shlex

            cmd_str = shlex.join(cmd)
        else:
            cmd_str = str(cmd)
        if len(cmd_str) > 80:
            cmd_str = cmd_str[:77] + "..."
        return f"<div style='{style}'><b>[Step {ev.get('index')}/{ev.get('total')}]</b> Starting: <code style='font-size: 0.75rem; background: var(--nika-secondary); padding: 2px 4px; border-radius: 4px;'>{cmd_str}</code></div>"
    elif event == "ui_step_done":
        ret = ev.get("returncode", "0")
        status_word = "Completed" if str(ret) == "0" else "Failed"
        color = "var(--nika-success)" if str(ret) == "0" else "var(--nika-danger)"
        return f"<div style='{style}'><b>[Step {ev.get('index')}/{ev.get('total')}]</b> <span style='color: {color}; font-weight: bold;'>{status_word}</span> (Exit: <code>{ret}</code>)</div>"
    elif event == "ui_run_done":
        code = ev.get("exit_code", "0")
        return f"<div style='{style}; font-weight: bold; color: var(--nika-text);'>Run Finished (Exit: <code>{code}</code>)</div>"
    elif event == "ui_run_resumed":
        return f"<div style='{style}; color: var(--nika-accent-text);'><b>Run Resumed</b> in-place with existing results.</div>"
    elif event == "ui_run_stopped":
        return f"<div style='{style}; color: var(--nika-warning);'><b>Run Stopped</b> by user. Starting next queued run if available.</div>"
    elif event == "benchmark_start":
        return _case_event_html(
            ev, style=style, verb="Starting", color="var(--nika-text)"
        )
    elif event == "benchmark_skip":
        return _case_event_html(
            ev,
            style=style,
            verb="Skipped existing result",
            color="var(--nika-muted)",
        )
    elif event == "benchmark_progress":
        return None
    elif event == "benchmark_done":
        return _case_event_html(
            ev, style=style, verb="Finished", color="var(--nika-success)"
        )
    elif event == "benchmark_failed":
        return _case_event_html(
            ev, style=style, verb="Failed", color="var(--nika-danger)"
        )
    elif event == "benchmark_aborted":
        return _case_event_html(
            ev, style=style, verb="Aborted", color="var(--nika-danger)"
        )
    elif event == "benchmark_stage_start":
        role_name = str(ev.get("role") or "benchmark").strip().lower()
        role = {"training": "Train", "evaluation": "Eval"}.get(
            role_name,
            role_name.title(),
        )
        role = html.escape(role)
        total = html.escape(str(ev.get("total") or "?"))
        pending = html.escape(str(ev.get("pending") or total))
        return (
            f"<div style='{style}; color: var(--nika-accent-text);'>"
            f"<b>{role} benchmark started</b> · {pending}/{total} cases pending"
            "</div>"
        )
    elif event == "benchmark_stage_done":
        role_name = str(ev.get("role") or "benchmark").strip().lower()
        role = {"training": "Train", "evaluation": "Eval"}.get(
            role_name,
            role_name.title(),
        )
        role = html.escape(role)
        completed = html.escape(str(ev.get("completed") or "0"))
        total = html.escape(str(ev.get("total") or "?"))
        failed = html.escape(str(ev.get("failed") or "0"))
        return (
            f"<div style='{style}; color: var(--nika-success);'>"
            f"<b>{role} benchmark completed</b> · {completed}/{total} finished, "
            f"{failed} failed</div>"
        )
    elif event in {"training_barrier_created", "training_barrier_reused"}:
        verb = "created" if event.endswith("created") else "reused"
        return (
            f"<div style='{style}; color: var(--nika-accent-text);'>"
            f"<b>Training snapshot {verb}</b> · evaluation will use the frozen "
            "module state.</div>"
        )
    elif event == "benchmark_pipeline_blocked":
        reason = html.escape(str(ev.get("reason") or "training incomplete"))
        return (
            f"<div style='{style}; color: var(--nika-danger);'>"
            f"<b>Evaluation blocked</b> · {reason}</div>"
        )
    elif event == "benchmark_pipeline_done":
        training = html.escape(str(ev.get("training_cases") or "0"))
        evaluation = html.escape(str(ev.get("evaluation_cases") or "0"))
        return (
            f"<div style='{style}; color: var(--nika-success);'>"
            f"<b>Benchmark pipeline completed</b> · {training} training cases, "
            f"{evaluation} evaluation cases</div>"
        )
    return None


st.markdown(
    '<div class="section-title" style="margin-top: 1.5rem;">Tracking</div>',
    unsafe_allow_html=True,
)
tab_progress, tab_logs = st.tabs(["Progress", "Logs"])

with tab_progress:
    formatted_msgs = []
    for ev in events:
        msg = format_event_message_html(ev)
        if msg:
            formatted_msgs.append(msg)

    if formatted_msgs:
        st.markdown('<div style="margin-top: 0.8rem;"></div>', unsafe_allow_html=True)
        # Render HTML block with all events inside to snapping them perfectly
        st.markdown(
            f"<div style='border: 1px solid var(--nika-line); border-radius: 10px; padding: 0.2rem 0.8rem; background: var(--nika-panel);'>{''.join(formatted_msgs[-12:])}</div>",
            unsafe_allow_html=True,
        )

with tab_logs:
    # Keep only the last 1000 lines to avoid UI freezes with large logs
    log_lines = log_text.splitlines() if log_text else []
    truncated_log = "\n".join(log_lines[-1000:])
    if len(log_lines) > 1000:
        truncated_log = (
            f"... [Truncated {len(log_lines) - 1000} lines from start] ...\n"
            + truncated_log
        )

    st.text_area(
        "Full log",
        value=truncated_log or "No log lines yet.",
        height=420,
        label_visibility="collapsed",
        key=log_key,
    )

st.markdown(
    '<div class="section-title" style="margin-top: 1.5rem;">Results</div>',
    unsafe_allow_html=True,
)

all_rows = _result_rows(benchmark_name=None)
training_rows = _result_rows(benchmark_name=None, stage="training")
evaluation_rows = _result_rows(benchmark_name=None, stage="evaluation")
standalone_rows = [row for row in all_rows if row.get("stage") == "all"]
training_experiments = {
    str(row.get("result_root") or "").removesuffix("-train")
    for row in training_rows
}
evaluation_rows = [
    row
    for row in evaluation_rows
    if str(row.get("result_root") or "").removesuffix("-eval")
    in training_experiments
]
result_rows = standalone_rows + training_rows + evaluation_rows
summary_tab, detail_tab = st.tabs(["Summary", "Details"])

with summary_tab:
    st.html(
        _results_table_html(
            _summary_result_rows(result_rows),
            RESULT_SUMMARY_COLUMNS,
        ),
        width="stretch",
    )

with detail_tab:
    st.html(
        _results_table_html(
            _summary_result_rows(result_rows),
            RESULT_DETAIL_COLUMNS,
        ),
        width="stretch",
    )
