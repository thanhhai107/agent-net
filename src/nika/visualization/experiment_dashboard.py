"""Streamlit experiment runner for NIKA agent baselines and modules."""

from __future__ import annotations

import json
import html
import re
import shlex
import time
from pathlib import Path
from typing import Any

import streamlit as st

from agent.llm.model_factory import DEFAULT_LLM_BACKEND, DEFAULT_MODEL
from nika.config import BENCHMARK_DIR, RESULTS_DIR
from nika.utils.agent_config import resolve_max_steps
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
from nika.workflows.benchmark.load_config import load_benchmark_yaml
from nika.workflows.benchmark.run import default_benchmark_yaml_path


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
        --ink: #0f172a;
        --muted: #64748b;
        --panel: rgba(255, 255, 255, 0.78);
        --panel-strong: #f1f5f9;
        --line: rgba(15, 23, 42, 0.08);
        --cyan: #0ea5e9;
        --blue: #2563eb;
        --red: #e11d48;
        --amber: #d97706;
        --violet: #7c3aed;
      }

      .stApp {
        background:
          radial-gradient(circle at 12% 4%, rgba(203, 213, 225, 0.4), transparent 24rem),
          radial-gradient(circle at 92% 12%, rgba(186, 230, 253, 0.35), transparent 26rem),
          #f8fafc;
        color: var(--ink);
      }
      header[data-testid="stHeader"] {display: none !important;}
      .block-container {max-width: 1480px; padding: 2.8rem 2rem 4rem !important;}
      section[data-testid="stSidebar"] {
        background: rgba(241, 245, 249, 0.96);
        border-right: 1px solid var(--line);
      }
      section[data-testid="stSidebar"] .block-container {padding: 1.35rem 1.15rem;}
      h1, h2, h3 {letter-spacing: -.025em;}
      h1 {font-size: clamp(2rem, 3vw, 3.15rem) !important; line-height: 1.05 !important;}
      h2 {font-size: 1.28rem !important;}
      h3 {font-size: 1.02rem !important; color: #334155 !important;}

      .section-card {
        border:1px solid var(--line); border-radius:12px; padding:1.2rem;
        background:var(--panel); margin:.55rem 0 1rem;
        box-shadow:0 14px 35px rgba(15, 23, 42, 0.04);
      }
      .section-title {
        color: #1e293b; font-weight: 750; font-size: 1.1rem; margin: 1.2rem 0 .9rem;
      }
      .mini-grid {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.55rem;}
      .mini-item {
        border:1px solid var(--line); border-radius:12px; background:rgba(255, 255, 255, 0.7);
        padding:.55rem .65rem; min-height:58px;
      }
      .mini-label {font-size:.7rem; color:var(--muted); font-weight:700;}
      .mini-value {font-size:.82rem; color:#1e293b; font-weight:750; overflow-wrap:anywhere;}

      .status-pill {
        display:inline-flex; align-items:center; gap:.45rem; padding:.34rem .65rem;
        border:1px solid var(--line); border-radius:999px; color:#b9c7d9;
        background:rgba(255,255,255,.7); font-size:.78rem; font-weight:700;
      }
      .status-dot {width:7px; height:7px; border-radius:50%; background:#8a98aa;}
      .status-running .status-dot {background:#10b981; box-shadow:0 0 12px rgba(16,185,129,.45);}
      .status-finished .status-dot {background:#3b82f6;}
      .status-failed .status-dot {background:#ff647c; box-shadow:0 0 12px rgba(255,100,124,.65);}
      .status-queued .status-dot {background:#ffb454; box-shadow:0 0 12px rgba(255,180,84,.65);}

      [data-testid="stMetric"] {
        min-height: 112px;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 17px 18px;
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.95), rgba(241, 245, 249, 0.9));
        box-shadow: 0 12px 35px rgba(15, 23, 42, 0.04);
      }
      [data-testid="stMetricLabel"] {color: var(--muted); font-size: .78rem;}
      [data-testid="stMetricValue"] {color: var(--ink); font-weight: 700;}

      [data-testid="stTabs"] {
        border: 1px solid var(--line) !important;
        border-radius: 16px !important;
        background: var(--panel) !important;
        padding: 1.2rem !important;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.03) !important;
      }
      [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: .35rem; background: rgba(241, 245, 249, 0.85); border: 1px solid var(--line);
        border-radius: 12px; padding: .32rem; margin: 0 0 1rem 0 !important;
      }
      [data-testid="stTabs"] [data-baseweb="tab-panel"] {
        padding-top: 0px !important;
        padding-bottom: 0px !important;
      }
      [data-testid="stTabs"] [data-baseweb="tab"] {
        height: 42px; border-radius: 10px; padding: 0 1.2rem;
        color: var(--muted);
      }
      [data-testid="stTabs"] [aria-selected="true"] {
        background: rgba(14, 165, 233, 0.1); color: #0284c7;
      }
      div[data-baseweb="tab-border"] {
        display: none !important;
      }
      [data-testid="stDataFrame"] {
        border: 1px solid var(--line); border-radius: 14px; overflow: hidden;
      }
      [data-testid="stExpander"] {
        border: 1px solid var(--line); border-radius: 13px;
        background: rgba(255, 255, 255, 0.7);
      }

      div[data-testid="stCheckbox"] label {font-weight:700;}

      .stButton > button, .stDownloadButton > button {
        border-radius: 11px;
        font-weight: 800;
        transition: all 0.2s ease;
      }
      /* Stable action colors. Target explicit widget keys, not column order. */
      div.st-key-studio_run_button button,
      div.stButton button[data-testid="baseButton-primary"] {
        border: 1px solid #16a34a !important;
        background: linear-gradient(135deg, #22c55e, #16a34a) !important;
        color: #ffffff !important;
        border-radius: 11px !important;
        font-weight: 800 !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
      }
      div.st-key-studio_run_button button:hover,
      div.stButton button[data-testid="baseButton-primary"]:hover {
        border-color: #15803d !important;
        background: linear-gradient(135deg, #4ade80, #22c55e) !important;
        color: #ffffff !important;
        transform: translateY(-1.5px) !important;
        box-shadow: 0 6px 20px rgba(34, 197, 94, 0.35) !important;
      }
      div.st-key-studio_run_button button:active,
      div.stButton button[data-testid="baseButton-primary"]:active {
        transform: translateY(0.5px) !important;
      }

      /* Secondary button styling */
      button[data-testid="baseButton-secondary"], .stDownloadButton > button {
        border: 1px solid rgba(14, 165, 233, .28) !important;
        background: rgba(14, 165, 233, .08) !important;
        color: #0284c7 !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
      }
      button[data-testid="baseButton-secondary"]:hover, .stDownloadButton > button:hover {
        border-color: #0ea5e9 !important;
        color: #0369a1 !important;
        background: rgba(14, 165, 233, .15) !important;
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
        border: 1px solid rgba(217, 119, 6, .38) !important;
        background: rgba(251, 191, 36, .18) !important;
        color: #b45309 !important;
      }
      div.st-key-studio_resume_button button:hover {
        border-color: #d97706 !important;
        background: rgba(251, 191, 36, .28) !important;
        color: #92400e !important;
        box-shadow: 0 6px 20px rgba(217, 119, 6, 0.18) !important;
      }
      div.st-key-studio_stop_button button {
        border: 1px solid rgba(220, 38, 38, .38) !important;
        background: rgba(239, 68, 68, .12) !important;
        color: #dc2626 !important;
      }
      div.st-key-studio_stop_button button:hover {
        border-color: #b91c1c !important;
        background: rgba(239, 68, 68, .22) !important;
        color: #b91c1c !important;
        box-shadow: 0 6px 20px rgba(239, 68, 68, 0.18) !important;
      }

      div[data-testid="stCodeBlock"] {
        border: 2px solid var(--cyan) !important;
        border-radius: 12px !important;
        background: #f0f9ff !important;
        box-shadow: 0 4px 18px rgba(14, 165, 233, 0.14) !important;
      }
      textarea {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
        font-size:.78rem !important;
        background-color: #f8fafc !important;
        color: var(--ink) !important;
        border: 1px solid var(--line) !important;
      }

      .nika-brand {display:flex; align-items:center; gap:.75rem; margin:.2rem 0 1.6rem;}
      .nika-mark {
        width:38px; height:38px; display:grid; place-items:center; border-radius:12px;
        color:#ffffff; background:linear-gradient(135deg,#38bdf8,#0284c7);
        font-size:1.15rem; font-weight:900; box-shadow:0 0 24px rgba(14, 165, 233, .2);
      }
      .nika-brand-title {font-size:1.05rem; font-weight:800; letter-spacing:.08em; color: #0f172a;}
      .nika-brand-sub {font-size:.72rem; color:var(--muted); letter-spacing:.04em;}

      .eyebrow {
        color:#0284c7; font-size:.74rem; font-weight:800; letter-spacing:.14em;
        text-transform:uppercase; margin-bottom:.55rem;
      }

      /* Custom styling for multiselect tags */
      div[data-baseweb="tag"] {
        background-color: rgba(14, 165, 233, .08) !important;
        border: 1px solid rgba(14, 165, 233, .2) !important;
        border-radius: 6px !important;
        color: #0369a1 !important;
        padding: 2px 8px !important;
      }
      div[data-baseweb="tag"] span {
        color: #0369a1 !important;
        font-weight: 600 !important;
      }
      div[data-baseweb="tag"] svg {
        fill: #0369a1 !important;
      }
      @media (max-width: 900px) {.mini-grid {grid-template-columns:1fr 1fr;}}
    </style>
    """,
    unsafe_allow_html=True,
)


def _benchmark_path_from_name(value: str) -> Path:
    raw = value.strip() or Path(default_benchmark_yaml_path()).stem
    path = Path(raw).expanduser()
    if path.is_absolute() or path.parent != Path("."):
        return path if path.suffix in {".yaml", ".yml"} else path.with_suffix(".yaml")
    name = re.sub(r"\.ya?ml$", "", raw)
    return BENCHMARK_DIR / f"{name}.yaml"


def _count_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return len(load_benchmark_yaml(path))
    except (OSError, ValueError):
        return None


def _status_html(status: str) -> str:
    return (
        f'<span class="status-pill status-{status}">'
        '<span class="status-dot"></span>'
        f"{status}"
        "</span>"
    )


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
    "detection_score",
    "localization_f1",
    "rca_f1",
    "tool_calls",
    "tool_errors",
    "total_tokens",
    "duration",
    "agent",
    "model",
    "updated",
]

RESULT_DETAIL_COLUMNS = [
    "result_root",
    "progress",
    "detection_score",
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
    "memory_reward",
    "memory_advantage",
    "memory_success",
    "memory_added_tokens",
    "memory_delta_tokens_step",
    "memory_selector",
    "memory_controller",
    "draft_planned",
    "draft_consumed",
    "duration",
    "modules",
    "agent",
    "model",
    "updated",
]

RESULT_NUMERIC_COLUMNS = {
    "detection_score",
    "localization_f1",
    "rca_f1",
    "localization_precision",
    "rca_precision",
    "tool_calls",
    "detection_precision",
    "detection_recall",
    "detection_f1",
    "detection_fpr",
    "paired_delta_detection",
    "paired_delta_localization_f1",
    "paired_delta_rca_f1",
    "memory_reward",
    "memory_advantage",
    "memory_success",
    "memory_delta_tokens_step",
}

RESULT_INTEGER_COLUMNS = {
    "cases",
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
    "memory_added_tokens",
    "draft_planned",
    "draft_consumed",
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


def _results_dataframe(rows: list[dict[str, object]], columns: list[str]):
    import pandas as pd

    display_rows = [
        {
            column: _result_display_value(column, row.get(column))
            for column in columns
        }
        for row in rows
    ]
    return pd.DataFrame(display_rows, columns=columns)


def _transposed_results_dataframe(rows: list[dict[str, object]], columns: list[str]):
    frame = _results_dataframe(rows, columns)
    if frame.empty:
        return frame
    result_names = [
        str(value) if value is not None and value != "" else f"Result {index + 1}"
        for index, value in enumerate(frame["result_root"].tolist())
    ]
    metric_columns = [column for column in columns if column != "result_root"]
    transposed = frame[metric_columns].transpose()
    transposed.columns = result_names
    transposed.insert(0, "metric", transposed.index)
    return transposed.reset_index(drop=True)


def _result_column_config() -> dict[str, object]:
    return {
        "metric": st.column_config.TextColumn("Metric", width="medium"),
        "result_root": st.column_config.TextColumn("Result", width="large"),
        "modules": st.column_config.TextColumn("Modules", width="medium"),
        "progress": st.column_config.TextColumn("Progress", width="small"),
        "cases": st.column_config.NumberColumn("Cases", format="%d"),
        "finished": st.column_config.NumberColumn("Finished", format="%d"),
        "failed": st.column_config.NumberColumn("Failed", format="%d"),
        "submitted": st.column_config.NumberColumn("Submitted", format="%d"),
        "detection_score": st.column_config.NumberColumn(
            "Detection", format="%.2f"
        ),
        "localization_f1": st.column_config.NumberColumn(
            "Loc F1", format="%.2f"
        ),
        "rca_f1": st.column_config.NumberColumn("RCA F1", format="%.2f"),
        "localization_precision": st.column_config.NumberColumn(
            "Loc Prec", format="%.2f"
        ),
        "rca_precision": st.column_config.NumberColumn("RCA Prec", format="%.2f"),
        "tool_calls": st.column_config.NumberColumn("Tool Calls", format="%.2f"),
        "tool_errors": st.column_config.NumberColumn("Tool Errors", format="%d"),
        "detection_precision": st.column_config.NumberColumn(
            "Detect Prec", format="%.2f"
        ),
        "detection_recall": st.column_config.NumberColumn(
            "Detect Recall", format="%.2f"
        ),
        "detection_f1": st.column_config.NumberColumn("Detect F1", format="%.2f"),
        "detection_fpr": st.column_config.NumberColumn("Detect FPR", format="%.2f"),
        "detection_fp": st.column_config.NumberColumn("FP", format="%d"),
        "detection_fn": st.column_config.NumberColumn("FN", format="%d"),
        "paired_baseline": st.column_config.TextColumn("Baseline", width="large"),
        "paired_cases": st.column_config.NumberColumn("Paired", format="%d"),
        "paired_delta_detection": st.column_config.NumberColumn(
            "Δ Detect", format="%+.2f"
        ),
        "paired_delta_localization_f1": st.column_config.NumberColumn(
            "Δ Loc F1", format="%+.2f"
        ),
        "paired_delta_rca_f1": st.column_config.NumberColumn(
            "Δ RCA F1", format="%+.2f"
        ),
        "paired_record": st.column_config.TextColumn("W/L/T", width="small"),
        "paired_wins": st.column_config.NumberColumn("Wins", format="%d"),
        "paired_losses": st.column_config.NumberColumn("Losses", format="%d"),
        "paired_ties": st.column_config.NumberColumn("Ties", format="%d"),
        "token_in": st.column_config.NumberColumn("Token In", format="%d"),
        "token_out": st.column_config.NumberColumn("Token Out", format="%d"),
        "total_tokens": st.column_config.NumberColumn("Total Tokens", format="%d"),
        "memory_reward": st.column_config.NumberColumn("Mem Reward", format="%.2f"),
        "memory_advantage": st.column_config.NumberColumn("Mem Adv", format="%.2f"),
        "memory_success": st.column_config.NumberColumn("Mem Success", format="%.2f"),
        "memory_added_tokens": st.column_config.NumberColumn("Mem Tokens", format="%d"),
        "memory_delta_tokens_step": st.column_config.NumberColumn(
            "Mem Δ Tokens/Step", format="%.2f"
        ),
        "memory_selector": st.column_config.TextColumn("Mem Selector"),
        "memory_controller": st.column_config.TextColumn("Mem Controller"),
        "draft_planned": st.column_config.NumberColumn("DRAFT Planned", format="%d"),
        "draft_consumed": st.column_config.NumberColumn("DRAFT Used", format="%d"),
        "duration": st.column_config.TextColumn("Duration", width="small"),
        "agent": st.column_config.TextColumn("Agent", width="small"),
        "model": st.column_config.TextColumn("Model", width="medium"),
        "updated": st.column_config.TextColumn("Updated", width="medium"),
    }


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
    return (
        not bool(meta.get("tool_evolution_enabled"))
        and str(meta.get("memory_mode") or "off") == "off"
    )


def _metric_total(metrics: dict) -> float | None:
    values = [
        _float(metrics.get("detection_score")),
        _float(metrics.get("localization_f1")),
        _float(metrics.get("rca_f1")),
    ]
    if any(value is None for value in values):
        return None
    return sum(value or 0.0 for value in values) / 3


def _root_case_map(run_paths: list[Path]) -> dict[object, dict[str, object]]:
    cases: dict[object, dict[str, object]] = {}
    for run_path in run_paths:
        meta = _read_json(run_path)
        metrics = _read_json(run_path.parent / "eval_metrics.json")
        key = _case_key(meta)
        if key is None:
            continue
        cases[key] = {"meta": meta, "metrics": metrics}
    return cases


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
        target_metrics = target_cases[key]["metrics"]
        baseline_metrics = root_cases[baseline_root][key]["metrics"]
        for metric, values in deltas.items():
            target_value = _float(target_metrics.get(metric))
            baseline_value = _float(baseline_metrics.get(metric))
            if target_value is not None and baseline_value is not None:
                values.append(target_value - baseline_value)
        target_total = _metric_total(target_metrics)
        baseline_total = _metric_total(baseline_metrics)
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


def _result_rows(*, benchmark_name: str | None = None) -> list[dict[str, object]]:
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

    root_cases = {
        root: _root_case_map(run_paths)
        for root, run_paths in grouped.items()
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
        detections: list[float] = []
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
        memory_rewards: list[float] = []
        memory_advantages: list[float] = []
        memory_successes: list[float] = []
        memory_added_tokens: list[float] = []
        memory_delta_tokens: list[float] = []
        draft_planned: list[float] = []
        draft_consumed: list[float] = []
        submitted = 0
        finished = 0
        failed = 0
        result_modules: set[str] = set()
        agents: set[str] = set()
        models: set[str] = set()
        memory_selectors: set[str] = set()
        memory_controllers: set[str] = set()
        updated = "-"

        for run_path in run_paths:
            session_dir = run_path.parent
            meta = _read_json(run_path)
            metrics = _read_json(session_dir / "eval_metrics.json")
            if meta.get("status") == "finished":
                finished += 1
            if (session_dir / "submission.json").exists():
                submitted += 1
            elif meta.get("status") != "running":
                failed += 1

            for key, target in (
                ("detection_score", detections),
                ("localization_f1", localization_f1s),
                ("rca_f1", rca_f1s),
                ("localization_precision", localization_precisions),
                ("rca_precision", rca_precisions),
                ("tool_calls", tool_calls),
                ("in_tokens", in_tokens),
                ("out_tokens", out_tokens),
                ("tool_errors", tool_errors),
            ):
                value = _float(metrics.get(key))
                if value is not None:
                    target.append(value)

            detection_tp += _float(metrics.get("detection_tp")) or 0.0
            detection_tn += _float(metrics.get("detection_tn")) or 0.0
            detection_fp += _float(metrics.get("detection_fp")) or 0.0
            detection_fn += _float(metrics.get("detection_fn")) or 0.0

            memory_update = metrics.get("memory_update") or {}
            if isinstance(memory_update, dict):
                for key, target in (
                    ("episode_reward", memory_rewards),
                    ("episode_advantage", memory_advantages),
                    ("total_added_tokens", memory_added_tokens),
                    ("delta_prompt_tokens_per_step", memory_delta_tokens),
                ):
                    value = _float(memory_update.get(key))
                    if value is not None:
                        target.append(value)
                if memory_update.get("episode_success") is not None:
                    memory_successes.append(
                        1.0 if memory_update.get("episode_success") else 0.0
                    )
            for key, target in (
                ("draft_planned_explorations", draft_planned),
                ("draft_consumed_explorations", draft_consumed),
            ):
                value = _float(metrics.get(key))
                if value is not None:
                    target.append(value)

            dur = _parse_duration(meta)
            if dur is not None:
                durations.append(dur)

            if meta.get("tool_evolution_enabled"):
                result_modules.add("Tool Evolution")
            if meta.get("memory_mode") and meta.get("memory_mode") != "off":
                result_modules.add("Memory Evolution")
                if meta.get("memory_skill_selector_mode"):
                    memory_selectors.add(str(meta["memory_skill_selector_mode"]))
                if meta.get("memory_meta_controller_mode"):
                    memory_controllers.add(str(meta["memory_meta_controller_mode"]))
            if meta.get("agent_type"):
                agent_name = str(meta["agent_type"])
                agents.add(agent_name)
            if meta.get("model"):
                models.add(str(meta["model"]))
            updated = str(meta.get("updated_at") or meta.get("created_at") or updated)

        display_name = gkey.name

        row = {
            "result_root": display_name,
            "cases": len(run_paths),
            "finished": finished,
            "failed": failed,
            "submitted": submitted,
            "detection_score": _avg(detections),
            "detection_precision": _ratio(detection_tp, detection_tp + detection_fp),
            "detection_recall": _ratio(detection_tp, detection_tp + detection_fn),
            "detection_f1": (
                "-"
                if detection_tp + detection_fp <= 0
                or detection_tp + detection_fn <= 0
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
            "memory_reward": _avg(memory_rewards),
            "memory_advantage": _avg(memory_advantages),
            "memory_success": _avg(memory_successes),
            "memory_added_tokens": _sum(memory_added_tokens),
            "memory_delta_tokens_step": _avg(memory_delta_tokens),
            "memory_selector": ", ".join(sorted(memory_selectors)) or "-",
            "memory_controller": ", ".join(sorted(memory_controllers)) or "-",
            "draft_planned": _sum(draft_planned),
            "draft_consumed": _sum(draft_consumed),
            "duration": f"{int(sum(durations))}s" if durations else "-",
            "modules": ", ".join(sorted(result_modules)) or "-",
            "agent": ", ".join(sorted(agents)) or "-",
            "model": ", ".join(sorted(models)) or "-",
            "updated": updated,
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


def _event_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _index_progress(value: object) -> tuple[int | None, int | None]:
    raw = str(value or "")
    if "/" not in raw:
        return None, None
    left, right = raw.split("/", 1)
    return _event_int(left), _event_int(right)


def _benchmark_progress_state(events: list[dict[str, str]]) -> tuple[int, int, str] | None:
    completed: int | None = None
    total: int | None = None
    failed = 0
    summary_seen = False

    for event in events:
        kind = event.get("event")
        if kind not in {
            "benchmark_start",
            "benchmark_skip",
            "benchmark_progress",
            "benchmark_failed",
            "benchmark_summary",
        }:
            continue

        index_value, index_total = _index_progress(event.get("index"))
        if index_total is not None:
            total = index_total

        event_total = _event_int(event.get("total"))
        if event_total is not None:
            total = event_total

        event_completed = _event_int(event.get("completed"))
        if event_completed is not None:
            completed = event_completed
        elif kind == "benchmark_start" and completed is None and index_value is not None:
            completed = max(0, index_value - 1)

        event_failed = _event_int(event.get("failed"))
        if event_failed is not None:
            failed = event_failed

        if kind == "benchmark_summary":
            summary_seen = True

    if completed is None or total is None:
        return None
    completed = max(0, min(completed, total))
    suffix = f"{completed}/{total} completed"
    if failed:
        suffix += f" ({failed} failed)"
    if summary_seen and completed >= total and not failed:
        suffix = "Completed"
    return completed, total, suffix


def _progress_fraction(events: list[dict[str, str]], command_count: int) -> tuple[float, str]:
    if not events:
        return 0.0, "Waiting"

    benchmark_state = _benchmark_progress_state(events)
    if benchmark_state is not None:
        completed, total, label = benchmark_state
        return completed / max(1, total), label

    if events[-1]["event"] == "ui_run_done":
        exit_code = events[-1].get("exit_code", "1")
        return (1.0, "Done") if str(exit_code) == "0" else (0.0, "Failed")

    step_index = 1
    step_total = max(1, command_count)
    for event in events:
        if event["event"] in {"ui_step_start", "ui_step_done"}:
            try:
                step_index = int(event.get("index", step_index))
                step_total = int(event.get("total", step_total))
            except ValueError:
                pass

    # Only show Step prefix if there are multiple steps in the plan
    prefix = f"Step {step_index}/{step_total}" if step_total > 1 else ""
    label = prefix if prefix else "Running"

    fraction = max(0, step_index - 1) / step_total
    return max(0.0, min(1.0, fraction)), label


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
                f"{html.escape(key.removeprefix('inject_'))}="
                f"{html.escape(str(value))}"
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
    case_summary = _event_case_summary(ev)
    inject_summary = _event_inject_summary(ev)
    return (
        f"<div style='{style}; color: {color};'>"
        f"<b>[Case {index}]</b> {verb}: "
        f"<code style='font-size: 0.75rem; background: #f1f5f9; "
        f"padding: 2px 4px; border-radius: 4px;'>{case_summary}</code>"
        f"<div style='margin-top: 2px; color: #64748b;'>inject: "
        f"<code style='font-size: 0.75rem;'>{inject_summary}</code></div>"
        f"</div>"
    )


st.markdown('<div class="eyebrow">Experimentation & Benchmarking Suite</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="nika-brand" style="margin-bottom: 2rem;">
      <div class="nika-mark">N</div>
      <div>
        <div class="nika-brand-title" style="font-size: 1.5rem; line-height: 1.1;">NIKA</div>
        <div class="nika-brand-sub" style="font-size: 0.9rem; letter-spacing: 0.12em;">EXPERIMENTS STUDIO</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">Studio</div>', unsafe_allow_html=True)

with st.expander("Baseline Settings", expanded=True):
    b_col1, b_col2, b_col3 = st.columns([1.2, 1, 1.5], gap="small")
    with b_col1:
        agent_type = st.selectbox(
            "Workflow",
            ["react", "plan-execute", "reflexion", "mock"],
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

    b_col4, b_col5, b_col6 = st.columns([1.5, 1, 0.8], gap="small")
    with b_col4:
        benchmark_name = st.text_input(
            "Benchmark",
            value=Path(default_benchmark_yaml_path()).stem,
        )
        benchmark_path = _benchmark_path_from_name(benchmark_name)
        row_count = _count_rows(benchmark_path)
    with b_col5:
        default_max_steps = resolve_max_steps(None)
        max_steps_str = st.text_input("Steps", value=str(default_max_steps))
        max_steps = int(max_steps_str) if max_steps_str.isdigit() else default_max_steps
    with b_col6:
        max_attempts_str = st.text_input("Attempts", value="3")
        max_attempts = int(max_attempts_str) if max_attempts_str.isdigit() else 3
 
st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
col_modules = st.columns(2, gap="medium")

with col_modules[0]:
    with st.expander("Tool Evolution Settings", expanded=False):
        tool_selected = st.checkbox("Enable Tool Evolution", value=False)
        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
        
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
                "Tool doc chars",
                min_value=100,
                max_value=2000,
                value=500,
                step=50,
                disabled=not tool_selected,
            )
            
        t_col3, t_col4 = st.columns(2, gap="small")
        with t_col3:
            tool_prompt_doc_limit = st.number_input(
                "DRAFT docs",
                min_value=1,
                max_value=20,
                value=6,
                disabled=not tool_selected,
            )
        with t_col4:
            tool_scoped_prompt_doc_limit = st.number_input(
                "Scoped docs",
                min_value=1,
                max_value=20,
                value=4,
                disabled=not tool_selected,
            )
            
        t_col5, t_col6, t_col7 = st.columns(3, gap="small")
        with t_col5:
            tool_planned_checks = st.number_input(
                "Planned checks",
                min_value=0,
                max_value=20,
                value=4,
                disabled=not tool_selected,
            )
        with t_col6:
            tool_next_checks = st.number_input(
                "Next checks",
                min_value=0,
                max_value=10,
                value=2,
                disabled=not tool_selected,
            )
        with t_col7:
            tool_convergence_threshold = st.number_input(
                "Convergence",
                min_value=0.0,
                max_value=1.0,
                value=0.75,
                step=0.05,
                disabled=not tool_selected,
            )

with col_modules[1]:
    with st.expander("Memory Evolution Settings", expanded=False):
        memory_selected = st.checkbox("Enable Memory Evolution", value=False)
        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
        
        m_col1, m_col2, m_col3 = st.columns([1.5, 1, 1.2], gap="small")
        with m_col1:
            memory_bank = st.text_input(
                "Memory bank", 
                value="",
                placeholder="auto",
                disabled=not memory_selected
            )
        with m_col2:
            memory_k = st.number_input(
                "Memory top-k", 
                min_value=1, 
                max_value=20, 
                value=5, 
                disabled=not memory_selected
            )
        with m_col3:
            memory_tokens = st.number_input(
                "Memory tokens", 
                min_value=100, 
                max_value=8000, 
                value=1500, 
                step=100, 
                disabled=not memory_selected
            )
            
        m_col4, m_col5, m_col6 = st.columns(3, gap="small")
        with m_col4:
            memory_selector = st.selectbox(
                "Memory selector",
                ["lcb", "llm_topk_lcb"],
                disabled=not memory_selected,
            )
        with m_col5:
            memory_meta_controller = st.selectbox(
                "Memory controller",
                ["heuristic", "llm"],
                disabled=not memory_selected,
            )
        with m_col6:
            memory_max_skill_age = st.number_input(
                "Skill max age",
                min_value=1,
                max_value=20,
                value=4,
                disabled=not memory_selected,
            )
            
        m_col7, m_col8, m_col9 = st.columns(3, gap="small")
        with m_col7:
            memory_selector_min_lcb = st.number_input(
                "Selector min LCB",
                value=-0.05,
                step=0.01,
                disabled=not memory_selected,
            )
        with m_col8:
            memory_selector_nominee_k = st.number_input(
                "Nominee k",
                min_value=1,
                max_value=20,
                value=3,
                disabled=not memory_selected,
            )
        with m_col9:
            memory_pool_size = st.number_input(
                "Skill pool",
                min_value=1,
                max_value=200,
                value=32,
                disabled=not memory_selected,
            )
            
        m_col10, m_col11, m_col12 = st.columns(3, gap="small")
        with m_col10:
            memory_evolution_threshold = st.number_input(
                "Evolution samples",
                min_value=1,
                max_value=50,
                value=3,
                disabled=not memory_selected,
            )
        with m_col11:
            memory_best_of_n = st.number_input(
                "Best of N",
                min_value=1,
                max_value=20,
                value=3,
                disabled=not memory_selected,
            )
        with m_col12:
            memory_ppo_epsilon = st.number_input(
                "PPO epsilon",
                min_value=0.0,
                value=0.2,
                step=0.05,
                disabled=not memory_selected,
            )

modules = []
if tool_selected:
    modules.append("tool_evolution")
if memory_selected:
    modules.append("memory_evolution")

# Evaluation Settings
st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
with st.expander("Evaluation Settings", expanded=False):
    run_judge = st.checkbox("Run LLM judge", value=False)
    e_col1, e_col2 = st.columns(2, gap="small")
    with e_col1:
        judge_backend = st.text_input("Judge backend", value=llm_backend, disabled=not run_judge)
    with e_col2:
        judge_model = st.text_input("Judge model", value=model, disabled=not run_judge)

config = {
    "benchmark_file": str(benchmark_path),
    "modules": modules,
    "agent_type": agent_type,
    "llm_backend": llm_backend,
    "model": model,
    "max_steps": int(max_steps),
    "max_attempts": int(max_attempts),
    "parallel": 1,
    "tool_library_id": tool_library_id,
    "tool_doc_chars": int(tool_doc_chars),
    "tool_prompt_doc_limit": int(tool_prompt_doc_limit),
    "tool_scoped_prompt_doc_limit": int(tool_scoped_prompt_doc_limit),
    "tool_planned_checks": int(tool_planned_checks),
    "tool_next_checks": int(tool_next_checks),
    "tool_convergence_threshold": float(tool_convergence_threshold),
    "memory_bank": memory_bank,
    "memory_k": int(memory_k),
    "memory_tokens": int(memory_tokens),
    "memory_selector": memory_selector,
    "memory_meta_controller": memory_meta_controller,
    "memory_max_skill_age": int(memory_max_skill_age),
    "memory_selector_min_lcb": float(memory_selector_min_lcb),
    "memory_selector_nominee_k": int(memory_selector_nominee_k),
    "memory_pool_size": int(memory_pool_size),
    "memory_evolution_threshold": int(memory_evolution_threshold),
    "memory_best_of_n": int(memory_best_of_n),
    "memory_ppo_epsilon": float(memory_ppo_epsilon),
    "run_judge": bool(run_judge),
    "judge_backend": judge_backend,
    "judge_model": judge_model,
}
prepared_config = prepare_experiment_config(config)
plan = build_command_plan(prepared_config)

# Renders the commands directly under the Current Config grid with a top spacing
st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
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
        disabled=row_count is None,
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
        disabled=row_count is None,
        width="stretch",
    ):
        run_dir = create_run(prepared_config)
        st.session_state["active_run_dir"] = str(run_dir)
        st.rerun()
if row_count is None:
    st.error("Benchmark YAML not found.")

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
        selected_status = run_statuses.get(selected, run_status(selected)).get("status") or "unknown"
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
            index=run_labels.index(selected_label) if selected_label in run_labels else 0,
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
    fraction, progress_label = _progress_fraction(events, len(spec.get("commands") or []))
    log_key = f"log-{selected.name}"
else:
    status = {}
    log_text = ""
    events = []
    fraction, progress_label = 0.0, "No active run"
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
            with st.spinner("Stopping selected run and wiping Kathara containers..."):
                stop_run(selected)
            st.rerun()



def format_event_message_html(ev: dict) -> str | None:
    event = ev.get("event")
    style = "font-size: 0.8rem; padding: 6px 0; border-bottom: 1px solid rgba(15, 23, 42, 0.05); color: #334155; line-height: 1.4;"
    
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
        return f"<div style='{style}'><b>[Step {ev.get('index')}/{ev.get('total')}]</b> Starting: <code style='font-size: 0.75rem; background: #f1f5f9; padding: 2px 4px; border-radius: 4px;'>{cmd_str}</code></div>"
    elif event == "ui_step_done":
        ret = ev.get("returncode", "0")
        status_word = "Completed" if str(ret) == "0" else "Failed"
        color = "#16a34a" if str(ret) == "0" else "#dc2626"
        return f"<div style='{style}'><b>[Step {ev.get('index')}/{ev.get('total')}]</b> <span style='color: {color}; font-weight: bold;'>{status_word}</span> (Exit: <code>{ret}</code>)</div>"
    elif event == "ui_run_done":
        code = ev.get("exit_code", "0")
        return f"<div style='{style}; font-weight: bold; color: #1e293b;'>Run Finished (Exit: <code>{code}</code>)</div>"
    elif event == "ui_run_resumed":
        return f"<div style='{style}; color: #2563eb;'><b>Run Resumed</b> in-place with existing results.</div>"
    elif event == "ui_run_stopped":
        return f"<div style='{style}; color: #d97706;'><b>Run Stopped</b> by user. Starting next queued run if available.</div>"
    elif event == "benchmark_start":
        return _case_event_html(ev, style=style, verb="Starting", color="#334155")
    elif event == "benchmark_skip":
        return _case_event_html(
            ev,
            style=style,
            verb="Skipped existing result",
            color="#64748b",
        )
    elif event == "benchmark_progress":
        return None
    elif event == "benchmark_done":
        return _case_event_html(ev, style=style, verb="Finished", color="#16a34a")
    elif event == "benchmark_failed":
        return _case_event_html(ev, style=style, verb="Failed", color="#dc2626")
    return None
 
st.markdown('<div class="section-title" style="margin-top: 1.5rem;">Tracking</div>', unsafe_allow_html=True)
tab_progress, tab_logs = st.tabs(["Progress", "Logs"])
 
with tab_progress:
    # Render a professional custom HTML/CSS progress bar with text inside
    pct = int(fraction * 100)
    st.markdown(
        f"""
        <div style="margin: 0.5rem auto 1rem auto; width: 98%; max-width: 1400px; height: 28px; background-color: rgba(15, 23, 42, 0.06); border-radius: 8px; position: relative; overflow: hidden; display: flex; align-items: center; box-shadow: inset 0 1px 2px rgba(0,0,0,0.06); box-sizing: border-box;">
          <div style="width: {pct}%; height: 100%; background: linear-gradient(90deg, #0ea5e9, #2563eb); border-radius: 8px; transition: width 0.4s ease;"></div>
          <div style="position: absolute; width: 100%; text-align: center; left: 0; top: 0; line-height: 28px; font-size: 0.82rem; font-weight: 800; color: #ffffff; text-shadow: 0 1.5px 4px rgba(0, 0, 0, 0.95); z-index: 2; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding: 0 12px; box-sizing: border-box; pointer-events: none; letter-spacing: 0.02em;">
            {progress_label} ({pct}%)
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    formatted_msgs = []
    for ev in events:
        msg = format_event_message_html(ev)
        if msg:
            formatted_msgs.append(msg)
    
    if formatted_msgs:
        st.markdown('<div style="margin-top: 0.8rem;"></div>', unsafe_allow_html=True)
        # Render HTML block with all events inside to snapping them perfectly
        st.markdown(
            f"<div style='border: 1px solid rgba(15, 23, 42, 0.08); border-radius: 10px; padding: 0.2rem 0.8rem; background: #ffffff;'>{''.join(formatted_msgs[-12:])}</div>",
            unsafe_allow_html=True
        )
        
with tab_logs:
    # Keep only the last 1000 lines to avoid UI freezes with large logs
    log_lines = log_text.splitlines() if log_text else []
    truncated_log = "\n".join(log_lines[-1000:])
    if len(log_lines) > 1000:
        truncated_log = f"... [Truncated {len(log_lines) - 1000} lines from start] ...\n" + truncated_log

    st.text_area(
        "Full log",
        value=truncated_log or "No log lines yet.",
        height=420,
        label_visibility="collapsed",
        key=log_key,
    )

st.markdown('<div class="section-title" style="margin-top: 1.5rem;">Results</div>', unsafe_allow_html=True)

result_rows = _result_rows(benchmark_name=None)
summary_tab, detail_tab = st.tabs(["Summary", "Details"])
column_config = _result_column_config()

with summary_tab:
    st.dataframe(
        _transposed_results_dataframe(
            _summary_result_rows(result_rows),
            RESULT_SUMMARY_COLUMNS,
        ),
        width="stretch",
        hide_index=True,
        column_config=column_config,
    )

with detail_tab:
    st.dataframe(
        _transposed_results_dataframe(
            _summary_result_rows(result_rows),
            RESULT_DETAIL_COLUMNS,
        ),
        width="stretch",
        hide_index=True,
        column_config=column_config,
    )

if has_running_run:
    time.sleep(2)
    st.rerun()
