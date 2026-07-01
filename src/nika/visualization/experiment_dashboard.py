"""Streamlit experiment runner for NIKA agent baselines and modules."""

from __future__ import annotations

import json
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
    read_run_log,
    read_run_spec,
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

      [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: .35rem; background: rgba(241, 245, 249, 0.85); border: 1px solid var(--line);
        border-radius: 14px; padding: .32rem; margin: .7rem 0 1.2rem;
      }
      [data-testid="stTabs"] [data-baseweb="tab"] {
        height: 42px; border-radius: 10px; padding: 0 1.2rem;
        color: var(--muted);
      }
      [data-testid="stTabs"] [aria-selected="true"] {
        background: rgba(14, 165, 233, 0.1); color: #0284c7;
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
      /* Primary button styling */
      button[data-testid="baseButton-primary"] {
        border: 1px solid #0284c7 !important;
        background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
        color: #ffffff !important;
      }
      button[data-testid="baseButton-primary"]:hover {
        border-color: #0284c7 !important;
        background: linear-gradient(135deg, #38bdf8, #0ea5e9) !important;
        color: #ffffff !important;
        box-shadow: 0 0 15px rgba(14, 165, 233, 0.25) !important;
      }
      /* Secondary button styling */
      button[data-testid="baseButton-secondary"], .stDownloadButton > button {
        border: 1px solid rgba(14, 165, 233, .28) !important;
        background: rgba(14, 165, 233, .08) !important;
        color: #0284c7 !important;
      }
      button[data-testid="baseButton-secondary"]:hover, .stDownloadButton > button:hover {
        border-color: #0ea5e9 !important;
        color: #0369a1 !important;
        background: rgba(14, 165, 233, .15) !important;
      }

      div[data-testid="stCodeBlock"] {
        border: 1px solid var(--line) !important;
        border-radius: 12px !important;
        background: #f8fafc !important;
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

    rows: list[dict[str, object]] = []
    for gkey, run_paths in sorted(
        grouped.items(),
        key=lambda item: item[0].stat().st_mtime if item[0].exists() else 0,
        reverse=True,
    ):
        detections: list[float] = []
        localizations: list[float] = []
        rcas: list[float] = []
        steps: list[float] = []
        tool_calls: list[float] = []
        in_tokens: list[float] = []
        out_tokens: list[float] = []
        tool_errors: list[float] = []
        durations: list[float] = []
        submitted = 0
        finished = 0
        failed = 0
        result_modules: set[str] = set()
        agents: set[str] = set()
        models: set[str] = set()
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
                ("localization_accuracy", localizations),
                ("rca_accuracy", rcas),
                ("steps", steps),
                ("tool_calls", tool_calls),
                ("in_tokens", in_tokens),
                ("out_tokens", out_tokens),
                ("tool_errors", tool_errors),
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
            if meta.get("agent_type"):
                agent_name = str(meta["agent_type"])
                agents.add(agent_name)
            if meta.get("model"):
                models.add(str(meta["model"]))
            updated = str(meta.get("updated_at") or meta.get("created_at") or updated)

        display_name = gkey.name

        rows.append(
            {
                "result_root": display_name,
                "cases": len(run_paths),
                "finished": finished,
                "failed": failed,
                "submitted": submitted,
                "detection": _avg(detections),
                "localization": _avg(localizations),
                "rca": _avg(rcas),
                "steps": _avg(steps),
                "tool_calls": _avg(tool_calls),
                "tool_errors": _sum(tool_errors),
                "token_in": _sum(in_tokens),
                "token_out": _sum(out_tokens),
                "duration": f"{int(sum(durations))}s" if durations else "-",
                "modules": ", ".join(sorted(result_modules)) or "-",
                "agent": ", ".join(sorted(agents)) or "-",
                "model": ", ".join(sorted(models)) or "-",
                "updated": updated,
            }
        )
    return rows


def _progress_fraction(events: list[dict[str, str]], command_count: int) -> tuple[float, str]:
    if not events:
        return 0.0, "Waiting"
    if events[-1]["event"] == "ui_run_done":
        return 1.0, "Done"

    step_index = 1
    step_total = max(1, command_count)
    for event in events:
        if event["event"] in {"ui_step_start", "ui_step_done"}:
            try:
                step_index = int(event.get("index", step_index))
                step_total = int(event.get("total", step_total))
            except ValueError:
                pass

    within_step = 0.0
    
    # Only show Step prefix if there are multiple steps in the plan
    prefix = f"Step {step_index}/{step_total}" if step_total > 1 else ""
    label = prefix

    for event in reversed(events):
        if event["event"] == "benchmark_progress":
            completed = event.get("completed", "0")
            total = event.get("total") or "30"
            try:
                within_step = int(completed) / max(1, int(total))
                suffix = f"{completed}/{total} completed"
            except (ValueError, TypeError):
                suffix = f"{completed} completed"
            label = f"{prefix}: {suffix}" if prefix else suffix
            break
        if event["event"] == "benchmark_summary":
            within_step = 1.0
            label = f"{prefix}: Completed" if prefix else "Completed"
            break
    if not label:
        label = prefix if prefix else "Running"

    fraction = (max(0, step_index - 1) + within_step) / step_total
    return max(0.0, min(1.0, fraction)), label


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

cfg_cols = st.columns(4, gap="medium")
with cfg_cols[0]:
    benchmark_name = st.text_input(
        "Benchmark",
        value=Path(default_benchmark_yaml_path()).stem,
    )
    benchmark_path = _benchmark_path_from_name(benchmark_name)
    row_count = _count_rows(benchmark_path)
with cfg_cols[1]:
    default_max_steps = resolve_max_steps(None)
    max_steps_str = st.text_input("Steps", value=str(default_max_steps))
    max_steps = int(max_steps_str) if max_steps_str.isdigit() else default_max_steps
with cfg_cols[2]:
    backend_options = ["custom", "openai", "deepseek", "ollama"]
    llm_backend = st.selectbox(
        "Backend",
        backend_options,
        index=backend_options.index(DEFAULT_LLM_BACKEND)
        if DEFAULT_LLM_BACKEND in backend_options
        else 0,
    )
with cfg_cols[3]:
    model = st.text_input("Model", value=DEFAULT_MODEL)

agent_cols = st.columns(2, gap="medium")
with agent_cols[0]:
    agent_type = st.selectbox(
        "Agent baseline",
        ["react", "plan-execute", "reflexion", "mock"],
    )

with agent_cols[1]:
    max_attempts_str = st.text_input("Attempts", value="3")
    max_attempts = int(max_attempts_str) if max_attempts_str.isdigit() else 3

st.markdown(
    "<div style='font-size: 0.8rem; font-weight: bold; color: var(--muted); "
    "margin-top: 0.5rem; margin-bottom: 0.25rem;'>Modules</div>",
    unsafe_allow_html=True,
)

col_t1, col_t2, col_t3 = st.columns([1, 1.5, 1.5], gap="medium")
with col_t1:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    tool_selected = st.checkbox("Tool Evolution", value=False)
with col_t2:
    tool_library_id = st.text_input("Tool library ID", value="tools-gptoss120-test", disabled=not tool_selected)
with col_t3:
    pass

col_mem1, col_mem2, col_mem3, col_mem4 = st.columns([1, 1.5, 0.75, 0.75], gap="medium")
with col_mem1:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    memory_selected = st.checkbox("Memory Evolution", value=False)
with col_mem2:
    memory_bank = st.text_input("Memory bank", value="memory-gptoss120-test", disabled=not memory_selected)
with col_mem3:
    memory_k = st.number_input("Memory top-k", min_value=1, max_value=20, value=5, disabled=not memory_selected)
with col_mem4:
    memory_tokens = st.number_input("Memory tokens", min_value=100, max_value=8000, value=1500, step=100, disabled=not memory_selected)

modules = []
if tool_selected:
    modules.append("tool_evolution")
if memory_selected:
    modules.append("memory_evolution")

# Evaluation Settings
st.markdown(
    "<div style='font-size: 0.8rem; font-weight: bold; color: var(--muted); "
    "margin-top: 1rem; margin-bottom: 0.25rem;'>Evaluation Settings</div>",
    unsafe_allow_html=True,
)
col_e1, col_e2, col_e3 = st.columns([1, 1.5, 1.5], gap="medium")
with col_e1:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    run_judge = st.checkbox("Run LLM judge", value=False)
with col_e2:
    judge_backend = st.text_input("Judge backend", value=llm_backend)
with col_e3:
    judge_model = st.text_input("Judge model", value=model)

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
    "memory_bank": memory_bank,
    "memory_k": int(memory_k),
    "memory_tokens": int(memory_tokens),
    "run_judge": bool(run_judge),
    "judge_backend": judge_backend,
    "judge_model": judge_model,
}
plan = build_command_plan(config)

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


# Run / Stop Button
all_runs = list_runs()
running_run = None
for r in all_runs[:3]:
    if run_status(r).get("status") == "running":
        running_run = r
        break

is_running = running_run is not None

if is_running:
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        if st.button("Queue Run", type="secondary", disabled=row_count is None, width="stretch"):
            run_dir = create_run(config)
            st.session_state["active_run_dir"] = str(run_dir)
            st.rerun()
    with col2:
        if st.button("Stop", type="primary", width="stretch"):
            with st.spinner("Stopping run and wiping Kathara containers..."):
                stop_run(running_run)
            st.rerun()
else:
    if st.button("Run", type="primary", disabled=row_count is None, width="stretch"):
        run_dir = create_run(config)
        st.session_state["active_run_dir"] = str(run_dir)
        st.rerun()
if row_count is None:
    st.error("Benchmark YAML not found.")

runs = list_runs()
selected = _selected_run_dir()
if runs:
    run_labels = []
    run_map = {}
    for path in runs:
        status_val = run_status(path).get("status") or "unknown"
        label = f"{path.name} ({status_val})"
        run_labels.append(label)
        run_map[label] = path

    selected_label = None
    if selected is not None:
        selected_status = run_status(selected).get("status") or "unknown"
        selected_label = f"{selected.name} ({selected_status})"

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
    is_running = (status.get("status") == "running")
else:
    status = {}
    log_text = ""
    events = []
    fraction, progress_label = 0.0, "No active run"
    log_key = "log-none"
    is_running = False



def format_event_message(ev: dict) -> str | None:
    event = ev.get("event")
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
        return f"**[Step {ev.get('index')}/{ev.get('total')}]** Starting: `{cmd_str}`"
    elif event == "ui_step_done":
        ret = ev.get("returncode", "0")
        status_word = "Completed" if str(ret) == "0" else "Failed"
        return f"**[Step {ev.get('index')}/{ev.get('total')}]** {status_word} (Exit: `{ret}`)"
    elif event == "ui_run_done":
        code = ev.get("exit_code", "0")
        return f"**Run Finished** (Exit: `{code}`)"
    elif event == "ui_run_stopped":
        return "**Run Stopped** by user. Starting next queued run if available."
    elif event == "benchmark_start":
        index = ev.get("index") or "?"
        scenario = ev.get("scenario") or "?"
        problem = ev.get("problem") or "?"
        return f"**[Scenario {index}]** Starting: `{scenario} / {problem}`"
    elif event == "benchmark_progress":
        completed = ev.get("completed", "0")
        failed = ev.get("failed", "0")
        total = ev.get("total") or "30"
        return f"**Benchmark Progress**: {completed}/{total} completed (Failed: {failed})"
    elif event == "benchmark_done":
        scenario = ev.get("scenario") or "?"
        problem = ev.get("problem") or "?"
        return f"**[Scenario]** Finished: `{scenario} / {problem}`"
    elif event == "benchmark_failed":
        scenario = ev.get("scenario") or "?"
        problem = ev.get("problem") or "?"
        return f"**[Scenario]** Failed: `{scenario} / {problem}`"
    return None

st.markdown('<div class="section-title" style="margin-top: 1.5rem;">Tracking</div>', unsafe_allow_html=True)
tab_progress, tab_logs = st.tabs(["Progress", "Logs"])

with tab_progress:
    st.markdown('<div style="margin-top: 0.5rem;"></div>', unsafe_allow_html=True)
    st.progress(fraction, text=progress_label)
    
    formatted_msgs = []
    for ev in events:
        msg = format_event_message(ev)
        if msg:
            formatted_msgs.append(msg)
    
    if formatted_msgs:
        st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
        for msg in formatted_msgs[-12:]:
            st.markdown(msg)
        
with tab_logs:
    st.markdown('<div style="margin-top: 0.5rem;"></div>', unsafe_allow_html=True)
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

# List and display all available experiment results automatically
result_rows = _result_rows(benchmark_name=None)
if not result_rows:
    import pandas as pd
    cols = [
        "result_root", "cases", "finished", "failed", "submitted",
        "detection", "localization", "rca", "steps", "tool_calls", "tool_errors",
        "token_in", "token_out", "duration", "modules", "agent", "model", "updated"
    ]
    df = pd.DataFrame(columns=cols)
    st.dataframe(df, width="stretch", hide_index=True)
else:
    st.dataframe(result_rows, width="stretch", hide_index=True)

if is_running:
    time.sleep(2)
    st.rerun()
