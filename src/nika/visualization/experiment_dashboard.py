"""Streamlit experiment runner for NIKA benchmark modules."""

from __future__ import annotations

import json
import shlex
import time
from pathlib import Path
from typing import Any

import streamlit as st

from nika.config import BENCHMARK_DIR, RESULTS_DIR
from nika.visualization.experiment_runner import (
    MODULE_LABELS,
    build_command_plan,
    create_run,
    experiment_label,
    list_runs,
    parse_progress_events,
    read_run_log,
    read_run_spec,
    run_status,
    stop_run,
)
from nika.workflows.benchmark.run import default_benchmark_csv_path


st.set_page_config(
    page_title="NIKA Experiment Studio",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      :root {--line:#dce3ed;--muted:#64748b;--soft:#f8fafc;--ink:#0f172a;}
      .block-container {max-width: 1520px; padding-top: 1.05rem;}
      .nika-title {
        display:flex; align-items:flex-end; justify-content:space-between;
        gap:1rem; border-bottom:1px solid var(--line); padding-bottom:.75rem;
        margin-bottom:.9rem;
      }
      .nika-title h1 {margin:0; font-size:1.85rem; letter-spacing:0;}
      .section-card {
        border:1px solid var(--line); border-radius:8px; padding:1rem;
        background:#fff; margin:.55rem 0 1rem;
      }
      .section-title {font-weight:800; font-size:.95rem; color:var(--ink); margin-bottom:.65rem;}
      .mini-grid {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.55rem;}
      .mini-item {
        border:1px solid var(--line); border-radius:8px; background:var(--soft);
        padding:.55rem .65rem; min-height:58px;
      }
      .mini-label {font-size:.7rem; color:var(--muted); font-weight:700;}
      .mini-value {font-size:.82rem; color:var(--ink); font-weight:750; overflow-wrap:anywhere;}
      .status-pill {
        display:inline-flex; align-items:center; gap:.45rem; padding:.36rem .65rem;
        border-radius:999px; border:1px solid #d7dee8; background:#f8fafc;
        font-weight:700; font-size:.78rem;
      }
      .status-dot {width:9px;height:9px;border-radius:999px;background:#8a98aa;}
      .status-running .status-dot {background:#059669;}
      .status-finished .status-dot {background:#2563eb;}
      .status-failed .status-dot {background:#dc2626;}
      .status-queued .status-dot {background:#d97706;}
      div[data-testid="stMetric"] {
        border:1px solid #d9e1ea; border-radius:8px; padding:.8rem; background:#ffffff;
      }
      section[data-testid="stSidebar"] {border-right:1px solid #e3e8ef;}
      section[data-testid="stSidebar"] .block-container {padding-top:1rem;}
      div[data-testid="stCheckbox"] label {font-weight:700;}
      .stButton > button {border-radius:8px; font-weight:800;}
      div[data-testid="stCodeBlock"] {
        border: 1px solid var(--line) !important;
        border-radius: 8px !important;
      }
      textarea {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
        font-size:.78rem !important;
      }
      /* Custom styling for multiselect tags */
      div[data-baseweb="tag"] {
        background-color: #eff6ff !important;
        border: 1px solid #bfdbfe !important;
        border-radius: 6px !important;
        color: #1e40af !important;
        padding: 2px 8px !important;
      }
      div[data-baseweb="tag"] span {
        color: #1e40af !important;
        font-weight: 600 !important;
      }
      div[data-baseweb="tag"] svg {
        fill: #1e40af !important;
      }
      @media (max-width: 900px) {.mini-grid {grid-template-columns:1fr 1fr;}}
    </style>
    """,
    unsafe_allow_html=True,
)


def _benchmark_path_from_name(value: str) -> Path:
    raw = value.strip() or Path(default_benchmark_csv_path()).stem
    path = Path(raw).expanduser()
    if path.is_absolute() or path.parent != Path("."):
        return path if path.suffix == ".csv" else path.with_suffix(".csv")
    name = raw[:-4] if raw.endswith(".csv") else raw
    return BENCHMARK_DIR / f"{name}.csv"


def _count_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return max(0, sum(1 for _ in path.open(encoding="utf-8")) - 1)
    except OSError:
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
    return runs[0] if runs else None


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
        grouped.setdefault(root, []).append(run_path)

    rows: list[dict[str, object]] = []
    for root, run_paths in sorted(
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
        failed_subs = 0
        submitted = 0
        finished = 0
        running = 0
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
            elif meta.get("status") == "running":
                running += 1
            if (session_dir / "submission.json").exists():
                submitted += 1
                det = _float(metrics.get("detection_score"))
                loc = _float(metrics.get("localization_accuracy"))
                rca = _float(metrics.get("rca_accuracy"))
                if (det is not None and det < 1.0) or (loc is not None and loc < 1.0) or (rca is not None and rca < 1.0):
                    failed_subs += 1

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
                result_modules.add("tool")
            if meta.get("memory_mode") and meta.get("memory_mode") != "off":
                result_modules.add(str(meta.get("memory_mode")))
            if meta.get("agent_type"):
                agents.add(str(meta["agent_type"]))
            if meta.get("model"):
                models.add(str(meta["model"]))
            updated = str(meta.get("updated_at") or meta.get("created_at") or updated)

        if not result_modules:
            result_modules.add("baseline")
        rows.append(
            {
                "result_root": root.name,
                "cases": len(run_paths),
                "finished": finished,
                "running": running,
                "submitted": submitted,
                "failed_subs": failed_subs,
                "detection": _avg(detections),
                "localization": _avg(localizations),
                "rca": _avg(rcas),
                "steps": _avg(steps),
                "tool_calls": _avg(tool_calls),
                "tool_errors": _sum(tool_errors),
                "token_in": _sum(in_tokens),
                "token_out": _sum(out_tokens),
                "duration": f"{int(sum(durations))}s" if durations else "-",
                "modules": ", ".join(sorted(result_modules)),
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
        if event["event"] == "evolve_generation_done":
            try:
                generation, total = event.get("generation", "0/1").split("/", 1)
                within_step = int(generation) / max(1, int(total))
                suffix = f"Generation {generation}/{total} completed"
            except ValueError:
                suffix = "Generation completed"
            label = f"{prefix}: {suffix}" if prefix else suffix
            break

    if not label:
        label = prefix if prefix else "Running"

    fraction = (max(0, step_index - 1) + within_step) / step_total
    return max(0.0, min(1.0, fraction)), label


st.markdown(
    """
    <div class="nika-title">
      <div><h1>NIKA Experiment Studio</h1></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">Configuration</div>', unsafe_allow_html=True)

# Grid layout for general configuration
col1, col2, col3, col4 = st.columns(4, gap="medium")
with col1:
    benchmark_name = st.text_input("Benchmark Name", value=Path(default_benchmark_csv_path()).stem)
    benchmark_path = _benchmark_path_from_name(benchmark_name)
    row_count = _count_rows(benchmark_path)
with col2:
    agent_type = st.selectbox("Agent", ["react", "plan-execute", "reflexion", "mock"])
with col3:
    llm_backend = st.selectbox("Backend", ["netmind", "openai", "deepseek", "ollama"])
with col4:
    model = st.text_input("Model", value="openai/gpt-oss-120b")

# Grid layout for execution parameters and modules selector
col5, col6, col7, col8 = st.columns(4, gap="medium")
with col5:
    max_steps = st.number_input("Steps", min_value=1, max_value=500, value=100)
with col6:
    max_attempts = st.number_input("Attempts", min_value=1, max_value=20, value=3)
with col7:
    parallel = st.number_input("Parallel", min_value=1, max_value=16, value=1)
with col8:
    modules_selected = st.multiselect(
        "Active Modules",
        options=["Tool Evolution", "Memory Evolution", "Agent Evolution"],
        default=[]
    )

tool_selected = "Tool Evolution" in modules_selected
memory_selected = "Memory Evolution" in modules_selected
agent_evolution_selected = "Agent Evolution" in modules_selected

modules = []
if tool_selected:
    modules.append("tool_evolution")
if memory_selected:
    modules.append("memory_evolution")
if agent_evolution_selected:
    modules.append("agent_evolution")

# Advanced configurations
if tool_selected or memory_selected or agent_evolution_selected:
    with st.expander("Advanced Module Settings", expanded=True):
        active_modules_count = sum([tool_selected, memory_selected, agent_evolution_selected])
        m_cols = st.columns(active_modules_count, gap="medium")
        col_idx = 0
        
        if tool_selected:
            with m_cols[col_idx]:
                st.markdown("**Tool Evolution**")
                tool_library_id = st.text_input("Tool library", value="tools-gptoss120-test")
                tool_mode = st.selectbox("Tool mode", ["dual", "mastery", "distill"])
            col_idx += 1
        else:
            tool_library_id = "tools-gptoss120-test"
            tool_mode = "dual"
            
        if memory_selected:
            with m_cols[col_idx]:
                st.markdown("**Memory Evolution**")
                memory_bank = st.text_input("Memory bank", value="memory-gptoss120-test")
                memory_k = st.number_input("Memory top-k", min_value=1, max_value=20, value=5)
                memory_tokens = st.number_input("Memory tokens", min_value=100, max_value=8000, value=1500, step=100)
                ensure_memory_services = st.checkbox("Start postgres/qdrant", value=True)
            col_idx += 1
        else:
            memory_bank = "memory-gptoss120-test"
            memory_k = 5
            memory_tokens = 1500
            ensure_memory_services = False
            
        if agent_evolution_selected:
            with m_cols[col_idx]:
                st.markdown("**Agent Evolution**")
                max_generations = st.number_input("Generations", min_value=1, max_value=20, value=3)
                feedback_mode = st.selectbox("Feedback mode", ["auto", "deterministic", "llm"])
                feedback_backend = st.text_input("Feedback backend", value=llm_backend)
                feedback_model = st.text_input("Feedback model", value=model)
            col_idx += 1
        else:
            max_generations = 3
            feedback_mode = "auto"
            feedback_backend = llm_backend
            feedback_model = model
else:
    tool_library_id = "tools-gptoss120-test"
    tool_mode = "dual"
    memory_bank = "memory-gptoss120-test"
    memory_k = 5
    memory_tokens = 1500
    ensure_memory_services = False
    max_generations = 3
    feedback_mode = "auto"
    feedback_backend = llm_backend
    feedback_model = model

# Evaluation Settings expander
with st.expander("Evaluation Settings", expanded=False):
    col_e1, col_e2, col_e3, col_e4 = st.columns(4, gap="medium")
    with col_e1:
        run_judge = st.checkbox("Run LLM judge", value=False)
    with col_e2:
        judge_backend = st.text_input("Judge backend", value=llm_backend)
    with col_e3:
        judge_model = st.text_input("Judge model", value=model)
    with col_e4:
        oracle_routing = st.checkbox("Oracle routing", value=False)

config = {
    "benchmark_file": str(benchmark_path),
    "modules": modules,
    "agent_type": agent_type,
    "llm_backend": llm_backend,
    "model": model,
    "max_steps": int(max_steps),
    "max_attempts": int(max_attempts),
    "parallel": int(parallel),
    "tool_library_id": tool_library_id,
    "tool_mode": tool_mode,
    "memory_bank": memory_bank,
    "memory_k": int(memory_k),
    "memory_tokens": int(memory_tokens),
    "ensure_memory_services": bool(ensure_memory_services),
    "max_generations": int(max_generations),
    "feedback_mode": feedback_mode,
    "feedback_backend": feedback_backend,
    "feedback_model": feedback_model,
    "run_judge": bool(run_judge),
    "judge_backend": judge_backend,
    "judge_model": judge_model,
    "oracle_routing": bool(oracle_routing),
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
    st.error("Benchmark CSV not found.")

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
    elif event == "evolve_generation_start":
        gen = ev.get("gen") or ev.get("generation") or "?"
        return f"**Starting Evolution Generation {gen}**"
    elif event == "evolve_generation_done":
        gen = ev.get("gen") or ev.get("generation") or "?"
        return f"**Completed Evolution Generation {gen}**"
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

if is_running:
    time.sleep(2)
    st.rerun()


st.markdown('<div class="section-title" style="margin-top: 1.5rem;">Results</div>', unsafe_allow_html=True)

# List and display all available experiment results automatically
result_rows = _result_rows(benchmark_name=None)
if not result_rows:
    import pandas as pd
    cols = [
        "result_root", "cases", "finished", "running", "submitted", "failed_subs",
        "detection", "localization", "rca", "steps", "tool_calls", "tool_errors",
        "token_in", "token_out", "duration", "modules", "agent", "model", "updated"
    ]
    df = pd.DataFrame(columns=cols)
    st.dataframe(df, width="stretch", hide_index=True)
else:
    st.dataframe(result_rows, width="stretch", hide_index=True)

