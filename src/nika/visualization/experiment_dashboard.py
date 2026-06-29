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
    initial_sidebar_state="expanded",
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
      div[data-testid="stMetric"] {
        border:1px solid #d9e1ea; border-radius:8px; padding:.8rem; background:#ffffff;
      }
      section[data-testid="stSidebar"] {border-right:1px solid #e3e8ef;}
      section[data-testid="stSidebar"] .block-container {padding-top:1rem;}
      div[data-testid="stCheckbox"] label {font-weight:700;}
      .stButton > button {border-radius:8px; font-weight:800;}
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
            for key, target in (
                ("detection_score", detections),
                ("localization_accuracy", localizations),
                ("rca_accuracy", rcas),
                ("steps", steps),
                ("tool_calls", tool_calls),
            ):
                value = _float(metrics.get(key))
                if value is not None:
                    target.append(value)
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
                "detection": _avg(detections),
                "localization": _avg(localizations),
                "rca": _avg(rcas),
                "steps": _avg(steps),
                "tool_calls": _avg(tool_calls),
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
    label = f"Step {step_index}/{step_total}"
    for event in reversed(events):
        if event["event"] == "benchmark_progress":
            raw_completed = event.get("completed", "0")
            try:
                if "/" in raw_completed:
                    done, total = raw_completed.split("/", 1)
                    within_step = int(done) / max(1, int(total))
                elif event.get("index") and "/" in event["index"]:
                    _, total = event["index"].split("/", 1)
                    within_step = int(raw_completed) / max(1, int(total))
                label = f"{label} - benchmark {event.get('completed', '-')}"
            except ValueError:
                pass
            break
        if event["event"] == "benchmark_summary":
            within_step = 1.0
            label = f"{label} - benchmark done"
            break
        if event["event"] == "evolve_generation_done":
            try:
                generation, total = event.get("generation", "0/1").split("/", 1)
                within_step = int(generation) / max(1, int(total))
                label = f"{label} - generation {generation}/{total}"
            except ValueError:
                pass
            break

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

with st.sidebar:
    st.header("Benchmark")
    benchmark_name = st.text_input("Name", value=Path(default_benchmark_csv_path()).stem)
    benchmark_path = _benchmark_path_from_name(benchmark_name)
    row_count = _count_rows(benchmark_path)


    st.header("Baseline")
    col_a, col_b = st.columns(2)
    with col_a:
        agent_type = st.selectbox("Agent", ["react", "plan-execute", "reflexion", "mock"])
    with col_b:
        llm_backend = st.selectbox("Backend", ["netmind", "openai", "deepseek", "ollama"])
    model = st.text_input("Model", value="openai/gpt-oss-120b")
    col_c, col_d, col_e = st.columns(3)
    with col_c:
        max_steps = st.number_input("Steps", min_value=1, max_value=500, value=100)
    with col_d:
        max_attempts = st.number_input("Attempts", min_value=1, max_value=20, value=3)
    with col_e:
        parallel = st.number_input("Parallel", min_value=1, max_value=16, value=1)

    st.header("Modules")
    tool_selected = st.checkbox("Tool Evolution", value=False)
    memory_selected = st.checkbox("Memory Evolution", value=False)
    agent_evolution_selected = st.checkbox("Agent Evolution", value=False)

    modules = []
    if tool_selected:
        modules.append("tool_evolution")
    if memory_selected:
        modules.append("memory_evolution")
    if agent_evolution_selected:
        modules.append("agent_evolution")

    if tool_selected:
        st.subheader("Tool Evolution")
        tool_library_id = st.text_input("Tool library", value="tools-gptoss120-test")
        tool_mode = st.selectbox("Tool mode", ["dual", "mastery", "distill"])
    else:
        tool_library_id = "tools-gptoss120-test"
        tool_mode = "dual"

    if memory_selected:
        st.subheader("Memory")
        memory_bank = st.text_input("Memory bank", value="memory-gptoss120-test")
        memory_k = st.number_input("Memory top-k", min_value=1, max_value=20, value=5)
        memory_tokens = st.number_input("Memory tokens", min_value=100, max_value=8000, value=1500, step=100)
        ensure_memory_services = st.checkbox("Start postgres/qdrant", value=True)
    else:
        memory_bank = "memory-gptoss120-test"
        memory_k = 5
        memory_tokens = 1500
        ensure_memory_services = False

    if agent_evolution_selected:
        st.subheader("Agent Evolution")
        max_generations = st.number_input("Generations", min_value=1, max_value=20, value=3)
        feedback_mode = st.selectbox("Feedback mode", ["auto", "deterministic", "llm"])
        feedback_backend = st.text_input("Feedback backend", value=llm_backend)
        feedback_model = st.text_input("Feedback model", value=model)
    else:
        max_generations = 3
        feedback_mode = "auto"
        feedback_backend = llm_backend
        feedback_model = model

    st.header("Evaluation")
    run_judge = st.checkbox("Run LLM judge", value=False)
    judge_backend = st.text_input("Judge backend", value=llm_backend)
    judge_model = st.text_input("Judge model", value=model)
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

st.markdown('<div class="section-title">Current Config</div>', unsafe_allow_html=True)
mode_text = experiment_label(config)
st.markdown(
    f"""
    <div class="mini-grid">
      <div class="mini-item"><div class="mini-label">Benchmark</div><div class="mini-value">{benchmark_path.stem}</div></div>
      <div class="mini-item"><div class="mini-label">Rows</div><div class="mini-value">{row_count if row_count is not None else '-'}</div></div>
      <div class="mini-item"><div class="mini-label">Agent</div><div class="mini-value">{agent_type}</div></div>
      <div class="mini-item"><div class="mini-label">Backend</div><div class="mini-value">{llm_backend}</div></div>
      <div class="mini-item"><div class="mini-label">Model</div><div class="mini-value">{model}</div></div>
      <div class="mini-item"><div class="mini-label">Steps</div><div class="mini-value">{int(max_steps)}</div></div>
      <div class="mini-item"><div class="mini-label">Mode</div><div class="mini-value">{mode_text}</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Renders the commands directly under the Current Config grid with a top spacing
st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
for item in plan:
    st.code(shlex.join(item.command), language="bash")

# Actions Section (rendered on a separate horizontal row containing all 4 buttons)
st.markdown('<div class="section-title" style="margin-top: 1.5rem;">Actions</div>', unsafe_allow_html=True)
action_cols = st.columns(4, gap="medium")
with action_cols[0]:
    # Clear Docker Button
    if st.button("Clear Docker", key="btn_clear_docker", width="stretch"):
        import subprocess
        try:
            res = subprocess.run(["docker", "compose", "down", "-v"], cwd="/home/ngthanhhai/projects/nika", capture_output=True, text=True)
            if res.returncode == 0:
                st.success("Docker services cleared successfully!")
            else:
                st.error(f"Failed to clear Docker: {res.stderr or res.stdout}")
        except Exception as e:
            st.error(f"Error: {e}")

with action_cols[1]:
    # Clear Memory Button
    if st.button("Clear Memory Bank", key="btn_clear_mem", width="stretch"):
        import socket
        postgres_running = False
        try:
            with socket.create_connection(("127.0.0.1", 5432), timeout=1.0):
                postgres_running = True
        except OSError:
            pass

        if not postgres_running:
            st.info(f"Memory database (Postgres) is not running, so bank '{config['memory_bank']}' is already empty/cleared!")
        else:
            import sys
            import subprocess
            try:
                res = subprocess.run([
                    sys.executable, "-m", "nika.codex_cli.main", 
                    "memory", "clear", "--bank", config["memory_bank"], "-y"
                ], cwd="/home/ngthanhhai/projects/nika", capture_output=True, text=True)
                if res.returncode == 0:
                    st.success(f"Cleared memory bank '{config['memory_bank']}'!")
                else:
                    st.error(f"Failed to clear memory bank: {res.stderr or res.stdout}")
            except Exception as e:
                st.error(f"Error: {e}")

with action_cols[2]:
    # Clear Tool Library Button
    if st.button("Clear Tool Library", key="btn_clear_tools", width="stretch"):
        from nika.config import TOOL_EVOLUTION_DIR
        lib_dir = TOOL_EVOLUTION_DIR / config["tool_library_id"]
        if not lib_dir.exists():
            st.info(f"Tool library '{config['tool_library_id']}' is already empty/cleared!")
        else:
            import sys
            import subprocess
            try:
                res = subprocess.run([
                    sys.executable, "-m", "nika.codex_cli.main", 
                    "tools", "reset", config["tool_library_id"], "-y"
                ], cwd="/home/ngthanhhai/projects/nika", capture_output=True, text=True)
                if res.returncode == 0:
                    st.success(f"Cleared tool library '{config['tool_library_id']}'!")
                else:
                    st.error(f"Failed to clear tool library: {res.stderr or res.stdout}")
            except Exception as e:
                st.error(f"Error: {e}")

with action_cols[3]:
    # Run / Stop Button
    active_run = _selected_run_dir()
    is_running = False
    if active_run is not None:
        active_status = run_status(active_run)
        if active_status.get("status") == "running":
            is_running = True

    if is_running:
        if st.button("Stop", type="primary", width="stretch"):
            stop_run(active_run)
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
    run_labels = [path.name for path in runs]
    selected_label = selected.name if selected is not None else None
    selected_label = st.selectbox(
        "Run history",
        options=run_labels,
        index=run_labels.index(selected_label) if selected_label in run_labels else 0,
    )
    selected = next(path for path in runs if path.name == selected_label)
    st.session_state["active_run_dir"] = str(selected)
else:
    selected = None

if selected is not None:
    status = run_status(selected)
    log_text = read_run_log(selected)
    spec = read_run_spec(selected)
    events = parse_progress_events(log_text)
    fraction, progress_label = _progress_fraction(events, len(spec.get("commands") or []))



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
        else:
            st.info("No progress events yet.")
            
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
            key=f"log-{selected.name}",
        )

    if status.get("status") == "running":
        time.sleep(2)
        st.rerun()


st.markdown('<div class="section-title" style="margin-top: 1.5rem;">Results</div>', unsafe_allow_html=True)

# List all available result directories for comparison & management
all_dirs = []
if RESULTS_DIR.exists():
    for run_path in RESULTS_DIR.rglob("run.json"):
        if "0_summary" in run_path.relative_to(RESULTS_DIR).parts:
            continue
        root = _top_result_root(run_path)
        all_dirs.append(root.name)
all_dirs = sorted(list(set(all_dirs)))

# Premium Toolbar UIUX for selection
if "compare_dirs" not in st.session_state:
    st.session_state["compare_dirs"] = all_dirs
else:
    # Auto-add newly created/imported folders to the selection
    current_stored = set(st.session_state["compare_dirs"])
    new_dirs = [d for d in all_dirs if d not in current_stored]
    if new_dirs:
        st.session_state["compare_dirs"] = list(current_stored) + new_dirs

# Clean up deleted folders from selection
st.session_state["compare_dirs"] = [d for d in st.session_state["compare_dirs"] if d in all_dirs]

# Callbacks for programmatic selection to avoid state errors
def select_all_dirs():
    st.session_state["compare_dirs"] = all_dirs

def clear_all_dirs():
    st.session_state["compare_dirs"] = []

col_tool_a, col_tool_b, col_tool_c = st.columns([0.76, 0.12, 0.12], gap="small")
with col_tool_a:
    selected_compare_dirs = st.multiselect(
        "Select experiments to compare (Add/Remove from view)",
        options=all_dirs,
        key="compare_dirs",
        placeholder="Choose experiment folders...",
        label_visibility="collapsed",
    )
with col_tool_b:
    st.button("Select All", key="btn_select_all", on_click=select_all_dirs, width="stretch")
with col_tool_c:
    st.button("Clear All", key="btn_clear_all", on_click=clear_all_dirs, width="stretch")

col_act_a, col_act_b = st.columns(2)
with col_act_a:
    with st.expander("Import external experiment folder", expanded=False):
        import_path = st.text_input("Source folder path", placeholder="/path/to/experiment_run")
        if st.button("Import Folder", width="stretch"):
            if import_path:
                src_path = Path(import_path).expanduser().resolve()
                if src_path.exists() and src_path.is_dir():
                    # Sanity check: must contain run.json or have subfolders with run.json
                    if (src_path / "run.json").exists() or any(src_path.rglob("run.json")):
                        import shutil
                        dest_path = RESULTS_DIR / src_path.name
                        if dest_path.exists():
                            st.error(f"Folder '{src_path.name}' already exists in results.")
                        else:
                            try:
                                shutil.copytree(src_path, dest_path)
                                st.success(f"Imported '{src_path.name}' successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to import: {e}")
                    else:
                        st.error("Target directory is not a valid experiment (must contain run.json).")
                else:
                    st.error("Invalid path or not a directory.")
with col_act_b:
    with st.expander("Delete experiment from disk", expanded=False):
        dir_to_delete = st.selectbox("Folder to delete", options=[""] + all_dirs)
        if dir_to_delete:
            st.warning(f"This will permanently delete '{dir_to_delete}'.")
            if st.button("Delete Permanently", type="primary", width="stretch"):
                import shutil
                path_to_del = RESULTS_DIR / dir_to_delete
                if path_to_del.exists():
                    try:
                        shutil.rmtree(path_to_del)
                        st.success(f"Deleted '{dir_to_delete}' successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete: {e}")

result_rows = _result_rows(benchmark_name=None)
if result_rows:
    # Filter by user multiselect choice
    if selected_compare_dirs:
        result_rows = [row for row in result_rows if row["result_root"] in selected_compare_dirs]
    else:
        result_rows = []
    
    if result_rows:
        st.dataframe(result_rows, width="stretch", hide_index=True)
    else:
        st.info("No selected experiments to display.")

