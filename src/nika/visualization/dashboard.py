"""Streamlit dashboard for NIKA troubleshooting sessions."""

from __future__ import annotations

import html
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
import streamlit as st
import importlib
import nika.visualization.data

importlib.reload(nika.visualization.data)

from nika.visualization.data import (  # noqa: E402
    discover_sessions,
    fault_endpoints,
    faulty_devices,
    load_session_bundle,
    parse_topology,
    replay_steps,
)
from nika.visualization.topology import render_topology_svg  # noqa: E402


st.set_page_config(
    page_title="NIKA · Network Arena",
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
        border: none !important;
        background: transparent !important;
        padding: 0px !important;
        box-shadow: none !important;
      }
      [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 1.5rem; background: transparent; border: none;
        border-bottom: 1px solid var(--line);
        border-radius: 0px; padding: 0px; margin: .7rem 0 1.2rem;
        display: flex; width: 100%;
      }
      [data-testid="stTabs"] [data-baseweb="tab"] {
        height: auto; border-radius: 0px; padding: 0.5rem 0;
        background: transparent !important;
        color: var(--muted);
        flex: 1; text-align: center; justify-content: center;
        border-bottom: 2px solid transparent !important;
      }
      [data-testid="stTabs"] [aria-selected="true"] {
        background: transparent !important; color: #0284c7;
        border-bottom: 2px solid #0284c7 !important;
      }
      [data-testid="stDataFrame"] {
        border: 1px solid var(--line); border-radius: 14px; overflow: hidden;
      }
      [data-testid="stExpander"] {
        border: 1px solid var(--line); border-radius: 13px;
        background: rgba(255, 255, 255, 0.7);
      }
      .stButton > button, .stDownloadButton > button {
        border-radius: 11px; border: 1px solid rgba(14, 165, 233, .28);
        background: rgba(14, 165, 233, .08);
        color: #0284c7;
      }
      .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--cyan); color: #0369a1;
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
      .hero-sub {color:var(--muted); font-size:.98rem; margin-top:.4rem;}
      .hero-meta {display:flex; flex-wrap:wrap; gap:.45rem; margin-top:1rem;}
      .chip {
        display:inline-flex; align-items:center; gap:.38rem; padding:.34rem .65rem;
        border:1px solid var(--line); border-radius:999px; color:#475569;
        background:rgba(255,255,255,.7); font-size:.76rem;
      }
      .dot {width:7px;height:7px;border-radius:50%;background:#94a3b8;}
      .dot.running {background:#10b981;box-shadow:0 0 12px rgba(16,185,129,.45);}
      .dot.finished {background:#3b82f6;}

      .section-label {
        color:#1e293b; font-weight:750; font-size:1rem; margin:.25rem 0 .9rem;
      }
      .glass-card {
        height:100%; border:1px solid var(--line); border-radius:16px;
        padding:1.15rem 1.2rem; background:var(--panel);
        box-shadow:0 14px 35px rgba(15, 23, 42, 0.04);
      }
      .card-kicker {
        color:var(--muted); font-size:.7rem; font-weight:800; letter-spacing:.11em;
        text-transform:uppercase; margin-bottom:.85rem;
      }
      .kv-grid {display:grid;grid-template-columns:minmax(95px,.72fr) 1.6fr;gap:.64rem .85rem;}
      .kv-grid {display:grid;grid-template-columns:minmax(95px,.72fr) 1.6fr;gap:.64rem .85rem;}
      .kv-key {color:var(--muted);font-size:.8rem;}
      .kv-value {color:#1e293b;font-size:.82rem;font-weight:600;overflow-wrap:anywhere;}
      .diagnosis-title {font-size:1.08rem;font-weight:800;margin-bottom:.85rem;color:#0f172a;}
      .diagnosis-row {display:flex;align-items:flex-start;gap:.7rem;margin:.62rem 0;}
      .diagnosis-icon {
        width:25px;height:25px;flex:0 0 25px;display:grid;place-items:center;
        border-radius:8px;font-size:.72rem;font-weight:900;
      }
      .diagnosis-label {color:var(--muted);font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;}
      .diagnosis-value {color:#1e293b;font-size:.86rem;font-weight:650;margin-top:.12rem;}
      .match-banner {
        margin-top:1rem;padding:.72rem .85rem;border-radius:11px;font-size:.8rem;font-weight:700;
      }
      .match-good {background:rgba(16,185,129,.1);color:#065f46;border:1px solid rgba(16,185,129,.2);}
      .match-bad {background:rgba(225,29,72,.09);color:#9f1239;border:1px solid rgba(225,29,72,.2);}
      .empty-state {
        padding:2.8rem 1rem;text-align:center;border:1px dashed rgba(15,23,42,.15);
        border-radius:16px;color:var(--muted);background:rgba(255,255,255,.5);
      }
      .legend-row {display:flex;flex-wrap:wrap;gap:.55rem;margin:.25rem 0 1rem;}
      .legend-chip {
        display:inline-flex;align-items:center;gap:.45rem;padding:.35rem .62rem;
        border:1px solid var(--line);border-radius:9px;color:#475569;font-size:.74rem;
        background:rgba(255,255,255,.65);
      }
      .legend-swatch {width:9px;height:9px;border-radius:50%;}
      .sidebar-note {
        margin-top:1.4rem;padding:1rem;border:1px solid var(--line);border-radius:13px;
        background:rgba(255,255,255,.58);color:var(--muted);font-size:.77rem;line-height:1.55;
      }
      .replay-card {
        border:1px solid var(--line);border-radius:16px;padding:1.15rem 1.25rem;
        background:linear-gradient(145deg,rgba(255,255,255,.95),rgba(241,245,249,.9));
        min-height:180px;
        box-shadow: 0 12px 35px rgba(15, 23, 42, 0.04);
      }
      .replay-top {display:flex;align-items:center;justify-content:space-between;gap:1rem;}
      .replay-kind {
        color:#0284c7;font-size:.69rem;font-weight:850;letter-spacing:.12em;text-transform:uppercase;
      }
      .replay-time {color:var(--muted);font-size:.72rem;}
      .replay-title {font-size:1.25rem;font-weight:800;margin:.55rem 0 .25rem;color:#0f172a;}
      .replay-agent {color:#475569;font-size:.82rem;}
      .device-chip {
        display:inline-flex;padding:.28rem .55rem;margin:.7rem .3rem 0 0;border-radius:8px;
        background:rgba(14,165,233,.1);border:1px solid rgba(14,165,233,.22);
        color:#0369a1;font-size:.72rem;font-weight:700;
      }
      .step-track {display:flex;gap:4px;margin:1rem 0 .4rem;overflow:hidden;}
      .step-segment {height:5px;flex:1;border-radius:99px;background:#e2e8f0;}
      .step-segment.done {background:#93c5fd;}
      .step-segment.active {background:#3b82f6;box-shadow:0 0 12px rgba(59,130,246,.5);}
      @media (max-width: 800px) {
        .block-container {padding:1.15rem .9rem 3rem;}
        .kv-grid {grid-template-columns:1fr;}
        .kv-key {margin-top:.3rem;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _safe(value: object, fallback: str = "—") -> str:
    if value is None or value == "" or value == []:
        return fallback
    if isinstance(value, list):
        value = ", ".join(str(item) for item in value)
    return html.escape(str(value))


def _fmt_score(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_time(value: object) -> str:
    if not value:
        return "—"
    raw = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(raw).strftime("%b %d, %Y · %H:%M:%S")
    except ValueError:
        return str(value)


def _session_label(session: dict[str, Any]) -> str:
    status = "●" if session.get("status") == "running" else "○"
    scenario = session.get("scenario_name") or "unknown"
    tier = session.get("scenario_topo_size")
    topology = f"{scenario}/{tier}" if tier else scenario
    return f"{status}  {topology}  ·  {session.get('session_id', '')}"


def _card(title: str, values: list[tuple[str, object]]) -> None:
    rows = "".join(
        f'<div class="kv-key">{html.escape(key)}</div><div class="kv-value">{_safe(value)}</div>'
        for key, value in values
    )
    st.markdown(
        f'<div class="glass-card"><div class="card-kicker">{html.escape(title)}</div>'
        f'<div class="kv-grid">{rows}</div></div>',
        unsafe_allow_html=True,
    )


def _diagnosis_card(title: str, payload: dict[str, Any], accent: str) -> None:
    anomaly = payload.get("is_anomaly")
    anomaly_text = "Anomaly detected" if anomaly else "No anomaly"
    devices = payload.get("faulty_devices") or []
    causes = payload.get("root_cause_name") or []
    st.markdown(
        f"""
        <div class="glass-card">
          <div class="card-kicker">{html.escape(title)}</div>
          <div class="diagnosis-title">{anomaly_text if payload else "Not available"}</div>
          <div class="diagnosis-row">
            <div class="diagnosis-icon" style="background:{accent}20;color:{accent}">N</div>
            <div><div class="diagnosis-label">Faulty devices</div>
            <div class="diagnosis-value">{_safe(devices)}</div></div>
          </div>
          <div class="diagnosis-row">
            <div class="diagnosis-icon" style="background:{accent}20;color:{accent}">R</div>
            <div><div class="diagnosis-label">Root cause</div>
            <div class="diagnosis-value">{_safe(causes)}</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _read_artifact_json(path: os.PathLike[str] | str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_artifact_text(
    path: os.PathLike[str] | str,
    *,
    max_lines: int = 500,
) -> str:
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    if len(lines) > max_lines:
        return "\n".join(
            [
                f"... truncated {len(lines) - max_lines} earlier lines ...",
                *lines[-max_lines:],
            ]
        )
    return "\n".join(lines)


def _module_labels(meta: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    if meta.get("tool_refinement_enabled"):
        labels.append("Tool Refinement")
    if (
        meta.get("procedural_memory_mode")
        and meta.get("procedural_memory_mode") != "off"
    ):
        labels.append("Procedural Memory")
    return labels or ["-"]


def _agent_label(meta: dict[str, Any]) -> str:
    return str(meta.get("agent_type") or "-")


def _render_empty() -> None:
    st.markdown(
        """
        <div class="empty-state">
          <div style="font-size:2rem;margin-bottom:.6rem">◇</div>
          <div style="color:#d8e4f2;font-weight:750">No troubleshooting sessions yet</div>
          <div style="margin-top:.4rem">Start one with <code>nika env run &lt;scenario&gt;</code>.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _move_replay(key: str, delta: int, maximum: int) -> None:
    current = int(st.session_state.get(key, 0))
    st.session_state[key] = min(maximum, max(0, current + delta))


def _set_replay(key: str, value: int) -> None:
    st.session_state[key] = value


def _render_dashboard() -> None:
    from pathlib import Path
    from nika.config import RESULTS_DIR

    sessions = discover_sessions()
    selected_id: str | None = None

    if not sessions:
        st.markdown(
            '<div class="eyebrow">Network troubleshooting observatory</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="nika-brand" style="margin-bottom: 2rem;">
              <div class="nika-mark">N</div>
              <div>
                <div class="nika-brand-title" style="font-size: 1.5rem; line-height: 1.1;">NIKA</div>
                <div class="nika-brand-sub" style="font-size: 0.9rem; letter-spacing: 0.12em;">NETWORK ARENA</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.title("See the network think.")
        _render_empty()
        return

    def get_group_name(session: dict[str, Any]) -> str:
        sess_dir_str = session.get("session_dir")
        if not sess_dir_str:
            return "Runtime / Standalone"
        try:
            s_dir = Path(sess_dir_str).resolve()
            res_dir = RESULTS_DIR.resolve()
            if s_dir.is_relative_to(res_dir):
                rel = s_dir.relative_to(res_dir)
                if len(rel.parts) > 1:
                    top_dir = rel.parts[0]
                    for part in rel.parts[1:-1]:
                        if part.startswith("gen_") and part[4:].isdigit():
                            return f"{top_dir}/{part}"
                    return top_dir
            return "Runtime / Standalone"
        except Exception:
            return "Runtime / Standalone"

    groups: dict[str, list[dict[str, Any]]] = {}
    for sess in sessions:
        gname = get_group_name(sess)
        groups.setdefault(gname, []).append(sess)

    def get_newest_session_time(group_sessions: list[dict[str, Any]]) -> str:
        return max(
            str(s.get("created_at") or s.get("session_id") or "")
            for s in group_sessions
        )

    sorted_group_names = sorted(
        groups.keys(),
        key=lambda gname: get_newest_session_time(groups[gname]),
        reverse=True,
    )

    requested_id = os.environ.get("NIKA_VISUALIZE_SESSION_ID")
    default_group_index = 0
    default_session_index = 0
    if requested_id:
        for g_idx, gname in enumerate(sorted_group_names):
            for s_idx, sess in enumerate(groups[gname]):
                if str(sess.get("session_id")) == requested_id:
                    default_group_index = g_idx
                    default_session_index = s_idx
                    break

    st.markdown(
        '<div class="eyebrow">Network troubleshooting observatory</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="nika-brand" style="margin-bottom: 2rem;">
          <div class="nika-mark">N</div>
          <div>
            <div class="nika-brand-title" style="font-size: 1.5rem; line-height: 1.1;">NIKA</div>
            <div class="nika-brand-sub" style="font-size: 0.9rem; letter-spacing: 0.12em;">NETWORK ARENA</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(
        [1.0, 1.3, 0.4], gap="medium", vertical_alignment="bottom"
    )
    with col1:
        selected_group = st.selectbox(
            "Experiment group",
            sorted_group_names,
            index=default_group_index,
        )

    group_sessions = groups[selected_group]
    session_ids = [str(item["session_id"]) for item in group_sessions]

    with col2:
        selected_id = st.selectbox(
            "Active session",
            session_ids,
            index=default_session_index
            if selected_group == sorted_group_names[default_group_index]
            else 0,
            format_func=lambda session_id: _session_label(
                next(
                    item for item in group_sessions if item["session_id"] == session_id
                )
            ),
        )

    with col3:
        if st.button("↻  Refresh", use_container_width=True):
            st.rerun()

    assert selected_id is not None
    bundle = load_session_bundle(selected_id)
    meta = bundle.meta
    metrics = bundle.metrics or meta.get("eval_metrics", {})
    problems = meta.get("problem_names") or []
    pairs = parse_topology(meta)
    nodes = {endpoint.split(":", 1)[0] for pair in pairs for endpoint in pair}

    st.write("")

    st.write("")
    metric_cols = st.columns(6)
    headline_metrics = [
        (
            "Detection",
            _fmt_score(metrics.get("detection_score")),
            "Anomaly classification",
        ),
        (
            "Localization F1",
            _fmt_score(metrics.get("localization_f1")),
            "Faulty device accuracy",
        ),
        ("RCA F1", _fmt_score(metrics.get("rca_f1")), "Root-cause accuracy"),
        ("Tool calls", metrics.get("tool_calls", "—"), "MCP invocations"),
        ("Tool errors", metrics.get("tool_errors", "—"), "Failed invocations"),
        ("Steps", metrics.get("steps", "—"), "Agent reasoning steps"),
    ]
    for column, (label, value, help_text) in zip(metric_cols, headline_metrics):
        column.metric(label, value, help=help_text)

    tab_names = ["Overview", "Topology", "Replay"]
    tabs = dict(zip(tab_names, st.tabs(tab_names)))
    overview_tab = tabs["Overview"]
    topology_tab = tabs["Topology"]
    replay_tab = tabs["Replay"]

    with overview_tab:
        st.markdown(
            '<div class="section-label">Topology Configuration</div>',
            unsafe_allow_html=True,
        )
        _card(
            "Topology Configuration",
            [
                ("Scenario Name", meta.get("scenario_name")),
                ("Topology Size / Tier", meta.get("scenario_topo_size") or "Fixed"),
                ("Lab Instance Name", meta.get("lab_name")),
                ("Devices Count", f"{len(nodes)} devices"),
                ("Links Count", f"{len(pairs)} links"),
            ],
        )

        st.write("")
        st.markdown(
            '<div class="section-label">Failure Injections</div>',
            unsafe_allow_html=True,
        )
        if bundle.failure_injections:
            for idx, injection in enumerate(bundle.failure_injections):
                _card(
                    f"Injection {idx + 1}: {injection.get('problem_name')}",
                    [
                        ("Category", injection.get("root_cause_category")),
                        ("Status", injection.get("status")),
                        ("Started", _fmt_time(injection.get("created_at"))),
                        (
                            "Parameters",
                            json.dumps(
                                injection.get("injection_params") or {},
                                ensure_ascii=False,
                                default=str,
                            ),
                        ),
                    ],
                )
        elif problems:
            st.info(
                "Detailed injection records are unavailable; problem names remain in the run metadata."
            )
        else:
            _render_empty()

        st.write("")
        st.markdown(
            '<div class="section-label">Agent Configuration</div>',
            unsafe_allow_html=True,
        )
        _card(
            "Agent Configuration",
            [
                ("Agent Baseline", _agent_label(meta)),
                ("Learning Modules", _module_labels(meta)),
                ("LLM Backend Provider", meta.get("llm_backend")),
                ("LLM Model Name", meta.get("model")),
                ("Execution Started", _fmt_time(meta.get("start_time"))),
                ("Execution Completed", _fmt_time(meta.get("end_time"))),
            ],
        )

        st.write("")
        st.markdown(
            '<div class="section-label">Final Diagnosis</div>', unsafe_allow_html=True
        )
        truth_col, prediction_col = st.columns(2, gap="medium")
        with truth_col:
            _diagnosis_card("Ground truth", bundle.ground_truth, "#ff647c")
        with prediction_col:
            _diagnosis_card("Agent submission", bundle.submission, "#ffb454")

    with topology_tab:
        topo_left, topo_right = st.columns([1.2, 0.8], gap="large")
        with topo_left:
            st.markdown(
                """
                <div class="legend-row">
                  <span class="legend-chip"><span class="legend-swatch" style="background:#ff647c"></span>Ground truth</span>
                  <span class="legend-chip"><span class="legend-swatch" style="background:#ffb454"></span>Agent prediction</span>
                  <span class="legend-chip"><span class="legend-swatch" style="background:#a78bfa"></span>Both</span>
                  <span class="legend-chip"><span style="color:#ff647c">┄</span>Fault interface</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            svg = render_topology_svg(
                pairs,
                actual_faulty=faulty_devices(bundle.ground_truth),
                predicted_faulty=faulty_devices(bundle.submission),
                fault_interfaces=fault_endpoints(bundle),
            )
            st.html(svg, width="stretch")
            st.caption(
                "Hover nodes and links for endpoint details. Colors compare truth with agent localization."
            )

        with topo_right:
            st.markdown(
                '<div class="section-label" style="margin-top:0;">Network Directory</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div style="font-weight:700; font-size:0.95rem; margin-bottom:0.6rem; color:var(--ink);">Devices</div>',
                unsafe_allow_html=True,
            )

            actual_f = faulty_devices(bundle.ground_truth) or set()
            pred_f = faulty_devices(bundle.submission) or set()

            def get_node_type(name: str) -> str:
                n_lower = name.lower()
                if "router" in n_lower or "rtr" in n_lower or n_lower.startswith("r"):
                    return "Router"
                if "switch" in n_lower or "sw" in n_lower or n_lower.startswith("s"):
                    return "Switch"
                if "host" in n_lower or n_lower.startswith("h"):
                    return "Host"
                if "server" in n_lower or "srv" in n_lower or n_lower.startswith("d"):
                    return "Server"
                return "Controller"

            sorted_nodes = sorted(nodes)
            device_items = []
            for node in sorted_nodes:
                kind = get_node_type(node)
                status_color = "#10b981"
                status_text = "Healthy"
                badge_style = ""

                is_actual = node in actual_f
                is_pred = node in pred_f

                if is_actual and is_pred:
                    status_color = "#a78bfa"
                    status_text = "Failed (Matched)"
                    badge_style = "background: rgba(167, 139, 250, 0.1); border: 1px solid rgba(167, 139, 250, 0.3);"
                elif is_actual:
                    status_color = "#ff647c"
                    status_text = "Failed (Ground Truth)"
                    badge_style = "background: rgba(255, 100, 124, 0.1); border: 1px solid rgba(255, 100, 124, 0.3);"
                elif is_pred:
                    status_color = "#ffb454"
                    status_text = "Predicted Failure"
                    badge_style = "background: rgba(255, 180, 84, 0.1); border: 1px solid rgba(255, 180, 84, 0.3);"
                else:
                    badge_style = "background: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.2);"

                symbol = kind[0]

                device_html = f"""
                <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.5rem 0.75rem; margin-bottom: 0.4rem; border-radius: 8px; {badge_style}">
                  <div style="display: flex; align-items: center; gap: 0.6rem;">
                    <div style="width: 24px; height: 24px; border-radius: 6px; display: grid; place-items: center; background: rgba(15, 23, 42, 0.05); color: var(--ink); font-weight: 800; font-size: 0.78rem;">
                      {symbol}
                    </div>
                    <div>
                      <div style="font-weight: 650; font-size: 0.88rem; color: var(--ink); line-height: 1.2;">{node}</div>
                      <div style="font-size: 0.75rem; color: var(--muted); line-height: 1.1;">{kind}</div>
                    </div>
                  </div>
                  <div style="display: flex; align-items: center; gap: 0.4rem;">
                    <span style="width: 7px; height: 7px; border-radius: 50%; background: {status_color}; display: inline-block;"></span>
                    <span style="font-size: 0.75rem; font-weight: 550; color: var(--muted);">{status_text}</span>
                  </div>
                </div>
                """
                device_items.append(device_html)

            st.markdown(
                f'<div style="max-height: 250px; overflow-y: auto; padding-right: 0.2rem;">{"".join(device_items)}</div>',
                unsafe_allow_html=True,
            )
            st.write("")
            st.markdown(
                '<div style="font-weight:700; font-size:0.95rem; margin-bottom:0.6rem; color:var(--ink);">Network Links</div>',
                unsafe_allow_html=True,
            )

            link_items = []
            f_interfaces = fault_endpoints(bundle) or set()

            for left_ep, right_ep in pairs:
                left, left_intf = (
                    left_ep.split(":", 1) if ":" in left_ep else (left_ep, "")
                )
                right, right_intf = (
                    right_ep.split(":", 1) if ":" in right_ep else (right_ep, "")
                )

                is_fault = (left, left_intf) in f_interfaces or (
                    right,
                    right_intf,
                ) in f_interfaces

                link_style = ""
                status_color = "#64748b"
                status_text = "Active"
                if is_fault:
                    link_style = "background: rgba(255, 100, 124, 0.1); border: 1px solid rgba(255, 100, 124, 0.3);"
                    status_color = "#ff647c"
                    status_text = "Faulty"
                else:
                    link_style = "background: rgba(15, 23, 42, 0.02); border: 1px solid var(--line);"

                link_html = f"""
                <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.5rem 0.75rem; margin-bottom: 0.4rem; border-radius: 8px; {link_style}">
                  <div style="display: flex; flex-direction: column;">
                    <div style="display: flex; align-items: center; gap: 0.4rem;">
                      <span style="font-weight: 600; font-size: 0.82rem; color: var(--ink);">{left}</span>
                      <span style="color: var(--muted); font-size: 0.7rem;">({left_intf})</span>
                      <span style="color: var(--muted); font-size: 0.8rem;">↔</span>
                      <span style="font-weight: 600; font-size: 0.82rem; color: var(--ink);">{right}</span>
                      <span style="color: var(--muted); font-size: 0.7rem;">({right_intf})</span>
                    </div>
                  </div>
                  <div style="display: flex; align-items: center; gap: 0.4rem;">
                    <span style="width: 7px; height: 7px; border-radius: 50%; background: {status_color}; display: inline-block;"></span>
                    <span style="font-size: 0.75rem; font-weight: 550; color: var(--muted);">{status_text}</span>
                  </div>
                </div>
                """
                link_items.append(link_html)

            st.markdown(
                f'<div style="max-height: 220px; overflow-y: auto; padding-right: 0.2rem;">{"".join(link_items)}</div>',
                unsafe_allow_html=True,
            )

    with replay_tab:
        steps = replay_steps(bundle, pairs)
        if not steps:
            _render_empty()
        else:
            slider_key = f"replay_slider_{selected_id}"
            st.session_state.setdefault(slider_key, 0)
            st.session_state[slider_key] = min(
                int(st.session_state[slider_key]), len(steps) - 1
            )

            previous_col, counter_col, next_col, latest_col = st.columns(
                [0.7, 1.6, 0.7, 0.9], vertical_alignment="center"
            )
            with previous_col:
                if st.button(
                    "← Previous",
                    disabled=st.session_state[slider_key] == 0,
                    width="stretch",
                    on_click=_move_replay,
                    args=(slider_key, -1, len(steps) - 1),
                ):
                    pass
            with counter_col:
                st.slider(
                    "Replay position",
                    min_value=0,
                    max_value=len(steps) - 1,
                    format=f"Step %d / {len(steps)}",
                    label_visibility="collapsed",
                    key=slider_key,
                )
            with next_col:
                if st.button(
                    "Next →",
                    disabled=st.session_state[slider_key] >= len(steps) - 1,
                    width="stretch",
                    on_click=_move_replay,
                    args=(slider_key, 1, len(steps) - 1),
                ):
                    pass
            with latest_col:
                if st.button(
                    "Latest",
                    width="stretch",
                    on_click=_set_replay,
                    args=(slider_key, len(steps) - 1),
                ):
                    pass

            step_index = st.session_state[slider_key]
            step = steps[step_index]
            inspected = {
                device
                for previous_step in steps[: step_index + 1]
                for device in previous_step.devices
            }
            segments = "".join(
                '<span class="step-segment active"></span>'
                if index == step_index
                else '<span class="step-segment done"></span>'
                if index < step_index
                else '<span class="step-segment"></span>'
                for index in range(len(steps))
            )
            device_chips = (
                "".join(
                    f'<span class="device-chip">{html.escape(device)}</span>'
                    for device in step.devices
                )
                or '<span class="device-chip" style="opacity:.6">No device referenced</span>'
            )
            st.markdown(
                f"""
                <div class="step-track">{segments}</div>
                <div class="replay-card">
                  <div class="replay-top">
                    <span class="replay-kind">{html.escape(step.kind)}</span>
                    <span class="replay-time">{html.escape(_fmt_time(step.timestamp))}</span>
                  </div>
                  <div class="replay-title">{html.escape(step.title)}</div>
                  <div class="replay-agent">{html.escape(step.agent)} · step {step_index + 1} of {len(steps)}</div>
                  <div>{device_chips}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            replay_left, replay_right = st.columns([1.25, 0.75], gap="medium")
            with replay_left:
                replay_svg = render_topology_svg(
                    pairs,
                    inspected_devices=inspected,
                    active_devices=set(step.devices),
                )
                st.html(replay_svg, width="stretch")
                st.caption(
                    "Cyan glow marks devices referenced in this step; blue marks devices inspected earlier."
                )
            with replay_right:
                if step.input:
                    st.markdown("### Input")
                    st.code(step.input, language="json")
                if step.output:
                    st.markdown("### Output")
                    st.code(step.output, language="text")
                if not step.input and not step.output:
                    st.info("This trace step contains lifecycle metadata only.")
                with st.expander("Raw event"):
                    st.json(step.raw, expanded=False)


_render_dashboard()
